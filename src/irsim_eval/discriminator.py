"""Real versus synthetic as a classification problem: the gap score (ME.6, §15 T4/T5).

If a classifier can tell a rendered patch from a real one, the difference it is using is a gap in
the simulator, and the area under its ROC curve measures how large that gap is: 0.5 is
indistinguishable, 1.0 is trivially separable. ADR 0068 sets the Tier 4 target at **AUC < 0.70**.

**This is a linear probe on interpretable statistics, and that is a deliberate lower bound.**
The features are the same display-domain quantities the rest of Tier 4 is built on -- noise scale,
spectral slope, blockiness, histogram shape, edge energy -- and the classifier is logistic
regression over them. A convolutional network would find more, and would find it in a way nobody
could read. Two consequences follow and both are the point:

* an AUC near 0.5 here means *these statistics* do not separate the two sets. It does not mean a
  detector cannot. ME.7's transfer protocols are the test that answers that question.
* an AUC near 1.0 here names the feature that did it, so the gap is actionable the same day.

Nothing here needs torch or scikit-learn: the whole model is a few lines of NumPy, it is
deterministic given its seed, and a Tier 4 report should not be blocked on a 2 GB install.

docs/physics-model.md §15 T4, T5; ADR 0068
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.validation.compare import auc_from_scores

__all__ = [
    "FEATURE_NAMES",
    "DiscriminatorResult",
    "patch_features",
    "feature_matrix",
    "LinearProbe",
    "gap_score",
]

#: In the order :func:`patch_features` returns them. Kept as data so a result can name the feature
#: a separation came from rather than reporting a number with no handle on it.
FEATURE_NAMES: tuple[str, ...] = (
    "noise_scale",
    "psd_slope",
    "blockiness_z",
    "mean",
    "std",
    "skew",
    "kurtosis",
    "gradient_median",
    "gradient_p95_over_median",
    "lag1_autocorrelation",
)


def patch_features(patch: Any) -> NDArray[np.float64]:
    """One feature vector for one 2-D patch, in :data:`FEATURE_NAMES` order.

    ``mean`` and ``std`` are included **even though they are not scale-free**, and that is a
    decision rather than an oversight: an unknown AGC makes them untrustworthy as *physics*, but a
    discriminator's job is to find any separation at all, and if the two sets differ in level that
    is a real difference somebody should see. The named features are what tell a reader whether a
    separation is physical or an artefact of the display mapping.
    """
    from irsim.validation.codec import blockiness
    from irsim.validation.flat import robust_noise_scale
    from irsim.validation.targets import clutter_slope

    image = np.asarray(patch, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 8:
        raise ValueError(f"a patch must be 2-D and at least 8x8, got {image.shape}")
    centred = image - image.mean()
    sigma = float(image.std())
    normalised = centred / sigma if sigma > 0.0 else centred
    grad = np.hypot(*np.gradient(image))
    grad_median = float(np.median(grad))
    flat = normalised.ravel()
    return np.array(
        [
            robust_noise_scale(image),
            clutter_slope(image),
            float(max(blockiness(image[None, ...]).z_h, blockiness(image[None, ...]).z_v)),
            float(image.mean()),
            sigma,
            float(np.mean(flat**3)),
            float(np.mean(flat**4) - 3.0),
            grad_median,
            float(np.percentile(grad, 95) / grad_median) if grad_median > 0.0 else 0.0,
            float(np.mean(flat[:-1] * flat[1:])) if flat.size > 1 else 0.0,
        ],
        dtype=np.float64,
    )


def feature_matrix(patches: Any) -> NDArray[np.float64]:
    """``(n, len(FEATURE_NAMES))`` for a sequence of patches."""
    rows = [patch_features(p) for p in patches]
    if not rows:
        raise ValueError("no patches")
    return np.asarray(rows, dtype=np.float64)


@dataclass
class LinearProbe:
    """Logistic regression by full-batch gradient descent, with standardised features.

    Standardisation is not cosmetic here: the features span ten orders of magnitude (a blockiness
    z against a kurtosis), and without it the descent would spend every step on the largest one.
    The statistics are computed on the **training** split alone and applied to the test split, so
    the fold cannot leak its own scale into the model.
    """

    weights: NDArray[np.float64] | None = None
    bias: float = 0.0
    mean: NDArray[np.float64] | None = None
    scale: NDArray[np.float64] | None = None
    l2: float = 1e-3
    iterations: int = 2000
    learning_rate: float = 0.5

    def fit(self, x: Any, y: Any) -> LinearProbe:
        features = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64).ravel()
        if features.ndim != 2 or features.shape[0] != labels.size:
            raise ValueError("x must be (n, d) and y must have n entries")
        if len(set(labels.tolist())) < 2:
            raise ValueError("both classes must be present in the training split")
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale = np.where(self.scale > 0.0, self.scale, 1.0)
        z = (features - self.mean) / self.scale
        w = np.zeros(z.shape[1], dtype=np.float64)
        b = 0.0
        n = float(z.shape[0])
        for _ in range(self.iterations):
            p = 1.0 / (1.0 + np.exp(-(z @ w + b)))
            error = p - labels
            w -= self.learning_rate * ((z.T @ error) / n + self.l2 * w)
            b -= self.learning_rate * float(error.mean())
        self.weights, self.bias = w, b
        return self

    def decision(self, x: Any) -> NDArray[np.float64]:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("fit the probe before scoring with it")
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return np.asarray(z @ self.weights + self.bias, dtype=np.float64)


@dataclass(frozen=True)
class DiscriminatorResult:
    auc: float
    n_real: int
    n_synthetic: int
    folds: int
    #: Mean absolute standardised weight per feature over the folds, largest first. This is what
    #: turns a failing AUC into a thing to go and fix.
    importance: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def null_sigma(self) -> float:
        """Standard error of the AUC under the null "the two sets are the same distribution".

        sqrt((n1 + n2 + 1) / (12 n1 n2)), the Mann-Whitney variance. **This is what a
        synthetic-versus-itself control has to be judged against, and it is why ME.6's planned
        "AUC 0.5 +/- 0.03" is not a usable criterion:** at 96 patches per class the null spread is
        already 0.042, so a control that lands at 0.54 is one sigma from chance and a report that
        called it a failure would be measuring its own sample size. The criterion that scales is
        ``abs(auc - 0.5) < 2 * null_sigma``, which :attr:`indistinguishable` applies.
        """
        n1, n2 = float(self.n_real), float(self.n_synthetic)
        return float(math.sqrt((n1 + n2 + 1.0) / (12.0 * n1 * n2)))

    @property
    def z(self) -> float:
        """How many null standard errors the AUC sits above chance. Signed."""
        return (self.auc - 0.5) / max(self.null_sigma, 1e-12)

    @property
    def indistinguishable(self) -> bool:
        """Within two null standard errors of chance -- the control's pass condition."""
        return abs(self.z) < 2.0

    @property
    def separable(self) -> bool:
        return self.auc > 0.70

    def top(self, k: int = 3) -> str:
        return ", ".join(f"{name} {value:.2f}" for name, value in self.importance[:k])


def gap_score(
    real: Any, synthetic: Any, *, folds: int = 5, seed: int = 20260915
) -> DiscriminatorResult:
    """Cross-validated AUC separating real patches from synthetic ones.

    **Cross-validated, not in-sample.** A probe with ten free parameters fitted and scored on the
    same forty patches will report an AUC well above 0.5 on two sets drawn from one distribution,
    which is the single easiest way to invent a sim-to-real gap that is not there. Each fold is
    fitted on the rest and scored on itself, and the AUC is taken over the pooled held-out scores,
    so every patch is scored exactly once by a model that never saw it.
    """
    x_real = feature_matrix(real)
    x_synth = feature_matrix(synthetic)
    if not np.all(np.isfinite(x_real)) or not np.all(np.isfinite(x_synth)):
        raise ValueError("a feature came out non-finite; check for a constant or empty patch")
    x = np.vstack([x_real, x_synth])
    y = np.concatenate([np.ones(x_real.shape[0]), np.zeros(x_synth.shape[0])])
    n = y.size
    if min(x_real.shape[0], x_synth.shape[0]) < 2:
        raise ValueError("need at least two patches of each class")
    folds = int(min(folds, x_real.shape[0], x_synth.shape[0]))
    if folds < 2:
        raise ValueError("need at least two folds")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=int)
    # Stratified: shuffle within each class and deal the folds round-robin, so a fold cannot come
    # out all-real and leave `fit` with one class.
    for label in (0.0, 1.0):
        idx = order[y[order] == label]
        assignment[idx] = np.arange(idx.size) % folds

    scores = np.empty(n, dtype=np.float64)
    weights: list[NDArray[np.float64]] = []
    for fold in range(folds):
        test = assignment == fold
        probe = LinearProbe().fit(x[~test], y[~test])
        scores[test] = probe.decision(x[test])
        assert probe.weights is not None
        weights.append(np.abs(probe.weights))

    mean_weights = np.mean(weights, axis=0)
    ranked = sorted(zip(FEATURE_NAMES, mean_weights.tolist(), strict=True), key=lambda kv: -kv[1])
    return DiscriminatorResult(
        auc=auc_from_scores(scores, y.astype(bool)),
        n_real=int(x_real.shape[0]),
        n_synthetic=int(x_synth.shape[0]),
        folds=folds,
        importance=tuple(ranked),
    )
