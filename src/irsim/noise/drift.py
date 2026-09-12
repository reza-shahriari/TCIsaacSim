"""Mean-reverting drift of the fixed-pattern components -- "pattern breathing" (M9.4).

docs/physics-model.md §10.3, §10.2, §8.2. ADR 0054.

A real focal plane's fixed-pattern noise is not fixed. Between corrections the pattern *breathes*:
it wanders, but it does not wander away. §10.3 asks for that behaviour and does not say which
process produces it, so ADR 0054 chooses the Ornstein-Uhlenbeck process, advanced by its **exact**
update

    x <- x e^{-dt/tau} + sigma sqrt(1 - e^{-2 dt/tau}) xi,      xi ~ N(0, 1)

with tau = ``noise.fpn_drift_tau_s``.

**Why mean-reverting and not a random walk.** A random walk is the obvious way to make a pattern
move and it is wrong here, for a reason that only shows up after a long run: its variance grows
without bound. The configured 3-D ratios (sigma_V, sigma_H, sigma_VH against sigma_TVH) are the
sensor's *identity* -- they are what makes the imagery look like it came from this camera rather
than another -- and a random walk silently destroys them, a little more every frame, while every
individual frame still looks plausible. The OU update above is stationary by construction: if
Var(x) = sigma^2 then Var(x') = sigma^2 e^{-2dt/tau} + sigma^2 (1 - e^{-2dt/tau}) = sigma^2,
exactly, at any step size. The ratios stay true for a ten-minute sequence and a ten-hour one alike.

**What this is not.** This is the stochastic part of pattern drift only. The *deterministic*,
ambient-driven part -- the residual that grows with Delta-T_FPA between FFC events -- is M9.6's NUC
residual, and ADR 0053 makes that the only Delta-T_FPA-driven mechanism in the chain. Likewise the
global offset level drifts because the housing and the focal plane physically warm (M3.3, M9.3),
not because a scalar is doing a random walk; nothing here touches the global term. Keeping the
three apart is what stops the same drift being counted three times.

**Which components drift (ADR 0054).** All three fixed terms -- V (rows), H (columns) and VH
(per pixel). They share one physical cause: the per-row, per-column and per-pixel spread in how
the detector and the ROIC respond to bias and to temperature. Breathing VH alone would leave the
column stripes frozen, and slowly-breathing stripes are among the most recognisable real artefacts
there is; it would also under-predict the whole-pattern growth ME.3 measures, which does not
separate the three. ``components`` can restrict the set for an ablation.

Units follow the pattern being drifted -- radiance for the bolometer chain, electrons for a photon
FPA -- never kelvin (non-negotiable #3). float32 throughout (non-negotiable #2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from irsim.noise.seeding import NoiseStream, sensor_rng
from irsim.noise.three_d import FixedPattern, Sigmas7

__all__ = ["FpnDrift", "DriftingPattern", "DRIFTING_COMPONENTS", "ou_step", "drift_rng"]

# ADR 0054: every fixed term breathes; the global offset is physical (M3.3/M9.3), not stochastic.
DRIFTING_COMPONENTS: tuple[str, ...] = ("v", "h", "vh")


def drift_rng(sensor_seed: int) -> np.random.Generator:
    """The one blessed generator for a sensor's pattern drift (ADR 0022 stream ``VH_DRIFT``).

    Drift is the one part of the noise chain that is **sequential** rather than counter-based: the
    OU state at frame N is the accumulation of every innovation before it, so it cannot be drawn
    from ``(sensor_seed, frame_index)`` the way §10.2's per-frame terms are. A frame sequence is
    therefore reproducible from the start of the sequence, not from an arbitrary frame -- which is
    also why M10.7's Warp twin is held to *statistical* equivalence with this path rather than
    bit-equality (ADR 0022).
    """
    return sensor_rng(sensor_seed, NoiseStream.VH_DRIFT)


def ou_step(
    x: np.ndarray, sigma: float, dt_s: float, tau_s: float, rng: np.random.Generator
) -> np.ndarray:
    """One exact Ornstein-Uhlenbeck step on ``x`` toward zero mean with stationary σ (§10.3).

    Exact for any ``dt_s``: the decay and the innovation variance are the closed-form solution of
    the OU SDE over the interval, not a first-order approximation, so a 10-frame step and ten
    1-frame steps have the same stationary law. An infinite ``tau_s`` is a frozen pattern and is
    returned unchanged without consuming a draw.
    """
    if sigma == 0.0 or not math.isfinite(tau_s):
        return np.asarray(x, dtype=np.float32)
    decay = math.exp(-float(dt_s) / float(tau_s))
    innovation = float(sigma) * math.sqrt(max(0.0, 1.0 - decay * decay))
    xi = rng.standard_normal(size=np.shape(x))
    out = np.asarray(x, dtype=np.float64) * decay + innovation * xi
    return np.asarray(out, dtype=np.float32)


@dataclass(frozen=True)
class FpnDrift:
    """The §10.3 drift process for one sensor's fixed pattern; see the module docstring.

    ``tau_s`` is ``noise.fpn_drift_tau_s``. ``float('inf')`` freezes the pattern, which is the
    right model for an ideal sensor and the control case for every drift test.
    """

    tau_s: float
    sigmas: Sigmas7
    components: tuple[str, ...] = DRIFTING_COMPONENTS

    def __post_init__(self) -> None:
        if not (self.tau_s > 0.0) or math.isnan(self.tau_s):
            raise ValueError(f"fpn_drift_tau_s must be positive (inf = frozen), got {self.tau_s}")
        unknown = set(self.components) - set(DRIFTING_COMPONENTS)
        if unknown:
            raise ValueError(
                f"only the fixed terms {DRIFTING_COMPONENTS} drift, got {sorted(unknown)}; the "
                "global offset drifts physically (M3.3, M9.3) and the Delta-T_FPA residual is "
                "M9.6's -- neither belongs here (ADR 0054)"
            )

    @property
    def frozen(self) -> bool:
        return not math.isfinite(self.tau_s)

    def advance(self, fixed: FixedPattern, dt_s: float, rng: np.random.Generator) -> FixedPattern:
        """Advance the pattern by ``dt_s`` seconds and return the new one (float32).

        The input is not mutated: ``FixedPattern`` is frozen, and a drifting pattern that aliased
        its own history would make a golden sequence depend on how many times it had been read.

        Components are drawn in ``DRIFTING_COMPONENTS`` order from the single generator, so the
        order is part of the reproducibility contract; use :func:`drift_rng` to build it.
        """
        if not np.isfinite(dt_s) or float(dt_s) < 0.0:
            raise ValueError(f"dt_s must be finite and non-negative, got {dt_s}")
        if self.frozen or float(dt_s) == 0.0:
            return fixed
        parts = {
            "v": fixed.v,
            "h": fixed.h,
            "vh": fixed.vh,
        }
        for name in DRIFTING_COMPONENTS:
            if name in self.components:
                parts[name] = ou_step(
                    parts[name], getattr(self.sigmas, name), dt_s, self.tau_s, rng
                )
        return FixedPattern(
            v=np.asarray(parts["v"], dtype=np.float32),
            h=np.asarray(parts["h"], dtype=np.float32),
            vh=np.asarray(parts["vh"], dtype=np.float32),
        )

    def autocorrelation(self, lag_s: float) -> float:
        """The process's own lag autocorrelation, e^{−lag/τ}.

        The oracle the tests measure the realised series against.
        """
        if self.frozen:
            return 1.0
        return math.exp(-float(lag_s) / float(self.tau_s))


@dataclass
class DriftingPattern:
    """A :class:`FixedPattern` that breathes, and that a shutter event recalibrates.

    :class:`FpnDrift` is the *process* and is immutable; this is the state it acts on. It exists
    because the two things reset differently and at different times: the process never changes,
    while the pattern is replaced wholesale whenever the FFC controller closes the shutter
    (M9.7). Keeping them in one object would make "reset the drift" ambiguous.

    ``reset`` redraws the pattern from a new epoch rather than zeroing it. A flat-field correction
    does not remove a camera's fixed-pattern noise, it re-measures and re-subtracts it, and what
    is left afterwards is a fresh realisation of the same distribution -- uncorrelated with what
    came before, and the same size.
    """

    drift: FpnDrift
    pattern: FixedPattern
    sensor_seed: int
    epoch: int = 0

    @classmethod
    def start(cls, drift: FpnDrift, shape: tuple[int, int], sensor_seed: int) -> DriftingPattern:
        return cls(
            drift=drift,
            pattern=FixedPattern.generate(shape, drift.sigmas, sensor_seed),
            sensor_seed=sensor_seed,
        )

    def advance(self, dt_s: float, rng: np.random.Generator) -> FixedPattern:
        self.pattern = self.drift.advance(self.pattern, dt_s, rng)
        return self.pattern

    def reset(self, frame_index: int) -> None:
        """Re-measure and re-subtract: a new realisation, uncorrelated with the old one."""
        del frame_index  # epochs are counted by FFC, not by frame (see NucResidual.reset)
        self.epoch += 1
        # A distinct seed per epoch, derived so that it cannot collide with another sensor's.
        seed = (int(self.sensor_seed) * 1_000_003 + self.epoch) % (2**63 - 1)
        self.pattern = FixedPattern.generate(self.pattern.shape, self.drift.sigmas, seed)
