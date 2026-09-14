"""What the storage path did to the numbers: quantisation, block coding, and the floor they set.

roadmap ME.2b; docs/physics-model.md §10.2, §15 Tier 4(c); ADR 0023.

Every public set this project can validate against is 8-bit and most are lossily coded (ADR 0003,
`data/validation/datasets.yaml`). A noise sigma measured on such a clip is therefore not the
sensor's: it is the sensor's convolved with a quantiser and an encoder that was designed to throw
sensor noise away. This module measures how much of the answer belongs to the storage path, so
that ME.5 can report every statistic with its floor beside it and ME.6 never compares a simulated
sigma against a number the codec set.

**The quantiser.** Rounding to a lattice of step ``d`` adds ``d^2/12`` to the variance (Sheppard's
correction) once the signal dithers across several levels. Measured on Gaussian noise quantised to
integers, ``sheppard_correct`` recovers the true sigma to better than 0.5 % for sigma >= 0.5 LSB
and to 3.5 % at 0.3 LSB, and breaks down below that (at 0.2 LSB it under-reads by 50 %): the
information is gone, not merely biased. Hence the floor ``d/sqrt(12)`` = 0.289 LSB and the
``margin`` rule -- a component within a factor ``margin`` of the floor is reported as
**codec-limited** rather than as a measurement, whether or not the correction happens to land.

**The lattice itself is measured, not assumed.** A recorder that mapped a 16-bit stream onto a
narrow 8-bit range leaves the values on a coarse lattice, and reading its step off the data is the
only way to know: ``quantiser_step`` takes the largest step for which nearly every sample shares
one residue.

**Block coding.** An 8 x 8 DCT leaves its grid in the image, so the mean absolute difference
across block boundaries differs from the one inside blocks. The null has to be built from the
*gap positions*, not from the pixels: a fixed column pattern makes some columns noisier than
others in every frame alike, so pixel-count error bars are far too tight and read blocking where
there is none (uncoded Boson-ratio cubes land at |z| <= 2 on the per-gap null and at |z| ~ 3 on the
per-pixel one).

**Blockiness is not a sufficient test, and this module does not pretend otherwise.** Measured on a
320x256x60 cube of Boson-ratio noise through x264: at CRF 18 the encoder removes 95 % of the
temporal noise (sigma_TVH 1.53 -> 0.08 DN) while the blocking z-score reaches only 3.4, and at
800 kbit/s it removes 43 % with no detectable blocking at all. On flat sky the dominant fingerprint
of a lossy codec is that the noise is *gone*, not that it is blocky. The decisive statement a
report can make is therefore that a noise statistic from a lossy set is a **lower bound**; the
indicators here (levels used, lattice step, blocking) say when that bound is visibly biting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.validation.noise import RATIO_ORDER, Decomposition3D

__all__ = [
    "Blockiness",
    "CodecFloor",
    "FlaggedDecomposition",
    "blockiness",
    "codec_floor",
    "flag_codec_limited",
    "quantisation_floor",
    "quantiser_step",
    "sheppard_correct",
]

#: Standard deviation of a uniform quantiser's error, in units of its step.
UNIFORM_STEP_SIGMA = 1.0 / np.sqrt(12.0)

#: Below this multiple of the floor a component is reported as codec-limited, not measured.
DEFAULT_MARGIN = 2.0

#: |z| on the per-gap null above which the 8-pixel grid is called visible.
DEFAULT_BLOCK_Z = 5.0


def quantisation_floor(step: float = 1.0) -> float:
    """The smallest noise sigma a lattice of ``step`` can carry: ``step / sqrt(12)``.

    At one 8-bit code that is 0.289 DN. It is the per-set noise floor of every indexed set, and it
    applies before any codec: a set stored at 8 bits cannot report a 0.05 DN fixed-pattern term
    however many frames it has.
    """
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"quantiser step must be finite and positive, got {step}")
    return float(step) * float(UNIFORM_STEP_SIGMA)


def sheppard_correct(sigma: float, step: float = 1.0) -> float:
    """Remove the quantiser's own variance from a measured sigma (Sheppard's correction).

    Valid where the signal dithers: 0.5 % at sigma >= 0.5 step, 3.5 % at 0.3 step, and useless
    below (it returns 0 for sigma <= step/sqrt(12)). Always report the floor beside the result.
    """
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    floor = quantisation_floor(step)
    return float(np.sqrt(max(sigma * sigma - floor * floor, 0.0)))


def quantiser_step(values: object, *, max_step: int = 16, coverage: float = 0.995) -> int:
    """The lattice spacing the samples live on: the largest step nearly all of them share.

    Returns 1 when the values are not on a coarser lattice, and also when there are too few
    distinct levels to tell (a constant patch is consistent with every step; :class:`CodecFloor`
    reports that case through ``levels_used`` and ``determinate`` instead of guessing).
    """
    if max_step < 1:
        raise ValueError(f"max_step must be positive, got {max_step}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    x = np.asarray(values)
    if not np.allclose(x, np.rint(x.astype(np.float64)), atol=1e-6):
        raise ValueError("values are not on an integer lattice; quantiser_step reads stored codes")
    codes = np.rint(x.astype(np.float64)).astype(np.int64).ravel()
    if np.unique(codes).size < 3:
        return 1
    base = codes - codes.min()
    best = 1
    for step in range(2, max_step + 1):
        counts = np.bincount(base % step, minlength=step)
        if counts.max() / codes.size >= coverage:
            best = step
    return best


@dataclass(frozen=True)
class Blockiness:
    """Mean absolute difference across block boundaries relative to inside blocks, per axis.

    1.0 is the null. Above 1 is the classic blocking artefact; *below* 1 is the deblocking filter
    having smoothed the boundaries, which is the same evidence of an 8-pixel grid, so the verdict
    is on |z|. ``nan`` means the frame has no pixel-to-pixel variation left to measure at all --
    itself a codec finding, and one ``CodecFloor`` reports through ``levels_used``.
    """

    block: int
    ratio_h: float
    ratio_v: float
    z_h: float
    z_v: float
    z_threshold: float

    @property
    def z(self) -> float:
        """The stronger of the two axes; nan when neither axis had anything to measure."""
        finite = [abs(value) for value in (self.z_h, self.z_v) if np.isfinite(value)]
        return max(finite) if finite else float("nan")

    @property
    def blocking(self) -> bool:
        return bool(np.isfinite(self.z) and self.z >= self.z_threshold)


def _gap_means(x: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Mean |difference| at each gap position along ``axis``, averaged over everything else."""
    d = np.abs(np.diff(x, axis=axis))
    others = tuple(a for a in range(x.ndim) if a != axis)
    return np.asarray(d.mean(axis=others), dtype=np.float64)


def _ratio_and_z(means: NDArray[np.float64], block: int) -> tuple[float, float]:
    on_boundary = ((np.arange(means.size) + 1) % block) == 0
    boundary, interior = means[on_boundary], means[~on_boundary]
    if boundary.size < 2 or interior.size < 2 or interior.mean() <= 0.0:
        return (float("nan"), float("nan"))
    ratio = float(boundary.mean() / interior.mean())
    error = float(
        np.hypot(
            boundary.std(ddof=1) / np.sqrt(boundary.size),
            interior.std(ddof=1) / np.sqrt(interior.size),
        )
        / interior.mean()
    )
    if error <= 0.0:
        return (ratio, float("nan"))
    return (ratio, (ratio - 1.0) / error)


def blockiness(
    frames: object, *, block: int = 8, z_threshold: float = DEFAULT_BLOCK_Z
) -> Blockiness:
    """Look for a ``block``-pixel coding grid in a frame ``(V, H)`` or a cube ``(T, V, H)``."""
    if block < 2:
        raise ValueError(f"block must be at least 2, got {block}")
    x = np.asarray(frames)
    if x.dtype == np.float16:
        raise TypeError("frames are float16; promote to float32 or better before analysis")
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"frames must be (V, H) or (T, V, H), got shape {x.shape}")
    ratio_h, z_h = _ratio_and_z(_gap_means(x, 2), block)
    ratio_v, z_v = _ratio_and_z(_gap_means(x, 1), block)
    return Blockiness(
        block=block,
        ratio_h=ratio_h,
        ratio_v=ratio_v,
        z_h=z_h,
        z_v=z_v,
        z_threshold=float(z_threshold),
    )


@dataclass(frozen=True)
class CodecFloor:
    """What the storage path costs: its lattice, its floor, and whether its grid is visible."""

    step: int
    sigma_floor: float
    levels_used: int
    blockiness: Blockiness

    @property
    def determinate(self) -> bool:
        """False when the data has too few distinct levels to say anything about its lattice."""
        return self.levels_used >= 3

    def limits(self, sigma: float, *, margin: float = DEFAULT_MARGIN) -> bool:
        """Whether ``sigma`` is too close to the floor to be attributed to the sensor."""
        if margin < 1.0:
            raise ValueError(f"margin must be at least 1, got {margin}")
        return bool(not self.determinate or sigma < margin * self.sigma_floor)

    def correct(self, sigma: float) -> float:
        """Sheppard-corrected ``sigma``; read it only where :meth:`limits` is False."""
        return sheppard_correct(sigma, self.step)


def codec_floor(
    frames: object, *, block: int = 8, z_threshold: float = DEFAULT_BLOCK_Z
) -> CodecFloor:
    """Measure the storage path of a frame or clip of **stored codes** (uint8 or uint16)."""
    x = np.asarray(frames)
    step = quantiser_step(x, max_step=16)
    return CodecFloor(
        step=step,
        sigma_floor=quantisation_floor(step),
        levels_used=int(np.unique(np.asarray(x)).size),
        blockiness=blockiness(x, block=block, z_threshold=z_threshold),
    )


@dataclass(frozen=True)
class FlaggedDecomposition:
    """A 3-D decomposition read against the floor of the path its cube came through."""

    decomposition: Decomposition3D
    floor: CodecFloor
    corrected: dict[str, float]
    limited: dict[str, bool]
    margin: float

    @property
    def any_limited(self) -> bool:
        return any(self.limited.values())

    def measurable(self) -> tuple[str, ...]:
        """The components this cube can actually support a claim about."""
        return tuple(name for name in RATIO_ORDER if not self.limited[name])


def flag_codec_limited(
    decomposition: Decomposition3D,
    floor: CodecFloor,
    *,
    margin: float = DEFAULT_MARGIN,
) -> FlaggedDecomposition:
    """Attach the codec-limited flag and the Sheppard correction to every 3-D component.

    This is the step that keeps ADR 0023's promise that a codec bias is *reported beside* the
    estimate and never silently subtracted: ``corrected`` holds the de-quantised sigmas and
    ``limited`` says, per component, whether that correction may be believed. On 8-bit data the
    Boson's small terms (sigma_T = 0.02 sigma_TVH, the TV and TH interactions) are always limited
    -- they sit an order of magnitude below one code -- and saying so is the point.
    """
    sigmas = dict(zip(RATIO_ORDER, decomposition.as_vector(), strict=True))
    return FlaggedDecomposition(
        decomposition=decomposition,
        floor=floor,
        corrected={name: floor.correct(value) for name, value in sigmas.items()},
        limited={name: floor.limits(value, margin=margin) for name, value in sigmas.items()},
        margin=float(margin),
    )
