"""Where a noise statistic is allowed to be measured: flat regions and per-pixel temporal noise.

roadmap ME.2b; docs/physics-model.md §10.2, §15 Tier 4(c); ADR 0023.

Every statistic in :mod:`irsim.validation.noise` assumes the cube it is handed contains **only**
the sensor: a horizon, a cloud edge or a drone inside the window is scene structure, and the
decomposition will happily report it as fixed-pattern noise. On synthetic data that never happens
because we choose the fixture. On somebody else's ten-second clip of the sky it happens
constantly, so the selection step is part of the measurement, not preparation for it.

**Everything here is measured relative to the patch's own noise**, never in DN. A clip whose
recorder stretched 16 bits into 8 has a different DN scale from one that did not, and a threshold
in DN would silently mean something different on each set. ``robust_noise_scale`` is the ruler:
the MAD of the first differences *about their own median*, so a linear gradient across the patch
shifts every difference equally and cancels, and the estimate is of the pixel-to-pixel
(uncorrelated) noise alone. The smaller of the two axis estimates is taken, because a column
pattern inflates the horizontal differences and a row pattern the vertical ones, and the aim is
the white part.

**Flatness is scene flatness, not noise flatness.** A patch is scored on

* ``ramp``   -- the peak-to-peak of a fitted plane across it (the sky's elevation gradient, or a
  slow shading), and
* ``structure`` -- the standard deviation of the de-ramped residual after averaging it into
  ``smooth`` x ``smooth`` blocks, with the white-noise contribution (``noise / smooth``) removed
  in quadrature, so a patch of pure noise scores zero and a cloud scores its own amplitude.

Fixed-pattern striping survives block averaging in one direction (a column pattern is reduced by
1/sqrt(smooth), not by 1/smooth), so a heavily striped patch scores a little structure. That is
deliberate and the default threshold sits well above it: striping is the thing being measured, and
a finder that rejected striped patches would make column noise unmeasurable by construction --
``test_flat_regions.py`` pins that behaviour at the Boson's ratios.

The outlier rule is what removes targets. A drone is a handful of pixels many sigma from the local
background; a patch containing any pixel beyond ``outlier_sigma`` of the fitted plane is dropped
rather than clipped, because a clipped target is still a target-shaped hole in the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FlatRegion",
    "TemporalNoise",
    "find_flat_regions",
    "representative_frame",
    "robust_noise_scale",
    "temporal_noise",
    "temporal_std_map",
]

#: MAD -> sigma for a Gaussian: 1 / Phi^-1(0.75).
_MAD_TO_SIGMA = 1.4826022185056018


@dataclass(frozen=True)
class FlatRegion:
    """One accepted window, with the statistics it was accepted on (all in the image's unit)."""

    row: int
    col: int
    size: int
    noise: float
    ramp: float
    structure: float
    outliers: int

    @property
    def slices(self) -> tuple[slice, slice]:
        return (slice(self.row, self.row + self.size), slice(self.col, self.col + self.size))

    @property
    def structure_ratio(self) -> float:
        """Excess structure in units of the patch's own noise -- the acceptance statistic."""
        return self.structure / self.noise if self.noise > 0.0 else float("inf")

    @property
    def ramp_ratio(self) -> float:
        return self.ramp / self.noise if self.noise > 0.0 else float("inf")

    def extract(self, frames: object) -> NDArray[np.float64]:
        """The window out of a frame ``(V, H)`` or a cube ``(T, V, H)``; float64, a fresh array."""
        x = np.asarray(frames)
        if x.ndim not in (2, 3):
            raise ValueError(f"frames must be (V, H) or (T, V, H), got shape {x.shape}")
        rows, cols = self.slices
        window = x[..., rows, cols]
        if window.shape[-2:] != (self.size, self.size):
            raise ValueError(
                f"region at ({self.row}, {self.col}) of size {self.size} does not fit "
                f"in an image of shape {x.shape[-2:]}"
            )
        return np.asarray(window, dtype=np.float64)

    def overlaps(self, other: FlatRegion) -> bool:
        return abs(self.row - other.row) < max(self.size, other.size) and abs(
            self.col - other.col
        ) < max(self.size, other.size)


def _as_frame(image: object) -> NDArray[np.float64]:
    x = np.asarray(image)
    if x.dtype == np.float16:
        raise TypeError("image is float16; promote to float32 or better before analysis")
    if x.ndim != 2:
        raise ValueError(f"expected one frame (V, H), got shape {x.shape}")
    if min(x.shape) < 2:
        raise ValueError("a frame needs at least two rows and two columns")
    return x.astype(np.float64)


def _axis_noise(frame: NDArray[np.float64], axis: int) -> float:
    d = np.diff(frame, axis=axis)
    mad = float(np.median(np.abs(d - np.median(d))))
    return float(mad * _MAD_TO_SIGMA / np.sqrt(2.0))


def robust_noise_scale(image: object) -> float:
    """Pixel-to-pixel noise sigma of a frame, immune to a linear gradient across it.

    The MAD of the first differences about their own median: a plane in the image makes every
    difference along an axis the same constant, which the median removes. Differencing two pixels
    doubles the variance, hence the sqrt(2). The smaller of the row-wise and column-wise estimates
    is returned -- column fixed-pattern noise enters only the horizontal differences and row noise
    only the vertical ones, so the minimum is the estimate least contaminated by striping.
    """
    frame = _as_frame(image)
    return min(_axis_noise(frame, 0), _axis_noise(frame, 1))


def representative_frame(frames: object) -> NDArray[np.float64]:
    """The frame flatness is judged on: the input itself, or a cube's per-pixel temporal median.

    The median over time is the right reduction for a clip, and not only because it is quieter
    than one frame: a target that moves is below the median in most frames, so it disappears,
    while the fixed pattern -- the thing that must stay visible for the outlier and structure
    tests to mean anything -- passes through untouched.
    """
    x = np.asarray(frames)
    if x.dtype == np.float16:
        raise TypeError("frames are float16; promote to float32 or better before analysis")
    if x.ndim == 2:
        return _as_frame(x)
    if x.ndim != 3:
        raise ValueError(f"frames must be (V, H) or (T, V, H), got shape {x.shape}")
    return _as_frame(np.median(x.astype(np.float64), axis=0))


def _plane_ramp(patch: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Least-squares plane through a patch: the residual and the plane's peak-to-peak."""
    size = patch.shape[0]
    coord = np.arange(size, dtype=np.float64) - 0.5 * (size - 1)
    denominator = float(np.sum(coord**2)) * size
    slope_v = float(np.sum(patch * coord[:, None])) / denominator
    slope_h = float(np.sum(patch * coord[None, :])) / denominator
    plane = patch.mean() + slope_v * coord[:, None] + slope_h * coord[None, :]
    ramp = (abs(slope_v) + abs(slope_h)) * (size - 1)
    return patch - plane, float(ramp)


def _block_means(residual: NDArray[np.float64], smooth: int) -> NDArray[np.float64]:
    size = residual.shape[0]
    blocks = size // smooth
    means = residual.reshape(blocks, smooth, blocks, smooth).mean(axis=(1, 3))
    return np.asarray(means, dtype=np.float64)


def _region(
    patch: NDArray[np.float64], row: int, col: int, smooth: int, sigma: float
) -> FlatRegion:
    size = patch.shape[0]
    noise = robust_noise_scale(patch)
    residual, ramp = _plane_ramp(patch)
    coarse = _block_means(residual, smooth)
    spread = float(coarse.std(ddof=1))
    expected = noise / smooth
    structure = float(np.sqrt(max(spread**2 - expected**2, 0.0)))
    outliers = int(np.count_nonzero(np.abs(residual) > sigma * noise)) if noise > 0.0 else 0
    return FlatRegion(
        row=row, col=col, size=size, noise=noise, ramp=ramp, structure=structure, outliers=outliers
    )


def find_flat_regions(
    frames: object,
    *,
    size: int = 32,
    stride: int | None = None,
    smooth: int = 4,
    max_structure: float = 0.25,
    max_ramp: float = 2.0,
    max_outliers: int = 0,
    outlier_sigma: float = 6.0,
    limit: int = 8,
) -> list[FlatRegion]:
    """Non-overlapping windows of ``size`` x ``size`` that contain no scene, best first.

    ``frames`` is one frame or a whole clip (reduced by :func:`representative_frame`).
    ``max_structure`` and ``max_ramp`` are in units of the window's own noise (see the module
    docstring); ``max_outliers`` counts pixels further than ``outlier_sigma`` from the fitted
    plane, and its default of zero is what keeps targets out. Returns at most ``limit`` windows,
    ordered by excess structure, none overlapping another.

    An empty list is a real answer -- a clip of a cloudy horizon has no flat sky in it -- and is
    better than the alternative of measuring the sensor on a cloud.
    """
    if size < 4 or size % smooth != 0:
        raise ValueError(f"size {size} must be at least 4 and a multiple of smooth {smooth}")
    if smooth < 2:
        raise ValueError(f"smooth must average at least 2x2 pixels, got {smooth}")
    if limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    frame = representative_frame(frames)
    if min(frame.shape) < size:
        raise ValueError(f"a {size}x{size} window does not fit in a {frame.shape} frame")
    step = size // 2 if stride is None else int(stride)
    if step < 1:
        raise ValueError(f"stride must be positive, got {stride}")

    candidates: list[FlatRegion] = []
    for row in range(0, frame.shape[0] - size + 1, step):
        for col in range(0, frame.shape[1] - size + 1, step):
            region = _region(
                frame[row : row + size, col : col + size], row, col, smooth, outlier_sigma
            )
            if region.noise <= 0.0:
                continue  # a window with no pixel-to-pixel variation says nothing about noise
            if region.outliers > max_outliers:
                continue
            if region.structure_ratio > max_structure or region.ramp_ratio > max_ramp:
                continue
            candidates.append(region)

    chosen: list[FlatRegion] = []
    for region in sorted(candidates, key=lambda r: (r.structure_ratio, r.ramp_ratio)):
        if any(region.overlaps(kept) for kept in chosen):
            continue
        chosen.append(region)
        if len(chosen) == limit:
            break
    return chosen


def _as_cube(cube: object) -> NDArray[np.float64]:
    x = np.asarray(cube)
    if x.dtype == np.float16:
        raise TypeError("cube is float16; promote to float32 or better before analysis")
    if x.ndim != 3:
        raise ValueError(f"cube must be (T, V, H), got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("a temporal statistic needs at least two frames")
    return x.astype(np.float64)


def temporal_std_map(cube: object) -> NDArray[np.float64]:
    """Per-pixel standard deviation along t -- the map a dead or flickering pixel shows up in."""
    return np.asarray(_as_cube(cube).std(axis=0, ddof=1), dtype=np.float64)


@dataclass(frozen=True)
class TemporalNoise:
    """Per-pixel temporal noise of a clip, and how much of the array disagrees with the rest."""

    per_pixel: NDArray[np.float64]
    median: float
    mean_variance: float
    outlier_fraction: float
    outlier_factor: float

    @property
    def rms(self) -> float:
        """sqrt(mean variance) -- in expectation sqrt(sigma_T^2 + sigma_TV^2 + sigma_TH^2 +
        sigma_TVH^2), the temporal half of the 3-D decomposition (ADR 0023)."""
        return float(np.sqrt(self.mean_variance))


def temporal_noise(cube: object, *, outlier_factor: float = 5.0) -> TemporalNoise:
    """Summarise :func:`temporal_std_map` robustly.

    ``mean_variance`` is the mean of the per-pixel variances, which is an unbiased estimate of
    ``sigma_T^2 + sigma_TV^2 + sigma_TH^2 + sigma_TVH^2``: the fixed terms (V, H, VH) do not vary
    with t and cancel in a per-pixel variance. That makes it an independent check on
    :func:`~irsim.validation.noise.decompose_3d` rather than a repetition of it.

    ``median`` is the robust counterpart -- on real data a few stuck or flickering pixels move the
    mean and not the median -- and ``outlier_fraction`` is how many pixels sit beyond
    ``outlier_factor`` times that median, which is the first-order bad-pixel count (the proper one
    is ME.3's).
    """
    if outlier_factor <= 1.0:
        raise ValueError(f"outlier_factor must exceed 1, got {outlier_factor}")
    per_pixel = temporal_std_map(cube)
    median = float(np.median(per_pixel))
    fraction = (
        float(np.count_nonzero(per_pixel > outlier_factor * median) / per_pixel.size)
        if median > 0.0
        else 0.0
    )
    return TemporalNoise(
        per_pixel=per_pixel,
        median=median,
        mean_variance=float(np.mean(per_pixel**2)),
        outlier_fraction=fraction,
        outlier_factor=float(outlier_factor),
    )
