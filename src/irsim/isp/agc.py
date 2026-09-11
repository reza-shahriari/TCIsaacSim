"""Automatic gain control: 16-bit linear → [0, 1] display range (docs/physics-model.md §11.3).

The mapping from a 14-16 bit radiometric image to 8 bits is **part of the sensor model**: it
changes the image drastically, it is global (a hot exhaust entering the frame collapses the
contrast of everything else), and it must match between training and deployment (§15 Tier 5).
Two operators, both global and histogram-based on 2^bit_depth integer bins so that a GPU port
and a real core compute the same thing (ADR 0027, 0028):

* :func:`agc_linear` -- percentile clipping with in-bin linear interpolation of the CDF, then
  optional gamma: y = clip((x − x_lo)/(x_hi − x_lo), 0, 1)^(1/γ);
* :func:`agc_plateau` -- histogram equalisation with every bin's count clipped at
  P = plateau · N_pixels before the CDF is integrated. Small P → the rank map of the occupied
  bins (a min-max stretch on a dense histogram); P ≥ max count → full equalisation. This is the
  algorithm behind the characteristic thermal "look".

Inputs are uint16 DN or float32 in [0, 2^bit_depth − 1]; float16 is refused; outputs float32.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "histogram_dn",
    "percentile_from_histogram",
    "agc_linear",
    "agc_plateau",
    "CONSTANT_FRAME_LEVEL",
]

Float32Array = NDArray[np.float32]

# A frame with no dynamic range (x_hi == x_lo) displays as mid-grey, as real cores do.
CONSTANT_FRAME_LEVEL = 0.5


def _as_dn(x: object, bit_depth: int) -> NDArray[np.float64]:
    if not 8 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in 8..16")
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError("AGC input is float16 (non-negotiable #2); use uint16 or float32")
    if arr.ndim != 2:
        raise ValueError("AGC works on one (H, W) frame")
    if not (np.issubdtype(arr.dtype, np.integer) or arr.dtype in (np.float32, np.float64)):
        raise TypeError(f"AGC input must be uint16 or float32, got {arr.dtype}")
    a = arr.astype(np.float64)
    top = float(2**bit_depth - 1)
    if not np.all(np.isfinite(a)):
        raise ValueError("AGC input contains NaN or inf")
    if a.min() < 0.0 or a.max() > top:
        raise ValueError(f"AGC input outside [0, {top:.0f}] for bit_depth {bit_depth}")
    return a


def histogram_dn(dn: NDArray[np.float64], bit_depth: int) -> NDArray[np.int64]:
    """Counts per integer DN bin (2^bit_depth bins); float inputs are floored to their bin."""
    n_bins = 2**bit_depth
    idx = np.floor(dn).astype(np.int64)
    return np.bincount(idx.ravel(), minlength=n_bins).astype(np.int64)


def percentile_from_histogram(counts: NDArray[np.int64], p: float) -> float:
    """Value at fraction ``p`` of the pixels with linear interpolation inside the bin
    (the value ``v`` such that ``p·N`` pixels lie below it, pixels spread uniformly within
    their bin). For a ramp with one pixel per bin this equals p·N up to the in-bin offset."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    n = int(counts.sum())
    if n == 0:
        raise ValueError("empty histogram")
    target = p * n
    cdf = np.cumsum(counts)
    b = int(np.searchsorted(cdf, target, side="left"))
    b = min(b, counts.size - 1)
    below = float(cdf[b - 1]) if b > 0 else 0.0
    c = float(counts[b])
    frac = (target - below) / c if c > 0 else 0.0
    return b + min(max(frac, 0.0), 1.0)


def agc_linear(
    x: object, p_lo: float, p_hi: float, gamma: float = 1.0, bit_depth: int = 16
) -> Float32Array:
    """Linear AGC with percentile clipping and gamma, global over the frame (§11.3)."""
    if not 0.0 <= p_lo < p_hi <= 1.0:
        raise ValueError("need 0 <= p_lo < p_hi <= 1")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    dn = _as_dn(x, bit_depth)
    counts = histogram_dn(dn, bit_depth)
    x_lo = percentile_from_histogram(counts, p_lo)
    x_hi = percentile_from_histogram(counts, p_hi)
    if x_hi <= x_lo or int(np.floor(x_hi)) == int(np.floor(x_lo)):
        # both percentiles fall in one integer bin: no dynamic range at DN resolution
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    y = np.clip((dn - x_lo) / (x_hi - x_lo), 0.0, 1.0)
    if gamma != 1.0:
        y = y ** (1.0 / gamma)
    return np.asarray(y, dtype=np.float32)


def agc_plateau(x: object, plateau: float, bit_depth: int = 16) -> Float32Array:
    """Plateau-equalisation AGC (§11.3, ADR 0028): each of the 2^bit_depth bins' counts is
    clipped at P = plateau · N_pixels, the clipped counts are integrated into an exclusive CDF,
    and pixels map to (CDF(bin) − CDF(bin_min)) / (CDF(bin_max) − CDF(bin_min)), so the darkest
    occupied bin is 0 and the brightest is 1."""
    if plateau <= 0.0:
        raise ValueError("plateau must be positive (fraction of N_pixels per bin)")
    dn = _as_dn(x, bit_depth)
    counts = histogram_dn(dn, bit_depth).astype(np.float64)
    n = float(counts.sum())
    clipped = np.minimum(counts, plateau * n)
    cdf_excl = np.concatenate(([0.0], np.cumsum(clipped)[:-1]))
    occupied = np.flatnonzero(counts)
    lo, hi = cdf_excl[occupied[0]], cdf_excl[occupied[-1]]
    if hi <= lo:
        return np.full(dn.shape, CONSTANT_FRAME_LEVEL, dtype=np.float32)
    idx = np.floor(dn).astype(np.int64)
    y = (cdf_excl[idx] - lo) / (hi - lo)
    return np.asarray(np.clip(y, 0.0, 1.0), dtype=np.float32)
