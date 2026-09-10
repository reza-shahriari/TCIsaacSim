"""The one ADC quantiser shared by the bolometer and photon paths.

DN = clip(floor(signal_dn), 0, 2^bits − 1) as uint16 (docs/physics-model.md §2 𝒬, §9.1). An ideal
ADC compares against thresholds, hence floor, not round: the quantisation error is uniform on
[0, 1) LSB with 0.29 LSB rms, which is what the Tier 2 SITF bench expects (ADR 0019). Saturation
clips and never wraps; negative signals clip to 0.

Noise is added to ``signal_dn`` *before* this call (spec issue S10, non-negotiable #3):
quantise last.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["quantise", "dn_max_for_bits"]


def dn_max_for_bits(bit_depth: int) -> int:
    if not 8 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in 8..16 for a uint16 DN")
    return int(2**bit_depth - 1)


def quantise(signal_dn: NDArray[np.floating] | float, bit_depth: int) -> NDArray[np.uint16]:
    """Floor and clip to [0, 2^bits − 1]; uint16 out; float16 in is refused."""
    s = np.asarray(signal_dn)
    if s.dtype == np.float16:
        raise TypeError("signal is float16; quantise from float32 or better")
    if not np.issubdtype(s.dtype, np.floating):
        raise TypeError(f"signal must be a float array in DN units, got {s.dtype}")
    if not np.all(np.isfinite(s)):
        raise ValueError("signal contains NaN or inf")
    top = dn_max_for_bits(bit_depth)
    floored = np.floor(s.astype(np.float64))
    return np.asarray(np.clip(floored, 0, top), dtype=np.uint16)
