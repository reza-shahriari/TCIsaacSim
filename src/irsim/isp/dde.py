"""Digital detail enhancement: an unsharp mask over the compressed image (§11.3, ADR 0029).

    y_out = clip(y + gain · (y − K ∗ y), 0, 1)

with K a 3×3 box (the cheapest high-pass real cores use; the position in the chain is after
gamma, ADR 0031). Boundary handling is edge replication by default; ``wrap`` is available for
spectral tests. A 0.5 step through the 3×3 box overshoots by ±gain/6 at the pixels adjacent to
the edge, and the transfer function on a stationary input is |1 + gain·(1 − K̂(f))|².
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["dde", "box3", "box3_transfer"]

Float32Array = NDArray[np.float32]
Boundary = Literal["edge", "wrap"]


def box3(y: NDArray[np.floating], boundary: Boundary = "edge") -> NDArray[np.float64]:
    """3×3 box mean of a 2-D image with edge replication or periodic wrap."""
    if y.ndim != 2:
        raise ValueError("DDE works on one (H, W) frame")
    y64 = y.astype(np.float64)
    p = np.pad(y64, 1, mode="wrap") if boundary == "wrap" else np.pad(y64, 1, mode="edge")
    acc = np.zeros(y.shape, dtype=np.float64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            acc += p[dy : dy + y.shape[0], dx : dx + y.shape[1]]
    return acc / 9.0


def box3_transfer(fx: NDArray[np.floating], fy: NDArray[np.floating]) -> NDArray[np.float64]:
    """K̂(f) of the 3×3 box, f in cycles/pixel: ((1 + 2cos 2πf_x)/3)·((1 + 2cos 2πf_y)/3)."""
    kx = (1.0 + 2.0 * np.cos(2.0 * np.pi * np.asarray(fx, dtype=np.float64))) / 3.0
    ky = (1.0 + 2.0 * np.cos(2.0 * np.pi * np.asarray(fy, dtype=np.float64))) / 3.0
    return np.asarray(kx * ky, dtype=np.float64)


def dde(y: object, gain: float, boundary: Boundary = "edge") -> Float32Array:
    """Unsharp mask with a 3×3 box: y + gain·(y − box(y)), clipped to [0, 1]; float32 out."""
    if gain < 0.0:
        raise ValueError("dde gain must be non-negative")
    arr = np.asarray(y)
    if arr.dtype == np.float16:
        raise TypeError("DDE input is float16 (non-negotiable #2)")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("DDE takes the float display image in [0, 1]")
    if gain == 0.0:
        return np.asarray(arr, dtype=np.float32)
    a = arr.astype(np.float64)
    out = a + gain * (a - box3(a, boundary))
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)
