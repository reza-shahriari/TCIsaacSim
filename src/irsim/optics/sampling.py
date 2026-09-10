"""Box-filter downsample from the supersampled render grid to the detector grid.

§8.3: render at k× and downsample with a box matching the pixel footprint. That single choice
gives the detector MTF (a sinc), correct aliasing above Nyquist and correct sub-pixel target
behaviour with no extra model. The box is the **full pitch**; the fill factor scales the active
area A_d, not the box (ADR 0020, spec issue S21). The downsample is a mean, so radiance units are
unchanged and a uniform field passes through exactly.

For a discrete k× grid the transfer of the k-sample mean is the Dirichlet kernel
sin(πξ)/(k sin(πξ/k)) (ξ in cycles per native pixel), which tends to |sinc(πξ)| as k → ∞; at k = 4
and ξ = 0.5 the two differ by 2.6 %. Tests check the exact discrete form and the sinc limit.

docs/physics-model.md §8.3, §13.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["box_downsample", "required_render_size", "box_transfer"]


def required_render_size(width: int, height: int, factor: int) -> tuple[int, int]:
    """(H·k, W·k): the G-buffer size the engine must render for a k× supersample."""
    if factor < 1:
        raise ValueError("supersample factor must be >= 1")
    return height * factor, width * factor


def box_downsample(image_ss: NDArray[np.floating], factor: int) -> NDArray[np.float32]:
    """Exact k×k block mean over the leading two axes; float32 out; float16 refused."""
    x = np.asarray(image_ss)
    if x.dtype == np.float16:
        raise TypeError("supersampled image is float16 (non-negotiable #2)")
    if factor < 1:
        raise ValueError("supersample factor must be >= 1")
    if x.ndim < 2:
        raise ValueError("image must be at least 2-D (H, W, ...)")
    h, w = x.shape[:2]
    if h % factor or w % factor:
        raise ValueError(f"shape {(h, w)} is not divisible by the supersample factor {factor}")
    if factor == 1:
        return np.asarray(x, dtype=np.float32)
    blocks = x.astype(np.float64).reshape(h // factor, factor, w // factor, factor, *x.shape[2:])
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def box_transfer(
    cycles_per_pixel: NDArray[np.floating] | float, factor: int
) -> NDArray[np.float64]:
    """|sin(πξ) / (k sin(πξ/k))|: the discrete k-sample box transfer (→ |sinc(πξ)| for large k)."""
    xi = np.asarray(cycles_per_pixel, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * xi)
        den = factor * np.sin(np.pi * xi / factor)
        out = np.where(np.abs(den) < 1e-15, 1.0, num / den)
    return np.asarray(np.abs(out), dtype=np.float64)
