"""Optical PSF synthesis from the MTF and its application at the supersampled pitch.

The PSF is built from the **optical** part of the cascade only -- MTF_diff · MTF_gauss -- as the
inverse Fourier transform of that radially symmetric, real, non-negative-definite transfer
function sampled at the supersampled pitch p/k. The detector footprint is *not* in it (the box
downsample supplies MTF_det, ADR 0020/0059), nor is motion (photon-only, applied separately;
bolometer smear is the IIR). Kernel: odd size, normalised to Σ = 1, tiny negative FFT ringing
clipped, radially symmetric. ``apply_psf`` convolves by FFT in float64 with edge replication and
casts back to the input dtype (float32 minimum; float16 refused). Convolving a uniform field
leaves it unchanged, so radiance is conserved.

docs/physics-model.md §8.3, §2 (MTF ∗ L)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.optics.mtf import mtf_diffraction, mtf_gaussian

__all__ = ["optical_psf", "apply_psf", "psf_radius_samples"]

FloatArray = NDArray[np.float64]


def psf_radius_samples(
    wavelength_um: float, f_number: float, sigma_um: float, sample_pitch_um: float
) -> int:
    """Kernel half-width: the larger of 6 Airy radii (1.22 λF) and 6σ, at least 4 samples."""
    airy_um = 1.22 * wavelength_um * f_number
    extent_um = max(6.0 * airy_um, 6.0 * sigma_um)
    return max(4, int(np.ceil(extent_um / sample_pitch_um)))


def optical_psf(
    wavelength_um: float,
    f_number: float,
    sigma_aberr_um: float,
    pitch_um: float,
    supersample: int = 1,
    radius: int | None = None,
) -> FloatArray:
    """Radially symmetric PSF kernel (float64, odd size, Σ = 1) sampled at pitch/supersample."""
    if pitch_um <= 0.0 or supersample < 1 or sigma_aberr_um < 0.0:
        raise ValueError("pitch must be positive, supersample >= 1, sigma non-negative")
    sample_um = pitch_um / supersample
    r = (
        radius
        if radius is not None
        else psf_radius_samples(wavelength_um, f_number, sigma_aberr_um, sample_um)
    )
    n = 4 * r + 1  # generous FFT grid so the truncation ringing is negligible
    f = np.fft.fftfreq(n, d=sample_um * 1e-3)  # cycles per mm
    fx, fy = np.meshgrid(f, f, indexing="xy")
    fr = np.sqrt(fx * fx + fy * fy)
    mtf = mtf_diffraction(fr, wavelength_um, f_number)
    if sigma_aberr_um > 0.0:
        mtf = mtf * mtf_gaussian(fr, sigma_aberr_um * 1e-3)
    psf = np.real(np.fft.fftshift(np.fft.ifft2(mtf)))
    c = n // 2
    kernel = psf[c - r : c + r + 1, c - r : c + r + 1]
    kernel = np.clip(kernel, 0.0, None)
    total = kernel.sum()
    if total <= 0.0:
        raise ValueError("degenerate PSF")
    return np.asarray(kernel / total, dtype=np.float64)


def apply_psf(image_ss: object, kernel: FloatArray) -> NDArray[np.floating]:
    """Convolve a (H, W) image with the kernel by FFT (edge-replicated padding); dtype preserved."""
    x = np.asarray(image_ss)
    if x.dtype == np.float16:
        raise TypeError("image is float16 (non-negotiable #2)")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("image must be a float array")
    if x.ndim != 2:
        raise ValueError("apply_psf works on one (H, W) plane")
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("kernel must be 2-D with odd sides")
    ry, rx = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(x.astype(np.float64), ((ry, ry), (rx, rx)), mode="edge")
    h, w = padded.shape
    # place the kernel centre on the origin of the padded grid (circular convolution then equals
    # the linear one inside the crop because the padding is at least the kernel radius)
    kern = np.zeros((h, w), dtype=np.float64)
    kern[: k.shape[0], : k.shape[1]] = k
    # np.roll is shape-preserving, but numpy 2.x types it as widening the shape parameter
    # from (int, int) to (int, ...), so the 2-D type is restated rather than lost.
    kern = np.asarray(np.roll(kern, (-ry, -rx), axis=(0, 1)), dtype=np.float64).reshape(h, w)
    fk = np.fft.rfft2(kern)
    out = np.fft.irfft2(np.fft.rfft2(padded) * fk, s=(h, w))
    out = out[ry : ry + x.shape[0], rx : rx + x.shape[1]]
    return np.asarray(out, dtype=x.dtype)
