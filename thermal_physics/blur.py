"""Point spread function / MTF -- optical + detector blur.

Implements docs/thermal_camera_model.md SS6. Apply BEFORE noise (SS8) --
noise originates at the detector plane at full resolution and should not be
blurred along with the true signal.

apply_system_blur uses a small hand-rolled separable Gaussian convolution
(no scipy dependency) -- fine at the resolutions this project targets.
"""
from __future__ import annotations
import numpy as np


def diffraction_mtf(spatial_freq: np.ndarray, wavelength_m: float, f_number: float) -> np.ndarray:
    """Diffraction-limited MTF for a circular aperture, cutoff f_c = 1/(lambda*N). SS6."""
    f_c = 1.0 / (wavelength_m * f_number)
    f = np.asarray(spatial_freq, dtype=np.float64)
    ratio = np.clip(f / f_c, 0.0, 1.0)
    mtf = (2.0 / np.pi) * (np.arccos(ratio) - ratio * np.sqrt(1.0 - ratio**2))
    return np.where(f <= f_c, mtf, 0.0)


def detector_mtf(spatial_freq: np.ndarray, pixel_pitch_m: float) -> np.ndarray:
    """|sinc(f * pixel_pitch)| -- finite pixel footprint MTF.

    np.sinc is the normalized sinc, sin(pi*x)/(pi*x), which is exactly the
    convention this formula uses. SS6.
    """
    return np.abs(np.sinc(np.asarray(spatial_freq, dtype=np.float64) * pixel_pitch_m))


def apply_system_blur(image: np.ndarray, psf_sigma_px: float) -> np.ndarray:
    """Convolve with an approximated system PSF (Gaussian, matched-width). SS6."""
    image = np.asarray(image, dtype=np.float64)
    if psf_sigma_px <= 0:
        return image
    radius = max(1, int(np.ceil(3 * psf_sigma_px)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * psf_sigma_px**2))
    kernel /= kernel.sum()
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=image)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=blurred)
    return blurred
