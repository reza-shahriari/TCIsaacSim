"""Point spread function / MTF -- optical + detector blur.

Implements docs/thermal_camera_model.md SS6. Apply BEFORE noise (SS7) --
noise originates at the detector plane at full resolution and should not be
blurred along with the true signal.
"""
from __future__ import annotations
import numpy as np


def diffraction_mtf(spatial_freq: np.ndarray, wavelength_m: float, f_number: float) -> np.ndarray:
    """Diffraction-limited MTF for a circular aperture. SS6."""
    raise NotImplementedError("TODO: implement diffraction MTF, SS6")


def detector_mtf(spatial_freq: np.ndarray, pixel_pitch_m: float) -> np.ndarray:
    """sinc(f * pixel_pitch) -- finite pixel footprint MTF. SS6."""
    raise NotImplementedError("TODO: implement detector footprint MTF, SS6")


def apply_system_blur(image: np.ndarray, psf_sigma_px: float) -> np.ndarray:
    """Convolve the noise-free image with the (approximated) system PSF. SS6."""
    raise NotImplementedError("TODO: implement system blur convolution, SS6")
