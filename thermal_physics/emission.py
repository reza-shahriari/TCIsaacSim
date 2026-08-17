"""Surface emission -- Planck's law, band integration, emission + reflection.

Implements docs/thermal_camera_model.md SS1.
"""
from __future__ import annotations
import numpy as np


def planck_radiance(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    """Blackbody spectral radiance L_bb(lambda, T). SS1.

    Returns:
        Spectral radiance in W / (m^2 sr m).
    """
    raise NotImplementedError("TODO: implement Planck's law, docs/thermal_camera_model.md SS1")


def band_radiance(temperature_k: float, wavelength_range_m: tuple[float, float],
                   n_samples: int = 200) -> float:
    """Band-integrated radiance L_band(T) over [lambda_min, lambda_max]. SS1.

    Intended to be precomputed as a lookup table over your working temperature
    range rather than called per-pixel per-frame -- see SS1's note on this.
    """
    raise NotImplementedError("TODO: integrate planck_radiance over the band, SS1")


def surface_leaving_radiance(temperature_k: float, emissivity: float, t_env_k: float,
                              wavelength_range_m: tuple[float, float]) -> float:
    """eps*L_band(T) + (1-eps)*L_band(T_env) -- opaque, diffuse, gray-body surface. SS1."""
    raise NotImplementedError("TODO: implement SS1's surface radiance equation")
