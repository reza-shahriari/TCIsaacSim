"""Surface emission -- Planck's law, band integration, emission + reflection.

Implements docs/thermal_camera_model.md SS1.

Valid range: terrestrial/combustion temperatures (~100-3000 K) and
LWIR/MWIR/SWIR wavelengths (~1-14 um). Outside that range the Planck
exponent can overflow float64 -- not guarded against here on purpose, since
staying in-range is a scene-design choice, not a numerical one.
"""
from __future__ import annotations
import numpy as np

from .constants import PLANCK_H, SPEED_OF_LIGHT, BOLTZMANN_K

# np.trapz was removed in numpy 2.0, renamed np.trapezoid. Support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def planck_radiance(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    """Blackbody spectral radiance L_bb(lambda, T). SS1.

    Returns:
        Spectral radiance in W / (m^2 sr m).
    """
    wl = np.asarray(wavelength_m, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    exponent = (PLANCK_H * SPEED_OF_LIGHT) / (wl * BOLTZMANN_K * t)
    return (2 * PLANCK_H * SPEED_OF_LIGHT**2) / (wl**5 * (np.exp(exponent) - 1))


def band_radiance(temperature_k: float, wavelength_range_m: tuple[float, float],
                   n_samples: int = 200) -> float:
    """Band-integrated radiance L_band(T) = integral of L_bb(lambda,T) dlambda over
    the band. SS1. R_sys(lambda) is left as 1 (unweighted) -- pass a real spectral
    response later if needed.

    temperature_k may be a scalar or an (H, W) array; the integration is
    vectorized over whatever shape it has.
    """
    lam_min, lam_max = wavelength_range_m
    wavelengths = np.linspace(lam_min, lam_max, n_samples)
    t = np.asarray(temperature_k, dtype=np.float64)
    wl = wavelengths.reshape((-1,) + (1,) * t.ndim)  # broadcast against t's shape
    radiances = planck_radiance(wl, t)
    return _trapz(radiances, wavelengths, axis=0)


def surface_leaving_radiance(temperature_k: float, emissivity: float, t_env_k: float,
                              wavelength_range_m: tuple[float, float]) -> float:
    """eps*L_band(T) + (1-eps)*L_band(T_env) -- opaque, diffuse, gray-body surface. SS1."""
    l_point = band_radiance(temperature_k, wavelength_range_m)
    l_env = band_radiance(t_env_k, wavelength_range_m)
    return emissivity * l_point + (1 - emissivity) * l_env
