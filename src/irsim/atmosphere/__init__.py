"""Atmosphere: Beer–Lambert transmittance, path radiance, humidity, presets, the stage.

docs/physics-model.md §7
"""

from irsim.atmosphere.beer_lambert import (
    apply_atmosphere,
    apply_tau_override,
    path_radiance,
    transmittance,
)
from irsim.atmosphere.humidity import (
    absolute_humidity_g_m3,
    gamma_molecular,
    saturation_vapour_pressure_hpa,
    vapour_pressure_hpa,
)
from irsim.atmosphere.spectral import (
    band_transmittance_spectral,
    effective_gamma,
    fit_grey_gamma,
    grey_fit_error,
)

__all__ = [
    "apply_atmosphere",
    "apply_tau_override",
    "path_radiance",
    "transmittance",
    "absolute_humidity_g_m3",
    "gamma_molecular",
    "saturation_vapour_pressure_hpa",
    "vapour_pressure_hpa",
    "band_transmittance_spectral",
    "effective_gamma",
    "fit_grey_gamma",
    "grey_fit_error",
]
