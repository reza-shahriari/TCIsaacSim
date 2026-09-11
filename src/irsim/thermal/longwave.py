"""Broadband longwave downwelling irradiance for the energy balance (§5.3, §6.1).

    Q_LW↓ = V_s · ε_sky σ T_air⁴  +  (1 − V_s) · σ T_surround⁴
    ε_sky = ε_clear + (1 − ε_clear) · cloud
    ε_clear = 0.52 + 0.065 √e            (Brunt 1932, e in hPa; default)
            = 0.70 + 5.95e-5 e exp(1500/T)   (Idso 1981)

This is the *broadband* (all-wavelength) sky emission the surface energy balance needs. It is
**not** the LWIR-window apparent sky temperature of §5.3(a): the 8–14 µm window is where the
clear sky is coldest, and using T_sky(zenith) ≈ T_air − 60 K as a broadband sky would put
Q_LW↓ at ~150 W m⁻² against the ~260–300 W m⁻² a pyrgeometer measures at 288 K -- a 100+ W m⁻²
error in the balance (ADR 0035). Vapour pressure comes from the same WeatherSeries sample as
the atmosphere's humidity (M8.2's Magnus form), so both paths see one humidity.

docs/physics-model.md §5.3, §6.1
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.weather import WeatherSample

__all__ = [
    "EmissivityFormula",
    "clear_sky_emissivity",
    "sky_emissivity",
    "longwave_down",
    "longwave_down_from_sample",
]

EmissivityFormula = Literal["brunt", "idso"]
BRUNT_A = 0.52
BRUNT_B = 0.065  # per sqrt(hPa)
IDSO_A = 0.70
IDSO_B = 5.95e-5  # per hPa
IDSO_C_K = 1500.0


def clear_sky_emissivity(
    e_vapour_hpa: Any, t_air_k: Any, formula: EmissivityFormula = "brunt"
) -> NDArray[np.float64]:
    """Effective broadband clear-sky emissivity (≤ 1), rising with vapour pressure."""
    e = np.asarray(e_vapour_hpa, dtype=np.float64)
    t = np.asarray(t_air_k, dtype=np.float64)
    if np.any(e < 0.0):
        raise ValueError("vapour pressure must be non-negative")
    if np.any(t <= 0.0):
        raise ValueError("t_air_k must be positive kelvin")
    if formula == "brunt":
        eps = BRUNT_A + BRUNT_B * np.sqrt(e)
    elif formula == "idso":
        eps = IDSO_A + IDSO_B * e * np.exp(IDSO_C_K / t)
    else:
        raise ValueError(f"unknown emissivity formula {formula!r}")
    return np.asarray(np.minimum(eps, 1.0), dtype=np.float64)


def sky_emissivity(
    e_vapour_hpa: Any,
    t_air_k: Any,
    cloud_fraction: Any,
    formula: EmissivityFormula = "brunt",
) -> NDArray[np.float64]:
    """ε_clear + (1 − ε_clear) · cloud: overcast radiates as a blackbody at T_air."""
    c = np.asarray(cloud_fraction, dtype=np.float64)
    if np.any((c < 0.0) | (c > 1.0)):
        raise ValueError("cloud_fraction must lie in [0, 1]")
    eps_clear = clear_sky_emissivity(e_vapour_hpa, t_air_k, formula)
    return np.asarray(eps_clear + (1.0 - eps_clear) * c, dtype=np.float64)


def longwave_down(
    t_air_k: Any,
    e_vapour_hpa: Any,
    cloud_fraction: Any,
    sky_view_factor: Any,
    t_surround_k: Any,
    formula: EmissivityFormula = "brunt",
) -> NDArray[np.float64]:
    """Q_LW↓ (W m⁻²) on a facet seeing the sky with factor V_s and surroundings at T_surround."""
    v_s = np.asarray(sky_view_factor, dtype=np.float64)
    if np.any((v_s < 0.0) | (v_s > 1.0)):
        raise ValueError("sky_view_factor must lie in [0, 1]")
    t_air = np.asarray(t_air_k, dtype=np.float64)
    t_sur = np.asarray(t_surround_k, dtype=np.float64)
    if np.any(t_sur <= 0.0):
        raise ValueError("t_surround_k must be positive kelvin")
    eps_sky = sky_emissivity(e_vapour_hpa, t_air, cloud_fraction, formula)
    q = v_s * eps_sky * SIGMA_SB * t_air**4 + (1.0 - v_s) * SIGMA_SB * t_sur**4
    return np.asarray(q, dtype=np.float64)


def longwave_down_from_sample(
    sample: WeatherSample,
    sky_view_factor: Any,
    t_surround_k: Any,
    formula: EmissivityFormula = "brunt",
) -> NDArray[np.float64]:
    """The same, fed by one WeatherSample (its vapour pressure is M8.2's Magnus form)."""
    return longwave_down(
        sample.t_air_k,
        sample.vapour_pressure_hpa,
        sample.cloud_fraction,
        sky_view_factor,
        t_surround_k,
        formula,
    )
