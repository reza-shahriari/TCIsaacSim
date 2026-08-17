"""Water / sea -- same lumped model, much larger effective thermal mass.

Implements docs/thermal_object_dynamics.md SS3. No new equation: construct a
LumpedThermalNode with water's constants (c_p ~= 4186 J/(kg*K), plus mixing
increases the *effective* C_th further -- see SS3) and add evaporative loss.
"""
from __future__ import annotations
import numpy as np

WATER_SPECIFIC_HEAT_J_PER_KG_K = 4186.0
LATENT_HEAT_VAPORIZATION_J_PER_KG = 2.26e6


def _saturation_vapor_pressure_pa(temperature_k: float) -> float:
    """Magnus/Tetens approximation, valid for typical terrestrial temperatures."""
    t_c = temperature_k - 273.15
    return 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))


def evaporative_loss_w(area_m2: float, wind_speed_m_s: float, t_water_k: float,
                        t_air_k: float, relative_humidity: float,
                        k_evap: float = 1e-8) -> float:
    """Bulk aerodynamic evaporative cooling term, Q_evap. SS3.

    k_evap is a tunable bulk transfer coefficient -- the illustrative default
    is not calibrated against any specific real site/conditions (Phase 7).
    """
    e_sat_water = _saturation_vapor_pressure_pa(t_water_k)
    e_air = relative_humidity * _saturation_vapor_pressure_pa(t_air_k)
    deficit = max(0.0, e_sat_water - e_air)
    return LATENT_HEAT_VAPORIZATION_J_PER_KG * area_m2 * k_evap * wind_speed_m_s * deficit
