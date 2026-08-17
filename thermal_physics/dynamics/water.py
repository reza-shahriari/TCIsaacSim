"""Water / sea -- same lumped model, much larger effective thermal mass.

Implements docs/thermal_object_dynamics.md SS3. No new equation: construct a
LumpedThermalNode with water's constants (c_p ~= 4186 J/(kg*K), plus mixing
increases the *effective* C_th further -- see SS3) and add evaporative loss.
"""
from __future__ import annotations


def evaporative_loss_w(area_m2: float, wind_speed_m_s: float, t_water_k: float,
                        t_air_k: float, relative_humidity: float) -> float:
    """Bulk aerodynamic evaporative cooling term, Q_evap. SS3."""
    raise NotImplementedError("TODO: implement the bulk aerodynamic formula, SS3")
