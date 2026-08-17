"""Ice -- phase change via enthalpy tracking.

Implements docs/thermal_object_dynamics.md SS4 (recommended enthalpy method,
not the effective-heat-capacity-spike shortcut).
"""
from __future__ import annotations


class EnthalpyThermalNode:
    """Tracks enthalpy H instead of T directly; T(H) has a flat plateau at
    T_melt while both phases coexist. SS4.

    Unit test target (docs/isaacsim_checklist.md 5.2): constant heat input
    should hold temperature_k at the melting point for m*L_f/heat_rate
    seconds before it starts rising.
    """

    def __init__(self, mass_kg: float, c_solid: float, c_liquid: float,
                 latent_heat_j_per_kg: float, t_melt_k: float,
                 initial_temp_k: float):
        raise NotImplementedError("TODO: implement SS4")

    def step(self, dt_s: float, heat_input_w: float) -> tuple[float, float]:
        """Advance one step. Returns (temperature_k, melt_fraction). SS4."""
        raise NotImplementedError("TODO: implement the piecewise H -> T map, SS4")
