"""Ice -- phase change via enthalpy tracking.

Implements docs/thermal_object_dynamics.md SS4 (the recommended enthalpy
method, not the effective-heat-capacity-spike shortcut).
"""
from __future__ import annotations

WATER_LATENT_HEAT_FUSION_J_PER_KG = 334_000.0


class EnthalpyThermalNode:
    """Tracks enthalpy H instead of T directly; T(H) has a flat plateau at
    T_melt while both phases coexist. SS4.

    H=0 is defined as "just starting to melt" (T=T_melt, melt_fraction=0),
    matching SS4's piecewise H -> T map exactly.
    """

    def __init__(self, mass_kg: float, c_solid: float, c_liquid: float,
                 latent_heat_j_per_kg: float, t_melt_k: float, initial_temp_k: float):
        self.mass_kg = mass_kg
        self.c_solid = c_solid
        self.c_liquid = c_liquid
        self.latent_heat_j_per_kg = latent_heat_j_per_kg
        self.t_melt_k = t_melt_k
        if initial_temp_k <= t_melt_k:
            self._h = mass_kg * c_solid * (initial_temp_k - t_melt_k)
        else:
            self._h = mass_kg * latent_heat_j_per_kg + mass_kg * c_liquid * (initial_temp_k - t_melt_k)

    def step(self, dt_s: float, heat_input_w: float) -> tuple[float, float]:
        """Advance one step. Returns (temperature_k, melt_fraction). SS4."""
        self._h += heat_input_w * dt_s
        return self._h_to_t()

    def _h_to_t(self) -> tuple[float, float]:
        h_melt_total = self.mass_kg * self.latent_heat_j_per_kg
        if self._h < 0:
            return self.t_melt_k + self._h / (self.mass_kg * self.c_solid), 0.0
        if self._h <= h_melt_total:
            return self.t_melt_k, self._h / h_melt_total
        return self.t_melt_k + (self._h - h_melt_total) / (self.mass_kg * self.c_liquid), 1.0
