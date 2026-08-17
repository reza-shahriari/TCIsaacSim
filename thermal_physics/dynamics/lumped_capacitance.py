"""General lumped thermal-capacitance model -- the engine-warm-up ODE.

Implements docs/thermal_object_dynamics.md SS1. Same equation family as
thermal_physics.detector.bolometer_temperature_rise -- a whole object
instead of a detector pixel.
"""
from __future__ import annotations


class LumpedThermalNode:
    """One thermal mass: C_th * dT/dt = Q_in(t) - G_th*(T - T_ambient). SS1."""

    def __init__(self, thermal_capacitance: float, thermal_conductance: float,
                 ambient_temp_k: float, initial_temp_k: float | None = None):
        self.thermal_capacitance = thermal_capacitance
        self.thermal_conductance = thermal_conductance
        self.ambient_temp_k = ambient_temp_k
        self._t = ambient_temp_k if initial_temp_k is None else initial_temp_k

    def step(self, dt_s: float, heat_input_w: float) -> float:
        """Advance one Euler step, return the updated temperature in K. SS1."""
        d_t = (heat_input_w - self.thermal_conductance * (self._t - self.ambient_temp_k)) \
            / self.thermal_capacitance * dt_s
        self._t += d_t
        return self._t

    @property
    def temperature_k(self) -> float:
        return self._t
