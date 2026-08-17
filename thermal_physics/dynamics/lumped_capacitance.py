"""General lumped thermal-capacitance model -- the engine-warm-up ODE.

Implements docs/thermal_object_dynamics.md SS1. Same equation family as
thermal_physics.detector.bolometer_temperature_rise -- a whole object
instead of a detector pixel.
"""
from __future__ import annotations


class LumpedThermalNode:
    """One thermal mass: C_th * dT/dt = Q_in(t) - G_th*(T - T_ambient). SS1.

    Unit test target (docs/isaacsim_checklist.md 3.1): with constant Q_in,
    step() repeatedly should match the closed form
        T(t) = T_ambient + (Q_run/G_th) * (1 - exp(-t/tau))
    within a small tolerance.
    """

    def __init__(self, thermal_capacitance: float, thermal_conductance: float,
                 ambient_temp_k: float, initial_temp_k: float | None = None):
        raise NotImplementedError("TODO: implement SS1")

    def step(self, dt_s: float, heat_input_w: float) -> float:
        """Advance one Euler step, return the updated temperature in K. SS1."""
        raise NotImplementedError("TODO: implement the discretized update, SS1")

    @property
    def temperature_k(self) -> float:
        raise NotImplementedError("TODO: implement SS1")
