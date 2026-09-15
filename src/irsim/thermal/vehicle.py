"""§6.6's active heat sources: scripted, not predicted, and where most of the signal lives.

§6.6 is explicit that these are "scripted, not predicted", so this module is a set of *schedules*
driven by a :class:`VehicleState` trace rather than a combustion model. That is a deliberate
fidelity choice and not a shortcut: predicting an exhaust tip temperature needs an engine map, a
catalyst model and a flow solver, and the result would still be authored parameters wearing a
physics costume. A schedule with §6.6's own ΔT ranges and time constants is honest about what it is.

Three laws, each chosen because it has a closed form to test against:

* **First-order warm-up and cool-down.** `T = T_air + ΔT_max (1 − e^{−t/τ})` while the source runs,
  Newton cooling after it stops. §6.6 lists a time constant for every row, so this is the law the
  table is already describing.
* **Brakes are an energy deposit, not a schedule.** A braking event dumps ½m(v₁² − v₂²) into the
  discs, so ΔT = E/(m c_p) is arithmetic, and only the *cool-down* is a time constant. Scripting a
  brake temperature directly would make it independent of how hard the car actually braked.
* **Tyres follow speed, not time.** §6.6's "+10 … +35 K, rises with speed" is a steady-state
  relation like the aerial heat sources of ADR 0072 — the contact patch reaches a temperature that
  depends on how fast it is being flexed, with a long time constant to get there.

**Everything here is ESTIMATED.** The defaults are §6.6's own table midpoints; no row is a
measurement, and the module says so in one place rather than in nine.

docs/physics-model.md §6.6; ADR 0038
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "VehicleState",
    "HeatSourceSpec",
    "VEHICLE_HEAT_SOURCES",
    "HUMAN_BODY",
    "first_order_rise",
    "newton_cool",
    "brake_temperature_rise_k",
    "tyre_delta_t_k",
    "source_temperature_k",
]


@dataclass(frozen=True)
class VehicleState:
    """What the vehicle is doing at one instant. The only input the schedules take."""

    t_s: float
    speed_m_s: float = 0.0
    accel_m_s2: float = 0.0
    ignition: bool = False
    braking: bool = False

    def __post_init__(self) -> None:
        if self.speed_m_s < 0.0:
            raise ValueError("speed must be non-negative (use a heading, not a sign)")


@dataclass(frozen=True)
class HeatSourceSpec:
    """One §6.6 row: how hot it gets, how fast, and how fast it forgets.

    ``delta_t_max_k`` is the steady rise over ambient at full load; ``tau_rise_s`` and
    ``tau_cool_s`` are the warm-up and cool-down constants. All ESTIMATED from §6.6's table.
    """

    name: str
    delta_t_max_k: float
    tau_rise_s: float
    tau_cool_s: float
    load_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.delta_t_max_k < 0.0:
            raise ValueError("delta_t_max_k must be non-negative")
        if self.tau_rise_s <= 0.0 or self.tau_cool_s <= 0.0:
            raise ValueError("time constants must be positive")


#: §6.6's table, as midpoints of its quoted ranges. ESTIMATED, every row.
VEHICLE_HEAT_SOURCES: dict[str, HeatSourceSpec] = {
    "engine_bay": HeatSourceSpec("engine_bay", 65.0, 750.0, 1800.0),
    "exhaust_manifold": HeatSourceSpec("exhaust_manifold", 250.0, 240.0, 600.0),
    "catalytic_converter": HeatSourceSpec("catalytic_converter", 200.0, 390.0, 900.0),
    "exhaust_pipe": HeatSourceSpec("exhaust_pipe", 120.0, 360.0, 700.0),
    "exhaust_tip": HeatSourceSpec("exhaust_tip", 90.0, 360.0, 500.0),
    "brake_disc": HeatSourceSpec("brake_disc", 225.0, 30.0, 300.0),
    "tyre": HeatSourceSpec("tyre", 22.5, 1200.0, 1800.0),
}

#: §6.6's clothed-human row: "+8 … +15 K, effective ε ≈ 0.98; face is warmest". The face runs at
#: the top of the range and covered skin at the bottom, which is the ordering a detector sees.
HUMAN_BODY: dict[str, float] = {"face": 15.0, "hands": 12.0, "clothed_torso": 8.0}


def first_order_rise(t_s: Any, delta_t_max_k: float, tau_s: float) -> NDArray[np.float64]:
    """ΔT(t) = ΔT_max (1 − e^{−t/τ}); exactly (1 − 1/e) of the way there at t = τ."""
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("elapsed time cannot be negative")
    return np.asarray(delta_t_max_k * (1.0 - np.exp(-t / tau_s)))


def newton_cool(t_s: Any, delta_t0_k: float, tau_s: float) -> NDArray[np.float64]:
    """ΔT(t) = ΔT₀ e^{−t/τ}. §6.6: "Newton cooling is exactly right here"."""
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    t = np.asarray(t_s, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("elapsed time cannot be negative")
    return np.asarray(delta_t0_k * np.exp(-t / tau_s))


def brake_temperature_rise_k(
    mass_kg: float,
    speed_from_m_s: float,
    speed_to_m_s: float,
    disc_mass_kg: float,
    disc_specific_heat_j_kgk: float = 500.0,
    fraction_to_discs: float = 0.9,
) -> float:
    """ΔT = f · ½ m (v₁² − v₂²) / (m_disc c_p) — arithmetic, not a schedule.

    Scripting a brake temperature directly would make it independent of how hard the car actually
    braked, which is the one thing a braking cue is supposed to carry. ``fraction_to_discs``
    accounts for the rest going into the pads, the tyres and the air (ESTIMATED).
    """
    if speed_to_m_s > speed_from_m_s:
        raise ValueError("braking means slowing down")
    for name, value in (("mass_kg", mass_kg), ("disc_mass_kg", disc_mass_kg)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < fraction_to_discs <= 1.0:
        raise ValueError("fraction_to_discs must lie in (0, 1]")
    energy = 0.5 * mass_kg * (speed_from_m_s**2 - speed_to_m_s**2)
    return float(fraction_to_discs * energy / (disc_mass_kg * disc_specific_heat_j_kgk))


def tyre_delta_t_k(
    speed_m_s: Any,
    reference_speed_m_s: float = 30.0,
    spec: HeatSourceSpec | None = None,
) -> NDArray[np.float64]:
    """§6.6's "+10 … +35 K, rises with speed", as a steady-state relation in u = v/v_ref.

    Like ADR 0072's aerial heat sources this has **no time constant in it**: the tyre reaches the
    temperature its current speed implies. That is defensible only where the speed varies slowly
    against the tyre's own 10–30 min constant, and a trace that swings the speed in seconds will
    produce a temperature swing no rubber could follow.
    """
    source = VEHICLE_HEAT_SOURCES["tyre"] if spec is None else spec
    if reference_speed_m_s <= 0.0:
        raise ValueError("reference_speed_m_s must be positive")
    v = np.asarray(speed_m_s, dtype=np.float64)
    if np.any(v < 0.0):
        raise ValueError("speed must be non-negative")
    u = np.clip(v / reference_speed_m_s, 0.0, 1.0)
    return np.asarray(10.0 + (source.delta_t_max_k * 2.0 - 10.0 - 10.0) * u**source.load_exponent)


@dataclass
class SourceHistory:
    """Integrates one source's first-order response along a :class:`VehicleState` trace."""

    spec: HeatSourceSpec
    delta_t_k: float = 0.0
    _last_t_s: float | None = field(default=None, repr=False)

    def step(self, state: VehicleState, load: float = 1.0) -> float:
        """Advance to ``state.t_s`` under a load fraction in [0, 1]; return ΔT over ambient."""
        if not 0.0 <= load <= 1.0:
            raise ValueError("load must lie in [0, 1]")
        if self._last_t_s is None:
            self._last_t_s = state.t_s
            return self.delta_t_k
        dt = state.t_s - self._last_t_s
        if dt < 0.0:
            raise ValueError("a vehicle trace must move forward in time")
        self._last_t_s = state.t_s
        target = self.spec.delta_t_max_k * load**self.spec.load_exponent
        tau = self.spec.tau_rise_s if target > self.delta_t_k else self.spec.tau_cool_s
        # exact exponential step, so the result does not depend on the trace's sample spacing
        alpha = 1.0 - math.exp(-dt / tau)
        self.delta_t_k += alpha * (target - self.delta_t_k)
        return self.delta_t_k


def source_temperature_k(t_air_k: float, delta_t_k: float) -> float:
    """T = T_air + ΔT. The one place ambient enters, so a source cannot drift off the weather."""
    return float(t_air_k + delta_t_k)
