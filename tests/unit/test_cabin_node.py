"""M6.14 — the cabin node, and the greenhouse it makes of a parked car (§6.6, ADR 0038).

A vehicle panel is not a slab on the ground: behind it is a closed volume of air the sun has been
heating through the glass all afternoon. Give the roof an adiabatic back and it comes out at
ambient-plus-solar; give it the cabin and it is **4.8 K hotter**, because it is losing heat into
air that is already hotter than the outside. That difference is what makes a parked car read the
way a parked car reads.

The trap the constructor now refuses is worth stating: glass is both the cabin's solar **inlet**
and one of its largest conduction **paths**. A cabin given the inlet without the path reaches
102 °C at noon instead of 80 °C — not obviously wrong, just wrong.

docs/physics-model.md §6.6; ADR 0038
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.thermal.balance import SurfaceForcing, ThermalProperties, steady_state_temperature
from irsim.thermal.cabin import CabinNode, CabinPanel, CabinState

PAINT = ThermalProperties(7800 * 470 * 0.0012, 0.90, 0.94)
GLASS = ThermalProperties(2500 * 840 * 0.005, 0.88, 0.10)
NOON = SurfaceForcing(t_air_k=298.15, h_w_m2_k=12.0, q_solar_w_m2=850.0, q_longwave_down_w_m2=340.0)
NIGHT = SurfaceForcing(t_air_k=288.15, h_w_m2_k=8.0, q_longwave_down_w_m2=300.0)


def _panels() -> list[CabinPanel]:
    return [
        CabinPanel("roof", PAINT, 2.2),
        CabinPanel("bonnet", PAINT, 1.6),
        CabinPanel("doors", PAINT, 3.2, 0.5),
        CabinPanel("glazing", GLASS, 2.6, 0.17),
    ]


def _cabin(**kwargs) -> CabinNode:  # type: ignore[no-untyped-def]
    kwargs.setdefault("glazing_panel", "glazing")
    return CabinNode(_panels(), **kwargs)


# ---------------------------------------------------------------------------------------------
# the greenhouse
# ---------------------------------------------------------------------------------------------


def test_the_cabin_makes_the_roof_hotter_than_an_adiabatic_back_would() -> None:
    """The headline: **+4.8 K** at noon, against §6.6's requirement of more than 2 K."""
    cabin = _cabin()
    state = cabin.equilibrium(NOON)
    roof = float(state.panels_k[0])
    adiabatic = steady_state_temperature(PAINT, NOON)
    assert roof - adiabatic > 2.0
    assert roof - adiabatic == pytest.approx(4.8, abs=1.0)
    assert state.cabin_k > roof, "the air behind the panel must be the hotter of the two"


def test_the_cabin_reaches_a_temperature_a_parked_car_actually_reaches() -> None:
    """80 °C in strong sun. A sealed car in the sun really does reach 60-80 °C."""
    state = _cabin().equilibrium(NOON)
    assert 60.0 < state.cabin_k - 273.15 < 90.0, state.cabin_k - 273.15


def test_the_glazing_stays_cool_because_it_transmits_rather_than_absorbs() -> None:
    """α_sol 0.10: the glass lets the sun through rather than taking it, so it sits far below the
    painted panels even while being the thing that heats the cabin."""
    state = _cabin().equilibrium(NOON)
    glazing = float(state.panels_k[-1])
    roof = float(state.panels_k[0])
    assert glazing < roof - 20.0


def test_a_cabin_without_its_glazing_in_the_panel_list_is_refused() -> None:
    """⚠️ The trap, as a constructor error. Glass is both the inlet and the path."""
    with pytest.raises(ValueError, match="glazing_panel must name"):
        CabinNode(_panels())
    with pytest.raises(ValueError, match="glazing_panel must name"):
        CabinNode(_panels(), glazing_panel="windscreen")

    # and the size of the mistake, measured: drop the glass path and the cabin runs ~22 K hot
    without_path = CabinNode(
        [p for p in _panels() if p.name != "glazing"] + [CabinPanel("glazing", GLASS, 2.6, 50.0)],
        glazing_panel="glazing",
    )
    assert without_path.equilibrium(NOON).cabin_k - _cabin().equilibrium(NOON).cabin_k > 15.0


# ---------------------------------------------------------------------------------------------
# the coupled system
# ---------------------------------------------------------------------------------------------


def test_the_coupled_equilibrium_is_a_root_of_both_equations() -> None:
    """Against `fsolve` on the full residual vector, < 0.1 K."""
    from scipy.optimize import fsolve

    cabin = _cabin()
    state = cabin.equilibrium(NOON)

    def residual(x: np.ndarray) -> np.ndarray:
        d_cabin, d_panels = cabin.derivative(CabinState(float(x[0]), x[1:]), NOON)
        return np.concatenate([[d_cabin], d_panels])

    start = np.concatenate([[state.cabin_k], state.panels_k]) + 5.0
    solved = fsolve(residual, start, full_output=False)
    assert abs(solved[0] - state.cabin_k) < 0.1
    assert float(np.max(np.abs(solved[1:] - state.panels_k))) < 0.1
    assert float(np.max(np.abs(residual(np.concatenate([[state.cabin_k], state.panels_k]))))) < 1e-9


def test_energy_is_conserved_across_the_cabin_and_its_panels() -> None:
    """What enters the panels and the glazing, minus what leaves through infiltration, equals what
    the whole assembly stored. To 1e-6."""
    from irsim.thermal.balance import net_flux

    cabin = _cabin()
    dt = 0.05
    state = CabinState(290.0, np.full(len(cabin.panels), 290.0))
    incoming = 0.0
    for _ in range(4000):
        d_cabin, d_panels = cabin.derivative(state, NOON)
        half = CabinState(state.cabin_k + 0.5 * dt * d_cabin, state.panels_k + 0.5 * dt * d_panels)
        outside = sum(
            panel.area_m2 * float(net_flux(t, panel.properties, NOON))
            for panel, t in zip(cabin.panels, half.panels_k, strict=True)
        )
        solar = cabin.glazing_transmittance * cabin.glazing_area_m2 * NOON.q_solar_w_m2
        infiltration = cabin.infiltration_w_k * (NOON.t_air_k - half.cabin_k)
        incoming += dt * (outside + solar + infiltration)
        state = cabin.advance(state, NOON, dt)
    stored = cabin.capacity_j_k * (state.cabin_k - 290.0) + float(
        sum(
            panel.area_m2 * panel.properties.heat_capacity_j_m2_k * (t - 290.0)
            for panel, t in zip(cabin.panels, state.panels_k, strict=True)
        )
    )
    assert abs(incoming - stored) / abs(stored) < 1e-6


def test_the_night_relaxation_follows_the_lumped_time_constant() -> None:
    """τ = C_cab / (Σ A/R + ṁ c_p) — **within 2 %**, with the panels pinned, which is what the
    formula assumes.

    ⚠️ Let the panels move and the coupled system relaxes **33 % slower** (12.0 min against the
    lumped 9.0 min), because a cooling cabin drags its panels down and they feed heat back. That
    is not an error in the formula; it is the formula's own assumption showing up as a number, and
    it is worth knowing which one a scene is quoting.
    """
    cabin = _cabin()
    equilibrium = cabin.equilibrium(NIGHT)
    tau = cabin.relaxation_time_constant_s(equilibrium, NIGHT)
    assert 300.0 < tau < 1800.0

    # panels pinned: exactly the lumped first-order system the formula describes
    dt, elapsed = tau / 400.0, 0.0
    cabin_k = equilibrium.cabin_k + 1.0
    while cabin_k - equilibrium.cabin_k > 1.0 / math.e:
        state = CabinState(cabin_k, equilibrium.panels_k.copy())
        d_cabin, _ = cabin.derivative(state, NIGHT)
        half = CabinState(cabin_k + 0.5 * dt * d_cabin, equilibrium.panels_k.copy())
        d_cabin, _ = cabin.derivative(half, NIGHT)
        cabin_k += dt * d_cabin
        elapsed += dt
    assert elapsed == pytest.approx(tau, rel=0.02)

    # panels free: slower, and the ratio is the thing to record
    state = CabinState(equilibrium.cabin_k + 1.0, equilibrium.panels_k.copy())
    coupled = 0.0
    while state.cabin_k - equilibrium.cabin_k > 1.0 / math.e:
        state = cabin.advance(state, NIGHT, dt)
        coupled += dt
    assert coupled / tau == pytest.approx(1.33, abs=0.1)


def test_a_parked_car_reads_below_ambient_on_a_clear_night() -> None:
    """The roof radiates to a −40 °C sky and drags the cabin with it — which is why cars get
    frost on the roof when the air never goes below freezing."""
    state = _cabin().equilibrium(NIGHT)
    assert state.cabin_k < NIGHT.t_air_k - 2.0
    assert float(state.panels_k[0]) < NIGHT.t_air_k - 2.0


def test_infiltration_pulls_the_cabin_back_toward_ambient() -> None:
    """The term that stops a sealed car reaching absurd temperatures."""
    sealed = _cabin(air_changes_per_hour=0.0).equilibrium(NOON).cabin_k
    leaky = _cabin(air_changes_per_hour=20.0).equilibrium(NOON).cabin_k
    assert sealed > leaky
    assert abs(leaky - NOON.t_air_k) < abs(sealed - NOON.t_air_k)


def test_bad_geometry_and_bad_state_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one bounding panel"):
        CabinNode([], glazing_panel="x")
    with pytest.raises(ValueError, match="transmittance"):
        _cabin(glazing_transmittance=1.5)
    with pytest.raises(ValueError, match="area must be positive"):
        CabinPanel("bad", PAINT, 0.0)
    cabin = _cabin()
    with pytest.raises(ValueError, match="expected 4 panel temperatures"):
        cabin.derivative(CabinState(300.0, np.full(2, 300.0)), NOON)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        cabin.advance(CabinState(300.0, np.full(4, 300.0)), NOON, 0.0)
