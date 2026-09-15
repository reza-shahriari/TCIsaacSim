"""M6.8 — §6.4's two-node solver, and the factor of 400 hiding in its stability bound.

One node gives a surface that tracks the air and forgets yesterday. Two give it somewhere to put
the day's heat, which is the difference between a road that is still warm at midnight and one that
is not — and thermal crossover, which §6.3 calls one of the most operationally important phenomena
in thermal imaging, is a consequence of materials having *different* amounts of somewhere.

**§6.4 states its stability bound correctly and evaluates it wrongly.** The formula
`Δt < 2C₁/(h + 4εσT³ + 1/R₁₂)` is right; the claim that it "lands around 60–200 s" for thin
painted metal is not. For the §16.2 car-paint row, `1/R₁₂` is **37 500 W m⁻² K⁻¹** against
`h + 4εσT³ ≈ 43`, and the bound is **0.235 s** — 400 times smaller. The quoted range is what the
single-node bound gives at high wind, i.e. the formula with the term that dominates it removed.
Spec issue S39.

docs/physics-model.md §6.3, §6.4; ADR 0036, spec issue S39
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.balance import (
    SurfaceForcing,
    ThermalProperties,
    net_flux,
    rk2_step,
    steady_state_temperature,
)
from irsim.thermal.two_node import (
    GUARD_T_MAX_K,
    LumpedTwoNodeSolver,
    NodeLayer,
    TwoNodeProperties,
    TwoNodeState,
    contact_resistance,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
H_MAX = 30.0
NOON = SurfaceForcing(
    t_air_k=298.15,
    h_w_m2_k=12.0,
    q_solar_w_m2=600.0,
    q_longwave_down_w_m2=340.0,
    q_internal_w_m2=0.0,
)


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


def _properties(library, name: str, **kwargs) -> TwoNodeProperties:  # type: ignore[no-untyped-def]
    material = library[name]
    layer = NodeLayer.from_material(material)
    return TwoNodeProperties(
        surface=layer,
        substrate=layer,
        optical=ThermalProperties.from_material(material, 300.0, data_dir=DATA),
        **kwargs,
    )


# ---------------------------------------------------------------------------------------------
# the bound, and the spec inconsistency it exposes
# ---------------------------------------------------------------------------------------------


def test_the_car_paint_bound_is_a_quarter_of_a_second_not_a_minute(library) -> None:  # type: ignore[no-untyped-def]
    """⚠️ §6.4's own formula on §6.4's own example: **0.235 s**, against its claimed 60–200 s."""
    props = _properties(library, "car_paint_black")
    assert props.stability_limit_s(H_MAX) == pytest.approx(0.235, abs=0.005)

    radiative = 4.0 * props.optical.emissivity * SIGMA_SB * GUARD_T_MAX_K**3
    substrate = 1.0 / props.r12_m2k_w
    assert substrate == pytest.approx(37_500.0, rel=0.01)
    assert substrate / (H_MAX + radiative) == pytest.approx(874.0, rel=0.05), (
        "the substrate term should dominate the other two by nearly three orders"
    )

    # and the 60-200 s range is the *single-node* bound at high wind -- the same formula with the
    # term that dominates it removed
    single_node = 2.0 * props.surface.heat_capacity_j_m2_k / (H_MAX + radiative)
    assert 60.0 < single_node < 400.0


def test_a_thick_surface_is_comfortable_at_a_one_hertz_tick(library) -> None:  # type: ignore[no-untyped-def]
    """The problem is specific to *thin, conductive* layers, which is why it is a design decision
    and not a blanket one: asphalt and concrete have bounds of an hour or more."""
    for name, floor in (("asphalt_dry", 1000.0), ("concrete", 1000.0)):
        assert _properties(library, name).stability_limit_s(H_MAX) > floor
    # glass sits in between: a second is fine, a minute is not
    glass = _properties(library, "glass_windshield").stability_limit_s(H_MAX)
    assert 10.0 < glass < 200.0


def test_r12_is_the_sum_of_two_half_thickness_resistances(library) -> None:  # type: ignore[no-untyped-def]
    a = NodeLayer(
        thickness_m=0.002, conductivity_w_mk=45.0, density_kg_m3=7800, specific_heat_j_kgk=470
    )
    b = NodeLayer(
        thickness_m=0.010, conductivity_w_mk=0.8, density_kg_m3=2000, specific_heat_j_kgk=900
    )
    assert contact_resistance(a, b) == pytest.approx(
        0.002 / (2 * 45.0) + 0.010 / (2 * 0.8), rel=1e-12
    )


def test_the_guard_fires_at_construction_and_not_on_frame_four_thousand(library) -> None:  # type: ignore[no-untyped-def]
    """0.9× the bound constructs, 1.1× raises. In the constructor, because a caller that has
    chosen a tick has chosen it once, and finding out mid-render is finding out too late."""
    props = _properties(library, "car_paint_black")
    limit = props.stability_limit_s(H_MAX)
    start = TwoNodeState(300.0, 300.0)
    LumpedTwoNodeSolver(props, 0.9 * limit, H_MAX, start)
    with pytest.raises(ValueError, match="exceeds the §6.4 explicit bound"):
        LumpedTwoNodeSolver(props, 1.1 * limit, H_MAX, start)
    with pytest.raises(ValueError, match="dt_s must be positive"):
        LumpedTwoNodeSolver(props, 0.0, H_MAX, start)


def test_a_finite_back_resistance_without_a_deep_temperature_is_refused(library) -> None:  # type: ignore[no-untyped-def]
    """R₂d without T_deep is a path to an unspecified reservoir (ADR 0036)."""
    layer = NodeLayer.from_material(library["asphalt_dry"])
    optical = ThermalProperties.from_material(library["asphalt_dry"], 300.0, data_dir=DATA)
    with pytest.raises(ValueError, match="deep_temperature_k"):
        TwoNodeProperties(layer, layer, optical, back_resistance_m2k_w=0.5)
    TwoNodeProperties(layer, layer, optical, back_resistance_m2k_w=0.5, deep_temperature_k=288.0)


# ---------------------------------------------------------------------------------------------
# reduction to one node
# ---------------------------------------------------------------------------------------------


def test_an_isolated_surface_node_reproduces_the_single_node_solver(library) -> None:  # type: ignore[no-untyped-def]
    """R₁₂ → ∞ decouples the two nodes, and the surface must then be M6.7's, to **0.1 mK**."""
    material = library["car_paint_black"]
    surface = NodeLayer.from_material(material)
    insulating = NodeLayer(
        thickness_m=1.0, conductivity_w_mk=1e-9, density_kg_m3=1.0, specific_heat_j_kgk=1.0
    )
    optical = ThermalProperties.from_material(material, 300.0, data_dir=DATA)
    props = TwoNodeProperties(surface, insulating, optical)
    single = ThermalProperties(
        heat_capacity_j_m2_k=surface.heat_capacity_j_m2_k,
        emissivity=optical.emissivity,
        solar_absorptivity=optical.solar_absorptivity,
    )

    dt = 0.5 * props.stability_limit_s(H_MAX)
    solver = LumpedTwoNodeSolver(props, dt, H_MAX, TwoNodeState(280.0, 280.0))
    scalar = 280.0
    for _ in range(4000):
        solver.advance(NOON)
        scalar = float(rk2_step(scalar, dt, single, NOON))
    assert abs(solver.state.surface_k - scalar) < 1e-4


def test_an_adiabatic_pair_equilibrates_where_the_single_node_does(library) -> None:  # type: ignore[no-untyped-def]
    """With no path out the back, at steady state nothing flows through R₁₂ and both nodes sit at
    the one-node answer — whatever R₁₂ is."""
    props = _properties(library, "asphalt_dry")
    solver = LumpedTwoNodeSolver(props, 1.0, H_MAX, TwoNodeState(300.0, 300.0))
    state = solver.equilibrium(NOON)
    single = steady_state_temperature(props.optical, NOON)
    assert state.surface_k == pytest.approx(single, abs=1e-6)
    assert state.substrate_k == pytest.approx(single, abs=1e-6)


# ---------------------------------------------------------------------------------------------
# the equilibrium and the integrator
# ---------------------------------------------------------------------------------------------


def test_the_two_node_equilibrium_is_a_root_of_the_coupled_system(library) -> None:  # type: ignore[no-untyped-def]
    """Against brentq on the same residual, < 0.1 K, with a finite back boundary so the two
    nodes genuinely differ."""
    from scipy.optimize import brentq

    props = _properties(library, "asphalt_dry", back_resistance_m2k_w=0.4, deep_temperature_k=288.0)
    solver = LumpedTwoNodeSolver(props, 1.0, H_MAX, TwoNodeState(300.0, 295.0))
    state = solver.equilibrium(NOON)
    assert state.surface_k != pytest.approx(state.substrate_k, abs=0.5), "nodes should differ"

    def residual(t1: float) -> float:
        ratio = props.back_resistance_m2k_w / (props.r12_m2k_w + props.back_resistance_m2k_w)
        t2 = 288.0 + ratio * (t1 - 288.0)
        return float(net_flux(t1, props.optical, NOON)) - (t1 - t2) / props.r12_m2k_w

    assert state.surface_k == pytest.approx(brentq(residual, 150.0, 900.0, xtol=1e-10), abs=0.1)


def test_the_integrator_matches_the_linearised_matrix_exponential(library) -> None:  # type: ignore[no-untyped-def]
    """Linearise about the equilibrium and step 6 h both ways: < 10 mK.

    `expm` solves the linear system exactly, so any disagreement is the RK2 truncation error on
    the non-linear part — which near equilibrium is small, and the test says how small.
    """
    from scipy.linalg import expm

    props = _properties(library, "concrete", back_resistance_m2k_w=0.6, deep_temperature_k=288.0)
    dt = 10.0
    solver = LumpedTwoNodeSolver(props, dt, H_MAX, TwoNodeState(300.0, 295.0))
    eq = solver.equilibrium(NOON)
    base = np.array([eq.surface_k, eq.substrate_k])

    # Jacobian by central differences on the solver's own derivative
    jac = np.zeros((2, 2))
    for j in range(2):
        step = np.zeros(2)
        step[j] = 1e-4
        jac[:, j] = (
            solver.derivative(base + step, NOON) - solver.derivative(base - step, NOON)
        ) / (2e-4)

    offset = np.array([2.0, -1.5])
    solver._state = base + offset  # noqa: SLF001 - the linearisation needs a known start
    hours = 6.0
    for _ in range(int(hours * 3600 / dt)):
        solver.advance(NOON)
    numerical = solver._state - base  # noqa: SLF001
    analytic = expm(jac * hours * 3600.0) @ offset
    assert float(np.max(np.abs(numerical - analytic))) < 0.010


def test_energy_is_conserved_across_the_pair(library) -> None:  # type: ignore[no-untyped-def]
    """What enters the surface, minus what leaves the back, equals what the two nodes stored."""
    props = _properties(library, "asphalt_dry", back_resistance_m2k_w=0.4, deep_temperature_k=288.0)
    dt = 0.2 * props.stability_limit_s(H_MAX)
    solver = LumpedTwoNodeSolver(props, dt, H_MAX, TwoNodeState(290.0, 290.0))
    start = solver._state.copy()  # noqa: SLF001
    incoming = 0.0
    for _ in range(1000):
        half = solver._state + 0.5 * dt * solver.derivative(solver._state, NOON)  # noqa: SLF001
        surface_in = float(net_flux(half[0], props.optical, NOON))
        back_out = (half[1] - props.deep_temperature_k) / props.back_resistance_m2k_w
        incoming += dt * (surface_in - back_out)
        solver.advance(NOON)
    stored = props.surface.heat_capacity_j_m2_k * (solver._state[0] - start[0]) + (  # noqa: SLF001
        props.substrate.heat_capacity_j_m2_k * (solver._state[1] - start[1])  # noqa: SLF001
    )
    assert abs(incoming - stored) / abs(stored) < 1e-6


def test_the_diurnal_swing_is_ordered_by_areal_heat_capacity(library) -> None:  # type: ignore[no-untyped-def]
    """A surface with more heat capacity per square metre swings less over a day — which is the
    whole reason the substrate node exists, and the mechanism behind thermal crossover.

    ⚠️ The ordering variable is **C = ρ c δ**, not the thermal inertia P = √(ρck) the literature
    reaches for first. P is the right measure for a *semi-infinite* solid, where the heat has
    unbounded depth to diffuse into. These are finite layers on an adiabatic back, and by P the
    ordering is exactly wrong: car paint has the **highest** P of the three (12 800 against
    concrete's 1 700, because it is backed by steel) and the **largest** swing, because 1.2 mm of
    it holds 4.4 kJ m⁻² K⁻¹ against concrete's 202. P returns as the right variable once the back
    boundary is a deep reservoir rather than a wall, which is M6.10's ground case.
    """
    names = ("car_paint_black", "asphalt_dry", "concrete")
    capacity = {}
    inertia = {}
    swing = {}
    for name in names:
        material = library[name]
        thermal = material.spec.thermal
        capacity[name] = thermal.heat_capacity_j_m2_k
        inertia[name] = math.sqrt(
            thermal.density_kg_m3 * thermal.specific_heat_j_kgk * thermal.conductivity_w_mk
        )
        props = _properties(library, name)
        dt = min(60.0, 0.5 * props.stability_limit_s(H_MAX))
        solver = LumpedTwoNodeSolver(props, dt, H_MAX, TwoNodeState(288.0, 288.0))
        temps = []
        for step in range(int(24 * 3600 / dt)):
            hour = step * dt / 3600.0
            sun = max(0.0, 900.0 * math.sin(math.pi * (hour - 6.0) / 12.0))
            forcing = SurfaceForcing(
                t_air_k=288.0 + 6.0 * math.sin(math.pi * (hour - 8.0) / 12.0),
                h_w_m2_k=10.0,
                q_solar_w_m2=sun,
                q_longwave_down_w_m2=320.0,
            )
            solver.advance(forcing)
            if hour > 12.0:  # second half of the day, past the start transient
                temps.append(solver.state.surface_k)
        swing[name] = max(temps) - min(temps)

    by_capacity = sorted(names, key=lambda n: capacity[n])
    assert [swing[n] for n in by_capacity] == sorted(
        (swing[n] for n in by_capacity), reverse=True
    ), {"capacity": capacity, "swing": swing}
    assert swing["car_paint_black"] > 2.0 * swing["concrete"], swing

    # and P would have got it backwards, which is the point of the note above
    by_inertia = sorted(names, key=lambda n: inertia[n])
    assert [swing[n] for n in by_inertia] != sorted((swing[n] for n in by_inertia), reverse=True), {
        "inertia": inertia,
        "swing": swing,
    }
