"""M6.7 — §6.1's surface energy balance: the equation, its step, and its fixed point.

    C dT/dt = α_sol Q_sol + Q_LW↓ − ε σ T⁴ − h (T − T_air) + q_int

Four checks, each of which fails for a different reason if the balance is wrong: the steady state
against an independent root, energy conservation across a long run, the linearised relaxation time,
and the one case with a closed-form answer (ε = 0, where the T⁴ term disappears and equilibrium is
pure algebra).

The one design rule worth stating: **a scene cannot author an emissivity here.** It comes from the
material's optical data through M7.8's total hemispherical value, because a `thermal:` block with
its own ε would let one scene radiate at 0.95 while the camera sees 0.88, and nothing would catch
it — the two numbers live in different files and are never compared.

docs/physics-model.md §6.1, §6.2, §15 T1; ADR 0036, ADR 0043
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
    linearised_time_constant_s,
    net_flux,
    rk2_step,
    stability_limit_s,
    steady_state_temperature,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"

#: A painted panel: 1.2 mm of steel-backed paint, ε 0.90, α_sol 0.94.
PAINT = ThermalProperties(
    heat_capacity_j_m2_k=7800 * 470 * 0.0012, emissivity=0.90, solar_absorptivity=0.94
)
NIGHT = SurfaceForcing(
    t_air_k=288.15, h_w_m2_k=8.0, q_longwave_down_w_m2=300.0, q_internal_w_m2=50.0
)
NOON = SurfaceForcing(
    t_air_k=298.15,
    h_w_m2_k=12.0,
    q_solar_w_m2=600.0,
    q_longwave_down_w_m2=340.0,
    q_internal_w_m2=50.0,
)


# ---------------------------------------------------------------------------------------------
# the equation
# ---------------------------------------------------------------------------------------------


def test_the_net_flux_is_the_sum_of_section_6_1s_five_terms() -> None:
    """Rebuilt term by term, so a sign error or a missing term shows up as a number."""
    t = 310.0
    expected = (
        PAINT.solar_absorptivity * NOON.q_solar_w_m2
        + PAINT.emissivity * NOON.q_longwave_down_w_m2
        - PAINT.emissivity * SIGMA_SB * t**4
        - NOON.h_w_m2_k * (t - NOON.t_air_k)
        + NOON.q_internal_w_m2
    )
    assert float(net_flux(t, PAINT, NOON)) == pytest.approx(expected, rel=1e-12)


def test_the_flux_is_strictly_decreasing_in_temperature() -> None:
    """Which is what makes the equilibrium unique and bisection safe."""
    temps = np.linspace(200.0, 500.0, 601)
    flux = net_flux(temps, PAINT, NOON)
    assert np.all(np.diff(flux) < 0.0)


def test_the_downwelling_longwave_is_absorbed_with_epsilon() -> None:
    """⚠️ §6.1 writes **ε Q_LW↓**, "absorbed sky/env" — and the first version of this module wrote
    `Q_LW↓`, with a test asserting that was correct.

    Kirchhoff: a surface absorbs the same fraction of incident longwave that it emits. Dropping
    that ε is invisible on a painted surface — 10 % of one term — and catastrophic on a metal,
    where it hands a panel with ε = 0.09 the full ~320 W m⁻² of sky radiation while letting it
    emit only a tenth of a blackbody. That put an aluminium panel **19 K above the air at 03:00**
    in the M6.13 facet scene, which is how it was found: not by re-reading §6.1, but by looking at
    a number that could not be true.

    `ε Q_LW↓ − ε σ T⁴` is algebraically the net-exchange form `ε σ (ε_sky T_air⁴ − T⁴)`. The two
    terms are kept apart in the code only because Q_LW↓ arrives from §6.5 already carrying the
    sky's own emissivity, its cloud fraction and the facet's view factor.
    """
    import dataclasses

    doubled = dataclasses.replace(NIGHT, q_longwave_down_w_m2=2.0 * NIGHT.q_longwave_down_w_m2)
    delta = float(net_flux(300.0, PAINT, doubled)) - float(net_flux(300.0, PAINT, NIGHT))
    assert delta == pytest.approx(PAINT.emissivity * NIGHT.q_longwave_down_w_m2, rel=1e-12)

    # a low-emissivity surface is nearly deaf to the sky, which is the whole point
    mirror = ThermalProperties(
        heat_capacity_j_m2_k=5000.0, emissivity=0.09, solar_absorptivity=0.15
    )
    mirror_delta = float(net_flux(300.0, mirror, doubled)) - float(net_flux(300.0, mirror, NIGHT))
    assert mirror_delta < 0.15 * delta

    # and the two forms agree: eps Q_LW - eps sigma T^4 == eps sigma (eps_sky T_air^4 - T^4)
    from irsim.radiometry.constants import SIGMA_SB

    eps_sky = NIGHT.q_longwave_down_w_m2 / (SIGMA_SB * NIGHT.t_air_k**4)
    separate = (
        PAINT.emissivity * NIGHT.q_longwave_down_w_m2 - PAINT.emissivity * SIGMA_SB * 300.0**4
    )
    folded = PAINT.emissivity * SIGMA_SB * (eps_sky * NIGHT.t_air_k**4 - 300.0**4)
    assert separate == pytest.approx(folded, rel=1e-12)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="kelvin"):
        SurfaceForcing(t_air_k=-5.0, h_w_m2_k=8.0)
    with pytest.raises(ValueError, match="emissivity"):
        ThermalProperties(heat_capacity_j_m2_k=1000.0, emissivity=1.5, solar_absorptivity=0.5)
    with pytest.raises(ValueError, match="capacity"):
        ThermalProperties(heat_capacity_j_m2_k=0.0, emissivity=0.9, solar_absorptivity=0.5)
    with pytest.raises(ValueError, match="positive"):
        net_flux(-10.0, PAINT, NIGHT)


# ---------------------------------------------------------------------------------------------
# the steady state
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("forcing", [NIGHT, NOON], ids=["night", "noon"])
def test_the_steady_state_is_a_root_of_the_flux(forcing: SurfaceForcing) -> None:
    """Against an independent root finder, < 0.1 K — and the residual flux is ~0."""
    from scipy.optimize import brentq

    ours = steady_state_temperature(PAINT, forcing)
    theirs = brentq(lambda t: float(net_flux(t, PAINT, forcing)), 150.0, 900.0, xtol=1e-10)
    assert abs(ours - theirs) < 0.1
    assert abs(float(net_flux(ours, PAINT, forcing))) < 1e-6


def test_a_zero_emissivity_surface_has_a_closed_form_equilibrium() -> None:
    """ε = 0 removes the T⁴ term and the answer is algebra: T = T_air + (αQ + q)/h, to 1e-9."""
    grey = ThermalProperties(heat_capacity_j_m2_k=5000.0, emissivity=0.0, solar_absorptivity=0.5)
    forcing = SurfaceForcing(t_air_k=295.0, h_w_m2_k=10.0, q_solar_w_m2=800.0, q_internal_w_m2=40.0)
    expected = 295.0 + (0.5 * 800.0 + 40.0) / 10.0
    assert steady_state_temperature(grey, forcing) == pytest.approx(expected, abs=1e-9)


def test_an_equilibrium_outside_the_bracket_raises_rather_than_returning_an_edge() -> None:
    hot = SurfaceForcing(t_air_k=300.0, h_w_m2_k=0.1, q_solar_w_m2=500_000.0)
    with pytest.raises(ValueError, match="outside the bracket"):
        steady_state_temperature(PAINT, hot)


def test_sunlight_raises_the_equilibrium_and_wind_pulls_it_back_to_air() -> None:
    import dataclasses

    hot = steady_state_temperature(PAINT, NOON)
    assert hot > NOON.t_air_k + 5.0
    windy = steady_state_temperature(PAINT, dataclasses.replace(NOON, h_w_m2_k=100.0))
    assert abs(windy - NOON.t_air_k) < abs(hot - NOON.t_air_k)


# ---------------------------------------------------------------------------------------------
# the integrator
# ---------------------------------------------------------------------------------------------


def test_rk2_conserves_energy_over_a_thousand_steps() -> None:
    """C ΔT = Σ dt · f(T_mid), to **1e-12** over 1000 steps — an identity, not a tolerance.

    The midpoint rule *is* `T_{n+1} = T_n + dt f(T_half)/C`, so the energy it puts into the node
    is exactly `dt f(T_half)` per step. Anything looser than machine precision here would mean
    the step and the flux had come apart.
    """
    dt = 0.05 * stability_limit_s(300.0, PAINT, NOON)
    capacity = PAINT.heat_capacity_j_m2_k
    t = 250.0
    energy = 0.0
    for _ in range(1000):
        half = t + 0.5 * dt * float(net_flux(t, PAINT, NOON)) / capacity
        energy += dt * float(net_flux(half, PAINT, NOON))
        t = float(rk2_step(t, dt, PAINT, NOON))
    stored = capacity * (t - 250.0)
    assert abs(energy - stored) / abs(stored) < 1e-12


def test_the_midpoint_and_trapezoid_quadratures_are_not_the_same_thing() -> None:
    """Which is why the test above uses the midpoint: RK2 is that rule and not the other one.

    Accounting the same run with a trapezoid of the endpoint fluxes leaves a **2.4e-3** relative
    residual — the difference between two second-order quadratures of a curved integrand, not an
    error in the step. A test that used the trapezoid and loosened its tolerance until it passed
    would be measuring that difference and calling it conservation.
    """
    dt = 0.05 * stability_limit_s(300.0, PAINT, NOON)
    capacity = PAINT.heat_capacity_j_m2_k
    t = 250.0
    trapezoid = 0.0
    for _ in range(1000):
        before = float(net_flux(t, PAINT, NOON))
        t = float(rk2_step(t, dt, PAINT, NOON))
        trapezoid += 0.5 * (before + float(net_flux(t, PAINT, NOON))) * dt
    stored = capacity * (t - 250.0)
    assert abs(trapezoid - stored) / abs(stored) == pytest.approx(2.4e-3, rel=0.2)


def test_the_step_relaxes_to_the_steady_state_it_predicts() -> None:
    equilibrium = steady_state_temperature(PAINT, NOON)
    dt = 0.1 * stability_limit_s(equilibrium, PAINT, NOON)
    t = 250.0
    for _ in range(20_000):
        t = float(rk2_step(t, dt, PAINT, NOON))
    assert t == pytest.approx(equilibrium, abs=1e-6)


def test_the_relaxation_follows_the_linearised_time_constant() -> None:
    """τ = C/(h + 4εσT³) within 1 %, measured from a small perturbation.

    The 4εσT³ term is not a correction: at this equilibrium (≈289 K, ε = 0.9) it is
    **4.88 W m⁻² K⁻¹** against a convection coefficient of 8, so a model that left it out would
    predict a surface **1.6× slower** than it is.
    """
    equilibrium = steady_state_temperature(PAINT, NIGHT)
    tau = linearised_time_constant_s(equilibrium, PAINT, NIGHT)
    radiative_h = 4.0 * PAINT.emissivity * SIGMA_SB * equilibrium**3
    assert radiative_h == pytest.approx(4.88, abs=0.2)
    assert radiative_h > 0.5 * NIGHT.h_w_m2_k, "the radiative term is not negligible"

    dt = tau / 200.0
    t = equilibrium + 0.5  # small, so the linearisation is the right reference
    elapsed = 0.0
    while t - equilibrium > 0.5 / math.e:
        t = float(rk2_step(t, dt, PAINT, NIGHT))
        elapsed += dt
    assert elapsed == pytest.approx(tau, rel=0.01)


def test_the_stability_limit_is_reported_rather_than_enforced() -> None:
    """A caller stepping a thin panel with a scene-sized tick usually wants the steady state."""
    limit = stability_limit_s(300.0, PAINT, NOON)
    assert limit == pytest.approx(2.0 * linearised_time_constant_s(300.0, PAINT, NOON), rel=1e-12)
    # a thin panel is fast: a few minutes of scene tick is orders of magnitude past its bound
    assert limit < 2000.0
    assert float(rk2_step(300.0, 10.0 * limit, PAINT, NOON)) != 300.0  # runs; does not raise
    with pytest.raises(ValueError, match="dt_s must be positive"):
        rk2_step(300.0, 0.0, PAINT, NOON)


# ---------------------------------------------------------------------------------------------
# where ε comes from
# ---------------------------------------------------------------------------------------------


def test_emissivity_comes_from_the_optical_data_and_not_from_the_thermal_block() -> None:
    """ADR 0043's rule, enforced by there being nowhere else to get one.

    `ThermalSpec` has no emissivity field at all, and `from_material` reaches through M7.8's
    hemispherical integral. Two authored copies of ε is two chances to disagree, in files that
    are never compared.
    """
    library = MaterialLibrary.load(REPO / "configs" / "materials")
    material = library["car_paint_black"]
    assert not hasattr(material.spec.thermal, "emissivity")
    props = ThermalProperties.from_material(material, 300.0, data_dir=DATA)
    assert props.heat_capacity_j_m2_k == pytest.approx(
        material.spec.thermal.heat_capacity_j_m2_k, rel=1e-12
    )
    assert props.solar_absorptivity == material.spec.thermal.solar_absorptivity

    from irsim.materials.hemispherical import total_hemispherical_emissivity

    expected = total_hemispherical_emissivity(material, 300.0, data_dir=DATA).value
    assert props.emissivity == pytest.approx(expected, rel=1e-12)
    # and it is the hemispherical value, not the normal-incidence one the camera sees
    assert props.emissivity < float(material.band_properties("lwir").emissivity)


def test_the_metal_is_the_one_material_whose_thermal_epsilon_exceeds_its_optical_one() -> None:
    """ε_hemi > ε(0) for a metal (M7.8), all the way through to the solver's properties."""
    library = MaterialLibrary.load(REPO / "configs" / "materials")
    aluminium = ThermalProperties.from_material(library["bare_aluminium"], 300.0, data_dir=DATA)
    assert aluminium.emissivity > float(
        library["bare_aluminium"].band_properties("lwir").emissivity
    )
    # and it is still a poor radiator, so it equilibrates far closer to the air than paint does
    # A poor radiator is nearly deaf to the sky: on a **passive** clear night it sits far closer
    # to the air than a painted panel does, because both its absorbed and its emitted longwave
    # are scaled by 0.11. NIGHT carries 50 W/m² of internal heat, and with that the ordering
    # reverses -- the mirror cannot shed it and runs 4.6 K *above* air while the paint sits 2.5 K
    # below. Both are right, and which one you measure depends on whether the surface is doing
    # anything, so the passive case is the one that says something about emissivity.
    import dataclasses

    passive = dataclasses.replace(NIGHT, q_internal_w_m2=0.0)
    metal_night = abs(steady_state_temperature(aluminium, passive) - passive.t_air_k)
    paint_night = abs(steady_state_temperature(PAINT, passive) - passive.t_air_k)
    assert metal_night < paint_night, "a passive mirror should track the air, not the sky"
    powered_metal = steady_state_temperature(aluminium, NIGHT) - NIGHT.t_air_k
    assert powered_metal > 3.0, "and with internal heat it cannot shed, it runs hot"
