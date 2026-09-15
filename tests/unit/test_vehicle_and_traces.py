"""M6.15 / M6.16 — §6.6's active heat sources, and the ghosts a scene leaves behind.

§6.6 calls these "scripted, not predicted, and they are where most of the useful signal lives", and
singles out the last row: *"Thermal shadows and residual heat traces are a signature phenomenon of
the band and they routinely confuse detectors trained only on synthetic data that lacks them."*

Two design decisions carry most of the weight, and both are about refusing to script something
that has an answer:

* **Brakes are an energy deposit, not a schedule.** ΔT = f·½m(v₁²−v₂²)/(m_disc c_p) is arithmetic.
  Scripting a brake temperature directly would make it independent of how hard the car actually
  braked, which is the one thing a braking cue is supposed to carry.
* **Traces are overlays, not conserved quantities** (ADR 0039). A trace adds ΔT to whatever the
  field says and takes that heat from nowhere. Making them conserved would need the ground solver
  to know a car had been parked there, which is exactly the state an overlay exists to avoid.

docs/physics-model.md §6.6; ADR 0038, ADR 0039
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.thermal.traces import (
    BODY_SHADOW_K,
    TYRE_TRACE_K,
    Footprint,
    GroundGrid,
    HeatTraceLayer,
    body_shadow_from_pose,
    tyre_trace_from_trajectory,
)
from irsim.thermal.vehicle import (
    HUMAN_BODY,
    VEHICLE_HEAT_SOURCES,
    SourceHistory,
    VehicleState,
    brake_temperature_rise_k,
    first_order_rise,
    newton_cool,
    source_temperature_k,
    tyre_delta_t_k,
)

T_AIR = 293.15


# ---------------------------------------------------------------------------------------------
# M6.15: the schedules
# ---------------------------------------------------------------------------------------------


def test_the_first_order_rise_reaches_one_minus_one_over_e_at_tau() -> None:
    """To **1 mK** — the identity every row of §6.6's time-constant column is describing."""
    for spec in VEHICLE_HEAT_SOURCES.values():
        at_tau = float(first_order_rise(spec.tau_rise_s, spec.delta_t_max_k, spec.tau_rise_s))
        assert at_tau == pytest.approx(spec.delta_t_max_k * (1.0 - 1.0 / math.e), abs=1e-3)
    assert float(first_order_rise(0.0, 100.0, 60.0)) == 0.0
    assert float(newton_cool(60.0, 100.0, 60.0)) == pytest.approx(100.0 / math.e, abs=1e-9)


def test_brake_heating_is_the_kinetic_energy_and_nothing_else() -> None:
    """ΔT = f·½m(v₁²−v₂²)/(m_disc c_p), to 1e-12, then Newton cooling."""
    mass, disc, c_p, fraction = 1600.0, 8.0, 500.0, 0.9
    rise = brake_temperature_rise_k(mass, 30.0, 0.0, disc, c_p, fraction)
    expected = fraction * 0.5 * mass * 30.0**2 / (disc * c_p)
    assert rise == pytest.approx(expected, rel=1e-12)
    assert 50.0 < rise < 400.0, "§6.6 quotes +50 … +400 K for a brake disc"

    # a harder stop is hotter, and the dependence is on v², which a schedule could not carry
    assert brake_temperature_rise_k(mass, 40.0, 0.0, disc) / brake_temperature_rise_k(
        mass, 20.0, 0.0, disc
    ) == pytest.approx(4.0, rel=1e-12)

    # and it cools by §6.6's 5 min constant
    spec = VEHICLE_HEAT_SOURCES["brake_disc"]
    assert float(newton_cool(spec.tau_cool_s, rise, spec.tau_cool_s)) == pytest.approx(
        rise / math.e, rel=1e-12
    )


def test_braking_uphill_is_refused_and_bad_masses_raise() -> None:
    with pytest.raises(ValueError, match="slowing down"):
        brake_temperature_rise_k(1600.0, 10.0, 20.0, 8.0)
    with pytest.raises(ValueError, match="positive"):
        brake_temperature_rise_k(1600.0, 20.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="fraction_to_discs"):
        brake_temperature_rise_k(1600.0, 20.0, 0.0, 8.0, fraction_to_discs=1.5)


def test_tyre_temperature_rises_monotonically_with_speed_inside_section_6_6s_range() -> None:
    """ "+10 … +35 K, rises with speed" — the strong ID cue, and the range is the assertion."""
    speeds = np.linspace(0.0, 40.0, 41)
    delta = tyre_delta_t_k(speeds)
    assert np.all(np.diff(delta) >= 0.0)
    assert float(delta.min()) == pytest.approx(10.0, abs=0.5)
    assert float(delta.max()) == pytest.approx(35.0, abs=0.5)


def test_the_exhaust_ordering_is_manifold_then_cat_then_pipe_then_tip() -> None:
    """Hottest nearest the engine, coolest at the tip — which is what makes an exhaust legible."""
    order = ["exhaust_manifold", "catalytic_converter", "exhaust_pipe", "exhaust_tip"]
    values = [VEHICLE_HEAT_SOURCES[n].delta_t_max_k for n in order]
    assert values == sorted(values, reverse=True), dict(zip(order, values, strict=True))
    assert all(80.0 <= v <= 250.0 for v in values), values


def test_the_human_body_row_puts_the_face_warmest() -> None:
    """§6.6: "+8 … +15 K … face is warmest". A detector's most reliable human cue."""
    assert HUMAN_BODY["face"] > HUMAN_BODY["hands"] > HUMAN_BODY["clothed_torso"]
    assert min(HUMAN_BODY.values()) >= 8.0 and max(HUMAN_BODY.values()) <= 15.0


def test_a_source_history_is_deterministic_and_sample_rate_independent() -> None:
    """The exponential step is exact, so a trace sampled at 1 Hz and at 10 Hz agree to 1e-9.

    That is not a nicety: a schedule integrated with a forward difference would give a different
    engine-bay temperature depending on how finely the vehicle trace happened to be logged.
    """
    spec = VEHICLE_HEAT_SOURCES["engine_bay"]
    results = []
    for dt in (1.0, 0.1):
        history = SourceHistory(spec)
        t = 0.0
        while t <= 1800.0 + 1e-9:
            history.step(VehicleState(t_s=t, ignition=True), load=1.0)
            t += dt
        results.append(history.delta_t_k)
    assert results[0] == pytest.approx(results[1], abs=1e-9)
    assert results[0] == pytest.approx(
        float(first_order_rise(1800.0, spec.delta_t_max_k, spec.tau_rise_s)), abs=0.05
    )


def test_a_source_cools_when_the_load_goes_away() -> None:
    spec = VEHICLE_HEAT_SOURCES["engine_bay"]
    history = SourceHistory(spec)
    for t in np.arange(0.0, 1800.0, 10.0):
        history.step(VehicleState(t_s=float(t), ignition=True), load=1.0)
    hot = history.delta_t_k
    for t in np.arange(1800.0, 1800.0 + spec.tau_cool_s, 10.0):
        history.step(VehicleState(t_s=float(t)), load=0.0)
    assert history.delta_t_k == pytest.approx(hot / math.e, rel=0.02)
    assert source_temperature_k(T_AIR, history.delta_t_k) > T_AIR


def test_a_trace_running_backwards_in_time_is_refused() -> None:
    history = SourceHistory(VEHICLE_HEAT_SOURCES["tyre"])
    history.step(VehicleState(t_s=10.0))
    with pytest.raises(ValueError, match="forward in time"):
        history.step(VehicleState(t_s=5.0))
    with pytest.raises(ValueError, match="non-negative"):
        VehicleState(t_s=0.0, speed_m_s=-1.0)


# ---------------------------------------------------------------------------------------------
# M6.16: the overlays
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def grid() -> GroundGrid:
    return GroundGrid(width=80, height=60, cell_m=0.10)


def test_a_footprint_decays_to_one_over_e_at_its_time_constant(grid: GroundGrid) -> None:
    mask = np.zeros(grid.shape, dtype=bool)
    mask[10:20, 10:20] = True
    footprint = Footprint(mask, 8.0, deposited_t_s=0.0, tau_s=600.0)
    assert footprint.delta_t_at(0.0) == pytest.approx(8.0, abs=1e-12)
    assert footprint.delta_t_at(600.0) == pytest.approx(8.0 / math.e, abs=1e-3)
    assert footprint.delta_t_at(-5.0) == 0.0, "a trace does not exist before it is deposited"


def test_superposition_is_linear_and_untouched_cells_are_bit_identical(grid: GroundGrid) -> None:
    """A cell no footprint touches must come back **exactly** as it went in.

    A layer that blended or clamped would make an untouched road pixel depend on whether a car
    drove past somewhere else in the frame, which is the kind of coupling that is impossible to
    notice and impossible to debug.
    """
    layer = HeatTraceLayer(grid)
    first = np.zeros(grid.shape, dtype=bool)
    first[5:15, 5:15] = True
    second = np.zeros(grid.shape, dtype=bool)
    second[10:20, 10:20] = True
    layer.deposit(Footprint(first, 4.0, 0.0, 600.0))
    layer.deposit(Footprint(second, 3.0, 0.0, 600.0))

    field = np.full(grid.shape, 295.0) + np.random.default_rng(0).normal(0.0, 0.5, grid.shape)
    out = layer.apply(field, 0.0)
    both = first & second
    untouched = ~(first | second)
    assert np.array_equal(out[untouched], field[untouched])
    assert np.allclose(out[both] - field[both], 7.0, atol=1e-12)
    assert np.allclose(out[first & ~second] - field[first & ~second], 4.0, atol=1e-12)

    # linear in amplitude, to 1e-12
    doubled = HeatTraceLayer(grid)
    doubled.deposit(Footprint(first, 8.0, 0.0, 600.0))
    doubled.deposit(Footprint(second, 6.0, 0.0, 600.0))
    assert np.allclose(doubled.delta_t(0.0), 2.0 * layer.delta_t(0.0), atol=1e-12)


def test_a_tyre_trace_is_two_stripes_at_the_track_gauge(grid: GroundGrid) -> None:
    """Two, not one: a single wide stripe is a skid; two at a known gauge are a vehicle, and a
    detector that has only seen the first will not learn the second."""
    path = np.column_stack([np.linspace(1.0, 7.0, 25), np.full(25, 3.0)])
    stripes = tyre_trace_from_trajectory(grid, path, track_width_m=1.55, tyre_width_m=0.20)
    assert len(stripes) == 2
    for stripe in stripes:
        assert stripe.mask.any()
        assert stripe.amplitude_k == pytest.approx(TYRE_TRACE_K)
        assert 3.0 <= stripe.amplitude_k <= 10.0, "§6.6 quotes +3 … +10 K"
    assert not (stripes[0].mask & stripes[1].mask).any(), "the stripes must not overlap"

    rows_a = np.flatnonzero(stripes[0].mask.any(axis=1))
    rows_b = np.flatnonzero(stripes[1].mask.any(axis=1))
    gauge_m = abs(float(rows_a.mean() - rows_b.mean())) * grid.cell_m
    assert gauge_m == pytest.approx(1.55, abs=0.15)


def test_the_stripes_do_not_depend_on_how_finely_the_path_was_sampled(grid: GroundGrid) -> None:
    """Rasterised by distance to the segment, not by walking it — so 1 Hz and 100 Hz agree."""
    coarse = np.column_stack([np.linspace(1.0, 7.0, 4), np.full(4, 3.0)])
    fine = np.column_stack([np.linspace(1.0, 7.0, 200), np.full(200, 3.0)])
    a = tyre_trace_from_trajectory(grid, coarse)
    b = tyre_trace_from_trajectory(grid, fine)
    assert np.array_equal(a[0].mask, b[0].mask)
    assert np.array_equal(a[1].mask, b[1].mask)


def test_a_body_shadow_is_a_cold_rectangle_matching_an_independent_mask(grid: GroundGrid) -> None:
    """Built by an axis-aligned rectangle test here, and by a rotated one in the module."""
    shadow = body_shadow_from_pose(grid, (4.0, 3.0), length_m=4.5, width_m=1.8, heading_rad=0.0)
    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    x = (cols + 0.5) * grid.cell_m
    y = (rows + 0.5) * grid.cell_m
    expected = (np.abs(x - 4.0) <= 2.25) & (np.abs(y - 3.0) <= 0.9)
    assert np.array_equal(shadow.mask, expected)
    assert shadow.amplitude_k == pytest.approx(BODY_SHADOW_K)
    assert -8.0 <= shadow.amplitude_k <= -3.0, "§6.6 quotes −3 … −8 K"
    assert int(shadow.mask.sum()) == pytest.approx(4.5 * 1.8 / grid.cell_m**2, rel=0.05)


def test_a_rotated_shadow_covers_the_same_area(grid: GroundGrid) -> None:
    straight = body_shadow_from_pose(grid, (4.0, 3.0), 4.5, 1.8, heading_rad=0.0)
    turned = body_shadow_from_pose(grid, (4.0, 3.0), 4.5, 1.8, heading_rad=math.radians(30.0))
    assert int(turned.mask.sum()) == pytest.approx(int(straight.mask.sum()), rel=0.05)
    assert not np.array_equal(turned.mask, straight.mask)


def test_the_tyre_stripes_fade_but_the_shadow_is_still_there_at_twenty_minutes(
    grid: GroundGrid,
) -> None:
    """⚠️ Both decay constants sit inside §6.6's "5–20 min", and only one is gone by 20 minutes.

    The stripes (τ = 7 min) fall under **0.5 K**; the body shadow (τ = 12 min) is still **1.0 K**
    — and that residual *is* the phenomenon §6.6 is pointing at. A ghost that had faded by the
    time anyone looked would not "routinely confuse detectors". Asserting both below 0.5 K would
    have meant choosing τ for the convenience of the test.
    """
    path = np.column_stack([np.linspace(1.0, 7.0, 10), np.full(10, 3.0)])
    stripes = HeatTraceLayer(grid)
    for stripe in tyre_trace_from_trajectory(grid, path):
        stripes.deposit(stripe)
    shadow = HeatTraceLayer(grid)
    shadow.deposit(body_shadow_from_pose(grid, (4.0, 3.0), 4.5, 1.8))

    assert float(np.abs(stripes.delta_t(0.0)).max()) > 3.0
    assert float(np.abs(shadow.delta_t(0.0)).max()) > 3.0

    twenty = 20.0 * 60.0
    assert float(np.abs(stripes.delta_t(twenty)).max()) < 0.5
    assert float(np.abs(shadow.delta_t(twenty)).max()) == pytest.approx(1.04, abs=0.2)

    # both constants inside §6.6's quoted 5-20 min
    for layer in (stripes, shadow):
        for footprint in layer.footprints:
            assert 300.0 <= footprint.tau_s <= 1200.0, footprint.tau_s
    # and by an hour even the shadow is gone
    assert float(np.abs(shadow.delta_t(3600.0)).max()) < 0.2


def test_a_mismatched_mask_is_refused(grid: GroundGrid) -> None:
    layer = HeatTraceLayer(grid)
    with pytest.raises(ValueError, match="does not match grid"):
        layer.deposit(Footprint(np.zeros((3, 3), dtype=bool), 1.0, 0.0, 60.0))
    with pytest.raises(ValueError, match="does not match grid"):
        layer.apply(np.zeros((3, 3)), 0.0)
    with pytest.raises(TypeError, match="boolean"):
        Footprint(np.zeros(grid.shape), 1.0, 0.0, 60.0)
