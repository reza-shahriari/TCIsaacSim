"""M6.11 — the object the engine glue reads: a fixed-tick solve behind a const query (§6.4, §1).

§6.4's "decouple this from the render loop entirely" is one sentence and two requirements. The
solve advances on its own clock, so 240 fps and 1 fps see the same temperatures. And **a query
never mutates anything** — a renderer asks many times per tick, per band, per AOV, per pass, and a
query that advanced the solver would make the answer depend on how often it was asked. That is not
a bug that shows up as an error message; it shows up as a scene that renders differently when you
add an output.

docs/physics-model.md §6.4, §1; CLAUDE.md #2; ADR 0036, ADR 0037
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.thermal.balance import ThermalProperties
from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver
from irsim.thermal.field import DEFAULT_TICK_S, ThermalField

MATERIALS = [
    ThermalProperties(2200 * 920 * 0.05, 0.94, 0.90),  # asphalt
    ThermalProperties(7800 * 470 * 0.0012, 0.90, 0.94),  # painted panel
    ThermalProperties(2500 * 840 * 0.005, 0.88, 0.10),  # glass
]


def _forcing(t_s: float) -> FacetForcing:
    hour = (t_s / 3600.0) % 24.0
    return FacetForcing(
        t_air_k=288.0 + 6.0 * math.sin(math.pi * (hour - 8.0) / 12.0),
        h_w_m2_k=10.0,
        q_solar_w_m2=max(0.0, 900.0 * math.sin(math.pi * (hour - 6.0) / 12.0)),
        q_longwave_down_w_m2=320.0,
    )


def _field(tick_s: float = DEFAULT_TICK_S, t0_s: float = 0.0) -> ThermalField:
    packed = FacetProperties.stack(MATERIALS)
    return ThermalField(packed, _forcing, t0_s, np.full(len(MATERIALS), 288.0), tick_s)


# ---------------------------------------------------------------------------------------------
# the query does not mutate
# ---------------------------------------------------------------------------------------------


def test_three_thousand_six_hundred_queries_leave_the_state_untouched() -> None:
    """A whole tick's worth of render-time queries, bit for bit."""
    field = _field(tick_s=1.0)
    field.advance_to(600.0)
    before = field.state_hash()
    ticks_before = field.n_ticks
    for i in range(3600):
        field.temperature_at(300.0 + (i % 100) * 0.01)
    assert field.state_hash() == before
    assert field.n_ticks == ticks_before


def test_a_query_past_the_last_tick_raises_rather_than_advancing() -> None:
    """The alternative — advancing on demand — is what makes the answer depend on the caller."""
    field = _field()
    field.advance_to(10.0)
    field.temperature_at(10.0)
    with pytest.raises(ValueError, match="call advance_to first"):
        field.temperature_at(10.5)
    with pytest.raises(ValueError, match="before the field's start"):
        field.temperature_at(-1.0)


def test_running_time_backwards_raises() -> None:
    """A thermal history is not reversible, and silently re-running it gives a different answer."""
    field = _field(t0_s=100.0)
    with pytest.raises(ValueError, match="not reversible"):
        field.advance_to(50.0)


# ---------------------------------------------------------------------------------------------
# interpolation
# ---------------------------------------------------------------------------------------------


def test_the_interpolation_is_exact_at_the_ticks() -> None:
    """Weight 0 and weight 1 return the stored value **exactly**, in float64, to 1e-12.

    Checked on the stored ticks rather than on `temperature_at`'s output, because that output is
    float32 — whose spacing at 288 K is 3e-5 — so a 1e-12 claim made through it would be a claim
    about the narrowing and not about the interpolation. The narrowing gets its own test.
    """
    field = _field(tick_s=10.0)
    field.advance_to(100.0)
    solver = FacetSolver(FacetProperties.stack(MATERIALS), np.full(len(MATERIALS), 288.0))
    for step in range(10):
        stored = field._ticks[step].temperatures_k  # noqa: SLF001
        assert np.allclose(stored, solver.temperatures_k, rtol=0.0, atol=1e-12)
        # and the query at that exact time agrees to float32 spacing
        assert np.allclose(
            field.temperature_at(step * 10.0).astype(np.float64), stored, rtol=0.0, atol=1e-4
        )
        solver.advance(_forcing(step * 10.0), 10.0)


def test_the_midpoint_is_the_mean_of_its_brackets() -> None:
    field = _field(tick_s=10.0)
    field.advance_to(100.0)
    lower = field.temperature_at(30.0).astype(np.float64)
    upper = field.temperature_at(40.0).astype(np.float64)
    middle = field.temperature_at(35.0).astype(np.float64)
    assert np.allclose(middle, 0.5 * (lower + upper), atol=1e-6)


def test_interpolation_error_is_far_below_a_millikelvin_at_one_hertz() -> None:
    """Linear rather than anything cleverer: the tick is 1 s and the fastest surface here has a
    time constant of minutes, so a higher-order scheme would store more ticks to buy nothing."""
    coarse = _field(tick_s=1.0)
    fine = _field(tick_s=0.05)
    coarse.advance_to(600.0)
    fine.advance_to(600.0)
    worst = 0.0
    for t in np.linspace(0.0, 600.0, 501):
        worst = max(
            worst,
            float(
                np.max(np.abs(coarse.temperature_at(t).astype(np.float64) - fine.temperature_at(t)))
            ),
        )
    assert worst < 1e-3, worst


# ---------------------------------------------------------------------------------------------
# the float32 boundary
# ---------------------------------------------------------------------------------------------


def test_the_query_returns_float32_and_the_solve_stays_float64() -> None:
    field = _field()
    field.advance_to(5.0)
    assert field.temperature_at(2.5).dtype == np.float32
    assert field._ticks[0].temperatures_k.dtype == np.float64  # noqa: SLF001


def test_narrowing_once_at_the_end_beats_carrying_float32_for_forty_eight_hours() -> None:
    """< 10 mK over 48 h — measured against a run that narrows to float32 at every step.

    This is CLAUDE.md #2 as an experiment rather than an assertion: the balance subtracts ~400
    W m⁻² terms to leave a few, and 48 h at 60 s is 2 880 of those subtractions in series.
    """
    packed = FacetProperties.stack(MATERIALS)
    exact = FacetSolver(packed, np.full(len(MATERIALS), 288.0))
    narrowed = FacetSolver(packed, np.full(len(MATERIALS), 288.0))
    dt = 60.0
    for i in range(int(48 * 3600 / dt)):
        forcing = _forcing(i * dt)
        exact.advance(forcing, dt)
        narrowed.advance(forcing, dt)
        narrowed._state = narrowed._state.astype(np.float32).astype(np.float64)  # noqa: SLF001
    drift = float(np.max(np.abs(exact.temperatures_k - narrowed.temperatures_k)))
    assert drift < 0.010, drift
    assert drift > 0.0, "if float32 made no difference at all, the experiment is not running"


# ---------------------------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------------------------


def test_two_identical_runs_are_bit_identical() -> None:
    """There is no randomness anywhere in the thermal path, and this is what says so."""
    a, b = _field(), _field()
    a.advance_to(300.0)
    b.advance_to(300.0)
    assert a.state_hash() == b.state_hash()
    assert np.array_equal(a.temperature_at(123.4), b.temperature_at(123.4))


def test_the_render_rate_does_not_change_the_answer() -> None:
    """240 fps and 1 fps must see the same temperatures: the solve is on its own clock."""
    fast, slow = _field(), _field()
    for i in range(1, 241):
        fast.advance_to(i / 240.0 * 60.0)
        fast.temperature_at(i / 240.0 * 60.0)
    slow.advance_to(60.0)
    assert np.allclose(
        fast.temperature_at(60.0).astype(np.float64),
        slow.temperature_at(60.0).astype(np.float64),
        atol=1e-9,
    )


def test_a_non_positive_tick_is_refused() -> None:
    packed = FacetProperties.stack(MATERIALS)
    with pytest.raises(ValueError, match="tick_s must be positive"):
        ThermalField(packed, _forcing, 0.0, np.full(len(MATERIALS), 288.0), 0.0)
