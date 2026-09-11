"""Temperature solvers (M6.6): Newton vs the analytic exponential at dt = tau/100 and dt = 5 tau
(no overshoot), a stepped ambient restarting analytically, a ramping ambient, prescribed nodes
exact, schedule guards, the Protocol, and the float64 guarantee."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.thermal import WeatherSample, WeatherSeries
from irsim.thermal.solvers import (
    SOLVER_TYPES,
    NewtonCoolingSolver,
    PrescribedSolver,
    SolverState,
    TemperatureSolver,
)


@pytest.mark.parametrize("dt_over_tau", [0.01, 0.5, 5.0])
def test_newton_matches_analytic_and_never_overshoots(dt_over_tau: float) -> None:
    tau, t0, t_inf = 600.0, 350.0, 290.0
    s = NewtonCoolingSolver(t0, tau, t_inf)
    dt = dt_over_tau * tau
    t = 0.0
    for _ in range(int(round(20.0 / dt_over_tau)) if dt_over_tau < 1 else 4):
        temp = s.advance(t, dt)
        t += dt
        analytic = t_inf + (t0 - t_inf) * math.exp(-t / tau)
        assert abs(temp - analytic) * 1e3 < 1.0, f"{abs(temp - analytic) * 1e3:.3f} mK at t={t}"
        assert temp >= t_inf, "no overshoot below the ambient"
    assert s.state == SolverState(t, s.temperature())


def test_stepped_ambient_restarts_analytically() -> None:
    tau, t0 = 300.0, 320.0
    step_at = 900.0
    ambient = lambda t: 290.0 if t < step_at else 270.0  # noqa: E731
    s = NewtonCoolingSolver(t0, tau, ambient)
    dt = 30.0
    t = 0.0
    while t < step_at - 1e-9:
        s.advance(t, dt)
        t += dt
    t_at_step = 290.0 + (t0 - 290.0) * math.exp(-step_at / tau)
    assert abs(s.temperature() - t_at_step) * 1e3 < 1e-6
    for _ in range(40):
        s.advance(t, dt)
        t += dt
    analytic = 270.0 + (t_at_step - 270.0) * math.exp(-(t - step_at) / tau)
    assert abs(s.temperature() - analytic) * 1e3 < 1e-6, "exact after the restart"


def test_ramping_ambient_within_1mK_of_the_closed_form() -> None:
    """T_inf = a + r t has the closed form T = T_inf(t) − r τ + (T0 − a + r τ) e^{−t/τ}; the
    midpoint-held update is second order in dt."""
    tau, t0, a, r = 600.0, 300.0, 290.0, 1.0 / 3600.0  # 1 K/h ramp
    s = NewtonCoolingSolver(t0, tau, lambda t: a + r * t)
    dt = 60.0
    t = 0.0
    for _ in range(120):
        s.advance(t, dt)
        t += dt
    closed = a + r * t - r * tau + (t0 - a + r * tau) * math.exp(-t / tau)
    assert abs(s.temperature() - closed) * 1e3 < 1.0


def test_weather_ambient_exposes_the_shared_object() -> None:
    w = WeatherSeries.constant(WeatherSample(283.15, 0.5, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 7200.0)
    s = NewtonCoolingSolver(300.0, 100.0, w)
    assert s.weather is w and s.ambient_at(10.0) == 283.15
    s.advance(0.0, 1000.0)
    assert abs(s.temperature() - (283.15 + (300.0 - 283.15) * math.exp(-10.0))) < 1e-9
    with pytest.raises(ValueError, match="extrapolation"):
        s.advance(7000.0, 1000.0)  # the weather ends at 7200 s: midpoint 7500 is outside
    assert PrescribedSolver([0.0, 1.0], [300.0, 301.0]).weather is None


def test_prescribed_nodes_exact_and_guards() -> None:
    times = np.array([0.0, 60.0, 180.0, 600.0])
    temps = np.array([300.0, 330.0, 345.0, 350.0])
    s = PrescribedSolver(times, temps)
    assert s.temperature() == 300.0
    for t, expect in zip(times[1:], temps[1:], strict=True):
        assert s.advance(float(t) - 1.0, 1.0) == expect
    assert s.advance(0.0, 30.0) == 315.0
    with pytest.raises(ValueError, match="outside"):
        s.advance(600.0, 1.0)
    with pytest.raises(ValueError, match="increasing"):
        PrescribedSolver([0.0, 10.0, 5.0], [300.0, 310.0, 320.0])
    with pytest.raises(ValueError, match="positive kelvin"):
        PrescribedSolver([0.0, 10.0], [300.0, -1.0])
    with pytest.raises(ValueError, match="tau"):
        NewtonCoolingSolver(300.0, 0.0, 290.0)


def test_protocol_and_float64_guarantee() -> None:
    solvers: list[TemperatureSolver] = [
        PrescribedSolver([0.0, 10.0], [300.0, 310.0]),
        NewtonCoolingSolver(np.float32(300.0), 10.0, 290.0),
    ]
    assert all(isinstance(s, TemperatureSolver) for s in solvers)
    assert {type(s) for s in solvers} == set(SOLVER_TYPES)
    for s in solvers:
        s.advance(0.0, 1.0)
        assert type(s.temperature()) is float and type(s.state.temperature_k) is float
    n = NewtonCoolingSolver(np.float32(300.1), 10.0, 290.0)
    assert n.temperature() == float(np.float32(300.1)), "widened, not re-rounded"
    with pytest.raises(TypeError):
        NewtonCoolingSolver("hot", 10.0, 290.0)  # type: ignore[arg-type]
