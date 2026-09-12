"""Aerial-target thermal signature: airframe at air temperature, heat sources above it.

An aircraft in flight is not a ground scene. It has no solar-loaded slab in contact with soil, no
diurnal storage and no sky-view geometry worth solving: forced convection at flight speed pins the
skin to within a degree or two of the air it is flying through, and everything a thermal camera
actually keys on -- the motors, the speed controllers, the battery pack, the exhaust -- is an
*internal dissipation* whose temperature is set by throttle, not by weather. So the target model
here is a set of **prescribed** nodes (M6.6 ``PrescribedSolver``), not an energy balance:

    T_node(t) = T_air(t) + ΔT(u(t)),      ΔT(u) = ΔT_max · u^n            (§6.6, ESTIMATED)

with u ∈ [0, 1] the throttle fraction. n = 2 is the ohmic reading: winding and MOSFET loss go as
I²R and current is roughly proportional to throttle, so dissipation -- and, at a fixed convective
conductance, the steady rise above ambient -- goes as u². The ΔT_max values are **ESTIMATED**
(ADR 0072): no public dataset of instrumented multirotor motor temperatures was available, and
they are the parameters a Tier 4 comparison against public aerial IR imagery should re-fit first.

The airframe node follows the shared ``WeatherSeries`` (CLAUDE.md #6: it is injected, never loaded
here), optionally with a small constant offset for a sun-soaked upper surface.

Everything returns a ``PrescribedSolver``, so an aerial target drops into the same
``Scene``/solver plumbing as a ground surface. The schedule is refined until piecewise-linear
interpolation between its nodes reproduces the analytic law to a stated tolerance, which is what
makes "prescribed" a discretisation of a model rather than a table someone typed.

docs/physics-model.md §6.6, §16.2; ADR 0072
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.solvers import PrescribedSolver
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "HeatSource",
    "MOTOR",
    "ESC",
    "BATTERY",
    "AERIAL_HEAT_SOURCES",
    "throttle_profile",
    "node_temperature",
    "refine_nodes",
    "prescribed_from_schedule",
    "heat_source_solver",
    "airframe_solver",
]

FloatArray = NDArray[np.float64]

# Linear interpolation of a C² function on a step h errs by at most h²/8·max|f''|, and for a
# quadratic the worst point is the segment midpoint -- so bisecting on the midpoint residual is
# both the cheapest test and, for the u² law, the exact one.
DEFAULT_TOLERANCE_K = 1e-3
MAX_SCHEDULE_NODES = 1 << 16


@dataclass(frozen=True)
class HeatSource:
    """One dissipating node on an aerial target: ΔT above air = ``delta_t_max_k`` · u^``exponent``.

    ``delta_t_max_k`` is the steady rise at full throttle in still air at the reference condition;
    it is ESTIMATED (ADR 0072). ``exponent`` = 2 is ohmic loss with current ∝ throttle.
    """

    name: str
    delta_t_max_k: float
    exponent: float = 2.0
    reference: str = "ESTIMATED (ADR 0072)"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("heat source needs a name")
        if not self.delta_t_max_k >= 0.0 or not math.isfinite(self.delta_t_max_k):
            raise ValueError(f"{self.name}: delta_t_max_k must be finite and non-negative")
        if not self.exponent > 0.0 or not math.isfinite(self.exponent):
            raise ValueError(f"{self.name}: exponent must be positive and finite")

    def delta_t_k(self, throttle: object) -> FloatArray:
        """ΔT above air (K) at throttle fraction u ∈ [0, 1]. Vectorised, analytic, float64."""
        u = np.asarray(throttle, dtype=np.float64)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"{self.name}: throttle must lie in [0, 1]")
        return np.asarray(self.delta_t_max_k * u**self.exponent, dtype=np.float64)


# ESTIMATED (ADR 0072). Ordering is the physically defensible part: the motor windings are the
# hottest node, the ESC MOSFETs sit below them, and the pack -- large thermal mass, spread over
# many cells -- is the coolest of the three. A Tier 4 fit against public aerial IR should move the
# magnitudes and keep the ordering.
MOTOR = HeatSource("motor", delta_t_max_k=45.0, exponent=2.0)
ESC = HeatSource("esc", delta_t_max_k=30.0, exponent=2.0)
BATTERY = HeatSource("battery", delta_t_max_k=15.0, exponent=2.0)

AERIAL_HEAT_SOURCES: dict[str, HeatSource] = {s.name: s for s in (MOTOR, ESC, BATTERY)}


def throttle_profile(times_s: object, throttle: object) -> Callable[[FloatArray], FloatArray]:
    """A piecewise-linear u(t) from breakpoints, clamped to the end values outside the range."""
    t = np.asarray(times_s, dtype=np.float64)
    u = np.asarray(throttle, dtype=np.float64)
    if t.ndim != 1 or t.size < 1 or u.shape != t.shape:
        raise ValueError("throttle profile needs 1-D times and values of equal length")
    if t.size > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("throttle breakpoint times must be strictly increasing")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("throttle must lie in [0, 1]")
    if not np.all(np.isfinite(t)):
        raise ValueError("throttle breakpoint times must be finite")

    def u_of_t(t_query: FloatArray) -> FloatArray:
        return np.asarray(np.interp(np.asarray(t_query, dtype=np.float64), t, u), dtype=np.float64)

    return u_of_t


def node_temperature(
    source: HeatSource,
    weather: WeatherSeries,
    throttle: Callable[[FloatArray], FloatArray],
) -> Callable[[FloatArray], FloatArray]:
    """The analytic node law T(t) = T_air(t) + ΔT(u(t)) as a vectorised callable.

    ``weather`` is the scene's shared series (CLAUDE.md #6) -- T_air comes from it, never from a
    file read here. This is the oracle the prescribed schedule is a discretisation of.
    """

    def t_of_t(t_query: FloatArray) -> FloatArray:
        t = np.asarray(t_query, dtype=np.float64)
        t_air = np.asarray(weather.interpolate(t)["t_air_k"], dtype=np.float64)
        return np.asarray(t_air + source.delta_t_k(throttle(t)), dtype=np.float64)

    return t_of_t


def refine_nodes(
    f: Callable[[FloatArray], FloatArray],
    nodes_s: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
    max_nodes: int = MAX_SCHEDULE_NODES,
) -> FloatArray:
    """Bisect ``nodes_s`` until linear interpolation of ``f`` errs by < ``tolerance_k`` everywhere.

    Each pass evaluates every segment midpoint and splits the segments that miss the tolerance;
    for the u^n law the midpoint is the worst point of the segment, so the returned grid carries a
    genuine bound, not a sampled one.
    """
    t = np.unique(np.asarray(nodes_s, dtype=np.float64))
    if t.ndim != 1 or t.size < 2:
        raise ValueError("refinement needs at least two distinct node times")
    if not tolerance_k > 0.0:
        raise ValueError("tolerance_k must be positive")
    while True:
        y = f(t)
        mid = 0.5 * (t[:-1] + t[1:])
        err = np.abs(f(mid) - 0.5 * (y[:-1] + y[1:]))
        split = err > tolerance_k
        if not np.any(split):
            return np.asarray(t, dtype=np.float64)
        if t.size + int(split.sum()) > max_nodes:
            raise ValueError(
                f"schedule needs more than {max_nodes} nodes to reach {tolerance_k} K; "
                "the profile is probably discontinuous"
            )
        t = np.unique(np.concatenate([t, mid[split]]))


def prescribed_from_schedule(
    f: Callable[[FloatArray], FloatArray],
    nodes_s: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
) -> PrescribedSolver:
    """A ``PrescribedSolver`` sampling ``f`` on a grid refined to ``tolerance_k`` (§6.6, M6.6)."""
    t = refine_nodes(f, nodes_s, tolerance_k)
    return PrescribedSolver(t, f(t))


def _base_nodes(weather: WeatherSeries, times_s: object) -> FloatArray:
    """Union of the profile breakpoints and the weather samples inside their span."""
    t = np.unique(np.asarray(times_s, dtype=np.float64))
    if t.size < 2:
        raise ValueError("need at least two distinct breakpoint times")
    inside = weather.time_s[(weather.time_s > t[0]) & (weather.time_s < t[-1])]
    return np.asarray(np.unique(np.concatenate([t, inside])), dtype=np.float64)


def heat_source_solver(
    source: HeatSource,
    weather: WeatherSeries,
    times_s: object,
    throttle: object,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
) -> PrescribedSolver:
    """Motor / ESC / battery node as a prescribed schedule over a throttle profile (§6.6).

    The returned solver reproduces T_air(t) + ΔT_max u(t)^n to ``tolerance_k`` at every instant in
    the profile's span, including between its nodes.
    """
    u_of_t = throttle_profile(times_s, throttle)
    law = node_temperature(source, weather, u_of_t)
    return prescribed_from_schedule(law, _base_nodes(weather, times_s), tolerance_k)


def airframe_solver(
    weather: WeatherSeries,
    times_s: object | None = None,
    offset_k: float = 0.0,
    tolerance_k: float = DEFAULT_TOLERANCE_K,
) -> PrescribedSolver:
    """The airframe node: T_air(t) + ``offset_k``, from the scene's shared weather (CLAUDE.md #6).

    Forced convection at flight speed holds an unpowered skin within a degree or so of the air;
    ``offset_k`` carries a sun-soaked upper surface or an unmodelled internal soak, and is 0 by
    default so an airframe is air temperature unless someone says otherwise.
    """
    if not math.isfinite(offset_k):
        raise ValueError("offset_k must be finite")
    span = weather.time_s if times_s is None else np.asarray(times_s, dtype=np.float64)

    def law(t_query: FloatArray) -> FloatArray:
        t_air = weather.interpolate(np.asarray(t_query, dtype=np.float64))["t_air_k"]
        return np.asarray(np.asarray(t_air, dtype=np.float64) + offset_k, dtype=np.float64)

    return prescribed_from_schedule(law, _base_nodes(weather, span), tolerance_k)
