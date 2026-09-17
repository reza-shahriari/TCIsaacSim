"""Housing temperature source (M9.3): the three §12.2 modes feeding §8.2's self-emission term.

These are physics tests, not API tests. Each one is written so that it fails if the housing node
stops being a first-order lag toward ``T_air + ΔT_self``:

* the coupled step response is checked against the closed-form exponential at τ, 2τ and 5τ, so a
  node that relaxed at the wrong rate, or toward the wrong asymptote, is caught at the 0.01 K
  level the roadmap sets — a fifth of an LWIR NETD, and 0.87 mK of apparent temperature through
  ADR 0016's 87 mK/K coupling;
* the steady state is checked to be ``T_air + ΔT_self`` and *not* ``T_air``, which is the mistake
  that makes a self-heating camera read cold;
* ``ambient`` mode is pinned to the weather to 1e-6 K, so a node that quietly introduced a lag
  where none was configured is caught;
* and a node holding a different ``WeatherSeries`` than the scene is refused at construction of
  the Scene (CLAUDE.md non-negotiable #6).

docs/physics-model.md §8.2, §12.2. ADR 0015, ADR 0016.
"""

from __future__ import annotations

import math
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.optics import HousingTemperature
from irsim.thermal import WeatherSeries

EPOCH = datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc)
REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

# The roadmap's tolerance for the step response and the steady state.
TOL_K = 0.01


def _weather(
    t_air_k: float | np.ndarray = 293.15, n: int = 49, step_s: float = 1800.0
) -> WeatherSeries:
    """A weather series whose only interesting column is T_air."""
    t = np.arange(n) * step_s
    air = np.full(n, float(t_air_k)) if np.isscalar(t_air_k) else np.asarray(t_air_k, float)
    return WeatherSeries.from_arrays(
        EPOCH,
        t,
        t_air_k=air,
        rh_fraction=np.full(n, 0.5),
        wind_speed_m_s=np.full(n, 2.0),
        cloud_fraction=np.zeros(n),
        dni_w_m2=np.zeros(n),
        dhi_w_m2=np.zeros(n),
        visibility_m=np.full(n, 23000.0),
        precip_mm_h=np.zeros(n),
    )


# -- fixed ---------------------------------------------------------------------------------


def test_fixed_mode_ignores_the_weather_entirely() -> None:
    """A bench-fixed housing is the mode in which a camera shows no self-emission drift."""
    swinging = _weather(293.15 + 15.0 * np.sin(np.arange(49) * 0.3))
    node = HousingTemperature(mode="fixed", fixed_temp_k=305.0, weather=swinging)
    for t in (0.0, 3600.0, 43200.0, 86400.0):
        assert node.at(t) == 305.0
    assert node.steady_state_k(0.0) == 305.0


def test_fixed_mode_requires_a_temperature() -> None:
    with pytest.raises(ValueError, match="requires a positive housing_temp_k"):
        HousingTemperature(mode="fixed")


# -- ambient -------------------------------------------------------------------------------


def test_ambient_mode_tracks_the_weather_to_1e_6_k() -> None:
    """No lag, no self-heating: the housing *is* the air, at every time including between nodes."""
    air = 293.15 + 8.0 * np.sin(np.arange(49) * 2 * np.pi / 48.0)
    w = _weather(air)
    node = HousingTemperature(mode="ambient", weather=w)
    for t in (0.0, 900.0, 1800.0, 5400.0, 37_000.0, 86_400.0):
        assert node.at(t) == pytest.approx(w.at(t).t_air_k, abs=1e-6)


def test_ambient_mode_may_be_read_out_of_order() -> None:
    """It is algebraic, so it carries no state that reading backwards could corrupt."""
    node = HousingTemperature(mode="ambient", weather=_weather(np.linspace(280.0, 300.0, 49)))
    late = node.at(80_000.0)
    early = node.at(1_000.0)
    assert early < late
    assert node.at(80_000.0) == late


def test_a_lagging_mode_needs_an_ambient_source() -> None:
    for mode in ("ambient", "coupled"):
        with pytest.raises(ValueError, match="needs an ambient source"):
            HousingTemperature(mode=mode, tau_s=100.0)


# -- coupled: the step response ------------------------------------------------------------


@pytest.mark.parametrize("n_tau", [1.0, 2.0, 5.0])
def test_coupled_step_response_matches_the_exponential(n_tau: float) -> None:
    """T(t) = T∞ + (T0 − T∞) e^{−t/τ} at τ, 2τ and 5τ, to 0.01 K.

    The node starts 20 K below a constant ambient and is stepped in 200 sub-steps, so an
    integrator that was merely first-order accurate, or that sampled the ambient once per call,
    would show up here as well as a wrong time constant.
    """
    tau = 600.0
    t_air, t_start = 293.15, 273.15
    node = HousingTemperature(
        mode="coupled", tau_s=tau, weather=_weather(t_air), t0_k=t_start, t0_s=0.0
    )

    target = t_air  # no self-heating in this case
    t_end = n_tau * tau
    for i in range(1, 201):
        node.at(t_end * i / 200.0)

    expected = target + (t_start - target) * math.exp(-n_tau)
    assert node.at(t_end) == pytest.approx(expected, abs=TOL_K)
    # And the approach is genuinely exponential, not merely close at these three points:
    fraction = (node.temperature_k - t_start) / (target - t_start)
    assert fraction == pytest.approx(1.0 - math.exp(-n_tau), abs=TOL_K / abs(target - t_start))


def test_coupled_step_size_does_not_change_the_answer() -> None:
    """The exact exponential update is unconditionally stable: one 5τ leap == 5000 small steps.

    A forward-Euler node with dt > 2τ would oscillate or diverge here rather than agree.
    """
    tau, t_air, t_start = 300.0, 295.0, 270.0
    end = 5.0 * tau

    one = HousingTemperature(mode="coupled", tau_s=tau, weather=_weather(t_air), t0_k=t_start)
    many = HousingTemperature(mode="coupled", tau_s=tau, weather=_weather(t_air), t0_k=t_start)
    leap = one.at(end)
    for i in range(1, 5001):
        many.at(end * i / 5000.0)
    assert leap == pytest.approx(many.temperature_k, abs=1e-9)

    # A leap of 1000 tau lands on the asymptote rather than overshooting or blowing up. The
    # ambient is an explicit constant here because the weather series refuses extrapolation --
    # correctly, but that guard is not what this test is about.
    giant = HousingTemperature(
        mode="coupled", tau_s=tau, ambient_provider=lambda _t: t_air, t0_k=t_start
    )
    assert giant.at(1000.0 * tau) == pytest.approx(t_air, abs=1e-9)


# -- coupled: the steady state -------------------------------------------------------------


@pytest.mark.parametrize("self_heating", [0.0, 1.5, 6.0])
def test_steady_state_is_air_plus_self_heating(self_heating: float) -> None:
    """The asymptote is T_air + ΔT_self. A node relaxing to bare T_air reads cold by ΔT_self."""
    t_air = 288.0
    node = HousingTemperature(
        mode="coupled",
        tau_s=120.0,
        self_heating_k=self_heating,
        weather=_weather(t_air),
        t0_k=t_air,
    )
    settled = node.at(50.0 * 120.0)
    assert settled == pytest.approx(t_air + self_heating, abs=TOL_K)
    assert node.steady_state_k(0.0) == pytest.approx(t_air + self_heating, abs=1e-12)
    if self_heating > 0.0:
        assert settled > t_air + 0.5 * self_heating


def test_settle_starts_the_node_at_its_steady_state() -> None:
    """Spin-up entry point: a camera that has been on does not need its warm-up simulated."""
    w = _weather(np.linspace(280.0, 300.0, 49))
    node = HousingTemperature(
        mode="coupled", tau_s=900.0, self_heating_k=2.0, weather=w, t0_k=250.0
    )
    assert node.temperature_k == 250.0
    node.settle(7200.0)
    assert node.temperature_k == pytest.approx(w.at(7200.0).t_air_k + 2.0, abs=1e-12)
    # A settled node stepped under frozen weather does not move.
    frozen = HousingTemperature(mode="coupled", tau_s=900.0, self_heating_k=2.0, weather=_weather())
    frozen.settle(0.0)
    start = frozen.temperature_k
    assert frozen.at(10_000.0) == pytest.approx(start, abs=1e-12)


def test_coupled_default_start_is_settled() -> None:
    node = HousingTemperature(
        mode="coupled", tau_s=60.0, self_heating_k=3.0, weather=_weather(291.0), t0_s=1800.0
    )
    assert node.temperature_k == pytest.approx(294.0, abs=1e-12)


def test_the_node_lags_a_swinging_ambient() -> None:
    """The housing is a first-order low-pass on air temperature, with the textbook Bode response.

    Driven by a 24 h sinusoid, a lag of time constant tau must damp the amplitude by exactly
    1/sqrt(1 + (omega tau)^2) and delay the peak by arctan(omega tau)/omega. Checking both is
    much stronger than checking "it is damped": a second-order node, a moving average, or a
    node with the wrong tau all reproduce "damped" and fail these two numbers.

    The lag is *why* the housing is a drift source at all. If it tracked the air exactly, a NUC
    table calibrated at one air temperature would still be valid at the next.
    """
    period = 86_400.0
    omega = 2.0 * np.pi / period
    tau = 7200.0
    step = 300.0
    n = int(3 * period / step) + 1  # three days, so the start-up transient is long dead
    t = np.arange(n) * step
    amp_air = 10.0
    w = _weather(293.15 + amp_air * np.sin(omega * t), n=n, step_s=step)

    node = HousingTemperature(mode="coupled", tau_s=tau, weather=w)
    node.settle(0.0)
    series = np.array([node.at(ti) for ti in t])

    last_day = t >= 2.0 * period
    amp_house = 0.5 * float(np.ptp(series[last_day]))
    assert amp_house / amp_air == pytest.approx(1.0 / math.sqrt(1.0 + (omega * tau) ** 2), rel=0.02)

    # Phase: the housing peak arrives arctan(omega tau)/omega after the air peak.
    day = t[last_day]
    peak_house = float(day[int(np.argmax(series[last_day]))])
    peak_air = float(day[int(np.argmax(np.sin(omega * day)))])
    assert peak_house - peak_air == pytest.approx(math.atan(omega * tau) / omega, abs=1.5 * step)

    # And a node far faster than the drive follows it instead -- to within half a sampling step.
    # That residual is not an error in this node: M6.6's solver evaluates the ambient at the step
    # midpoint (second order for a smooth ambient), so a node with tau << dt relaxes onto the
    # midpoint air temperature rather than the endpoint one. The bound is therefore derived from
    # how far the air moves in half a step, not guessed.
    half_step_k = 0.5 * step * amp_air * omega
    fast = HousingTemperature(mode="coupled", tau_s=1.0, weather=w)
    fast.settle(0.0)
    followed = np.array([fast.at(ti) for ti in t])
    air = np.array([w.at(ti).t_air_k for ti in t])
    assert np.max(np.abs(followed - air)) <= 1.05 * half_step_k
    assert np.max(np.abs(followed - air)) < 0.2 * amp_house  # still far tighter than the slow node


# -- guards --------------------------------------------------------------------------------


def test_coupled_requires_a_positive_time_constant() -> None:
    for tau in (None, 0.0, -5.0, float("inf")):
        with pytest.raises(ValueError, match="positive housing_tau_s"):
            HousingTemperature(mode="coupled", tau_s=tau, weather=_weather())


def test_self_heating_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HousingTemperature(mode="coupled", tau_s=10.0, self_heating_k=-1.0, weather=_weather())


def test_two_ambient_sources_are_refused() -> None:
    """CLAUDE.md #6: one weather object. Two sources is two weathers waiting to disagree."""
    with pytest.raises(ValueError, match="not both"):
        HousingTemperature(
            mode="coupled",
            tau_s=10.0,
            weather=_weather(),
            ambient_provider=lambda t: 300.0,
        )


def test_the_coupled_node_refuses_to_run_time_backwards() -> None:
    node = HousingTemperature(mode="coupled", tau_s=100.0, weather=_weather())
    node.at(500.0)
    assert node.at(500.0) == node.temperature_k  # re-reading the current time is a read
    with pytest.raises(ValueError, match="cannot step back"):
        node.at(499.0)
    node.settle(0.0)
    assert node.at(10.0) > 0.0  # settle() restarts it


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown housing_temp_mode"):
        HousingTemperature(mode="floating")  # type: ignore[arg-type]


# -- wiring --------------------------------------------------------------------------------


def test_from_optics_reads_the_sensor_yaml() -> None:
    """The camera's housing behaviour lives with the camera, not in calling code."""
    optics = SensorConfig.model_validate(BOSON).sensor.optics
    w = _weather(290.0)
    node = HousingTemperature.from_optics(optics, weather=w, t0_s=0.0)
    assert node.mode == optics.housing_temp_mode
    if optics.housing_temp_mode == "coupled":
        assert node.tau_s == optics.housing_tau_s
        assert node.self_heating_k == optics.housing_self_heating_k
        settled = node.at(86_400.0)  # ~96 tau: fully settled, and inside the weather window
        assert settled == pytest.approx(290.0 + optics.housing_self_heating_k, abs=TOL_K)


def test_an_ambient_provider_is_accepted_and_reports_no_weather() -> None:
    """An explicit provider is for tests and for engine-side ambient; it holds no WeatherSeries,
    so the Scene identity check passes it over rather than comparing against None."""
    node = HousingTemperature(mode="coupled", tau_s=50.0, ambient_provider=lambda t: 300.0 + t)
    assert node.weather is None
    assert node.at(0.0) == pytest.approx(300.0, abs=1e-9)


def test_band_radiance_rises_with_housing_temperature(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """Phi_self consumes L_B(T_housing); a hotter housing must emit more, monotonically."""
    cold = HousingTemperature(mode="fixed", fixed_temp_k=280.0)
    hot = HousingTemperature(mode="fixed", fixed_temp_k=320.0)
    assert hot.band_radiance(0.0, boson_lut) > cold.band_radiance(0.0, boson_lut) > 0.0


def test_scene_refuses_a_housing_node_on_a_different_weather() -> None:
    """Non-negotiable #6, end to end: the housing registers as a Scene consumer via ``.weather``.

    A lens warming on a different day than the atmosphere would be invisible in the image -- the
    self-emission term is a smooth pedestal -- so it has to be refused at construction.
    """
    from irsim.scene import Scene

    scene_weather = _weather(293.15)
    other_weather = _weather(273.15)  # a different object *and* a different day

    good = HousingTemperature(mode="coupled", tau_s=100.0, weather=scene_weather)
    bad = HousingTemperature(mode="coupled", tau_s=100.0, weather=other_weather)

    def _scene(housing: HousingTemperature) -> Scene:
        return Scene(
            spec=_MinimalSceneSpec(),
            weather=scene_weather,
            grey_atmosphere=_NoWeather(),
            targets={},
            t0_s=0.0,
            extra_consumers={"housing": housing},
        )

    _scene(good)  # accepted
    with pytest.raises(ValueError, match="different WeatherSeries"):
        _scene(bad)


class _NoWeather:
    """A stand-in consumer that holds no weather, so the test isolates the housing node."""

    weather = None


class _MinimalSceneSpec:
    """Only what ``Scene.__post_init__`` touches."""

    weather_file = "unused.csv"
