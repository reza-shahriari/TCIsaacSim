"""FPA temperature node and gain/offset(T_FPA) polynomials (M9.2, ADR 0053): the steady state, the
time constant recovered from the 63.2 % crossing, normalisation at T_cal, the uniform DN shift a
5 K ambient rise produces (and its absence when TEC-pinned), per-step energy conservation, and the
one-weather-object guard (CLAUDE.md #6)."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import SCHEMA_VERSION
from irsim.detector.fpa_thermal import FpaThermalModel, gain_of_t, offset_of_t
from irsim.thermal import WeatherSample, WeatherSeries

TAU_S = 30.0
SELF_HEATING_K = 4.0
T_AMB = 290.0


def _weather(t_air: float) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, 0.4, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _coupled(ambient: float | None = None, **over: object) -> FpaThermalModel:
    kw: dict[str, object] = {
        "mode": "coupled",
        "tau_s": TAU_S,
        "self_heating_k": SELF_HEATING_K,
        "ambient_provider": (lambda _t: T_AMB) if ambient is None else (lambda _t: ambient),
    }
    kw.update(over)
    return FpaThermalModel(**kw)  # type: ignore[arg-type]


def test_schema_version_bumped_for_the_fpa_node() -> None:
    # A floor, not the current number: this guards that the FPA node bumped the version, and
    # later steps bump it again (M9.3 took it to 6). The current value is pinned once, in
    # tests/unit/test_optics_schema.py.
    assert SCHEMA_VERSION >= 5


def test_steady_state_is_ambient_plus_self_heating() -> None:
    """T -> T_amb + P/h to 1 mK, from either side, and it is the fixed point of step()."""
    node = _coupled()
    assert node.steady_state_k(0.0) == pytest.approx(T_AMB + SELF_HEATING_K, abs=1e-12)

    for start in (T_AMB - 20.0, T_AMB + 20.0):
        n = _coupled()
        n.temperature_k = start
        t = 0.0
        for _ in range(4000):  # 400 s = 13 tau
            t += 0.1
            n.step(t, 0.1)
        assert n.step(t, 0.1) == pytest.approx(T_AMB + SELF_HEATING_K, abs=1e-3), start

    settled = _coupled()
    settled.settle(0.0)
    before = settled.temperature_k
    assert settled.step(1.0, 1.0) == pytest.approx(before, abs=1e-12), "steady state is a fixed pt"


def test_time_constant_from_the_63_2_percent_crossing() -> None:
    """Settled at one ambient, stepped to another, the node crosses 63.2 % of the change at t = tau
    within 1 % -- the observable definition of the time constant, not the one the code was given."""
    step_k = 10.0
    ambient = T_AMB
    node = _coupled(ambient=None, ambient_provider=lambda _t: ambient)
    node.settle(0.0)
    start = node.temperature_k
    assert start is not None
    ambient = T_AMB + step_k  # the step
    final = ambient + SELF_HEATING_K
    target = start + 0.632 * (final - start)

    dt, t, prev = 0.01, 0.0, start
    crossing = None
    for _ in range(20000):
        t += dt
        cur = node.step(t, dt)
        if crossing is None and cur >= target:
            # linear interpolation between the bracketing samples
            crossing = t - dt + dt * (target - prev) / (cur - prev)
            break
        prev = cur
    assert crossing is not None
    assert crossing == pytest.approx(TAU_S, rel=0.01), crossing


def test_energy_is_conserved_each_step() -> None:
    """Each RK2 step satisfies C dT = (P - h (T_stage - T_amb)) dt to 1e-6 relative.

    Written in C, h and P rather than tau and DeltaT_self so it checks the integrator against the
    energy balance the spec states, not against the algebra the implementation happens to use.
    """
    c_th = 12.0  # J/K
    h = c_th / TAU_S  # W/K, so tau = C/h
    power_w = h * SELF_HEATING_K  # so P/h = DeltaT_self

    node = _coupled()
    node.temperature_k = T_AMB - 15.0
    dt, t = 0.5, 0.0
    for _ in range(200):
        t0 = node.temperature_k
        assert t0 is not None
        t += dt
        t1 = node.step(t, dt)
        k1 = (T_AMB + SELF_HEATING_K - t0) / TAU_S
        t_stage = 0.5 * (t0 + (t0 + dt * k1))  # the Heun stage mean
        stored = c_th * (t1 - t0) / dt
        net = power_w - h * (t_stage - T_AMB)
        assert stored == pytest.approx(net, rel=1e-6, abs=1e-12), (t, stored, net)


def test_gain_and_offset_are_normalised_at_t_cal() -> None:
    """g(T_cal) == 1 and o(T_cal) == 0 exactly -- structural, since the coefficients carry no
    constant term and so cannot be authored with one."""
    t_cal = 300.0
    gains = (2.0e-3, -1.5e-5)
    offsets = (-8.0, 0.25)
    assert gain_of_t(t_cal, t_cal, gains) == 1.0
    assert offset_of_t(t_cal, t_cal, offsets) == 0.0
    assert gain_of_t(t_cal, t_cal, ()) == 1.0
    assert offset_of_t(t_cal, t_cal, ()) == 0.0

    dt = 5.0
    assert gain_of_t(t_cal + dt, t_cal, gains) == pytest.approx(
        1.0 + gains[0] * dt + gains[1] * dt**2, rel=1e-12
    )
    assert offset_of_t(t_cal - dt, t_cal, offsets) == pytest.approx(
        offsets[0] * -dt + offsets[1] * dt**2, rel=1e-12
    )
    # the drift is signed: warming and cooling by the same amount do not cancel
    assert gain_of_t(t_cal + dt, t_cal, gains) != gain_of_t(t_cal - dt, t_cal, gains)


def test_five_kelvin_ambient_rise_shifts_dn_uniformly_and_not_when_tec_pinned() -> None:
    """The phenomenology M9.2 exists for: shutterless drift. A 5 K ambient rise moves every pixel
    of a uniform scene by the same DN, and a TEC-pinned core does not move at all."""
    t_cal = T_AMB + SELF_HEATING_K
    # deliberately not (2e-3, -8.0): at 4000 DN those cancel exactly (+40 gain, -40 offset), which
    # would let a broken model pass by reading zero shift for the wrong reason
    gains, offsets = (2.0e-3,), (-3.0,)
    scene = np.full((8, 12), 4000.0)

    def settled_dn(node: FpaThermalModel, t_s: float) -> np.ndarray:
        node.settle(t_s)
        return scene * node.gain(t_cal, gains) + node.offset(t_cal, offsets)

    ambient = T_AMB
    coupled = _coupled(ambient=None, ambient_provider=lambda _t: ambient)
    cold = settled_dn(coupled, 0.0)
    assert np.allclose(cold, scene), "calibrated at the settled temperature, so no shift yet"

    ambient = T_AMB + 5.0
    warm = settled_dn(coupled, 1.0)
    shift = warm - cold
    assert np.ptp(shift) < 1e-9, "the shift must be uniform across the array"
    expected = scene[0, 0] * gains[0] * 5.0 + offsets[0] * 5.0
    assert float(shift[0, 0]) == pytest.approx(expected, rel=1e-12)
    assert abs(float(shift[0, 0])) > 1.0, "and large enough to matter in DN"

    pinned = FpaThermalModel(mode="fixed", fixed_temp_k=t_cal)
    assert np.allclose(settled_dn(pinned, 0.0), settled_dn(pinned, 1.0))
    assert np.allclose(settled_dn(pinned, 1.0), scene), "a TEC-pinned core does not drift"


def test_ambient_mode_tracks_air_temperature_with_no_lag() -> None:
    node = FpaThermalModel(mode="ambient", weather=_weather(295.0))
    assert node.step(0.0, 1.0) == pytest.approx(295.0)
    assert node.steady_state_k(0.0) == pytest.approx(295.0)


def test_one_weather_object_only() -> None:
    """CLAUDE.md #6: the FPA node takes the shared WeatherSeries or a provider, never both, and
    never neither -- two ambient sources is exactly the summer/winter split that rule forbids."""
    with pytest.raises(ValueError, match="not both"):
        FpaThermalModel(
            mode="coupled",
            tau_s=TAU_S,
            weather=_weather(T_AMB),
            ambient_provider=lambda _t: 250.0,
        )
    with pytest.raises(ValueError, match="ambient source"):
        FpaThermalModel(mode="coupled", tau_s=TAU_S)
    with pytest.raises(ValueError, match="ambient source"):
        FpaThermalModel(mode="ambient")

    shared = _weather(T_AMB)
    node = FpaThermalModel(mode="coupled", tau_s=TAU_S, self_heating_k=1.0, weather=shared)
    assert node.weather is shared, "the node holds the object, not a copy"
    assert node.ambient_k(0.0) == pytest.approx(T_AMB)


def test_configuration_guards() -> None:
    with pytest.raises(ValueError, match="fpa_temp_k"):
        FpaThermalModel(mode="fixed")
    with pytest.raises(ValueError, match="fpa_tau_s"):
        FpaThermalModel(mode="coupled", ambient_provider=lambda _t: T_AMB)
    with pytest.raises(ValueError, match="tick"):
        _coupled().step(1.0, 0.0)
    with pytest.raises(ValueError, match="no temperature yet"):
        _coupled().gain(300.0, (1e-3,))
    # 'fixed' needs no ambient source at all and settles immediately
    pinned = FpaThermalModel(mode="fixed", fixed_temp_k=303.0)
    assert pinned.temperature_k == 303.0
    with pytest.raises(ValueError, match="fixed"):
        pinned.ambient_k(0.0)
