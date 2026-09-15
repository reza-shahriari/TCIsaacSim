"""The sea skin temperature: cool skin and diurnal warm layer (MM.4, ADR 0080).

The camera never sees the bulk SST a scenario authors. It sees the top fraction of a millimetre,
which is colder whenever the ocean is losing heat and can be warmer on a calm sunny afternoon.
Both corrections are sub-kelvin and both are several times a 50 mK NETD, so the tests here are
about *size and sign* against independent arithmetic, not about reproducing remembered numbers.

The load-bearing test is :func:`test_sst_accuracy_matters_at_nadir_and_barely_at_the_horizon`. It
is the deliverable MM.4 was specified around: it says where an SST error shows up in a frame and
where the atmosphere and the reflected sky have already swallowed it.

docs/physics-model.md §6.1, §6.5; ADR 0080
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.atmosphere.sea import SeaModel, horizon_depression_rad, slant_range_m
from irsim.config.environment import load_environment_preset
from irsim.materials.nk import load_nk_table
from irsim.radiometry.constants import (
    DENSITY_AIR_SEA_LEVEL,
    DENSITY_SEAWATER,
    THERMAL_CONDUCTIVITY_SEAWATER,
)
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal.sea_skin import (
    DEFAULT_SEA_SKIN,
    SeaSkinParams,
    cool_skin_deficit_k,
    cool_skin_thickness_m,
    friction_velocity_water_m_s,
    net_longwave_up_w_m2,
    skin_temperature_k,
    warm_layer_k,
)
from irsim.thermal.weather import WeatherSample, WeatherSeries

BOSON_RESPONSE = "data/spectra/responses/boson_vox.csv"
SST_K = 290.0
T_AIR_K = 291.0
#: Net longwave a clear night over a 290 K sea comes to, from the scene's own sky model. Quoted
#: here so the deficits below are readable as "at a realistic flux", not at a round number.
Q_NET_TYPICAL = 79.0


@pytest.fixture(scope="module")
def rig():
    response = load_spectral_response(BOSON_RESPONSE)
    return response, BandLUT.build(response), load_nk_table("water")


def _sea(rig, wind=5.0, sst=SST_K, dni=0.0, dhi=0.0, **kwargs):
    response, lut, table = rig
    weather = WeatherSeries.constant(
        WeatherSample(T_AIR_K, 0.5, wind, 0.0, dni, dhi, 23000.0, 0.0),
        86400.0,
        epoch_utc=datetime(2026, 6, 21, 0, 0, tzinfo=timezone.utc),
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut})
    sky = SkyModel(atm, load_environment_preset("clear_dry"), "lwir", lut)
    return sky, SeaModel(sky, table, response, bulk_sst_k=sst, **kwargs)


# --- the conductive sublayer ---------------------------------------------------------------


def test_the_friction_velocity_is_continuous_across_the_surface() -> None:
    """Same stress both sides: rho_a C_D U^2 = rho_w u*^2. Checked against the definition."""
    wind, c_d = 7.0, 1.3e-3
    u_star = float(friction_velocity_water_m_s(wind, c_d))
    stress_air = DENSITY_AIR_SEA_LEVEL * c_d * wind**2
    stress_water = DENSITY_SEAWATER * u_star**2
    assert stress_water == pytest.approx(stress_air, rel=1e-12)
    # and the water-side value is ~1/1000 of the wind, which is why the sublayer is millimetric
    assert 1e-3 < u_star / wind < 2e-3


def test_the_sublayer_is_a_millimetre_at_a_moderate_wind() -> None:
    """Saunders' form, checked against the measurement it is famous for reproducing.

    The cool skin is observed at 0.3-2 mm thick, about 1 mm in a moderate breeze. A model off by
    a factor of ten here would still produce a plausible-looking deficit, because the deficit is
    also linear in a flux nobody measures directly in this scene.
    """
    assert 0.8e-3 < float(cool_skin_thickness_m(5.0)) < 1.2e-3
    assert 0.4e-3 < float(cool_skin_thickness_m(10.0)) < 0.7e-3
    assert float(cool_skin_thickness_m(15.0)) < float(cool_skin_thickness_m(5.0))


def test_the_sublayer_is_bounded_where_saunders_diverges() -> None:
    """u* -> 0 sends lambda nu / u* to infinity; the observed cool skin stays millimetric.

    Without the bound a dead-calm scene gets an unbounded deficit -- and calm, clear nights are
    exactly the maritime conditions a long-range IR scenario cares about most.
    """
    assert float(cool_skin_thickness_m(0.0)) == pytest.approx(DEFAULT_SEA_SKIN.max_thickness_m)
    assert np.all(np.isfinite(cool_skin_thickness_m(np.array([0.0, 1e-9, 1e-3]))))
    # and the bound does not intrude on the wind-stirred regime it exists to protect
    tanh_free = DEFAULT_SEA_SKIN.saunders_lambda * 1.05e-6 / float(friction_velocity_water_m_s(8.0))
    assert float(cool_skin_thickness_m(8.0)) == pytest.approx(tanh_free, rel=0.05)


def test_the_deficit_falls_with_wind_and_never_turns_into_a_surplus() -> None:
    """Monotone in wind, and clamped at zero under net warming.

    The clamp is a division of labour, not a convenience: conduction against an *outgoing* flux is
    what makes the skin cool, so under net warming this mechanism simply stops. A surface warmer
    than the water beneath it is the warm layer, and letting the deficit go negative would count
    that twice.
    """
    winds = np.linspace(0.5, 25.0, 400)
    deficit = cool_skin_deficit_k(Q_NET_TYPICAL, winds)
    assert np.all(np.diff(deficit) < 0.0), "not strictly decreasing in wind"
    assert np.all(deficit > 0.0)
    assert float(cool_skin_deficit_k(-150.0, 2.0)) == 0.0
    assert float(cool_skin_deficit_k(0.0, 2.0)) == 0.0


def test_the_deficit_has_the_observed_size_and_scales_with_the_flux() -> None:
    """0.1-0.6 K is the observed range; within it, dT is exactly linear in Q_net.

    Linearity is worth asserting separately because it is what makes the missing turbulent fluxes
    a *bounded* omission rather than an unknown one: a caller who adds sensible and latent heat to
    Q_net gets a proportionally larger deficit and nothing else changes.
    """
    assert 0.08 < float(cool_skin_deficit_k(Q_NET_TYPICAL, 5.0)) < 0.20
    assert 0.15 < float(cool_skin_deficit_k(200.0, 5.0)) < 0.45
    assert float(cool_skin_deficit_k(150.0, 1.0)) < 0.6
    doubled = float(cool_skin_deficit_k(2.0 * Q_NET_TYPICAL, 5.0))
    assert doubled == pytest.approx(2.0 * float(cool_skin_deficit_k(Q_NET_TYPICAL, 5.0)), rel=1e-12)


def test_the_deficit_is_the_gradient_that_conducts_the_flux() -> None:
    """Fourier's law, evaluated independently: dT = Q delta / k."""
    q, wind = 120.0, 6.0
    expected = q * float(cool_skin_thickness_m(wind)) / THERMAL_CONDUCTIVITY_SEAWATER
    assert float(cool_skin_deficit_k(q, wind)) == pytest.approx(expected, rel=1e-12)


# --- the diurnal warm layer ----------------------------------------------------------------


def test_the_warm_layer_is_zero_at_night_and_in_wind() -> None:
    """Its two hard limits. A warm layer at 3 a.m. is the failure that would look like a bug."""
    assert float(warm_layer_k(0.0, 1.0)) == 0.0
    assert float(warm_layer_k(900.0, 6.0)) == 0.0
    assert float(warm_layer_k(900.0, 12.0)) == 0.0
    assert np.all(warm_layer_k(900.0, np.linspace(6.0, 30.0, 50)) == 0.0)


def test_the_warm_layer_reaches_about_three_kelvin_on_a_calm_clear_noon() -> None:
    """The observed upper end, and monotone in both drivers on the way there."""
    calm_noon = float(warm_layer_k(950.0, 1.0))
    assert 2.0 < calm_noon <= DEFAULT_SEA_SKIN.warm_layer_max_k
    winds = np.linspace(0.0, 5.9, 200)
    assert np.all(np.diff(warm_layer_k(950.0, winds)) < 0.0)
    irradiance = np.linspace(0.0, 900.0, 200)
    assert np.all(np.diff(warm_layer_k(irradiance, 2.0)) > 0.0)


def test_the_warm_layer_saturates_rather_than_extrapolating() -> None:
    """An irradiance above the reference cannot drive an unbounded rise.

    A tropical noon with a bright cloud edge can put more than 1000 W/m2 on the surface for a few
    minutes; a linear form would turn that into a 5 K warm layer nothing has observed.
    """
    assert float(warm_layer_k(3000.0, 0.0)) == pytest.approx(DEFAULT_SEA_SKIN.warm_layer_max_k)
    with pytest.raises(ValueError, match="cannot be negative"):
        warm_layer_k(-1.0, 1.0)


# --- the two together ----------------------------------------------------------------------


def test_the_two_mechanisms_oppose_each_other_and_compose() -> None:
    """T_skin = T_bulk - cool + warm, with the same wind in both."""
    bulk, q_net, q_sw, wind = 290.0, Q_NET_TYPICAL, 800.0, 2.0
    got = float(skin_temperature_k(bulk, q_net, q_sw, wind))
    expect = bulk - float(cool_skin_deficit_k(q_net, wind)) + float(warm_layer_k(q_sw, wind))
    assert got == pytest.approx(expect, rel=1e-12)
    # a sunny calm day is warmer than the bulk; a clear night is colder
    assert float(skin_temperature_k(bulk, q_net, 800.0, 1.0)) > bulk
    assert float(skin_temperature_k(bulk, q_net, 0.0, 1.0)) < bulk


def test_a_custom_parameterisation_is_honoured_everywhere() -> None:
    """The params object is one knob-set, not a default repeated at each call site."""
    flat = SeaSkinParams(warm_layer_max_k=0.0, saunders_lambda=12.0)
    assert float(warm_layer_k(900.0, 0.0, flat)) == 0.0
    assert float(cool_skin_thickness_m(10.0, flat)) > float(cool_skin_thickness_m(10.0))


# --- on the sea model, driven by the scene's own weather -----------------------------------


def test_the_sea_model_derives_its_skin_from_the_shared_weather(rig) -> None:
    """The wind that roughens the surface is the wind that thins the sublayer.

    This is CLAUDE.md #6 in miniature. Before MM.4 the deficit was a constructor argument, so a
    scene could roughen its surface with 15 m/s while holding a dead-calm skin offset; now both
    read the same `WeatherSeries` and cannot disagree.
    """
    _, calm = _sea(rig, wind=1.0)
    _, windy = _sea(rig, wind=15.0)
    assert calm.cool_skin_deficit_k(0.0) > windy.cool_skin_deficit_k(0.0)
    assert calm.skin_temperature_k(0.0) < calm.bulk_sst_k
    assert calm.tilt_sigma(0.0) < windy.tilt_sigma(0.0)  # same wind, the other consumer


def test_the_net_longwave_comes_from_the_scenes_own_sky(rig) -> None:
    """Not a typed-in flux: a clear night over a 290 K sea loses ~80 W/m2, and cloud closes it.

    Checked as a *response* rather than a value -- overcast raises the downwelling toward a
    blackbody at air temperature, so the net loss must collapse and the cool skin with it.
    """
    _, clear = _sea(rig)
    q_clear = clear.net_longwave_up_w_m2(0.0)
    assert 40.0 < q_clear < 120.0

    response, lut, table = rig
    overcast = WeatherSeries.constant(
        WeatherSample(T_AIR_K, 0.95, 5.0, 1.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), overcast, {"lwir": lut})
    sky = SkyModel(atm, load_environment_preset("clear_dry"), "lwir", lut)
    cloudy = SeaModel(sky, table, response, bulk_sst_k=SST_K)
    assert cloudy.net_longwave_up_w_m2(0.0) < 0.5 * q_clear
    assert cloudy.cool_skin_deficit_k(0.0) < 0.5 * clear.cool_skin_deficit_k(0.0)


def test_without_a_site_there_is_no_warm_layer(rig) -> None:
    """A sea model that does not know where it is cannot know whether the sun is up.

    It returns zero rather than guessing from the irradiance alone, and that is a deliberate
    choice: DNI and DHI without an elevation cannot distinguish noon from a bright dawn.
    """
    _, blind = _sea(rig, dni=900.0, dhi=100.0, wind=1.0)
    assert blind.absorbed_solar_w_m2(0.0) == 0.0
    assert blind.warm_layer_k(0.0) == 0.0


def test_with_a_site_the_warm_layer_follows_the_sun(rig) -> None:
    """Midsummer at 57 N: nothing at midnight, a real warm layer at local noon, calm wind."""
    _, sea = _sea(rig, dni=900.0, dhi=100.0, wind=1.0, latitude_deg=57.0, longitude_deg=0.0)
    midnight, noon = 0.0, 12.0 * 3600.0
    assert sea.absorbed_solar_w_m2(midnight) == 0.0
    assert sea.warm_layer_k(midnight) == 0.0
    assert sea.skin_temperature_k(midnight) < sea.bulk_sst_k

    assert sea.absorbed_solar_w_m2(noon) > 500.0
    assert sea.warm_layer_k(noon) > 1.0
    assert sea.skin_temperature_k(noon) > sea.bulk_sst_k


def test_wind_switches_the_warm_layer_off_on_the_sea_model(rig) -> None:
    """The same sunlit noon, blown away: above the cutoff the skin is back below the bulk."""
    noon = 12.0 * 3600.0
    _, calm = _sea(rig, dni=900.0, dhi=100.0, wind=1.0, latitude_deg=57.0, longitude_deg=0.0)
    _, breezy = _sea(rig, dni=900.0, dhi=100.0, wind=8.0, latitude_deg=57.0, longitude_deg=0.0)
    assert calm.skin_temperature_k(noon) > calm.bulk_sst_k
    assert breezy.warm_layer_k(noon) == 0.0
    assert breezy.skin_temperature_k(noon) < breezy.bulk_sst_k


def test_the_net_longwave_helper_uses_the_absorbed_share_of_the_downwelling() -> None:
    """eps sigma T^4 - eps Q_down. Reflected downwelling never enters the water."""
    t, down, eps = 290.0, 320.0, 0.98
    from irsim.radiometry.constants import SIGMA_SB

    assert float(net_longwave_up_w_m2(t, down, eps)) == pytest.approx(
        eps * (SIGMA_SB * t**4 - down), rel=1e-12
    )
    # a blackbody sky at the surface's own temperature gives zero net loss, for any eps
    balanced = float(net_longwave_up_w_m2(t, SIGMA_SB * t**4, 0.7))
    assert balanced == pytest.approx(0.0, abs=1e-9)


# --- the deliverable: where SST accuracy matters --------------------------------------------


def test_sst_accuracy_matters_at_nadir_and_barely_at_the_horizon(rig) -> None:
    """MM.4's real output: a 1 K SST error is a 1 K frame error looking down and a 0.2 K one at
    the horizon.

    This is the sensitivity ordering, and it is the answer to "how well do we need to know the
    SST". Two things suppress it away from nadir and they compound: the emissivity collapses
    toward grazing, so most of the signal is reflected sky rather than water; and the slant range
    grows as the depression shrinks, so the atmosphere replaces more of what is left.

    ⚠️ The ordering is what MM.4 promised and it holds with room to spare. The *threshold* the
    roadmap wrote down -- under 0.15 K at 0.2 degrees of depression -- does not, and cannot at any
    camera height, which is why it is measured here rather than asserted. The sensitivity near the
    horizon is set by slant **range**, not by the angle: at 20 m the 0.2 degree ray is 6.8 km out
    and reads 0.20; getting under 0.15 needs about 13 km of path, which at 20 m is past the
    horizon entirely, and at the 100 m height where 13 km *is* 0.5 degrees, 0.2 degrees is above
    the horizon and sees no water at all.
    """
    _, cold = _sea(rig, sst=SST_K)
    _, warm = _sea(rig, sst=SST_K + 1.0)

    def sensitivity(depression_deg: float) -> float:
        d = math.radians(depression_deg)
        a = float(np.atleast_1d(cold.apparent_temperature_k(0.0, d))[0])
        b = float(np.atleast_1d(warm.apparent_temperature_k(0.0, d))[0])
        return b - a

    nadir = sensitivity(90.0)
    assert nadir > 0.9, nadir

    horizon_deg = math.degrees(horizon_depression_rad(cold.camera_height_m))
    assert 0.1 < horizon_deg < 0.2, horizon_deg
    near_horizon = sensitivity(0.2)
    assert float(slant_range_m(cold.camera_height_m, math.radians(0.2))) > 5e3
    assert near_horizon < 0.25, near_horizon
    assert nadir > 4.0 * near_horizon, (nadir, near_horizon)

    # and it falls monotonically between the two, so "near the horizon" is a regime and not a
    # single unlucky angle
    profile = [sensitivity(d) for d in (90.0, 45.0, 10.0, 1.0, 0.5, 0.2)]
    assert all(b < a for a, b in zip(profile, profile[1:], strict=False)), profile


def test_the_skin_correction_is_larger_than_the_sensor_can_ignore(rig) -> None:
    """The reason MM.4 exists at all: 0.1-0.3 K is several times a 50 mK NETD.

    Rendering the bulk SST as if it were the skin is not a rounding error -- it is a bias of a few
    noise-equivalent temperature differences, in one direction, across the whole lower half of
    every maritime frame.
    """
    netd_k = 0.05
    for wind in (1.0, 5.0, 10.0):
        _, sea = _sea(rig, wind=wind)
        deficit = sea.bulk_sst_k - sea.skin_temperature_k(0.0)
        assert deficit > netd_k, (wind, deficit)
        assert deficit < 0.6, (wind, deficit)
