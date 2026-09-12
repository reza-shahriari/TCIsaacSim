"""Tier 3 atmosphere phenomenology and the solar-path stub (M8.8).

The §7.2 table's decisive claims are not about absolute transmittance -- they are about *which
band wins*: LWIR beats everything in fog, SWIR beats LWIR in humid clear air, and the visible
dies first in both. A simulator that reproduces those orderings from the weather alone is doing
real work (§7.2); one that merely lands each number in its row could still have the bands in the
wrong order. These tests assert the orderings and the crossovers over every committed preset,
plus the geometric self-consistency of the layered model and the Bouguer solar stub.

docs/physics-model.md §15 (Tier 3), §7.2, §7.3, §5.4; ADR 0048, 0049, 0051, 0071
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere.extinction import (
    MAX_SOLAR_ZENITH_DEG,
    airmass,
    solar_transmittance,
    transmittance_per_band,
)
from irsim.atmosphere.humidity import absolute_humidity_g_m3, saturation_vapour_pressure_hpa
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import available_presets, load_atmosphere_preset
from irsim.atmosphere.model import Atmosphere
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries

BANDS_SHORT_TO_LONG = ("visible", "nir", "swir", "mwir", "lwir")
T_AIR = 288.15


def _weather(t_air: float, rh: float, visibility_m: float) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, rh, 1.0, 0.0, 0.0, 0.0, visibility_m, 0.0), 3600.0
    )


# -- §7.3 humidity known answers -------------------------------------------------------------


def test_magnus_known_values_and_humidity_ordering() -> None:
    """The Magnus/Bolton anchors the whole humidity chain rests on (§7.3)."""
    assert saturation_vapour_pressure_hpa(0.0) == pytest.approx(6.112, rel=1e-12)
    assert saturation_vapour_pressure_hpa(20.0) == pytest.approx(23.39, rel=5e-3)
    assert saturation_vapour_pressure_hpa(30.0) == pytest.approx(42.47, rel=5e-3)
    assert absolute_humidity_g_m3(303.15, 0.80) == pytest.approx(24.3, rel=1e-2)
    # warmer or wetter air always holds more water
    assert absolute_humidity_g_m3(303.15, 0.8) > absolute_humidity_g_m3(293.15, 0.8)
    assert absolute_humidity_g_m3(293.15, 0.8) > absolute_humidity_g_m3(293.15, 0.3)


# -- §7.2 band orderings, over every preset --------------------------------------------------


@pytest.mark.parametrize("name", sorted(available_presets()))
def test_every_preset_orders_the_bands_by_particle_size(name: str) -> None:
    """Aerosol and droplet extinction both fall with wavelength, so at any one weather the
    transmittance must rise monotonically from the visible to the SWIR. (LWIR and MWIR can sit
    below SWIR: their molecular water-vapour absorption is a different mechanism, which is
    exactly why the humid crossover exists.)"""
    preset = load_atmosphere_preset(name)
    fog = preset.aerosol_regime == "droplet"
    tau = transmittance_per_band(preset, T_AIR, 0.5, 200.0 if fog else 5000.0, 200.0)
    assert tau["visible"] < tau["nir"] < tau["swir"], tau
    assert all(0.0 < v <= 1.0 for v in tau.values())


def test_fog_ordering_and_ranges_at_200_m() -> None:
    """§7.2's light-fog row, and the ordering that is the commercial argument for LWIR."""
    fog = load_atmosphere_preset("fog_light_200m")
    tau = transmittance_per_band(fog, 283.15, 1.0, 200.0, 200.0)
    assert tau["lwir"] > tau["mwir"] > tau["swir"] > tau["nir"] > tau["visible"], tau
    assert 0.35 <= tau["lwir"] <= 0.60, tau["lwir"]
    assert 0.02 <= tau["visible"] <= 0.10, tau["visible"]
    dense = load_atmosphere_preset("fog_dense_50m")
    assert transmittance_per_band(dense, 283.15, 1.0, 50.0, 200.0)["visible"] < 1e-4


def test_humid_and_fog_crossover_from_one_preset_and_the_weather() -> None:
    """The §7.2 crossover: water vapour hits 8-12 um hardest, so humid clear air makes LWIR
    *worse* than SWIR, while fog droplets reverse it. One preset, weather alone."""
    preset = load_atmosphere_preset("us_standard_clear")
    humid = transmittance_per_band(preset, 303.15, 0.80, 23000.0, 200.0)
    assert humid["lwir"] < humid["swir"], humid
    fog = transmittance_per_band(preset, 283.15, 1.0, 200.0, 200.0)
    assert fog["lwir"] > fog["swir"], fog
    dry = transmittance_per_band(preset, 303.15, 0.10, 23000.0, 200.0)
    assert dry["lwir"] > humid["lwir"], "drying the air helps LWIR most"
    assert dry["lwir"] - humid["lwir"] > dry["swir"] - humid["swir"]


# -- geometry: the layered model against itself ----------------------------------------------


def _apparent(lut: BandLUT, radiance: float) -> float:
    return float(lut.apparent_temperature(np.asarray(radiance))[()])


@pytest.mark.parametrize("distance_m", [0.0, 10.0, 300.0, 5000.0])
def test_a_target_at_air_temperature_is_distance_invariant(
    tophat_lwir_lut: BandLUT, distance_m: float
) -> None:
    """The isothermal identity on a horizontal path: a target radiating exactly L_B(T_air) is
    indistinguishable from the air at any range, for the grey and the layered model alike. Any
    error here is a bookkeeping bug in attenuation vs path radiance (§7.1)."""
    weather = _weather(T_AIR, 0.5, 23000.0)
    preset = load_atmosphere_preset("us_standard_clear")
    l_air = float(tophat_lwir_lut.lookup(np.float64(T_AIR))[()])
    grey = Atmosphere(preset, weather, {"lwir": tophat_lwir_lut})
    assert float(grey.apply("lwir", 0.0, np.float64(l_air), distance_m)) == pytest.approx(
        l_air, rel=1e-9
    )
    layered = LayeredAtmosphere(preset, weather, {"lwir": tophat_lwir_lut})
    assert float(layered.apply("lwir", 0.0, np.float64(l_air), distance_m, 0.0)) == pytest.approx(
        l_air, rel=1e-9
    )


def test_short_slant_path_matches_the_horizontal_one(tophat_lwir_lut: BandLUT) -> None:
    """A 300 m path at 10 deg rises only 52 m -- 2.6 % of the water-vapour scale height -- so it
    must read within 0.1 K of the same length horizontally. This is the self-consistency check
    on MS.1's column integrals: get the geometry wrong and the two diverge immediately."""
    weather = _weather(T_AIR, 0.5, 23000.0)
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": tophat_lwir_lut}
    )
    l_target = float(tophat_lwir_lut.lookup(np.float64(320.0))[()])
    horizontal = float(atm.apply("lwir", 0.0, np.float64(l_target), 300.0, 0.0))
    slant = float(atm.apply("lwir", 0.0, np.float64(l_target), 300.0, math.radians(10.0)))
    difference_k = abs(_apparent(tophat_lwir_lut, horizontal) - _apparent(tophat_lwir_lut, slant))
    assert difference_k < 0.1, difference_k
    # A slant ray climbs into thinner air, so it attenuates less and carries less path radiance.
    # Which way that moves the reading depends on the target: a target hotter than the air keeps
    # more of its own signal (warmer), a colder one is warmed less by the path (colder). Both
    # signs must appear, or attenuation and path radiance are not being kept apart.
    assert slant > horizontal, (slant, horizontal)
    l_cold = float(tophat_lwir_lut.lookup(np.float64(260.0))[()])
    cold_horizontal = float(atm.apply("lwir", 0.0, np.float64(l_cold), 300.0, 0.0))
    cold_slant = float(atm.apply("lwir", 0.0, np.float64(l_cold), 300.0, math.radians(10.0)))
    assert cold_slant < cold_horizontal, (cold_slant, cold_horizontal)
    # Over 300 m the geometry barely matters (0.085 K between horizontal and zenith), which is
    # why the grey horizontal model was adequate to ~500 m. Stretch the path to 5 km and the two
    # diverge by more than a kelvin -- that gap is why MS.1's column integrals exist.
    far_h = float(atm.apply("lwir", 0.0, np.float64(l_target), 5000.0, 0.0))
    far_z = float(atm.apply("lwir", 0.0, np.float64(l_target), 5000.0, math.radians(90.0)))
    spread_k = abs(_apparent(tophat_lwir_lut, far_z) - _apparent(tophat_lwir_lut, far_h))
    assert spread_k > 1.0, spread_k


# -- §5.4 solar path stub ---------------------------------------------------------------------


def test_bouguer_airmass_and_the_60_degree_identity() -> None:
    """τ_sun(θ) = τ_zenith^{sec θ}: at 60° the airmass is exactly 2, so the transmittance is the
    zenith value squared. Pinning that identity is what makes the stub checkable before M11
    replaces it (ADR 0051)."""
    assert airmass(0.0) == pytest.approx(1.0, rel=1e-12)
    assert airmass(math.radians(60.0)) == pytest.approx(2.0, rel=1e-12)
    for name in available_presets():
        preset = load_atmosphere_preset(name)
        for band in BANDS_SHORT_TO_LONG:
            zenith = preset.solar.zenith_transmittance[band]
            assert solar_transmittance(preset, band, 0.0) == pytest.approx(zenith, rel=1e-12)
            assert solar_transmittance(preset, band, math.radians(60.0)) == pytest.approx(
                zenith**2, rel=1e-12
            )
    preset = load_atmosphere_preset("us_standard_clear")
    # monotone in zenith angle, and the sun never gains flux by being lower
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
    taus = [solar_transmittance(preset, "visible", math.radians(a)) for a in angles]
    assert all(b < a for a, b in zip(taus[:-1], taus[1:], strict=True)), taus
    with pytest.raises(ValueError, match="plane-parallel"):
        airmass(math.radians(MAX_SOLAR_ZENITH_DEG + 1.0))
    with pytest.raises(KeyError):
        solar_transmittance(preset, "thz", 0.0)


def test_solar_stub_orders_the_presets_by_how_much_they_block_the_sun() -> None:
    """Clear air passes more sun than haze, which passes more than fog -- in every band. A
    preset table that got this backwards would produce daytime MWIR glint in dense fog."""
    for band in BANDS_SHORT_TO_LONG:
        clear = load_atmosphere_preset("us_standard_clear").solar.zenith_transmittance[band]
        haze = load_atmosphere_preset("haze").solar.zenith_transmittance[band]
        light_fog = load_atmosphere_preset("fog_light_200m").solar.zenith_transmittance[band]
        dense_fog = load_atmosphere_preset("fog_dense_50m").solar.zenith_transmittance[band]
        assert clear > haze > light_fog > dense_fog, band
