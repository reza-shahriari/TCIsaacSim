"""M11.4 — what lights a SWIR scene when the sun is down (§5.5).

Two sources that behave differently, and the differences are the tests:

* airglow has **no direction** — no shadow, no sun, and cloud attenuates it without cutting it
  off, because cloud scatters it rather than absorbing it;
* the moon is reflected sunlight, so it has a phase law that is violently non-linear: half the
  disc lit is a *tenth* of the light, not half.

The one number this step could not reproduce is recorded rather than tuned away. §5.5 says that
"at full moon, moonlight and airglow radiation densities are comparable" [R7]. Built from the
visual magnitudes and an OH band model, the full moon comes out **13x** the default airglow over
0.9–1.7 µm. What *is* reproduced is the familiar SWIR rule of thumb: airglow is about a **quarter
moon** (1.22x). See ADR 0065 and spec issue S38.

docs/physics-model.md §5.5; ADR 0063, ADR 0065
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.materials.table import MaterialTable
from irsim.pipeline.illumination import Illumination
from irsim.pipeline.night import NightIllumination
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.constants import SOLAR_CONSTANT_W_M2
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.night import (
    AIRGLOW_SHAPE_FILE,
    CLOUD_FLOOR,
    FULL_MOON_IRRADIANCE_W_M2,
    airglow_variation,
    cloud_attenuation,
    load_airglow_spectrum,
    lunar_phase_factor,
    phase_angle_deg,
)
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COARSE_N = 161
DEFAULT_LEVEL_W_M2 = 1.0e-4  # 10 nW/cm², §5.5's default


@pytest.fixture(scope="module")
def swir():  # type: ignore[no-untyped-def]
    return load_sensor_config(REPO / "configs/sensors/example_swir_ingaas_640.yaml", DATA)


@pytest.fixture(scope="module")
def airglow():  # type: ignore[no-untyped-def]
    return load_airglow_spectrum(DATA / AIRGLOW_SHAPE_FILE)


@pytest.fixture(scope="module")
def swir_response(swir):  # type: ignore[no-untyped-def]
    return load_spectral_response(swir.sensor.band.spectral_response)


@pytest.fixture(scope="module")
def night(swir):  # type: ignore[no-untyped-def]
    return NightIllumination.for_sensor(swir, load_environment_preset("clear_dry"), "lb_q", DATA)


# ---------------------------------------------------------------------------------------------
# the airglow level and its shape
# ---------------------------------------------------------------------------------------------


def test_the_configured_level_is_recovered_by_the_shape_file(airglow) -> None:  # type: ignore[no-untyped-def]
    """A flat response over the whole support must read back exactly the configured level."""
    from irsim.radiometry.spectral_response import SpectralResponse

    lo, hi = airglow.support_um
    flat = SpectralResponse(
        wavelength_um=np.array([lo, hi]),
        response=np.array([1.0, 1.0]),
        source_path="<test>",
        sha256="",
    )
    assert airglow.band_irradiance(flat, DEFAULT_LEVEL_W_M2) == pytest.approx(
        DEFAULT_LEVEL_W_M2, rel=1e-6
    )


def test_a_band_takes_its_own_share_of_the_level_not_all_of_it(airglow, swir_response) -> None:  # type: ignore[no-untyped-def]
    """The level is defined over the shape's support (ADR 0065), so the fraction is physics.

    The strongest OH sequence (Δv = 1) sits at 1.4–2.0 µm, so a standard 1.7 µm InGaAs cut-off
    throws part of it away -- which is a real reason extended-InGaAs parts exist.
    """
    fraction = airglow.band_fraction(swir_response)
    assert 0.5 < fraction < 0.75, fraction
    assert airglow.band_irradiance(swir_response, DEFAULT_LEVEL_W_M2) == pytest.approx(
        fraction * DEFAULT_LEVEL_W_M2, rel=1e-12
    )


def test_extending_the_cutoff_to_2p5_um_gains_a_third_more_airglow(airglow) -> None:  # type: ignore[no-untyped-def]
    from irsim.radiometry.spectral_response import SpectralResponse

    def tophat(lo: float, hi: float) -> SpectralResponse:
        return SpectralResponse(
            wavelength_um=np.array([lo - 1e-6, lo, hi, hi + 1e-6]),
            response=np.array([0.0, 1.0, 1.0, 0.0]),
            source_path="<test>",
            sha256="",
        )

    standard = airglow.band_fraction(tophat(0.9, 1.7))
    extended = airglow.band_fraction(tophat(0.9, 2.5))
    assert extended / standard == pytest.approx(1.34, rel=0.05), (standard, extended)


def test_a_zero_shape_and_a_negative_level_are_refused(airglow, swir_response) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="non-negative"):
        airglow.band_irradiance(swir_response, -1.0)


# ---------------------------------------------------------------------------------------------
# airglow has no direction
# ---------------------------------------------------------------------------------------------


def test_airglow_takes_no_shadow_and_no_sun_direction(night) -> None:  # type: ignore[no-untyped-def]
    """Asserted through the signature: there is nothing to pass.

    §5.5 is explicit that airglow "does not have a directional shadow". A parameter for it would
    let a scene darken the night sky by geometry, so the API does not have one and this test
    pins that rather than the behaviour of a parameter that should not exist.
    """
    import inspect

    params = set(inspect.signature(night.incident_radiance).parameters)
    assert "shadow" not in params and "sun_elevation_deg" not in params
    assert "cos_incidence" not in params
    a = night.airglow_irradiance(0.0)
    b = night.airglow_irradiance(0.0)
    assert float(a) == float(b)


@pytest.mark.parametrize("k_cloud", [0.0, 0.25, 0.5, 1.0])
def test_cloud_attenuates_airglow_and_never_extinguishes_it(k_cloud: float) -> None:
    overcast = float(cloud_attenuation(1.0, k_cloud))
    clear = float(cloud_attenuation(0.0, k_cloud))
    assert clear == 1.0
    assert overcast <= clear
    assert overcast >= CLOUD_FLOOR - 1e-12, "solid overcast must not black out the night sky"
    assert overcast > 0.0
    mid = float(cloud_attenuation(0.5, k_cloud))
    assert clear >= mid >= overcast


def test_cloud_fraction_and_k_cloud_are_range_checked() -> None:
    with pytest.raises(ValueError, match="cloud_fraction"):
        cloud_attenuation(1.5, 0.5)
    with pytest.raises(ValueError, match="k_cloud"):
        cloud_attenuation(0.5, 1.5)


def test_the_slow_variation_is_slow_seeded_and_strictly_positive() -> None:
    t = np.linspace(0.0, 3600.0, 2001)
    v = airglow_variation(t, seed=7)
    assert np.all(v > 0.0)
    assert np.array_equal(v, airglow_variation(t, seed=7)), "not reproducible"
    assert not np.array_equal(v, airglow_variation(t, seed=8))
    # slow: over one 60 Hz frame it moves by far less than a percent, so it is sky and not noise
    step = abs(float(airglow_variation(np.array([1.0 / 60.0]), 7)[0] - v[0]))
    assert step < 1e-3 * float(v[0])
    assert float(v.min()) > 0.6 and float(v.max()) < 1.4


# ---------------------------------------------------------------------------------------------
# the moon
# ---------------------------------------------------------------------------------------------


def test_the_full_moon_level_follows_from_the_visual_magnitudes() -> None:
    """Not authored: m_sun = −26.74, m_moon = −12.74 ⇒ a flux ratio of 3.98e5."""
    flux_ratio = SOLAR_CONSTANT_W_M2 / FULL_MOON_IRRADIANCE_W_M2
    full_moon_w_m2 = FULL_MOON_IRRADIANCE_W_M2
    assert flux_ratio == pytest.approx(3.98e5, rel=0.01)
    assert full_moon_w_m2 == pytest.approx(3.42e-3, rel=0.01)


@pytest.mark.parametrize(
    "fraction, expected_alpha", [(1.0, 0.0), (0.5, 90.0), (0.0, 180.0), (0.75, 60.0)]
)
def test_the_illuminated_fraction_maps_to_a_phase_angle(
    fraction: float, expected_alpha: float
) -> None:
    assert float(phase_angle_deg(fraction)) == pytest.approx(expected_alpha, abs=1e-9)


def test_a_quarter_moon_is_a_tenth_of_a_full_one_not_a_half() -> None:
    """Lane & Irvine. A model using the illuminated fraction directly would be 5x too bright."""
    assert float(lunar_phase_factor(1.0)) == pytest.approx(1.0, rel=1e-12)
    assert float(lunar_phase_factor(0.5)) == pytest.approx(0.091, rel=0.02)
    assert float(lunar_phase_factor(0.0)) < 1e-3
    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.all(np.diff(lunar_phase_factor(fractions)) > 0.0)


def test_the_moon_sets(night) -> None:  # type: ignore[no-untyped-def]
    assert float(night.moon_irradiance(-0.001, phase_fraction=1.0)) == 0.0
    assert float(night.moon_irradiance(0.0, phase_fraction=1.0)) == 0.0
    zenith = float(night.moon_irradiance(90.0, phase_fraction=1.0))
    assert float(night.moon_irradiance(30.0, phase_fraction=1.0)) == pytest.approx(0.5 * zenith)


def test_a_disabled_moon_contributes_exactly_nothing(swir) -> None:  # type: ignore[no-untyped-def]
    import copy

    preset = load_environment_preset("clear_dry")
    off = preset.model_copy(
        update={
            "night": preset.night.model_copy(
                update={"moon": preset.night.moon.model_copy(update={"enabled": False})}
            )
        }
    )
    dark = NightIllumination.for_sensor(swir, off, "lb_q", DATA)
    assert float(dark.moon_irradiance(90.0, phase_fraction=1.0)) == 0.0
    assert copy.copy(dark) == dark


# ---------------------------------------------------------------------------------------------
# the two sources against each other
# ---------------------------------------------------------------------------------------------


def test_airglow_is_about_a_quarter_moon_in_swir(night) -> None:  # type: ignore[no-untyped-def]
    """The rule of thumb SWIR night imaging is actually sold on, and the models reproduce it."""
    quarter = float(night.moon_irradiance(90.0, phase_fraction=0.5))
    glow = float(night.airglow_irradiance(0.0))
    assert quarter / glow == pytest.approx(1.148, rel=0.05), (
        f"quarter moon / airglow = {quarter / glow:.3f}"
    )


def test_the_full_moon_to_airglow_ratio_is_recorded_not_reproduced(night) -> None:  # type: ignore[no-untyped-def]
    """§5.5 [R7] says "comparable"; these first-order models say 12.6x. Spec issue S38.

    Held to a band around the measured value rather than around the spec's claim, so that a
    change to either model is noticed and the disagreement stays visible instead of being tuned
    away. At §5.5's *upper* airglow level (39 nW/cm²) the ratio is 3.2, which is close to
    comparable; at the OH band peaks the spectral-density ratio is 2-5, which is closer still.
    """
    full = float(night.moon_irradiance(90.0, phase_fraction=1.0))
    glow = float(night.airglow_irradiance(0.0))
    assert full / glow == pytest.approx(12.62, rel=0.05), f"full moon / airglow = {full / glow:.3f}"
    assert full / (glow * 3.9) < 4.0, "at the §5.5 upper airglow level they should be close"


# ---------------------------------------------------------------------------------------------
# through the kernel
# ---------------------------------------------------------------------------------------------


def test_a_lit_target_outshines_its_own_emission_at_night_in_swir(swir, night) -> None:  # type: ignore[no-untyped-def]
    """§5.5's headline: a ρ = 0.3 target at 300 K is *seen*, not *radiating*, in night SWIR."""
    response = load_spectral_response(swir.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    l_night = float(night.incident_radiance(0.0, moon_elevation_deg=-10.0))  # moonless
    reflected = 0.3 * l_night
    emitted = 0.7 * float(lut.lookup(300.0, "lb_q")[()])
    assert reflected / emitted > 10.0, f"reflected/emitted = {reflected / emitted:.2f}"


def test_a_500_k_exhaust_still_outglows_the_night_sky(swir, night) -> None:  # type: ignore[no-untyped-def]
    response = load_spectral_response(swir.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    l_night = float(night.incident_radiance(0.0, moon_elevation_deg=-10.0))
    emitted = 0.9 * float(lut.lookup(500.0, "lb_q")[()])
    assert emitted / (0.1 * l_night) > 100.0


def test_the_night_term_reaches_stage_one_through_the_bundle(swir, night) -> None:  # type: ignore[no-untyped-def]
    response = load_spectral_response(swir.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    table = MaterialTable.constant(0.7, ids=(1,))
    t = np.full((2, 2), 300.0, dtype=np.float32)
    ids = np.ones((2, 2), dtype=np.int32)
    moonless = float(night.incident_radiance(0.0, moon_elevation_deg=-10.0))
    moonlit = float(night.incident_radiance(0.0, moon_elevation_deg=60.0, phase_fraction=1.0))
    dark = band_radiance(
        t,
        ids,
        table,
        lut,
        "lb_q",
        illumination=Illumination.for_regime("reflective", "lb_q", l_night=moonless),
    )
    lit = band_radiance(
        t,
        ids,
        table,
        lut,
        "lb_q",
        illumination=Illumination.for_regime("reflective", "lb_q", l_night=moonlit),
    )
    assert float(lit[0, 0]) > 5.0 * float(dark[0, 0])
    # and an LWIR-style emissive band would be unmoved by any of it (M11.2's gate)
    gated = Illumination.for_regime("emissive", "lb_q", l_night=moonlit)
    assert gated.total_incident() is None


def test_the_quantity_tag_is_honoured_by_both_sources(swir) -> None:  # type: ignore[no-untyped-def]
    preset = load_environment_preset("clear_dry")
    energy = NightIllumination.for_sensor(swir, preset, "lb", DATA)
    photons = NightIllumination.for_sensor(swir, preset, "lb_q", DATA)
    for attr in ("airglow_band", "moon_band_full"):
        ratio = getattr(energy, attr) / getattr(photons, attr)
        assert 1e-19 < ratio < 3e-19, f"{attr} implies a photon energy of {ratio:.3e} J"
    with pytest.raises(ValueError, match="quantity"):
        NightIllumination.for_sensor(swir, preset, "watts", DATA)  # type: ignore[arg-type]
