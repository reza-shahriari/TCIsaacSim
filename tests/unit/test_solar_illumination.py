"""M11.3 — the reflected solar term, and the fact that decides which band you are in.

The spectral crossover between reflected sunlight and self-emission sits at **4.1 µm** for a
300 K, ρ = 0.3 surface in full sun. That single number is why SWIR is a "reflective" band and
LWIR an "emissive" one, why MWIR is neither, and why §12.1 has a `regime` column at all. It is
also the one thing in this step that is not an artefact of the modelled spectrum: it moves by a
few tens of nanometres if the solar shape changes by a few percent, because both sides are steep.

The solar files are **modelled, not measured** (ADR 0064) and their integrals are normalisations,
so the tests that matter are the ones that are not: the opacity of the 1.38 µm and 1.87 µm water
bands, which is why a SWIR camera is blind there and which no part of the fit constrained; the
exact zeros at the horizon; and the Lambertian identity.

docs/physics-model.md §5.4, §5.2, §7.1; ADR 0063, ADR 0064
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.extinction import airmass, airmass_kasten_young
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.loader import load_sensor_config
from irsim.materials.table import MaterialTable
from irsim.pipeline.illumination import Illumination
from irsim.pipeline.radiance import band_radiance
from irsim.pipeline.solar import SolarIllumination, beam_transmittance
from irsim.radiometry.constants import C1L, C2, SOLAR_CONSTANT_W_M2
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.solar import (
    AM15_DIRECT_FILE,
    TOA_FILE,
    TRUSTED_MIN_UM,
    load_solar_spectrum,
)
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COARSE_N = 161

#: G-173's direct + circumsolar reference total at AM1.5.
AM15_DIRECT_W_M2 = 900.1


@pytest.fixture(scope="module")
def toa():  # type: ignore[no-untyped-def]
    return load_solar_spectrum(DATA / TOA_FILE)


@pytest.fixture(scope="module")
def ground():  # type: ignore[no-untyped-def]
    return load_solar_spectrum(DATA / AM15_DIRECT_FILE)


@pytest.fixture(scope="module")
def swir():  # type: ignore[no-untyped-def]
    return load_sensor_config(REPO / "configs/sensors/example_swir_ingaas_640.yaml", DATA)


@pytest.fixture(scope="module")
def lwir():  # type: ignore[no-untyped-def]
    return load_sensor_config(REPO / "configs/sensors/flir_boson_640_lwir.yaml", DATA)


# ---------------------------------------------------------------------------------------------
# the files
# ---------------------------------------------------------------------------------------------


def test_the_toa_spectrum_carries_the_solar_constant(toa) -> None:  # type: ignore[no-untyped-def]
    """This is the file's normalisation, not an independent check -- and it is worth asserting
    anyway, because a regenerated file that misses it is a file with a broken integral."""
    assert toa.total_w_m2() == pytest.approx(SOLAR_CONSTANT_W_M2, rel=0.01)


def test_the_ground_spectrum_carries_the_g173_direct_total(ground) -> None:  # type: ignore[no-untyped-def]
    assert ground.total_w_m2() == pytest.approx(AM15_DIRECT_W_M2, rel=0.005)


def test_the_toa_spectrum_peaks_where_wien_says_it_should(toa) -> None:  # type: ignore[no-untyped-def]
    """2897.77 / 5778 = 0.5015 µm. Catches a file written on the wrong wavelength unit."""
    peak_um = float(toa.wavelength_um[int(np.argmax(toa.irradiance_w_m2_um))])
    assert peak_um == pytest.approx(2897.771955 / 5778.0, rel=0.01)


@pytest.mark.parametrize("lambda_um", [1.38, 1.87])
def test_the_strong_water_bands_are_opaque_at_the_ground(toa, ground, lambda_um: float) -> None:  # type: ignore[no-untyped-def]
    """Not fitted, and the reason a SWIR camera is blind at 1.38 µm.

    The common water strength was calibrated against the *total*; nothing in that fit says where
    the absorption has to sit. If the band centres were wrong this would pass 1.0.
    """
    e_toa = float(np.interp(lambda_um, toa.wavelength_um, toa.irradiance_w_m2_um))
    e_gnd = float(np.interp(lambda_um, ground.wavelength_um, ground.irradiance_w_m2_um))
    assert e_gnd / e_toa < 0.05, f"tau({lambda_um} um) = {e_gnd / e_toa:.3f}"


@pytest.mark.parametrize("lambda_um", [1.06, 1.55, 2.2])
def test_the_swir_imaging_windows_stay_open(toa, ground, lambda_um: float) -> None:  # type: ignore[no-untyped-def]
    """The other half of the same statement: 1.06 and 1.55 µm are where SWIR cameras work."""
    e_toa = float(np.interp(lambda_um, toa.wavelength_um, toa.irradiance_w_m2_um))
    e_gnd = float(np.interp(lambda_um, ground.wavelength_um, ground.irradiance_w_m2_um))
    assert e_gnd / e_toa > 0.85, f"tau({lambda_um} um) = {e_gnd / e_toa:.3f}"


def test_the_ground_spectrum_never_exceeds_the_extraterrestrial_one(toa, ground) -> None:  # type: ignore[no-untyped-def]
    e_toa = np.interp(ground.wavelength_um, toa.wavelength_um, toa.irradiance_w_m2_um)
    assert np.all(ground.irradiance_w_m2_um <= e_toa * (1.0 + 1e-9))


def test_a_response_below_the_trusted_range_is_refused(toa) -> None:  # type: ignore[no-untyped-def]
    """The model's ultraviolet is wrong by tens of percent, so it refuses rather than integrates."""
    from irsim.radiometry.spectral_response import SpectralResponse

    visible = SpectralResponse(
        wavelength_um=np.array([0.40, 0.50, 0.70]),
        response=np.array([0.0, 1.0, 0.0]),
        source_path="<test>",
        sha256="",
    )
    with pytest.raises(ValueError, match="trusted"):
        toa.band_irradiance(visible)
    assert TRUSTED_MIN_UM == 0.70


# ---------------------------------------------------------------------------------------------
# band integrals
# ---------------------------------------------------------------------------------------------


def test_the_photon_form_is_a_band_average_not_a_single_wavelength(toa, swir) -> None:  # type: ignore[no-untyped-def]
    """Same trap as M11.2's: hc/λ̄ outside the integral is a several-percent error here."""
    from irsim.radiometry.constants import C_LIGHT, H_PLANCK

    response = load_spectral_response(swir.sensor.band.spectral_response)
    e_energy = toa.band_irradiance(response)
    e_photons = toa.band_photon_irradiance(response)
    implied = e_energy / e_photons
    single = H_PLANCK * C_LIGHT / (response.mean_wavelength_um() * 1e-6)
    assert 1e-19 < implied < 3e-19
    assert abs(implied / single - 1.0) > 0.01, (implied, single)


def test_the_two_quantities_are_dispatched_by_the_lut_tag(toa, swir) -> None:  # type: ignore[no-untyped-def]
    response = load_spectral_response(swir.sensor.band.spectral_response)
    assert toa.band(response, "lb") == toa.band_irradiance(response)
    assert toa.band(response, "lb_q") == toa.band_photon_irradiance(response)
    with pytest.raises(ValueError, match="quantity"):
        toa.band(response, "watts")


# ---------------------------------------------------------------------------------------------
# the beam
# ---------------------------------------------------------------------------------------------


def test_the_beam_is_exactly_zero_below_the_horizon() -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    for elevation in (0.0, -0.001, -10.0, -90.0):
        assert float(beam_transmittance(preset, "swir", elevation)) == 0.0


def test_transmittance_falls_monotonically_as_the_sun_sets() -> None:
    preset = load_atmosphere_preset("us_standard_clear")
    elevations = np.array([90.0, 60.0, 30.0, 15.0, 5.0, 1.0])
    tau = beam_transmittance(preset, "swir", elevations)
    assert np.all(np.diff(tau) < 0.0)
    assert tau[0] == pytest.approx(preset.solar.zenith_transmittance["swir"], rel=1e-3)


def test_kasten_young_stays_finite_where_the_plane_parallel_airmass_gives_up() -> None:
    """ADR 0051 refuses beyond 85° because sec θ diverges. The beam needs an answer at 09:00."""
    assert airmass_kasten_young(math.radians(90.0)) == pytest.approx(37.9, rel=0.02)
    assert airmass_kasten_young(math.radians(60.0)) == pytest.approx(2.0, rel=0.01)
    # at zenith the empirical fit gives 0.99972, not 1 -- a known 2.8e-4 artefact, 0.03 % in tau
    assert airmass_kasten_young(0.0) == pytest.approx(1.0, rel=5e-4)
    with pytest.raises(ValueError):
        airmass(math.radians(89.0))


def test_the_reflected_term_is_exactly_zero_in_all_three_dark_cases(toa, swir) -> None:  # type: ignore[no-untyped-def]
    sun = SolarIllumination.for_sensor(swir, "lb_q", DATA)
    assert sun.e_band > 0.0
    assert float(sun.incident_radiance(1.0, elevation_deg=-0.001)) == 0.0
    assert float(sun.incident_radiance(-0.2, elevation_deg=45.0)) == 0.0
    assert float(sun.incident_radiance(1.0, elevation_deg=45.0, shadow=0.0)) == 0.0
    assert float(sun.incident_radiance(0.0, elevation_deg=45.0)) == 0.0


def test_the_lambertian_identity_holds_to_machine_precision(toa, swir) -> None:  # type: ignore[no-untyped-def]
    """ρ = 1, τ = 1, normal incidence, unshadowed: the surface leaves exactly E_B/π."""
    sun = SolarIllumination.for_sensor(swir, "lb", DATA)
    leaving = float(sun.incident_radiance(1.0, elevation_deg=90.0, tau_sun=1.0, shadow=1.0))
    assert leaving == pytest.approx(sun.e_band / math.pi, rel=1e-12)


def test_the_beam_scales_with_the_cosine_and_the_shadow(swir) -> None:  # type: ignore[no-untyped-def]
    sun = SolarIllumination.for_sensor(swir, "lb", DATA)
    full = float(sun.incident_radiance(1.0, elevation_deg=90.0))
    assert float(sun.incident_radiance(0.5, elevation_deg=90.0)) == pytest.approx(0.5 * full)
    assert float(sun.incident_radiance(1.0, elevation_deg=90.0, shadow=0.25)) == pytest.approx(
        0.25 * full
    )
    cos_plane = np.array([[1.0, 0.5], [0.0, -1.0]])
    out = sun.incident_radiance(cos_plane, elevation_deg=90.0)
    assert out.shape == (2, 2)
    assert out[1, 1] == 0.0 and out[1, 0] == 0.0


def test_out_of_range_inputs_are_refused(swir) -> None:  # type: ignore[no-untyped-def]
    sun = SolarIllumination.for_sensor(swir, "lb", DATA)
    with pytest.raises(ValueError, match="cos_incidence"):
        sun.incident_radiance(1.5, elevation_deg=45.0)
    with pytest.raises(ValueError, match="shadow"):
        sun.incident_radiance(1.0, elevation_deg=45.0, shadow=1.5)
    with pytest.raises(ValueError, match="tau_sun"):
        sun.incident_radiance(1.0, elevation_deg=45.0, tau_sun=-0.1)


# ---------------------------------------------------------------------------------------------
# the two-regime known answer
# ---------------------------------------------------------------------------------------------


def _reflected_over_emitted(config, quantity: str, rho: float = 0.3) -> float:  # type: ignore[no-untyped-def]
    preset = load_atmosphere_preset("us_standard_clear")
    response = load_spectral_response(config.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    sun = SolarIllumination.for_sensor(config, quantity, DATA)  # type: ignore[arg-type]
    tau = float(beam_transmittance(preset, config.sensor.band.band_id, 90.0))
    l_sun = float(sun.incident_radiance(1.0, elevation_deg=90.0, tau_sun=tau))
    emitted = (1.0 - rho) * float(lut.lookup(300.0, quantity)[()])  # type: ignore[arg-type]
    return rho * l_sun / emitted


def test_a_300_k_surface_in_full_sun_is_reflective_in_swir_and_emissive_in_lwir(swir, lwir) -> None:  # type: ignore[no-untyped-def]
    swir_ratio = _reflected_over_emitted(swir, "lb_q")
    lwir_ratio = _reflected_over_emitted(lwir, "lb")
    assert swir_ratio > 1e3, f"SWIR reflected/emitted = {swir_ratio:.3e}"
    assert lwir_ratio < 0.01, f"LWIR reflected/emitted = {lwir_ratio:.4f}"


def test_the_spectral_crossover_lands_in_the_mwir_band(toa) -> None:  # type: ignore[no-untyped-def]
    """Where ρ E(λ) τ cos θ / π equals ε B_λ(300 K): the reason §12.1 has three regimes.

    Computed spectrally, so it does not depend on any band being configured. The answer -- 4.1 µm
    -- is what puts MWIR on both sides of the line at once and is why the MWIR config is the only
    one marked `mixed`.
    """
    rho, eps, tau, cos_s = 0.3, 0.7, 0.8, 1.0
    lam = toa.wavelength_um
    reflected = rho * toa.irradiance_w_m2_um * tau * cos_s / math.pi
    emitted = eps * C1L / lam**5 / np.expm1(C2 / (lam * 300.0))
    window = (lam > 2.0) & (lam < 8.0)
    crossover = float(lam[window][int(np.argmin(np.abs(reflected[window] - emitted[window])))])
    assert 3.0 < crossover < 5.0, f"crossover at {crossover:.3f} um"
    # and it really is a crossover: reflection dominates well below it, emission well above
    below = np.argmin(np.abs(lam - 2.0))
    above = np.argmin(np.abs(lam - 10.0))
    assert reflected[below] / emitted[below] > 1e3
    assert reflected[above] / emitted[above] < 1e-2


def test_the_spectral_model_and_the_preset_agree_on_swir_band_transmittance(
    toa, ground, swir
) -> None:  # type: ignore[no-untyped-def]
    """A cross-check between two independent estimates, kept because they *disagree* by 11 %.

    The preset's `solar.zenith_transmittance['swir'] = 0.88` is an ESTIMATED §7.2 table midpoint.
    Band-integrating this step's own spectra gives tau(AM1.5) = E_ground,B / E_toa,B, hence a
    zenith value of that raised to 1/1.5 -- and it comes out near 0.79, because the 1.38 µm water
    band sits **inside** 0.9-1.7 µm and a broadband midpoint does not know that. Neither number is
    measured, so this is a flag rather than a correction (ADR 0064); the test holds them to 20 %
    so that a future change to either is noticed.
    """
    preset = load_atmosphere_preset("us_standard_clear")
    response = load_spectral_response(swir.sensor.band.spectral_response)
    tau_am15 = ground.band_irradiance(response) / toa.band_irradiance(response)
    implied_zenith = tau_am15 ** (1.0 / 1.5)
    assert implied_zenith == pytest.approx(preset.solar.zenith_transmittance["swir"], rel=0.20), (
        f"spectral model says {implied_zenith:.3f}, preset says 0.88"
    )
    assert implied_zenith < preset.solar.zenith_transmittance["swir"]


# ---------------------------------------------------------------------------------------------
# through the kernel
# ---------------------------------------------------------------------------------------------


def test_the_solar_term_reaches_stage_one_through_the_bundle(swir) -> None:  # type: ignore[no-untyped-def]
    response = load_spectral_response(swir.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    sun = SolarIllumination.for_sensor(swir, "lb_q", DATA)
    table = MaterialTable.constant(0.7, ids=(1,))
    t = np.full((2, 2), 300.0, dtype=np.float32)
    ids = np.ones((2, 2), dtype=np.int32)
    shadow = np.array([[1.0, 1.0], [0.0, 0.0]])
    l_sun = sun.incident_radiance(np.full((2, 2), 0.8), elevation_deg=40.0, shadow=shadow)

    lit = band_radiance(
        t,
        ids,
        table,
        lut,
        "lb_q",
        illumination=Illumination.for_regime("reflective", "lb_q", l_sun=l_sun),
    )
    assert lit[0, 0] > 10.0 * lit[1, 0], "the shadowed half is not darker"
    emission_only = band_radiance(t, ids, table, lut, "lb_q")
    assert float(lit[1, 0]) == pytest.approx(float(emission_only[0, 0]), rel=1e-6)
