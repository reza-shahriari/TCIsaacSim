"""M11.2 — one shading core, three incident terms, and a gate that is data rather than code.

§5.2 puts the emissive/reflective choice behind a compile-time flag. ADR 0063 makes it a runtime
function of ``band.regime``, which is the only version compatible with "bands are data": a
compile-time flag cannot be set by a YAML file that did not exist when the binary was built.

The tests that matter here are the ones about what the gate must *not* do:

* it must not cull self-emission, or a jet exhaust goes dark in night SWIR;
* it must not leave a trace when it drops a term, or an emissive frame with a solar plane
  attached is *nearly* the same as one without, and nobody notices the difference is real;
* it must not let an energy-unit source into a photon-unit kernel, a mistake that is ~1e19 wide
  and completely invisible after AGC.

docs/physics-model.md §5.2, §5.3, §5.4, §5.5, §12.1; ADR 0063
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson

from irsim.config.bands import enabled_illumination_terms
from irsim.materials.table import MaterialTable
from irsim.pipeline.illumination import Illumination, illumination_from_planes
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.constants import C1L, C2, C_LIGHT, H_PLANCK
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

SWIR_RESPONSE = "spectra/responses/ingaas.csv"
LWIR_RESPONSE = "spectra/responses/boson_vox.csv"
COARSE_N = 161

#: §5.5's airglow level, 10 nW/cm² in band = 1.0e-4 W/m² of downwelling irradiance. As an
#: isotropic incident radiance that is E/π.
AIRGLOW_E_W_M2 = 1.0e-4
AIRGLOW_L = AIRGLOW_E_W_M2 / np.pi


@pytest.fixture(scope="module")
def swir_response():  # type: ignore[no-untyped-def]
    from irsim.config.loader import resolve_data_dir

    return load_spectral_response(resolve_data_dir("data") / SWIR_RESPONSE)


@pytest.fixture(scope="module")
def swir_lut(swir_response):  # type: ignore[no-untyped-def]
    return BandLUT.build(swir_response, n=COARSE_N)


def _airglow_photon_radiance(response) -> float:
    """§5.5's 10 nW/cm² in band as an isotropic incident *photon* radiance.

    Converted at the band's mean wavelength, which is a stand-in: airglow is a line spectrum
    (OH-Meinel) and M11.4 gives it its real shape, normalised so the in-band integral is this
    same level. What is being tested here is the regime gate, not the airglow model, and the
    conversion only has to be the right order of magnitude for that.
    """
    photon_j = H_PLANCK * C_LIGHT / (response.mean_wavelength_um() * 1e-6)
    return AIRGLOW_L / photon_j


@pytest.fixture(scope="module")
def unit_emissivity() -> MaterialTable:
    return MaterialTable.constant(1.0, ids=(1,))


# ---------------------------------------------------------------------------------------------
# the units tag
# ---------------------------------------------------------------------------------------------


def test_the_bundle_carries_its_quantity_and_the_kernel_checks_it(
    swir_lut: BandLUT, unit_emissivity: MaterialTable
) -> None:
    t = np.full((2, 2), 300.0, dtype=np.float32)
    ids = np.ones((2, 2), dtype=np.int32)
    energy_bundle = Illumination.for_regime("reflective", "lb", l_sun=1.0)
    with pytest.raises(ValueError, match="1e19"):
        band_radiance(t, ids, unit_emissivity, swir_lut, "lb_q", illumination=energy_bundle)
    # and the matching one goes through
    photon_bundle = Illumination.for_regime("reflective", "lb_q", l_sun=1.0)
    band_radiance(t, ids, unit_emissivity, swir_lut, "lb_q", illumination=photon_bundle)


def test_an_unknown_quantity_is_refused() -> None:
    with pytest.raises(ValueError, match="quantity"):
        Illumination(quantity="watts")  # type: ignore[arg-type]


def test_l_env_and_illumination_together_are_refused(
    swir_lut: BandLUT, unit_emissivity: MaterialTable
) -> None:
    t = np.full((2, 2), 300.0, dtype=np.float32)
    ids = np.ones((2, 2), dtype=np.int32)
    bundle = Illumination.for_regime("reflective", "lb_q", l_env=1.0)
    with pytest.raises(ValueError, match="not both"):
        band_radiance(
            t, ids, unit_emissivity, swir_lut, "lb_q", l_env=np.ones((2, 2)), illumination=bundle
        )


def test_float16_and_negative_incident_radiance_are_refused() -> None:
    with pytest.raises(TypeError, match="float16"):
        Illumination.for_regime("reflective", "lb_q", l_sun=np.zeros((2, 2), dtype=np.float16))
    with pytest.raises(ValueError, match="non-negative"):
        Illumination.for_regime("reflective", "lb_q", l_sun=-1.0)


def test_terms_that_disagree_on_shape_are_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        Illumination.for_regime("reflective", "lb_q", l_env=np.ones((2, 2)), l_sun=np.ones((3, 3)))


# ---------------------------------------------------------------------------------------------
# the regime gate
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regime, kept_sun, kept_night",
    [("emissive", False, False), ("reflective", True, True), ("mixed", True, True)],
)
def test_the_gate_follows_the_registry_not_the_band(
    regime: str, kept_sun: bool, kept_night: bool
) -> None:
    bundle = Illumination.for_regime(regime, "lb", l_env=1.0, l_sun=2.0, l_night=3.0)  # type: ignore[arg-type]
    assert (bundle.l_sun is not None) is kept_sun
    assert (bundle.l_night is not None) is kept_night
    assert bundle.terms == enabled_illumination_terms(regime)  # type: ignore[arg-type]
    # The environment is never gated: it is what every surface in every band sees.
    assert bundle.l_env is not None


def test_a_dropped_term_is_absent_rather_than_zero() -> None:
    """The distinction that makes the next test a bitwise one instead of a tolerance."""
    bundle = Illumination.for_regime("emissive", "lb", l_sun=7.0, l_night=9.0)
    assert bundle.l_sun is None and bundle.l_night is None
    assert bundle.total_incident() is None
    assert bundle.dropped_terms == frozenset({"solar", "night"})


def test_an_emissive_band_is_bitwise_unchanged_by_illumination(
    unit_emissivity: MaterialTable,
) -> None:
    """A solar plane attached to an LWIR camera must change *nothing*, not merely little."""
    from irsim.config.loader import resolve_data_dir

    lut = BandLUT.build(
        load_spectral_response(resolve_data_dir("data") / LWIR_RESPONSE), n=COARSE_N
    )
    t = np.linspace(280.0, 320.0, 16, dtype=np.float32).reshape(4, 4)
    ids = np.ones((4, 4), dtype=np.int32)
    table = MaterialTable.constant(0.9, ids=(1,))
    env = np.full((4, 4), float(lut.lookup(250.0)[()]), dtype=np.float32)

    dark = Illumination.for_regime("emissive", "lb", l_env=env)
    lit = Illumination.for_regime(
        "emissive", "lb", l_env=env, l_sun=np.full((4, 4), 1e4), l_night=np.full((4, 4), 5.0)
    )
    a = band_radiance(t, ids, table, lut, "lb", illumination=dark)
    b = band_radiance(t, ids, table, lut, "lb", illumination=lit)
    assert np.array_equal(a, b), "an emissive band saw the sun"

    # and the same scene in a reflective band does move, so the test above is not vacuous
    mixed_dark = Illumination.for_regime("mixed", "lb", l_env=env)
    mixed_lit = Illumination.for_regime("mixed", "lb", l_env=env, l_sun=np.full((4, 4), 1e4))
    assert not np.array_equal(
        band_radiance(t, ids, table, lut, "lb", illumination=mixed_dark),
        band_radiance(t, ids, table, lut, "lb", illumination=mixed_lit),
    )


def test_an_unilluminated_bundle_takes_the_emission_only_path_exactly(
    swir_lut: BandLUT, unit_emissivity: MaterialTable
) -> None:
    t = np.full((3, 3), 400.0, dtype=np.float32)
    ids = np.ones((3, 3), dtype=np.int32)
    empty = Illumination.for_regime("reflective", "lb_q")
    assert empty.total_incident() is None
    through_bundle = band_radiance(t, ids, unit_emissivity, swir_lut, "lb_q", illumination=empty)
    emission_only = band_radiance(t, ids, unit_emissivity, swir_lut, "lb_q")
    assert np.array_equal(through_bundle, emission_only)


def test_the_three_terms_sum_before_reflection_not_after() -> None:
    """ρ multiplies the *sum*: one reflectance, three sources, the Lambertian handshake of §5.4."""
    bundle = Illumination.for_regime("mixed", "lb", l_env=2.0, l_sun=3.0, l_night=5.0)
    assert float(np.asarray(bundle.total_incident())) == pytest.approx(10.0, rel=0, abs=0)


# ---------------------------------------------------------------------------------------------
# emission is never culled
# ---------------------------------------------------------------------------------------------


def test_a_500_k_exhaust_outglows_airglow_in_a_reflective_band(
    swir_response, swir_lut: BandLUT
) -> None:
    """§12.1's "reflective" is a claim about 300 K scenes, not a licence to drop ε L_B.

    A 500 K exhaust nozzle at night, in SWIR, with no sun: the only two things in the pixel are
    its own emission and the airglow it reflects. If the regime switch culled emission the pixel
    would be the airglow alone and the exhaust would be invisible -- which is the failure this
    test exists to catch.
    """
    table = MaterialTable.constant(0.9, ids=(1,))
    ids = np.ones((1, 1), dtype=np.int32)
    lb_q_500 = float(swir_lut.lookup(500.0, "lb_q")[()])
    airglow_q = _airglow_photon_radiance(swir_response)

    night = Illumination.for_regime("reflective", "lb_q", l_night=airglow_q)
    pixel = float(
        band_radiance(
            np.full((1, 1), 500.0, dtype=np.float32),
            ids,
            table,
            swir_lut,
            "lb_q",
            illumination=night,
        )[0, 0]
    )
    emission = 0.9 * lb_q_500
    reflected = 0.1 * airglow_q
    assert pixel == pytest.approx(emission + reflected, rel=1e-6)
    assert emission / reflected > 100.0, f"emission/airglow = {emission / reflected:.1f}"


def test_a_300_k_surface_in_the_same_band_is_the_other_way_round(
    swir_response, swir_lut: BandLUT
) -> None:
    """The converse, so the test above is about temperature and not about the band.

    Same band, same airglow, 300 K instead of 500 K: now the reflected term wins, by the >10x
    M11.4 asks for (measured 15.5x for a rho = 0.1 surface). Between the two tests lies the
    crossover this simulator has to get right without a band switch anywhere.
    """
    airglow_q = _airglow_photon_radiance(swir_response)
    emission = 0.9 * float(swir_lut.lookup(300.0, "lb_q")[()])
    reflected = 0.1 * airglow_q
    assert reflected / emission > 10.0, f"airglow/emission = {reflected / emission:.1f}"


# ---------------------------------------------------------------------------------------------
# the photon path is the energy path divided by a band-averaged photon energy
# ---------------------------------------------------------------------------------------------


def _mean_photon_energy_j(response, temperature_k: float) -> float:
    """∫R B_λ dλ / ∫R B_λ λ/(hc) dλ, quadratured here from Planck's law directly.

    Independent of ``irsim.radiometry.band_integration``: this is the whole point -- the photon
    table must be the energy table divided by a band *average*, not by hc/λ at one wavelength.
    """
    lam_um = response.wavelength_um
    r = response.response
    planck = C1L / lam_um**5 / np.expm1(C2 / (lam_um * temperature_k))
    energy = simpson(r * planck, x=lam_um)
    photons = simpson(r * planck * (lam_um * 1e-6) / (H_PLANCK * C_LIGHT), x=lam_um)
    return float(energy / photons)


@pytest.mark.parametrize("temperature_k", [300.0, 600.0])
def test_the_energy_and_photon_kernels_differ_by_the_band_mean_photon_energy(
    swir_response, swir_lut: BandLUT, unit_emissivity: MaterialTable, temperature_k: float
) -> None:
    t = np.full((2, 2), temperature_k, dtype=np.float32)
    ids = np.ones((2, 2), dtype=np.int32)
    energy = band_radiance(t, ids, unit_emissivity, swir_lut, "lb")
    photons = band_radiance(t, ids, unit_emissivity, swir_lut, "lb_q")
    measured = float(energy[0, 0]) / float(photons[0, 0])
    assert measured == pytest.approx(_mean_photon_energy_j(swir_response, temperature_k), rel=1e-6)


def test_the_band_mean_photon_energy_is_not_the_centre_wavelength_one(swir_response) -> None:
    """If it were, a single-λ shortcut would pass the test above and be wrong by a few percent."""
    centre_um = 0.5 * (0.9 + 1.7)
    single = H_PLANCK * C_LIGHT / (centre_um * 1e-6)
    mean = _mean_photon_energy_j(swir_response, 600.0)
    assert abs(mean / single - 1.0) > 0.02, (mean, single)


# ---------------------------------------------------------------------------------------------
# the plane-dict path
# ---------------------------------------------------------------------------------------------


def test_the_source_planes_are_optional_exactly_like_motion_px() -> None:
    bundle = illumination_from_planes("reflective", "lb_q", {}, l_env=None)
    assert bundle.total_incident() is None
    with_sun = illumination_from_planes(
        "reflective", "lb_q", {"l_sun": np.full((2, 2), 4.0)}, l_env=np.full((2, 2), 1.0)
    )
    assert float(np.asarray(with_sun.total_incident())[0, 0]) == pytest.approx(5.0)


def test_the_planes_are_gated_by_the_regime_too() -> None:
    bundle = illumination_from_planes(
        "emissive", "lb", {"l_sun": np.full((2, 2), 4.0), "l_night": np.full((2, 2), 1.0)}
    )
    assert bundle.total_incident() is None
