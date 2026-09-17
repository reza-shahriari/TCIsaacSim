"""A photon camera's noise comes from its electron datasheet, not from a NETD anchor (SC.1).

ADR 0025 made NETD the calibration handle: the physical chain sets the *structure* of the noise
and the datasheet number sets its *magnitude*, by solving for one scene-independent Gaussian. That
is right for a microbolometer, whose datasheet quotes NETD and nothing else usable.

Its M11.6 addendum says it is wrong for a photon FPA, and `irsim.noise.electron.electron_budget`
was written to implement the reversal -- but nothing in `src/` ever called it. Every photon camera
this project has rendered was anchored, which means:

* the **MWIR InSb** rendered at sigma 533.3 e- against the 350 e- read noise its own config
  authors -- **1.52x** -- because the solver was free to invent whatever Gaussian reached 20 mK;
* the **SWIR InGaAs** rendered with dark = 0 against the ~200 e- per integration its own Arrhenius
  block implies -- a term *larger than its 120 e- read noise*, simply absent.

Neither is visible in an image. Both are exactly the kind of error that makes a sensor trade study
come out confidently wrong, which is what this simulator is for.

These tests drive `PipelineConfig.from_sensor` -- the selection site -- rather than
`electron_budget` directly, because the defect was never in the physics. It was that the physics
had no caller.

docs/physics-model.md §9.4, §10.1; ADR 0025 and its M11.6 addendum.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.detector.netd import predict_netd_k
from irsim.detector.params import PhotonParams, fpa_params_from_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.noise.electron import InfeasibleNoiseConfigError, dark_electrons_for
from irsim.pipeline.core import PipelineConfig
from irsim.radiometry.lut_files import load_band_lut_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSORS = REPO / "configs" / "sensors"
LUTS = REPO / "data" / "lut"

PHOTON = ("example_mwir_insb_640.yaml", "example_swir_ingaas_640.yaml", "example_nir_si_1280.yaml")
BOLOMETER = "flir_boson_640_lwir.yaml"


@pytest.fixture(scope="module")
def library():  # type: ignore[no-untyped-def]
    return MaterialLibrary.load()


def _build(library, name: str, **kwargs):  # type: ignore[no-untyped-def]
    sensor = load_sensor_config(SENSORS / name)
    band = sensor.sensor.band.band_id
    lut = load_band_lut_for_config(sensor, LUTS)
    form = "photon" if sensor.sensor.quantity == "lb_q" else "energy"
    table = MaterialTable.from_library(library, band, form=form)
    return sensor, PipelineConfig.from_sensor(sensor, table, lut, **kwargs)


# --- the handle is chosen by what the datasheet offers ----------------------------------------


@pytest.mark.parametrize("name", PHOTON)
def test_a_photon_camera_uses_the_read_noise_its_config_authors(library, name) -> None:
    """sigma_gaussian *is* read_noise_e -- used, not solved for."""
    sensor, config = _build(library, name)
    params = fpa_params_from_config(sensor)
    assert isinstance(params, PhotonParams)
    assert params.read_noise_e is not None

    budget = config.detector.budget
    assert budget.kind == "photon"
    assert budget.sigma_gaussian == pytest.approx(params.read_noise_e)


def test_the_mwir_insb_stops_rendering_at_twice_its_datasheet_read_noise(library) -> None:
    """The headline defect, measured on the shipped config.

    The anchor is free to pick any Gaussian that reaches 20 mK, and what it picks is 533 e-. The
    datasheet says 350. Nothing in the image says which one it was.

    Measured through `from_sensor`, so the cold shield's 3.26e5 background electrons are in the
    shot term the anchor solves against -- as they are in a render. Solved without them the anchor
    lands at 781 e-, which is why this test drives the pipeline rather than `anchor_noise`.
    """
    _, anchored = _build(library, "example_mwir_insb_640.yaml", noise_handle="netd")
    _, electrons = _build(library, "example_mwir_insb_640.yaml")

    assert anchored.detector.budget.sigma_gaussian == pytest.approx(533.27, rel=1e-3)
    assert electrons.detector.budget.sigma_gaussian == pytest.approx(350.0)
    ratio = anchored.detector.budget.sigma_gaussian / electrons.detector.budget.sigma_gaussian
    assert ratio == pytest.approx(1.5236, rel=1e-3)


def test_the_swir_ingaas_gets_the_dark_current_its_own_config_implies(library) -> None:
    """~200 e- per integration, larger than the 120 e- read noise, and previously just absent.

    Dark current is both an offset and a Poisson term, so leaving it out understates the noise
    *and* moves the pedestal the NUC is supposed to remove.
    """
    sensor, config = _build(library, "example_swir_ingaas_640.yaml")
    params = fpa_params_from_config(sensor)
    assert isinstance(params, PhotonParams)

    expected = dark_electrons_for(params)
    assert expected == pytest.approx(199.7, rel=1e-3), "the shipped config moved; update the bound"
    assert config.detector.budget.dark_electrons == pytest.approx(expected)
    assert expected > params.read_noise_e, "the term left out was the larger of the two"

    _, anchored = _build(library, "example_swir_ingaas_640.yaml", noise_handle="netd")
    assert anchored.detector.budget.dark_electrons == 0.0


def test_a_bolometer_keeps_the_netd_anchor(library) -> None:
    """ADR 0025 stands where it was right: NETD is a bolometer's only usable handle."""
    sensor, config = _build(library, BOLOMETER)
    budget = config.detector.budget
    assert budget.kind == "bolometer"
    assert budget.dark_electrons == 0.0

    # Anchored, i.e. it reproduces the datasheet by construction.
    lut = load_band_lut_for_config(sensor, LUTS)
    predicted = predict_netd_k(300.0, sensor.sensor, lut, budget)
    assert predicted * 1e3 == pytest.approx(sensor.sensor.noise.netd_mk_at_300k, rel=1e-6)


def test_a_bolometer_cannot_be_forced_onto_an_electron_budget(library) -> None:
    with pytest.raises(ValueError, match="photon FPA"):
        _build(library, BOLOMETER, noise_handle="electrons")


# --- NETD stops being the handle and becomes the cross-check ----------------------------------


def test_the_mwir_now_beats_its_own_datasheet_instead_of_reproducing_it(library) -> None:
    """The point of the reversal: NETD is now a prediction that can disagree.

    Anchored, the predicted NETD equals the claim to machine precision -- it was solved for, so it
    carries no information. Built from electrons it comes out at 19.5 mK against a 20 mK claim:
    a number the datasheet could have contradicted, and did not.
    """
    sensor, anchored = _build(library, "example_mwir_insb_640.yaml", noise_handle="netd")
    _, electrons = _build(library, "example_mwir_insb_640.yaml")
    lut = load_band_lut_for_config(sensor, LUTS)
    claim_k = sensor.sensor.noise.netd_mk_at_300k * 1e-3

    from_anchor = predict_netd_k(300.0, sensor.sensor, lut, anchored.detector.budget)
    assert from_anchor == pytest.approx(claim_k, rel=1e-9)

    predicted = predict_netd_k(300.0, sensor.sensor, lut, electrons.detector.budget)
    assert predicted * 1e3 == pytest.approx(19.471, rel=1e-3)
    assert predicted < claim_k, "the camera as specified is better than it claims"


def test_a_camera_that_cannot_reach_its_own_claim_is_refused(library) -> None:
    """The cross-check has to be able to fail, or it is decoration.

    The InSb is nearly shot-limited at its 90 %-efficient cold shield, so a datasheet claiming
    10 mK describes a camera that cannot exist at this aperture and integration time.
    Poisson terms are never rescaled (ADR 0025), so there is nothing to tune and it raises.
    """
    sensor = load_sensor_config(SENSORS / "example_mwir_insb_640.yaml")
    dumped = sensor.model_dump(mode="json")
    dumped["sensor"]["noise"]["netd_mk_at_300k"] = 10.0
    impossible = type(sensor).model_validate(dumped)

    lut = load_band_lut_for_config(impossible, LUTS)
    table = MaterialTable.from_library(library, impossible.sensor.band.band_id, form="photon")
    with pytest.raises(InfeasibleNoiseConfigError, match="worse than the datasheet"):
        PipelineConfig.from_sensor(impossible, table, lut)


def test_the_reflective_bands_are_not_checked_against_a_vacuous_netd(library) -> None:
    """§9.4's 300 K definition says nothing in a band a 300 K scene does not emit in.

    The SWIR's authored NETD is 9.76e5 mK and the NIR's 2.4e16 mK -- derived placeholders, not
    claims. Checking against them would be theatre; the guard is that they still *build*.
    """
    for name in ("example_swir_ingaas_640.yaml", "example_nir_si_1280.yaml"):
        sensor, config = _build(library, name)
        assert sensor.sensor.band.regime == "reflective"
        assert config.detector.budget.sigma_gaussian > 0.0


# --- and it reaches the rendered frame --------------------------------------------------------


def test_the_change_reaches_the_pixels_and_not_only_the_budget(library) -> None:
    """A budget nothing reads is a comment. This measures the detector's own sigma.

    Photon shot noise dominates at a bright signal, so the two handles are compared at a *dark*
    one, where the read term is what is left -- which is the regime a sensor trade study cares
    about and the one the 2.23x error would have distorted.
    """
    _, anchored = _build(library, "example_mwir_insb_640.yaml", noise_handle="netd")
    _, electrons = _build(library, "example_mwir_insb_640.yaml")

    faint = np.full((8, 8), 1.0e6, dtype=np.float64)  # photons/s on the pixel
    sigma_anchored = anchored.detector.sigma_electrons(faint)
    sigma_electrons = electrons.detector.sigma_electrons(faint)

    assert np.all(sigma_anchored > sigma_electrons)
    # sqrt(N_e + dark + bg + read^2), with the read term the only difference between the two.
    mean_e = electrons.detector.electrons(faint)
    expected = np.sqrt(mean_e + 350.0**2)
    np.testing.assert_allclose(sigma_electrons, expected, rtol=1e-12)
