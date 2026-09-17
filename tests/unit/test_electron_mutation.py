"""Perturbing ``read_noise_e`` moves the rendered noise, which it could not before (SC.2).

`SC.1` reversed which of NETD and the electron datasheet is derived for a photon FPA. This is the
mutation test that says the reversal reached the pixels rather than only the budget object.

The pre-`SC.1` behaviour is the sharper half of it and is kept here as a negative control. Under
the ADR 0025 anchor the solver picks whatever Gaussian reproduces the datasheet NETD, so
``read_noise_e`` is an input to *nothing*: change it by 10 %, by 50 %, by any amount, and the
rendered sigma is bit-identical. A configuration field that cannot move the output is not a
parameter, it is a comment -- and it had been one in every photon render this project has made.

The measured sensitivity is not 10 % out for 10 % in, and that is physics rather than a weak
test. sigma_total is sqrt(N_e + N_dark + N_bg + sigma_read^2), so how much a read-noise change
shows depends entirely on where the camera is sitting on that curve: the cooled InSb at a bright
scene carries 4.9e5 signal electrons and 3.3e5 from its own cold shield, so the read term is a
small part of the quadrature and 10 % in gives 2.1 % out. Take the same camera to a faint scene
and the same change gives the full 10 %. Both are checked.

docs/physics-model.md §9.4, §10.1; ADR 0025 and its M11.6 addendum.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.detector.params import PhotonParams, fpa_params_from_config
from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig
from irsim.radiometry.lut_files import load_band_lut_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSORS = REPO / "configs" / "sensors"
PHOTON = ("example_mwir_insb_640.yaml", "example_swir_ingaas_640.yaml", "example_nir_si_1280.yaml")

#: The perturbation. 10 % is small enough that nothing else in the chain reacts to it and large
#: enough to sit far outside any sampling floor here.
MUTATION = 1.10


@pytest.fixture(scope="module")
def library():  # type: ignore[no-untyped-def]
    return MaterialLibrary.load()


def _build(library, name: str, scale: float = 1.0, handle: str = "auto", shape=(32, 32)):  # type: ignore[no-untyped-def]
    """The committed config with its read noise scaled, through the real `from_sensor`."""
    dumped = load_sensor_config(SENSORS / name).model_dump(mode="json")
    dumped["sensor"]["fpa"]["read_noise_e"] *= scale
    dumped["sensor"]["fpa"].update(width=shape[1], height=shape[0])
    sensor = SensorConfig.model_validate(dumped)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    form = "photon" if sensor.sensor.quantity == "lb_q" else "energy"
    table = MaterialTable.from_library(library, sensor.sensor.band.band_id, form=form)
    return sensor, PipelineConfig.from_sensor(sensor, table, lut, noise_handle=handle)


# --- the mutation is alive ---------------------------------------------------------------------


@pytest.mark.parametrize("name", PHOTON)
def test_the_budget_carries_the_perturbation_exactly(library, name) -> None:
    """sigma_gaussian *is* read_noise_e, so a 10 % change is a 10 % change."""
    base, _ = _build(library, name)
    _, mutated = _build(library, name, MUTATION)
    params = fpa_params_from_config(base)
    assert isinstance(params, PhotonParams) and params.read_noise_e is not None

    assert mutated.detector.budget.sigma_gaussian == pytest.approx(
        MUTATION * params.read_noise_e, rel=1e-12
    )


@pytest.mark.parametrize("name", PHOTON)
def test_the_rendered_sigma_moves_by_the_quadrature_prediction(library, name) -> None:
    """Not "it moved" -- by how much, against sqrt(N_e + N_dark + N_bg + sigma_read^2).

    Stating the prediction rather than a stored number is what makes this a test of the noise
    model instead of a record of a previous run.
    """
    sensor, base = _build(library, name)
    _, mutated = _build(library, name, MUTATION)
    params = fpa_params_from_config(sensor)
    assert isinstance(params, PhotonParams) and params.read_noise_e is not None

    flux = np.full((32, 32), 1.0e8)
    mean_e = base.detector.electrons(flux)
    np.testing.assert_allclose(mutated.detector.electrons(flux), mean_e, rtol=1e-12)

    read = float(params.read_noise_e)
    expected_base = np.sqrt(mean_e + read**2)
    expected_mutated = np.sqrt(mean_e + (MUTATION * read) ** 2)

    np.testing.assert_allclose(base.detector.sigma_electrons(flux), expected_base, rtol=1e-12)
    np.testing.assert_allclose(mutated.detector.sigma_electrons(flux), expected_mutated, rtol=1e-12)
    assert float(expected_mutated.mean()) > float(expected_base.mean())


def test_how_much_it_moves_depends_on_where_the_camera_is_sitting(library) -> None:
    """The InSb at two operating points, because "10 % in, 2 % out" is not a weak result.

    Bright: 4.9e5 signal electrons plus 3.3e5 from the camera's own 90 %-efficient cold shield,
    so the shot term dominates the quadrature and a 10 % read change shows as 2.1 %. Faint: the
    read term is nearly all of it, and the same change shows in full. A test that only ever
    measured the bright case would look like a weak mutation rather than a strong one.
    """
    name = "example_mwir_insb_640.yaml"
    _, base = _build(library, name)
    _, mutated = _build(library, name, MUTATION)

    def shift(flux_value: float) -> float:
        flux = np.full((8, 8), flux_value)
        a = float(base.detector.sigma_electrons(flux).mean())
        b = float(mutated.detector.sigma_electrons(flux).mean())
        return b / a - 1.0

    bright = shift(1.0e8)
    assert bright == pytest.approx(0.0209, abs=0.002), bright

    # Faint: dark and background are fixed offsets, so this camera never reaches read-limited on
    # its own cold-shield load. The trend is still the point, and it is monotone.
    assert shift(1.0e6) > bright


def test_a_faint_photon_camera_shows_the_whole_perturbation(library) -> None:
    """The NIR at 0.004 mean electrons: the read term *is* the noise, so 10 % in gives 10 % out."""
    _, base = _build(library, "example_nir_si_1280.yaml")
    _, mutated = _build(library, "example_nir_si_1280.yaml", MUTATION)
    flux = np.full((8, 8), 1.0e2)
    assert float(base.detector.electrons(flux).mean()) < 0.01
    ratio = float(mutated.detector.sigma_electrons(flux).mean()) / float(
        base.detector.sigma_electrons(flux).mean()
    )
    assert ratio == pytest.approx(MUTATION, rel=1e-3)


def test_the_perturbation_survives_to_rendered_frames(library) -> None:
    """Monte Carlo on the real `response()`, not on the analytic sigma.

    400 frames of a uniform faint scene through the whole detector: the measured per-pixel
    temporal sigma in DN moves by the same 10 %. The analytic checks above could all pass with a
    `response()` that ignored the budget, which is exactly the failure mode SC.1 fixed one layer
    up -- a correct quantity nothing reads.
    """
    _, base = _build(library, "example_nir_si_1280.yaml")
    _, mutated = _build(library, "example_nir_si_1280.yaml", MUTATION)
    flux = np.full((32, 32), 1.0e2)

    def measured_sigma(config) -> float:  # type: ignore[no-untyped-def]
        frames = np.stack([config.detector.response(flux, i, 11).signal_dn for i in range(400)])
        return float(frames.std(axis=0).mean())

    a, b = measured_sigma(base), measured_sigma(mutated)
    assert b / a == pytest.approx(MUTATION, rel=0.02), (a, b)


# --- and it was dead before -------------------------------------------------------------------


@pytest.mark.parametrize("name", PHOTON)
def test_under_the_netd_anchor_the_perturbation_does_nothing_at_all(library, name) -> None:
    """The pre-SC.1 behaviour, kept as the negative control.

    The anchor solves for whatever Gaussian reproduces the datasheet NETD, so the configured read
    noise never enters. Bit-identical, not merely close: there is no path by which the field could
    have had a small effect.
    """
    _, base = _build(library, name, handle="netd")
    _, mutated = _build(library, name, MUTATION, handle="netd")
    assert mutated.detector.budget.sigma_gaussian == base.detector.budget.sigma_gaussian

    flux = np.full((16, 16), 1.0e8)
    np.testing.assert_array_equal(
        mutated.detector.sigma_electrons(flux), base.detector.sigma_electrons(flux)
    )
    frame_a = base.detector.response(flux, 3, 11).dn
    frame_b = mutated.detector.response(flux, 3, 11).dn
    np.testing.assert_array_equal(frame_a, frame_b)


def test_even_halving_the_read_noise_is_invisible_to_the_anchor(library) -> None:
    """10 % could be lost in a rounding somewhere. A factor of two could not."""
    _, base = _build(library, "example_mwir_insb_640.yaml", handle="netd")
    _, halved = _build(library, "example_mwir_insb_640.yaml", 0.5, handle="netd")
    assert halved.detector.budget.sigma_gaussian == base.detector.budget.sigma_gaussian

    # ... while the electron budget tracks it exactly, which is the contrast.
    _, electrons = _build(library, "example_mwir_insb_640.yaml", 0.5)
    assert electrons.detector.budget.sigma_gaussian == pytest.approx(175.0)
