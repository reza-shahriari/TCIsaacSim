"""Ideal photon-detector transfer (M3.8): independent quadrature, the 17/5 aperture identity,
saturation without wrap, exact quantisation points, and the MWIR order of magnitude."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.detector import PhotonParams, electrons_to_dn, fpa_params_from_config, photoelectrons
from irsim.detector.photon import electrons_to_signal_dn
from irsim.radiometry.planck import band_photon_radiance_tophat, spectral_photon_radiance

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _mwir(**over: Any) -> PhotonParams:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 640,
        "height": 512,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.7,
        "well_capacity_e": 1.0e7,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
        **over,
    }
    p = fpa_params_from_config(SensorConfig.model_validate(d))
    assert isinstance(p, PhotonParams)
    return p


def test_photoelectrons_match_independent_trapezoid() -> None:
    """300 K, 3-5 µm top-hat, η 0.7, F/2, τ 0.9, 15 µm, 5 ms: vs trapezoid of
    spectral_photon_radiance × (π/17) × 0.9 × 0.7 × A_d × t_int to 1e-4."""
    p = _mwir()
    lam = np.linspace(3.0, 5.0, 40_001)
    lq = float(np.trapezoid(spectral_photon_radiance(lam, 300.0), lam))
    expected = lq * (np.pi / 17.0) * 0.9 * 0.7 * p.active_area_m2 * p.integration_time_s
    got = float(photoelectrons(band_photon_radiance_tophat(3.0, 5.0, 300.0), p, 2.0, 0.9))
    assert abs(got / expected - 1.0) < 1e-4


def test_aperture_identity_f1_over_f2_is_17_over_5() -> None:
    p = _mwir()
    lq = band_photon_radiance_tophat(3.0, 5.0, 300.0)
    ratio = float(photoelectrons(lq, p, 1.0, 0.9) / photoelectrons(lq, p, 2.0, 0.9))
    assert abs(ratio / 3.4 - 1.0) < 1e-9, "a hand-written 4F^2 gives 4.0"


def test_saturation_without_wrap_and_monotone() -> None:
    p = _mwir()
    hot = photoelectrons(np.array([band_photon_radiance_tophat(3.0, 5.0, 1500.0)]), p, 2.0, 0.9)
    assert electrons_to_dn(hot, p).tolist() == [p.dn_max]
    assert int(electrons_to_dn(np.array([0.0]), p)[0]) == 0
    temps = np.linspace(200.0, 1000.0, 41)
    lq = np.array([band_photon_radiance_tophat(3.0, 5.0, float(t)) for t in temps])
    dn = electrons_to_dn(photoelectrons(lq, p, 2.0, 0.9), p)
    assert dn.dtype == np.uint16
    assert np.all(np.diff(dn.astype(np.int64)) >= 0) and int(dn.max()) == p.dn_max


def test_quantisation_points_exact() -> None:
    p = _mwir()
    half = np.array([p.well_capacity_e / 2.0])
    assert electrons_to_dn(half, p).tolist() == [2 ** (p.bit_depth - 1)]
    almost_full = np.array([p.well_capacity_e * (1 - 1e-9)])
    assert electrons_to_dn(almost_full, p).tolist() == [2**p.bit_depth - 1]
    assert electrons_to_signal_dn(half, p).dtype == np.float32


def test_mwir_order_of_magnitude() -> None:
    p = _mwir(quantum_efficiency=0.8)
    n_e = float(photoelectrons(band_photon_radiance_tophat(3.0, 5.0, 300.0), p, 2.0, 0.9))
    assert 1e6 < n_e < 1e8, n_e


def test_energy_form_input_refused_and_dtype_guards() -> None:
    p = _mwir()
    with pytest.raises(ValueError, match="energy-form"):
        photoelectrons(np.array([55.49]), p, 2.0, 0.9)
    with pytest.raises(TypeError, match="float16"):
        photoelectrons(np.array([1.0], dtype=np.float16), p, 2.0, 0.9)
    with pytest.raises(ValueError, match="negative"):
        photoelectrons(np.array([-1e21]), p, 2.0, 0.9)
    out = photoelectrons(
        np.full((3, 4), 1e21, dtype=np.float32), p, 2.0, 0.9, cos4=np.full((3, 4), 0.8)
    )
    assert out.shape == (3, 4) and out.dtype == np.float64
