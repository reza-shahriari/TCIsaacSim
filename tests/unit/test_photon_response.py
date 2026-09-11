"""PhotonDetector.response (M4.7, ADR 0026): Poisson statistics in electron space, Arrhenius
dark current, the two-blackbody NETD bench against the anchor and the M4.5 prediction, seeded
replay, saturation and dtype gates."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.detector import (
    NoiseBudget,
    PhotonDetector,
    anchor_noise,
    dark_current_a,
    dark_electrons,
    fpa_params_from_config,
    predict_netd_k,
)
from irsim.detector.params import PhotonParams
from irsim.optics import fpa_irradiance
from irsim.radiometry.constants import BAND_GAP_INSB_EV, EV_PER_K
from irsim.radiometry.lut import BandLUT
from irsim.validation import measured_netd_k

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _mwir(netd_mk: float = 25.0, **fpa: Any) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["optics"].update(f_number=2.0, transmittance=0.9)
    d["sensor"]["noise"]["netd_mk_at_300k"] = netd_mk
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 64,
        "height": 64,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.7,
        "well_capacity_e": 3.0e7,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
        "read_noise_e": 30.0,
        **fpa,
    }
    return SensorConfig.model_validate(d).sensor


def _detector(
    sensor: SensorSpec, lut: BandLUT, budget: NoiseBudget | None = None
) -> PhotonDetector:
    p = fpa_params_from_config(SensorConfig(sensor=sensor))
    assert isinstance(p, PhotonParams)
    return PhotonDetector(p, budget or anchor_noise(sensor, lut))


def _flux(sensor: SensorSpec, lut: BandLUT, t: float) -> np.ndarray:
    a_d = sensor.detector_active_area_m2
    lq = float(lut.lookup(t, "lb_q")[()])
    rate = float(fpa_irradiance(lq, sensor.optics.f_number, sensor.optics.transmittance)) * a_d
    return np.full(sensor.fpa_shape, rate, dtype=np.float32)


def test_poisson_variance_equals_mean(tophat_mwir_lut: BandLUT) -> None:
    """Read noise zero: temporal variance of electrons at a uniform 300 K equals the mean to 3 %."""
    s = _mwir()
    det = _detector(s, tophat_mwir_lut, NoiseBudget(kind="photon", sigma_gaussian=0.0))
    flux = _flux(s, tophat_mwir_lut, 300.0)
    mean_e = float(det.electrons(flux)[0, 0])
    frames = np.stack([det.response(flux, f, 11).signal_dn.astype(np.float64) for f in range(16)])
    to_e = det.params.well_capacity_e / 2**det.params.bit_depth
    electrons = frames * to_e
    assert abs(electrons.var(axis=0, ddof=1).mean() / mean_e - 1.0) < 0.03
    assert abs(electrons.mean() / mean_e - 1.0) < 1e-3


def test_insb_dark_current_ratio_matches_closed_form() -> None:
    i77 = dark_current_a(77.0, 1e-13, 77.0, BAND_GAP_INSB_EV)
    i150 = dark_current_a(150.0, 1e-13, 77.0, BAND_GAP_INSB_EV)
    expected = (150.0 / 77.0) ** 1.5 * math.exp(
        -BAND_GAP_INSB_EV / (2 * EV_PER_K) * (1 / 150.0 - 1 / 77.0)
    )
    assert i150 / i77 == pytest.approx(expected, rel=1e-9)
    assert i150 / i77 > 1e3
    assert dark_electrons(1e-13, 5e-3) == pytest.approx(1e-13 * 5e-3 / 1.602176634e-19, rel=1e-12)
    with pytest.raises(ValueError, match="celsius"):
        dark_current_a(25.0, 1e-13, 77.0, BAND_GAP_INSB_EV)


def test_two_blackbody_netd_matches_anchor_and_prediction(tophat_mwir_lut: BandLUT) -> None:
    """NETD(300 K) within 10 % of the 25 mK anchor; at 373 K equal to the M4.5 prediction within
    5 % on 200 frames -- fails if noise were added in kelvin or the anchor scaled the wrong term."""
    s = _mwir(netd_mk=25.0)
    det = _detector(s, tophat_mwir_lut)
    n = 200
    for t0, expect in (
        (300.0, 0.025),
        (373.0, predict_netd_k(373.0, s, tophat_mwir_lut, det.budget)),
    ):
        lo = np.stack(
            [det.response(_flux(s, tophat_mwir_lut, t0), f, 3).signal_dn for f in range(n)]
        )
        hi = np.stack(
            [det.response(_flux(s, tophat_mwir_lut, t0 + 2.0), f, 4).signal_dn for f in range(n)]
        )
        netd = measured_netd_k(lo, hi, 2.0)
        tol = 0.10 if t0 == 300.0 else 0.05
        assert abs(netd / expect - 1.0) < tol, f"T={t0}: {netd * 1e3:.2f} mK vs {expect * 1e3:.2f}"
    assert predict_netd_k(373.0, s, tophat_mwir_lut, det.budget) < 0.025


def test_determinism_and_dtype_gates(tophat_mwir_lut: BandLUT) -> None:
    s = _mwir()
    det = _detector(s, tophat_mwir_lut)
    flux = _flux(s, tophat_mwir_lut, 300.0)
    a, b = det.response(flux, 7, 1), det.response(flux, 7, 1)
    assert np.array_equal(a.dn, b.dn) and np.array_equal(a.signal_dn, b.signal_dn)
    assert not np.array_equal(a.dn, det.response(flux, 8, 1).dn)
    assert (
        a.dn.dtype == np.uint16
        and a.signal_dn.dtype == np.float32
        and a.sigma_dn.dtype == np.float32
    )
    with pytest.raises(TypeError, match="float16"):
        det.response(flux.astype(np.float16), 0, 1)
    with pytest.raises(ValueError):
        PhotonDetector(det.params, NoiseBudget(kind="bolometer", sigma_gaussian=1.0))


def test_saturation_with_noise(tophat_mwir_lut: BandLUT) -> None:
    s = _mwir()
    det = _detector(s, tophat_mwir_lut)
    out = det.response(_flux(s, tophat_mwir_lut, 1000.0), 0, 1)
    assert np.all(out.dn == det.params.dn_max)
    assert np.all(out.signal_dn >= 0.0)
    assert np.all(det.response(np.zeros(s.fpa_shape, np.float32), 0, 1).signal_dn >= 0.0)


def test_sigma_dn_reports_the_per_pixel_total(tophat_mwir_lut: BandLUT) -> None:
    s = _mwir()
    det = _detector(s, tophat_mwir_lut)
    flux = _flux(s, tophat_mwir_lut, 300.0)
    n_e = float(det.electrons(flux)[0, 0])
    expected_e = math.sqrt(n_e + det.budget.sigma_gaussian**2)
    to_dn = 2**det.params.bit_depth / det.params.well_capacity_e
    assert float(det.response(flux, 0, 1).sigma_dn[0, 0]) == pytest.approx(
        expected_e * to_dn, rel=1e-6
    )
