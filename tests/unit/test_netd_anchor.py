"""NETD anchoring (M4.6, ADR 0025): the anchor reproduces the datasheet number exactly, leaves the
Poisson structure untouched, refuses unattainable targets, and lets F enter only through the
transfer."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.detector import anchor_noise, predict_netd_k, shot_variance, signal_derivative_per_k
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _boson(
    netd_mk: float = 50.0, f_number: float = 1.0, netd_ref_f: float | None = None
) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["noise"]["netd_mk_at_300k"] = netd_mk
    d["sensor"]["noise"]["netd_ref_f_number"] = netd_ref_f
    d["sensor"]["optics"]["f_number"] = f_number
    return SensorConfig.model_validate(d).sensor


def _mwir(
    netd_mk: float = 20.0, t_int_ms: float = 5.0, qe: float = 0.7, **extra: Any
) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["optics"].update(f_number=2.0, transmittance=0.9)
    d["sensor"]["noise"]["netd_mk_at_300k"] = netd_mk
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 640,
        "height": 512,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": qe,
        "well_capacity_e": 1.0e7,
        "integration_time_ms": t_int_ms,
        "dark_current_model": "arrhenius",
        "read_noise_e": 30.0,
        **extra,
    }
    return SensorConfig.model_validate(d).sensor


def test_anchor_reproduces_target_exactly(
    tophat_lwir_lut: BandLUT, tophat_mwir_lut: BandLUT
) -> None:
    s = _boson()
    b = anchor_noise(s, tophat_lwir_lut)
    assert b.kind == "bolometer" and b.sigma_gaussian > 0.0
    assert predict_netd_k(300.0, s, tophat_lwir_lut, b) == pytest.approx(0.050, rel=1e-9)
    assert "netd_temperature_fluctuation_k" in b.floors
    # 1/(4 tau) on [R24]'s nominal 8 ms membrane (SC.3); it was 25.0 Hz at the ESTIMATED 10 ms.
    assert b.floors["enbw_hz"] == pytest.approx(31.25)
    m = _mwir(netd_mk=25.0)
    bm = anchor_noise(m, tophat_mwir_lut)
    assert bm.kind == "photon"
    assert predict_netd_k(300.0, m, tophat_mwir_lut, bm) == pytest.approx(0.025, rel=1e-9)


def test_structure_preserved_and_slope_negative(
    tophat_mwir_lut: BandLUT, tophat_lwir_lut: BandLUT
) -> None:
    m = _mwir(netd_mk=25.0)
    before = shot_variance(
        300.0, m, tophat_mwir_lut, anchor_noise(m, tophat_mwir_lut, target_netd_k=0.030)
    )
    after = shot_variance(
        300.0, m, tophat_mwir_lut, anchor_noise(m, tophat_mwir_lut, target_netd_k=0.025)
    )
    assert before == after, "anchoring must not touch the Poisson terms"
    bm = anchor_noise(m, tophat_mwir_lut)
    assert predict_netd_k(373.0, m, tophat_mwir_lut, bm) < predict_netd_k(
        300.0, m, tophat_mwir_lut, bm
    )
    s = _boson()
    b = anchor_noise(s, tophat_lwir_lut)
    assert predict_netd_k(373.0, s, tophat_lwir_lut, b) < 0.050


def test_unattainable_target_raises(tophat_mwir_lut: BandLUT) -> None:
    """A photon config whose shot-noise NETD exceeds the target must raise, not go negative."""
    m = _mwir(netd_mk=20.0, t_int_ms=0.05, qe=0.05)  # few electrons: shot-limited NETD ≫ 20 mK
    shot_only = predict_netd_k(
        300.0, m, tophat_mwir_lut, anchor_noise(m, tophat_mwir_lut, target_netd_k=10.0)
    )
    assert shot_only > 0.020, shot_only
    with pytest.raises(ValueError, match="unattainable"):
        anchor_noise(m, tophat_mwir_lut)


def test_grade_sweep_scales_sigma_analytically(
    tophat_lwir_lut: BandLUT, tophat_mwir_lut: BandLUT
) -> None:
    """Editing only netd_mk_at_300k to 60/50/40 mK: σ_read = √((NETD·∂S/∂T)² − σ_shot²)."""
    for grade in (60.0, 50.0, 40.0):
        s = _boson(netd_mk=grade)
        b = anchor_noise(s, tophat_lwir_lut)
        expected = grade * 1e-3 * signal_derivative_per_k(300.0, s, tophat_lwir_lut)
        assert b.sigma_gaussian == pytest.approx(expected, rel=1e-9)
    for grade in (40.0, 30.0, 25.0):
        m = _mwir(netd_mk=grade)
        b = anchor_noise(m, tophat_mwir_lut)
        d = signal_derivative_per_k(300.0, m, tophat_mwir_lut)
        expected = math.sqrt((grade * 1e-3 * d) ** 2 - shot_variance(300.0, m, tophat_mwir_lut, b))
        assert b.sigma_gaussian == pytest.approx(expected, rel=1e-9)


def test_f_number_enters_only_through_the_transfer(tophat_lwir_lut: BandLUT) -> None:
    """Anchored at F_ref = 1.0, a camera run at F/1.4 keeps the same σ_signal and predicts
    NETD(F=1.4)/NETD(F=1.0) = (4·1.96+1)/(4+1) = 1.768 -- not the paraxial 1.96."""
    at_f1 = anchor_noise(_boson(f_number=1.0), tophat_lwir_lut)
    at_f14 = anchor_noise(_boson(f_number=1.4, netd_ref_f=1.0), tophat_lwir_lut)
    assert at_f14.sigma_gaussian == pytest.approx(at_f1.sigma_gaussian, rel=1e-12)
    netd_f14 = predict_netd_k(300.0, _boson(f_number=1.4, netd_ref_f=1.0), tophat_lwir_lut, at_f14)
    assert netd_f14 / 0.050 == pytest.approx(1.768, rel=1e-9)
    # without netd_ref_f_number the datasheet number is taken at the configured F
    native = anchor_noise(_boson(f_number=1.4), tophat_lwir_lut)
    assert predict_netd_k(300.0, _boson(f_number=1.4), tophat_lwir_lut, native) == pytest.approx(
        0.050, rel=1e-9
    )


def test_bad_targets(tophat_lwir_lut: BandLUT) -> None:
    with pytest.raises(ValueError):
        anchor_noise(_boson(), tophat_lwir_lut, target_netd_k=0.0)
