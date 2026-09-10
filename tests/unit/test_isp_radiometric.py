"""Radiometric branch (M3.10, ADR 0021): blackbody round trip through the ideal chain, the
grey-body gap through the DN route, radiance units, and the omit-not-zero contract."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.isp import RadiometricCalibration, apparent_temperature, apparent_temperature_from_dn
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import band_radiance_tophat

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


@pytest.fixture(scope="module")
def sensor() -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=32, height=16)
    return SensorConfig.model_validate(d).sensor


@pytest.fixture(scope="module")
def cal(sensor: SensorSpec, tophat_lwir_lut: BandLUT) -> RadiometricCalibration:
    return RadiometricCalibration.from_scene_range(sensor, tophat_lwir_lut, 233.15, 473.15, 300.0)


def _uniform(sensor: SensorSpec, value: float) -> np.ndarray:
    return np.full(sensor.fpa_shape, value, dtype=np.float32)


@pytest.mark.parametrize("t", [250.0, 300.0, 350.0, 450.0])
def test_blackbody_round_trip_float32_and_dn_routes(
    sensor: SensorSpec, cal: RadiometricCalibration, tophat_lwir_lut: BandLUT, t: float
) -> None:
    """ε = 1 through the ideal chain: float32 route < 10 mK; DN16 route < max(10 mK, ½ LSB)."""
    lut = tophat_lwir_lut
    radiance = _uniform(sensor, float(lut.lookup(t)[()]))
    signal = cal.signal_from_radiance(radiance)
    t_float = apparent_temperature(cal.radiance_from_signal(signal), lut).astype(np.float64)
    assert np.max(np.abs(t_float - t)) * 1e3 < 10.0
    dn = cal.transfer.dn(cal.transfer.power_from_signal_w(signal))
    t_dn = apparent_temperature_from_dn(dn, cal, lut).astype(np.float64)
    lsb_w = 1.0 / cal.transfer.gain_dn_per_w
    slope_w_per_k = (
        sensor.detector_active_area_m2 * np.pi / 5.0 * 0.92 * float(lut.lookup(t, "dlb_dt")[()])
    )
    half_lsb_mk = 0.5 * lsb_w / slope_w_per_k * 1e3
    bound_mk = max(10.0, half_lsb_mk)
    err_mk = np.max(np.abs(t_dn - t)) * 1e3
    assert err_mk < bound_mk, f"T={t}: {err_mk:.2f} mK vs bound {bound_mk:.2f} mK (½ LSB)"


def test_grey_body_with_environment_reads_the_mixed_radiance(
    sensor: SensorSpec, cal: RadiometricCalibration, tophat_lwir_lut: BandLUT
) -> None:
    """ε = 0.9 at 300 K with a 260 K environment: T_app = Lb⁻¹(0.9 Lb(300) + 0.1 Lb(260)) within
    5 mK, and more than 1 K below 300 K (kinetic T must not leak)."""
    lut = tophat_lwir_lut
    mixed = 0.9 * band_radiance_tophat(7.5, 13.5, 300.0) + 0.1 * band_radiance_tophat(
        7.5, 13.5, 260.0
    )
    signal = cal.signal_from_radiance(_uniform(sensor, mixed))
    t_app = float(apparent_temperature(cal.radiance_from_signal(signal), lut)[0, 0])
    expected = float(lut.apparent_temperature(np.float32(mixed))[()])
    assert abs(t_app - expected) * 1e3 < 5.0
    assert 300.0 - t_app > 1.0


def test_radiance_output_matches_closed_form(
    sensor: SensorSpec, cal: RadiometricCalibration
) -> None:
    lb = band_radiance_tophat(7.5, 13.5, 320.0)
    back = cal.radiance_from_signal(cal.signal_from_radiance(_uniform(sensor, lb)))
    assert back.dtype == np.float32
    assert np.max(np.abs(back.astype(np.float64) / lb - 1.0)) < 1e-4


def test_calibration_spans_the_adc(
    sensor: SensorSpec, cal: RadiometricCalibration, tophat_lwir_lut: BandLUT
) -> None:
    lut = tophat_lwir_lut
    lo = cal.signal_from_radiance(_uniform(sensor, float(lut.lookup(233.15)[()])))
    hi = cal.signal_from_radiance(_uniform(sensor, float(lut.lookup(473.15)[()])))
    assert abs(float(lo[8, 16])) < 1.0 and abs(float(hi[8, 16]) - sensor.dn_max) < 1.0
    with pytest.raises(ValueError):
        RadiometricCalibration.from_scene_range(sensor, lut, 400.0, 300.0, 300.0)


def test_dtype_guards(cal: RadiometricCalibration, tophat_lwir_lut: BandLUT) -> None:
    with pytest.raises(TypeError, match="float16"):
        apparent_temperature(np.array([50.0], dtype=np.float16), tophat_lwir_lut)
    with pytest.raises(TypeError, match="integer"):
        cal.radiance_from_dn(np.array([[1.0]], dtype=np.float32))
