"""Tier 2 3-D noise bench (M4.4): the synthesiser + detector + NoiseStage round trip through the
ME.2a decomposition on a uniform scene. Tolerances are 3x the estimator's own sampling floors
(ADR 0023)."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import RATIO_ORDER, SensorConfig, SensorSpec
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    MicrobolometerDetector,
    anchor_noise,
    fpa_params_from_config,
)
from irsim.noise import NoiseStage, Sigmas7, measure_from_uniform_scene
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT
from irsim.validation import decompose_3d, estimate_floors

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


@pytest.fixture(scope="module")
def sensor() -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=64, height=64)
    return SensorConfig.model_validate(d).sensor


def _flux(sensor: SensorSpec, lut: BandLUT, t: float) -> np.ndarray:
    phi = float(
        pixel_power(
            lut.lookup(t),
            sensor.optics.f_number,
            sensor.optics.transmittance,
            sensor.detector_active_area_m2,
        )
    )
    return np.full(sensor.fpa_shape, phi, dtype=np.float32)


@pytest.fixture(scope="module")
def chain(
    sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> tuple[MicrobolometerDetector, NoiseStage]:
    p = fpa_params_from_config(SensorConfig(sensor=sensor))
    assert isinstance(p, BolometerParams)
    lo, hi = (
        float(_flux(sensor, tophat_lwir_lut, 233.15)[0, 0]),
        float(_flux(sensor, tophat_lwir_lut, 473.15)[0, 0]),
    )
    det = MicrobolometerDetector(
        p, BolometerTransfer.from_power_range(lo, hi, 16), anchor_noise(sensor, tophat_lwir_lut)
    )
    return det, NoiseStage.from_sensor(sensor, sensor_seed=21)


def test_end_to_end_3d_ratios_recovered_within_floors(
    chain: tuple[MicrobolometerDetector, NoiseStage], sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    det, stage = chain
    cube, sigma_tvh = measure_from_uniform_scene(
        det, stage, _flux(sensor, tophat_lwir_lut, 300.0), 200
    )
    assert cube.dtype == np.float32 and cube.shape == (200, 64, 64)
    injected = Sigmas7.from_ratios(sigma_tvh, sensor.noise.sigma_ratios())
    d = decompose_3d(cube)
    floors = estimate_floors(cube.shape, injected.as_vector())
    for name, true_sigma in zip(RATIO_ORDER, injected.as_vector(), strict=True):
        assert abs(d.raw_variances[name] - true_sigma**2) <= 3 * floors[name] + 1e-6, (
            name,
            d.raw_variances[name],
            true_sigma**2,
        )
    ratios = d.ratios()
    assert abs(ratios[-1] - 1.0) < 1e-9 and abs(d.tvh / sigma_tvh - 1.0) < 0.01
    assert abs(ratios[5] / 0.30 - 1.0) < 0.05  # VH: 4096 samples
    assert abs(d.total / injected.total - 1.0) < 0.03


def test_stage_adds_only_correlated_terms_and_replays(
    chain: tuple[MicrobolometerDetector, NoiseStage], sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    det, stage = chain
    flux = _flux(sensor, tophat_lwir_lut, 300.0)
    frame = det.response(flux, 3, stage.sensor_seed)
    added = stage.apply(frame.signal_dn, frame.sigma_dn, 3) - frame.signal_dn
    sig = stage.sigmas(float(frame.sigma_dn.mean()))
    assert sig.tvh == 0.0 and sig.vh == pytest.approx(0.30 * float(frame.sigma_dn.mean()))
    expected_std = np.sqrt(sum(s * s for s in sig.as_vector()))
    assert abs(added.std() / expected_std - 1.0) < 0.15
    again = stage.apply(frame.signal_dn, frame.sigma_dn, 3)
    assert np.array_equal(again, stage.apply(frame.signal_dn, frame.sigma_dn, 3))
    disabled = NoiseStage.from_sensor(sensor, sensor_seed=21, enabled=False)
    assert np.array_equal(disabled.apply(frame.signal_dn, frame.sigma_dn, 3), frame.signal_dn)
    with pytest.raises(TypeError, match="float32"):
        stage.apply(frame.signal_dn.astype(np.float64), frame.sigma_dn, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        stage.apply(frame.signal_dn[:8], frame.sigma_dn[:8], 3)


def test_fixed_patterns_are_per_sensor_and_scale_with_sigma(sensor: SensorSpec) -> None:
    a = NoiseStage.from_sensor(sensor, sensor_seed=1)
    b = NoiseStage.from_sensor(sensor, sensor_seed=2)
    assert not np.array_equal(a.unit_fixed.vh, b.unit_fixed.vh)
    assert a.unit_fixed.vh.std() == pytest.approx(0.30, rel=0.05)  # unit sigma_TVH times r_VH
    zeros = np.zeros(sensor.fpa_shape, np.float32)
    small = a.apply(zeros, np.full(sensor.fpa_shape, 1.0, np.float32), 0)
    big = a.apply(zeros, np.full(sensor.fpa_shape, 10.0, np.float32), 0)
    assert np.allclose(big, 10.0 * small, rtol=1e-5, atol=1e-5)
