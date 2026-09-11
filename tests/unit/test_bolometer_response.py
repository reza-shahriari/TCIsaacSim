"""MicrobolometerDetector.response (M4.8, ADR 0026): the two-blackbody NETD bench reproduces the
anchor at 300 K and the derivative ratio at 373 K; DN noise is scene-independent while NETD in
kelvin falls (the pair that distinguishes signal-space from Kelvin-space noise); SITF linear in
L_B; determinism; float16 refused."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    MicrobolometerDetector,
    NoiseBudget,
    anchor_noise,
    fpa_params_from_config,
)
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT
from irsim.validation import measured_netd_k

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
def detector(sensor: SensorSpec, tophat_lwir_lut: BandLUT) -> MicrobolometerDetector:
    p = fpa_params_from_config(SensorConfig(sensor=sensor))
    assert isinstance(p, BolometerParams)
    lo, hi = (
        float(_flux(sensor, tophat_lwir_lut, 233.15)[0, 0]),
        float(_flux(sensor, tophat_lwir_lut, 473.15)[0, 0]),
    )
    return MicrobolometerDetector(
        p, BolometerTransfer.from_power_range(lo, hi, 16), anchor_noise(sensor, tophat_lwir_lut)
    )


def _cube(det: MicrobolometerDetector, flux: np.ndarray, n: int, seed: int) -> np.ndarray:
    return np.stack([det.response(flux, f, seed).signal_dn for f in range(n)])


def test_two_blackbody_netd_at_300k_and_the_derivative_ratio_at_373k(
    detector: MicrobolometerDetector, sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    n = 200
    lut = tophat_lwir_lut
    netd_300 = measured_netd_k(
        _cube(detector, _flux(sensor, lut, 300.0), n, 1),
        _cube(detector, _flux(sensor, lut, 303.0), n, 2),
        3.0,
    )
    assert abs(netd_300 / 0.050 - 1.0) < 0.10, netd_300
    netd_373 = measured_netd_k(
        _cube(detector, _flux(sensor, lut, 373.0), n, 3),
        _cube(detector, _flux(sensor, lut, 376.0), n, 4),
        3.0,
    )
    expected_ratio = float(lut.lookup(300.0, "dlb_dt")[()]) / float(lut.lookup(373.0, "dlb_dt")[()])
    assert expected_ratio == pytest.approx(0.576, abs=0.01)
    assert abs((netd_373 / netd_300) / expected_ratio - 1.0) < 0.05, (netd_373, netd_300)


def test_dn_noise_scene_independent_while_netd_falls(
    detector: MicrobolometerDetector, sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    """Signal-space noise: the DN std is the same at 300 and 373 K (2 %) although NETD in K fell."""
    std_300 = (
        _cube(detector, _flux(sensor, tophat_lwir_lut, 300.0), 100, 5).std(axis=0, ddof=1).mean()
    )
    std_373 = (
        _cube(detector, _flux(sensor, tophat_lwir_lut, 373.0), 100, 6).std(axis=0, ddof=1).mean()
    )
    assert abs(std_373 / std_300 - 1.0) < 0.02
    assert std_300 == pytest.approx(detector.sigma_signal_dn, rel=0.03)


def test_noiseless_sitf_linear_in_band_radiance(
    detector: MicrobolometerDetector, sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    temps = np.linspace(250.0, 450.0, 10)
    lb = tophat_lwir_lut.lookup(temps).astype(np.float64)
    dn = np.array(
        [
            float(detector.noiseless_signal_dn(_flux(sensor, tophat_lwir_lut, float(t)))[0, 0])
            for t in temps
        ]
    )
    a, b = np.polyfit(lb, dn, 1)
    resid = dn - (a * lb + b)
    assert np.max(np.abs(resid)) / dn.max() < 1e-6
    assert np.all(np.diff(dn) > 0)


def test_determinism_dtypes_and_float16_refused(
    detector: MicrobolometerDetector, sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    flux = _flux(sensor, tophat_lwir_lut, 300.0)
    a, b = detector.response(flux, 9, 21), detector.response(flux, 9, 21)
    assert np.array_equal(a.dn, b.dn) and np.array_equal(a.signal_dn, b.signal_dn)
    assert not np.array_equal(a.dn, detector.response(flux, 10, 21).dn)
    assert not np.array_equal(a.dn, detector.response(flux, 9, 22).dn)
    assert (
        a.dn.dtype == np.uint16
        and a.signal_dn.dtype == np.float32
        and a.sigma_dn.dtype == np.float32
    )
    assert np.all(a.sigma_dn == np.float32(detector.sigma_signal_dn))
    with pytest.raises(TypeError, match="float16"):
        detector.response(flux.astype(np.float16), 0, 1)
    with pytest.raises(ValueError):
        MicrobolometerDetector(
            detector.params, detector.transfer, NoiseBudget(kind="photon", sigma_gaussian=1.0)
        )
