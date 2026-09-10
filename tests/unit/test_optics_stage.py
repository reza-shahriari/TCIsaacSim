"""Box downsample and the composed optics stage (M3.9, ADR 0020): blackbody round trip, the
on-axis power known answer, cos⁴ field, housing drift, detector-MTF transfer and aliasing."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.optics import (
    aperture_factor,
    apply_optics,
    box_downsample,
    box_transfer,
    invert_optics,
    required_render_size,
)
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import band_radiance_tophat

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _sensor(**optics: Any) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["optics"].update(optics)
    return SensorConfig.model_validate(d).sensor


def _small(width: int = 32, height: int = 16, **optics: Any) -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"].update(optics)
    return SensorConfig.model_validate(d).sensor


# --- box downsample -----------------------------------------------------------------------


def test_box_downsample_conserves_mean_and_uniform_field() -> None:
    rng = np.random.default_rng(1)
    img = rng.random((64, 96)).astype(np.float32)
    out = box_downsample(img, 4)
    assert out.shape == (16, 24) and out.dtype == np.float32
    assert abs(float(out.mean()) / float(img.astype(np.float64).mean()) - 1.0) < 1e-7
    uni = box_downsample(np.full((8, 8), 3.5, dtype=np.float32), 4)
    assert np.all(uni == 3.5)
    assert required_render_size(640, 512, 4) == (2048, 2560)
    with pytest.raises(ValueError, match="divisible"):
        box_downsample(np.zeros((10, 8), np.float32), 4)
    with pytest.raises(TypeError, match="float16"):
        box_downsample(np.zeros((8, 8), np.float16), 4)


def _boxed_cosine_amplitude(xi: float, k: int, n: int = 64) -> float:
    """Amplitude of a unit-amplitude cosine at ξ cycles/native pixel after the k× box, measured
    in the spatial domain. The cosine peaks at pixel centres (x = 0.5), so the native samples
    land on peaks and troughs for ξ ∈ {0.25, 0.5, 0.75, 1.0, 1.5} and (max − min)/2 is the
    sampled amplitude -- an FFT estimate is phase-sensitive at the Nyquist bin."""
    x = (np.arange(n * k) + 0.5) / k
    row = np.cos(2 * np.pi * xi * (x - 0.5))
    out = box_downsample(np.tile(row, (k, 1)), k)[0].astype(np.float64)
    return float((out.max() - out.min()) / 2.0)


@pytest.mark.parametrize("xi", [0.25, 0.5, 0.75, 1.0])
def test_box_transfer_matches_discrete_sinc_and_its_limit(xi: float) -> None:
    """A supersampled sinusoid at ξ cycles/pixel is attenuated by the discrete box transfer
    sin(πξ)/(k sin(πξ/k)) exactly (1e-6 at k = 4) and by |sinc(πξ)| in the limit (1e-3 at
    k = 32); one cycle per pixel averages to nothing."""
    assert abs(_boxed_cosine_amplitude(xi, 4) - float(box_transfer(xi, 4))) < 1e-6
    assert abs(_boxed_cosine_amplitude(xi, 32) - abs(float(np.sinc(xi)))) < 1e-3
    if xi == 1.0:
        assert _boxed_cosine_amplitude(xi, 4) < 1e-9


def test_aliasing_above_nyquist_folds_to_half_pixel() -> None:
    """1.5 cycles/pixel comes out as 0.5 cycles/pixel (sign alternating per pixel) with the
    amplitude the box transfer predicts at 1.5 -- a blur-then-decimate scheme would remove it."""
    k, n, xi = 4, 64, 1.5
    x = (np.arange(n * k) + 0.5) / k
    row = np.cos(2 * np.pi * xi * (x - 0.5))
    out = box_downsample(np.tile(row, (k, 1)), k)[0].astype(np.float64)
    spec = np.abs(np.fft.rfft(out))
    peak_bin = int(np.argmax(spec[1:])) + 1
    assert peak_bin == n // 2, "must alias to the Nyquist bin"
    assert abs((out.max() - out.min()) / 2.0 - float(box_transfer(xi, k))) < 1e-6
    assert (
        abs(
            _boxed_cosine_amplitude(xi, k)
            - _boxed_cosine_amplitude(0.5, k) * box_transfer(xi, k) / box_transfer(0.5, k)
        )
        < 1e-6
    )


def test_sub_pixel_point_source_flux_is_conserved() -> None:
    k, n = 4, 8
    total = []
    peaks = []
    for shift in range(k):
        img = np.zeros((n * k, n * k))
        img[4 * k : 5 * k, 4 * k + shift : 5 * k + shift] = (
            1.0  # a 1-pixel source shifted by shift/k
        )
        out = box_downsample(img, k)
        total.append(float(out.sum()))
        peaks.append(float(out.max()))
    assert max(total) - min(total) < 1e-6
    # aligned: all in one pixel; half-pixel shift: split 50/50
    assert peaks[0] == pytest.approx(1.0) and min(peaks) == pytest.approx(0.5)
    assert sorted(peaks) == pytest.approx([0.5, 0.75, 0.75, 1.0])


# --- composed stage -------------------------------------------------------------------------


def test_on_axis_power_known_answer_without_vignetting() -> None:
    """Φ = A_d π/5 (0.92 Lb(300) + 0.08 Lb(T_h)) on the Boson with cos⁴ off, to 1e-9."""
    s = _small(vignetting_cos4=False)
    lb300, lb310 = band_radiance_tophat(7.5, 13.5, 300.0), band_radiance_tophat(7.5, 13.5, 310.0)
    phi = apply_optics(np.full((16 * 4, 32 * 4), lb300, dtype=np.float32), s, lb310)
    expected = s.detector_active_area_m2 * np.pi / 5.0 * (0.92 * lb300 + 0.08 * lb310)
    assert phi.dtype == np.float32 and phi.shape == (16, 32)
    assert abs(float(phi[8, 16]) / expected - 1.0) < 1e-6  # float32 output
    phi64 = apply_optics(np.full((16, 32), lb300, dtype=np.float64), s, lb310, supersample=1)
    assert abs(float(phi64[8, 16]) / expected - 1.0) < 1e-6


def test_blackbody_round_trip_under_1mK_everywhere(tophat_lwir_lut: BandLUT) -> None:
    """radiance → Φ → invert → T_app recovers 300 K within 1 mK at every pixel (cos⁴ divided
    out)."""
    s = _sensor()
    lb300 = float(tophat_lwir_lut.lookup(300.0)[()])
    radiance = np.full((512, 640), lb300, dtype=np.float32)
    phi = apply_optics(radiance, s, float(tophat_lwir_lut.lookup(305.0)[()]), supersample=1)
    back = invert_optics(phi, s, float(tophat_lwir_lut.lookup(305.0)[()]))
    t_app = tophat_lwir_lut.apparent_temperature(back).astype(np.float64)
    err_mk = np.abs(t_app - 300.0) * 1e3
    assert err_mk[256, 320] < 1.0 and err_mk.max() < 1.0, f"max {err_mk.max():.3f} mK"


def test_cos4_field_corner_over_centre() -> None:
    s = _sensor()
    phi = apply_optics(np.ones((512, 640), dtype=np.float32), s, 0.0, supersample=1)
    centre = float(phi[255:257, 319:321].mean())
    # corner *pixel centre* is half a pixel inside the format corner (0.79240): 0.7930 (ADR 0015)
    assert float(phi[0, 0]) / centre == pytest.approx(0.7930, abs=1e-4)
    x_mm, y_mm = (0.5 - 320) * 0.012, (256.5 - 256) * 0.012  # edge pixel centre
    edge = (14.0**2 / (14.0**2 + x_mm**2 + y_mm**2)) ** 2  # 0.8653 (format edge: 0.86496)
    assert float(phi[256, 0]) / centre == pytest.approx(edge, abs=1e-5)


def test_housing_step_reads_as_87mK(tophat_lwir_lut: BandLUT) -> None:
    s = _small()
    lb = tophat_lwir_lut
    radiance = np.full((16, 32), float(lb.lookup(300.0)[()]), dtype=np.float32)
    phi_ref = apply_optics(radiance, s, float(lb.lookup(300.0)[()]), supersample=1)
    phi_hot = apply_optics(radiance, s, float(lb.lookup(301.0)[()]), supersample=1)
    # the camera still assumes the calibration housing temperature when inverting
    t_ref = lb.apparent_temperature(invert_optics(phi_ref, s, float(lb.lookup(300.0)[()])))
    t_hot = lb.apparent_temperature(invert_optics(phi_hot, s, float(lb.lookup(300.0)[()])))
    shift_mk = float((t_hot.astype(np.float64) - t_ref.astype(np.float64)).mean()) * 1e3
    assert shift_mk == pytest.approx(87.0, abs=2.0), shift_mk


def test_ramp_is_monotone_and_shapes_checked(
    gbuffer_ramp: dict[str, np.ndarray], boson_lut: BandLUT
) -> None:
    s = _sensor()
    radiance = boson_lut.lookup(gbuffer_ramp["temperature_k"])
    with pytest.raises(ValueError, match="detector grid"):
        apply_optics(radiance, s, 0.0, supersample=1)  # 256x256 is not 512x640
    small = _small(256, 256, vignetting_cos4=False)
    phi = apply_optics(radiance, small, 0.0, supersample=1)
    assert np.all(np.diff(phi[0].astype(np.float64)) > 0)
    assert aperture_factor(1.0) == pytest.approx(np.pi / 5.0)
