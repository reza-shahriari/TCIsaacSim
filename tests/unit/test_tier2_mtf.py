"""Tier 2 MTF bench (MS.5 / VAL-16): the hotplate/aluminium step edge at 4x through the Boson
optical PSF and the box filter, measured by the slant-edge estimator against the analytic
cascade; the supersampled path aliases while a native-resolution blur does not; the PSF-before-
box order matters; the edge contrast is the radiometric one."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics import (
    apply_psf,
    box_downsample,
    mtf_detector,
    mtf_diffraction,
    nyquist_frequency_cyc_per_mm,
    optical_psf,
)
from irsim.radiometry.lut import BandLUT
from irsim.validation import slant_edge_mtf

LAMBDA_UM, F, PITCH_UM, K = 10.5, 1.0, 12.0, 4


@pytest.fixture(scope="module")
def edge_radiance(tophat_lwir_lut: BandLUT):  # type: ignore[no-untyped-def]
    """Radiance (epsilon = 1 for both materials) of a 512x512-at-4x step edge: 373 K hotplate on the
    right of a 5.5 deg tilted edge, 293 K sheet on the left."""
    n = 128 * K
    yy, xx = np.mgrid[0:n, 0:n]
    edge_x = 64.0 * K + yy * np.tan(np.radians(5.5))
    hot = (xx + 0.5) > edge_x
    t = np.where(hot, 373.0, 293.0).astype(np.float32)
    return tophat_lwir_lut.lookup(t)


def test_measured_mtf_at_nyquist_matches_cascade(edge_radiance: np.ndarray) -> None:
    psf = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=K)
    native = box_downsample(apply_psf(edge_radiance, psf), K)
    res = slant_edge_mtf(native, oversample=4, pitch_um=PITCH_UM)
    xi_n = nyquist_frequency_cyc_per_mm(PITCH_UM)
    expected_n = float(mtf_diffraction(xi_n, LAMBDA_UM, F) * mtf_detector(xi_n, PITCH_UM * 1e-3))
    assert expected_n == pytest.approx(0.31, abs=0.02)
    measured_n = res.at(0.5)
    assert abs(measured_n - expected_n) < 0.05, (measured_n, expected_n)
    f_mm = res.freq_cyc_per_mm
    keep = f_mm <= xi_n
    cascade = mtf_diffraction(f_mm[keep], LAMBDA_UM, F) * mtf_detector(f_mm[keep], PITCH_UM * 1e-3)
    assert np.max(np.abs(res.mtf[keep] - cascade)) < 0.05


def test_supersampled_path_carries_the_detector_footprint_and_aliasing(
    edge_radiance: np.ndarray,
) -> None:
    """The supersampled + box path measures the full cascade (0.31 at Nyquist) and its estimated
    MTF above Nyquist follows |MTF_diff · sinc| -- real aliasing energy the estimator resolves from
    the edge tilt. A native-resolution point-sampled render blurred by the optical PSF lacks the
    detector footprint: it measures ~0.485 at Nyquist instead (the wrong path of §8.3)."""
    psf_ss = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=K)
    ss_path = box_downsample(apply_psf(edge_radiance, psf_ss), K)
    psf_native = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=1)
    native_path = apply_psf(_native_edge(edge_radiance), psf_native)
    a = slant_edge_mtf(ss_path, oversample=4, pitch_um=PITCH_UM)
    b = slant_edge_mtf(native_path, oversample=4, pitch_um=PITCH_UM)
    xi_n = nyquist_frequency_cyc_per_mm(PITCH_UM)
    diff_only = float(mtf_diffraction(xi_n, LAMBDA_UM, F))
    # the native-pitch PSF kernel is itself sampled coarsely, so its transfer at Nyquist is the
    # folded sum (0.51 rather than 0.46): the wrong path reads close to diffraction-only, not
    # the 0.31 of the full cascade
    assert abs(b.at(0.5) - diff_only) < 0.08, (b.at(0.5), diff_only)
    assert a.at(0.5) < b.at(0.5) - 0.1
    f_mm = a.freq_cyc_per_mm
    band = (f_mm > xi_n) & (f_mm < 0.9 / (PITCH_UM * 1e-3))  # between Nyquist and the sinc zero
    predicted = mtf_diffraction(f_mm[band], LAMBDA_UM, F) * mtf_detector(
        f_mm[band], PITCH_UM * 1e-3
    )
    assert np.max(np.abs(a.mtf[band] - predicted)) < 0.05
    assert float(np.trapezoid(a.mtf[band], f_mm[band])) > 0.0


def _native_edge(edge_radiance: np.ndarray) -> np.ndarray:
    """The wrong path: render at native resolution (point-sample the centre supersample)."""
    return np.ascontiguousarray(edge_radiance[K // 2 :: K, K // 2 :: K])


def test_psf_after_box_changes_the_mtf(edge_radiance: np.ndarray) -> None:
    psf_ss = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=K)
    right = box_downsample(apply_psf(edge_radiance, psf_ss), K)
    psf_native = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=1)
    wrong = apply_psf(box_downsample(edge_radiance, K), psf_native)
    a, b = slant_edge_mtf(right, oversample=4), slant_edge_mtf(wrong, oversample=4)
    keep = a.freq_cyc_per_px <= 0.5
    assert np.max(np.abs(a.mtf[keep] - b.mtf[keep])) > 0.02


def test_edge_contrast_matches_radiometry(
    edge_radiance: np.ndarray, tophat_lwir_lut: BandLUT
) -> None:
    psf = optical_psf(LAMBDA_UM, F, 0.0, PITCH_UM, supersample=K)
    native = box_downsample(apply_psf(edge_radiance, psf), K)
    contrast = float(native[:, -8:].mean() - native[:, :8].mean())
    expected = float(tophat_lwir_lut.lookup(373.0)[()] - tophat_lwir_lut.lookup(293.0)[()])
    assert abs(contrast / expected - 1.0) < 0.01
