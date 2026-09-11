"""Slant-edge MTF estimator (MS.5 / VAL-15): Gaussian and box-sampled known answers, polarity and
noise invariance."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.validation import slant_edge_mtf


def _edge(n: int = 128, angle_deg: float = 5.5, k: int = 8, x0: float = 64.0) -> np.ndarray:
    """A 0/1 vertical edge rendered at k x oversampling and box-averaged to n x n (area
    sampling)."""
    yy, xx = np.mgrid[0 : n * k, 0 : n * k]
    edge_x = (x0 + (yy + 0.5) / k * np.tan(np.radians(angle_deg))) * k
    img = ((xx + 0.5) > edge_x).astype(np.float64)
    return img.reshape(n, k, n, k).mean(axis=(1, 3))


def _gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    r = int(np.ceil(4 * sigma_px))
    x = np.arange(-r, r + 1)
    g = np.exp(-0.5 * (x / sigma_px) ** 2)
    g /= g.sum()
    tmp = np.apply_along_axis(
        lambda row: np.convolve(np.pad(row, r, mode="edge"), g, mode="valid"), 1, img
    )
    return np.apply_along_axis(
        lambda col: np.convolve(np.pad(col, r, mode="edge"), g, mode="valid"), 0, tmp
    )


def test_recovers_gaussian_mtf_within_0p02() -> None:
    sigma = 0.8
    img = _gaussian_blur(_edge(), sigma)
    res = slant_edge_mtf(img, oversample=4)
    assert abs(res.edge_angle_deg - 5.5) < 0.2
    f = res.freq_cyc_per_px
    keep = f <= 0.5
    # the blur is a Gaussian *sampled at the pixel pitch*, so its transfer is the aliased sum
    # sum_k G(f - k) (2 G(0.5) at Nyquist, 0.027 above the continuous formula); times the area
    # sampling sinc of the edge render
    gauss = sum(np.exp(-2 * np.pi**2 * sigma**2 * (f[keep] - kk) ** 2) for kk in (-1, 0, 1))
    expected = gauss * np.abs(np.sinc(f[keep]))
    assert np.max(np.abs(res.mtf[keep] - expected)) < 0.02


def test_recovers_detector_sinc_within_0p03() -> None:
    res = slant_edge_mtf(_edge(), oversample=4)
    f = res.freq_cyc_per_px
    keep = f <= 0.5
    assert np.max(np.abs(res.mtf[keep] - np.abs(np.sinc(f[keep])))) < 0.03
    assert res.at(0.5) == pytest.approx(2.0 / np.pi, abs=0.03)


def test_polarity_and_noise_invariance() -> None:
    img = _gaussian_blur(_edge(), 0.6)
    a = slant_edge_mtf(img, oversample=4)
    b = slant_edge_mtf(1.0 - img, oversample=4)
    assert np.max(np.abs(a.mtf - b.mtf)) < 1e-9
    rng = np.random.default_rng(1)
    noisy = img + rng.normal(0.0, 0.01, img.shape)  # SNR 100 on a unit step
    c = slant_edge_mtf(noisy, oversample=4)
    keep = a.freq_cyc_per_px <= 0.5
    assert np.max(np.abs(a.mtf[keep] - c.mtf[keep])) < 0.03
    with pytest.raises(TypeError, match="float16"):
        slant_edge_mtf(img.astype(np.float16))
    assert slant_edge_mtf(img, oversample=4, pitch_um=12.0).freq_cyc_per_mm[-1] == pytest.approx(
        2.0 / 0.012
    )
