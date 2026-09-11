"""DDE unsharp mask (M5.3, ADR 0029): identity at zero gain, mean preservation, the analytic
step overshoot of a 3x3 box, the spectral transfer |1 + g(1 - K)|^2, and clipping."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.isp.dde import box3, box3_transfer, dde


def test_gain_zero_is_bit_identical() -> None:
    rng = np.random.default_rng(1)
    y = rng.random((32, 32)).astype(np.float32)
    assert np.array_equal(dde(y, 0.0), y)


@pytest.mark.parametrize("gain", [0.35, 2.0])
def test_constant_preserved(gain: float) -> None:
    y = np.full((16, 16), 0.37, dtype=np.float32)
    assert np.max(np.abs(dde(y, gain) - 0.37)) < 1e-7


def test_step_overshoot_is_gain_over_six() -> None:
    """0.25 -> 0.75 vertical step: the 3x3 box sees 3 of 9 low pixels next to the edge, so the
    high-pass there is +0.5/3 = 1/6 (and -1/6 on the low side): overshoot = +/- gain/6."""
    y = np.full((16, 16), 0.25, dtype=np.float32)
    y[:, 8:] = 0.75
    gain = 0.6
    out = dde(y, gain).astype(np.float64)
    assert abs(out[8, 8] - (0.75 + gain / 6.0)) < 1e-6
    assert abs(out[8, 7] - (0.25 - gain / 6.0)) < 1e-6
    assert abs(out[8, 12] - 0.75) < 1e-6 and abs(out[8, 2] - 0.25) < 1e-6


def test_transfer_function_matches_1_plus_g_1_minus_k() -> None:
    """On white noise (periodic), |Y_out/Y_in|^2 equals |1 + g(1 - K(f))|^2 within 3 % averaged
    over 100 seeded realisations -- an independent FFT route to the same operator."""
    gain, n = 1.5, 32
    fx = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(fx, fx, indexing="xy")
    expected = np.abs(1.0 + gain * (1.0 - box3_transfer(kx, ky))) ** 2
    num = np.zeros((n, n))
    den = np.zeros((n, n))
    for seed in range(100):
        rng = np.random.default_rng(seed)
        y = (0.5 + 0.05 * rng.standard_normal((n, n))).astype(np.float32)
        out = y.astype(np.float64) + gain * (y.astype(np.float64) - box3(y, "wrap"))  # unclipped
        num += np.abs(np.fft.fft2(out - out.mean())) ** 2
        den += np.abs(np.fft.fft2(y.astype(np.float64) - y.mean())) ** 2
    ratio = num / np.where(den > 0, den, 1.0)
    mask = den > 0
    mask[0, 0] = False
    assert np.max(np.abs(ratio[mask] / expected[mask] - 1.0)) < 0.03


def test_output_clipped_and_guards() -> None:
    y = np.zeros((8, 8), dtype=np.float32)
    y[:, 4:] = 1.0
    out = dde(y, 5.0)
    assert out.min() >= 0.0 and out.max() <= 1.0 and out.dtype == np.float32
    with pytest.raises(TypeError, match="float16"):
        dde(y.astype(np.float16), 1.0)
    with pytest.raises(ValueError):
        dde(y, -1.0)
