"""Linear AGC (M5.1, ADR 0027) and plateau equalisation (M5.2, ADR 0028): oracle checks against
np.percentile and an argsort rank map, the analytic limits, monotonicity, entropy, bit-depth
handling, and the §15 Tier 3 hot-exhaust collapse as a scalar assertion."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.isp.agc import (
    CONSTANT_FRAME_LEVEL,
    agc_linear,
    agc_plateau,
    histogram_dn,
    percentile_from_histogram,
)


def _ramp(n: int = 256) -> np.ndarray:
    return np.arange(n * n, dtype=np.uint16).reshape(n, n)  # uniform histogram, one pixel per bin


# --- linear -----------------------------------------------------------------------------------


def test_linear_matches_percentile_oracle_on_uniform_ramp() -> None:
    x = _ramp()
    y = agc_linear(x, 0.005, 0.995).astype(np.float64)

    # independent route (sorted order statistics, same p*N convention as the histogram: the value
    # below which p*N pixels lie, pixels uniform within their DN bin; np.percentile's default
    # interpolates at p*(N-1) and differs by up to one DN at the top, i.e. 1.5e-5 in y)
    def threshold(p: float) -> float:
        s = np.sort(x.ravel()).astype(np.float64)
        t = p * s.size
        k = int(np.floor(t))
        return float(s[min(k, s.size - 1)] + (t - k))

    lo, hi = threshold(0.005), threshold(0.995)
    oracle = np.clip((x.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    assert np.max(np.abs(y - oracle)) < 1e-6
    assert abs(lo - np.percentile(x, 0.5)) < 1.0 and abs(hi - np.percentile(x, 99.5)) < 1.0
    # and the closed form for a uniform ramp, with a pixel's fraction taken at its bin centre
    n = x.size
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        d = int(round(p * n))
        expected = ((d + 0.5) / n - 0.005) / 0.99
        assert abs(float(y.ravel()[d]) - expected) < 2e-5, (p, d)
    counts = histogram_dn(x.astype(np.float64), 16)
    assert abs(percentile_from_histogram(counts, 0.5) - np.percentile(x, 50)) < 1.0


def test_linear_clip_fractions() -> None:
    x = _ramp()
    y = agc_linear(x, 0.005, 0.995)
    n = x.size
    assert abs(np.mean(y == 0.0) - 0.005) <= 1.0 / n + 1e-12
    assert abs(np.mean(y == 1.0) - 0.005) <= 1.0 / n + 1e-12


def test_linear_gamma_midpoint() -> None:
    x = _ramp()
    y = agc_linear(x, 0.0, 1.0, gamma=2.2).astype(np.float64)
    median = float(np.sort(y.ravel())[y.size // 2])
    assert abs(median - 0.5 ** (1.0 / 2.2)) < 1e-3  # 0.7297; exponent is 1/gamma, not gamma
    assert abs(median - 0.5**2.2) > 0.3


def test_linear_affine_invariance() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(1000, 9000, size=(64, 64)).astype(np.float32)
    y1 = agc_linear(x, 0.01, 0.99)
    y2 = agc_linear(3.7 * x + 1234.0, 0.01, 0.99)
    assert np.max(np.abs(y1 - y2)) < 1e-3  # bin quantisation of the percentile: ~1/range


def test_linear_constant_frame_and_dtype_guard() -> None:
    const = np.full((8, 8), 4000, dtype=np.uint16)
    y = agc_linear(const, 0.005, 0.995)
    assert np.all(np.isfinite(y)) and np.all(y == CONSTANT_FRAME_LEVEL) and y.dtype == np.float32
    with pytest.raises(TypeError, match="float16"):
        agc_linear(np.zeros((4, 4), np.float16), 0.005, 0.995)
    assert agc_linear(np.zeros((4, 4), np.float32), 0.005, 0.995).dtype == np.float32
    with pytest.raises(ValueError):
        agc_linear(np.full((4, 4), 70000.0, np.float32), 0.005, 0.995, bit_depth=16)


# --- plateau -----------------------------------------------------------------------------------


def test_plateau_full_he_matches_rank_oracle() -> None:
    rng = np.random.default_rng(5)
    x = rng.choice(65536, size=64 * 64, replace=False).astype(np.uint16).reshape(64, 64)
    y = agc_plateau(x, plateau=1.0).astype(np.float64)
    rank = np.argsort(np.argsort(x.ravel())).reshape(x.shape) / (x.size - 1)
    assert np.max(np.abs(y - rank)) < 1.0 / x.size + 1e-9


def test_plateau_small_limit_is_linear_stretch_on_dense_histogram() -> None:
    x = (2000 + (np.arange(64 * 64) % 500)).astype(np.uint16).reshape(64, 64)  # every bin occupied
    y = agc_plateau(x, plateau=1.0 / x.size).astype(np.float64)
    stretch = (x.astype(np.float64) - x.min()) / (x.max() - x.min())
    assert np.max(np.abs(y - stretch)) < 1.0 / 499 + 1e-9
    # sparse histogram: the rank map of the distinct occupied bins (ADR 0028)
    xs = np.array([[10, 10, 20], [20, 1000, 1000]], dtype=np.uint16)
    ys = agc_plateau(xs, plateau=1.0 / xs.size)
    assert ys.tolist() == [[0.0, 0.0, 0.5], [0.5, 1.0, 1.0]]


@pytest.mark.parametrize("plateau", [1e-4, 1e-2, 1.0])
def test_plateau_monotone_in_input(plateau: float) -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 65536, size=(48, 48)).astype(np.uint16)
    y = agc_plateau(x, plateau)
    order = np.argsort(x.ravel(), kind="stable")
    assert np.all(np.diff(y.ravel()[order]) >= 0.0)


def _entropy_bits(y: np.ndarray) -> float:
    dn8 = np.rint(np.clip(y, 0, 1) * 255).astype(int)
    p = np.bincount(dn8.ravel(), minlength=256) / dn8.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def test_plateau_entropy_nondecreasing() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(12000, 150, size=(64, 32))
    b = rng.normal(14000, 150, size=(64, 32))
    x = np.clip(np.concatenate([a, b], axis=1), 0, 65535).astype(np.uint16)
    ent = [_entropy_bits(agc_plateau(x, p)) for p in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
    assert all(b2 >= a2 - 0.01 for a2, b2 in zip(ent[:-1], ent[1:], strict=True)), ent


def test_hot_exhaust_collapses_linear_agc_but_not_plateau() -> None:
    """§15 Tier 3 as a scalar: 5 % of pixels at a saturating 600 K collapse the background's
    8-bit std under linear AGC (percentile clip lands inside the hot population) to < 20 % of
    baseline, while plateau equalisation (P = 0.012) keeps > 50 %."""
    rng = np.random.default_rng(9)
    background = np.clip(rng.normal(20000, 400, size=(128, 128)), 0, 65535).astype(np.uint16)
    scene = background.copy()
    hot = rng.random(scene.shape) < 0.05
    scene[hot] = 65535
    base_lin = agc_linear(background, 0.005, 0.995)[~hot].std()
    base_pl = agc_plateau(background, 0.012)[~hot].std()
    with_lin = agc_linear(scene, 0.005, 0.995)[~hot].std()
    with_pl = agc_plateau(scene, 0.012)[~hot].std()
    assert with_lin < 0.20 * base_lin, (with_lin, base_lin)
    assert with_pl > 0.50 * base_pl, (with_pl, base_pl)


def test_plateau_respects_bit_depth_and_guards() -> None:
    x = np.array([[0, 8000], [16000, 16383]], dtype=np.uint16)
    y = agc_plateau(x, 1.0, bit_depth=14)
    assert float(y[1, 1]) == 1.0 and float(y[0, 0]) == 0.0
    with pytest.raises(ValueError):
        agc_plateau(x, 0.0)
    with pytest.raises(TypeError, match="float16"):
        agc_plateau(np.zeros((4, 4), np.float16), 0.01)
    const = np.full((8, 8), 123, dtype=np.uint16)
    assert np.all(agc_plateau(const, 0.01) == CONSTANT_FRAME_LEVEL)
