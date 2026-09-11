"""Noise analysers (ME.2a, ADR 0023): the leakage-corrected 3-D decomposition recovers injected
components from the M4.3 synthesiser within its own sampling floors, single-component and white
cubes behave, the sum-of-squares identity holds, and the PSD diagnostics see white noise as flat
and column noise on the k_v = 0 line.

Tolerances are derived, not guessed: ``estimate_floors`` gives the standard deviation of each
variance estimate for the cube size; every assertion is 3x that floor. A fixed column pattern
of a 64-wide cube has 64 samples, so sigma_H is known only to about 9 % from one cube -- no
estimator can do better, and a flat 5 % bound there would be a coin toss.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import RATIO_ORDER
from irsim.noise import FixedPattern, Sigmas7, synthesize_frame
from irsim.validation import compare_psd, decompose_3d, estimate_floors, spatial_psd, temporal_psd

BOSON_RATIOS = (0.02, 0.08, 0.15, 0.05, 0.05, 0.30, 1.0)


def _cube(sigmas: Sigmas7, shape: tuple[int, int], n_frames: int, seed: int = 5) -> np.ndarray:
    fixed = FixedPattern.generate(shape, sigmas, seed)
    return np.stack([synthesize_frame(shape, sigmas, fixed, seed, f) for f in range(n_frames)])


def _only(name: str, value: float = 1.0) -> Sigmas7:
    return Sigmas7(**{k: (value if k == name else 0.0) for k in RATIO_ORDER})


def _assert_within_floors(cube: np.ndarray, injected: Sigmas7, k_sigma: float = 3.0) -> None:
    d = decompose_3d(cube)
    floors = estimate_floors(cube.shape, injected.as_vector())
    for name, true_sigma in zip(RATIO_ORDER, injected.as_vector(), strict=True):
        err = abs(d.raw_variances[name] - true_sigma**2)
        assert err <= k_sigma * floors[name] + 1e-9, (
            name,
            d.raw_variances[name],
            true_sigma**2,
            floors[name],
        )


def test_round_trip_recovers_boson_ratios_within_sampling_floors() -> None:
    """200 frames of 64x64 with the Boson ratios: every variance estimate within 3 floors. The
    large components are tight (TVH 0.6 %, VH 3 %), the fixed row/column terms are 64-sample
    quantities (about 27 % at 3 sigma on sigma_H)."""
    injected = Sigmas7.from_ratios(1.0, BOSON_RATIOS)
    cube = _cube(injected, (64, 64), 200, seed=17)
    _assert_within_floors(cube, injected)
    d = decompose_3d(cube)
    assert abs(d.tvh - 1.0) < 0.01 and abs(d.vh / 0.30 - 1.0) < 0.05
    assert abs(d.total / injected.total - 1.0) < 0.03
    floors = estimate_floors(cube.shape, injected.as_vector())
    assert 3 * floors["h"] / 0.15**2 > 0.25, "the sigma_H floor really is that wide at 64 columns"
    assert 3 * floors["tvh"] / 1.0 < 0.01


def test_white_cube_has_no_directional_components() -> None:
    """Pure i.i.d.: all six directional variance estimates within 3 floors of zero, i.e. the
    directional sigmas < ~0.03 sigma_TVH (the naive std of a directional mean would read
    sigma_TVH/sqrt(N_t N_h) plus bias; the leakage correction removes it)."""
    injected = _only("tvh")
    cube = _cube(injected, (64, 64), 200, seed=3)
    _assert_within_floors(cube, injected)
    d = decompose_3d(cube)
    assert abs(d.tvh - 1.0) < 0.01
    for k in ("t", "v", "h", "tv", "th", "vh"):
        assert getattr(d, k) < 0.03, (k, getattr(d, k))


@pytest.mark.parametrize("name", RATIO_ORDER)
def test_single_component_cubes(name: str) -> None:
    """A cube built from only sigma_X returns every variance within 3 floors: X itself to its
    sampling precision, the lower-order terms to the leakage noise floor, the rest ~0."""
    injected = _only(name)
    cube = _cube(injected, (64, 64), 100, seed=11)
    _assert_within_floors(cube, injected)
    d = decompose_3d(cube)
    unrelated = [k for k in RATIO_ORDER if not set(k) <= set(name) and k != name]
    for k in unrelated:
        assert getattr(d, k) < 0.02, (name, k, getattr(d, k))


def test_sum_of_squares_identity() -> None:
    """The seven sums of squares equal the total sum of squares to 1e-6 (orthogonal
    decomposition), reconstructed from the unbiased variances."""
    cube = _cube(Sigmas7.from_ratios(1.0, BOSON_RATIOS), (16, 24), 30, seed=2)
    nt, nv, nh = cube.shape
    raw = decompose_3d(cube).raw_variances
    ms_tvh = raw["tvh"]
    ms_tv, ms_th, ms_vh = raw["tv"] * nh + ms_tvh, raw["th"] * nv + ms_tvh, raw["vh"] * nt + ms_tvh
    ms_t = raw["t"] * nv * nh + ms_tv + ms_th - ms_tvh
    ms_v = raw["v"] * nt * nh + ms_tv + ms_vh - ms_tvh
    ms_h = raw["h"] * nt * nv + ms_th + ms_vh - ms_tvh
    ss = (
        ms_t * (nt - 1)
        + ms_v * (nv - 1)
        + ms_h * (nh - 1)
        + ms_tv * (nt - 1) * (nv - 1)
        + ms_th * (nt - 1) * (nh - 1)
        + ms_vh * (nv - 1) * (nh - 1)
        + ms_tvh * (nt - 1) * (nv - 1) * (nh - 1)
    )
    total_ss = float(np.sum((cube - cube.mean()) ** 2))
    assert abs(ss / total_ss - 1.0) < 1e-6


def test_floors_scale_with_cube_size() -> None:
    small = estimate_floors((100, 64, 64), Sigmas7.from_ratios(1.0, BOSON_RATIOS).as_vector())
    wide = estimate_floors((100, 64, 1024), Sigmas7.from_ratios(1.0, BOSON_RATIOS).as_vector())
    assert wide["h"] < small["h"] / 3.5  # 16x more columns: sqrt(16) on the variance floor
    assert wide["t"] < small["t"]  # more pixels average the leakage into T


def test_uint16_input_promoted_and_guards() -> None:
    cube = (1000.0 + 5.0 * _cube(_only("tvh"), (8, 8), 20)).astype(np.uint16)
    d = decompose_3d(cube)
    assert 3.5 < d.tvh < 6.5 and d.mean == pytest.approx(1000.0, abs=1.0)
    with pytest.raises(TypeError, match="float16"):
        decompose_3d(np.zeros((4, 4, 4), np.float16))
    with pytest.raises(ValueError):
        decompose_3d(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        decompose_3d(np.zeros((1, 4, 4)))


def test_white_noise_spatial_psd_is_flat_and_parseval() -> None:
    cube = _cube(_only("tvh"), (64, 64), 200, seed=7)
    p = spatial_psd(cube)
    assert p.radial.max() / p.radial.min() < 1.3
    assert p.total_power == pytest.approx(float(cube.var(axis=(1, 2)).mean()), rel=1e-6)
    assert compare_psd(p, p) == 1.0


def test_column_noise_lives_on_the_kv0_line() -> None:
    p = spatial_psd(_cube(_only("h"), (64, 64), 20, seed=9))
    assert p.fraction_on_kv0() > 0.95
    assert p.fraction_on_kh0() < 0.05
    rows = spatial_psd(_cube(_only("v"), (64, 64), 20, seed=9))
    assert rows.fraction_on_kh0() > 0.95
    white = spatial_psd(_cube(_only("tvh"), (64, 64), 20, seed=9))
    assert white.fraction_on_kv0() < 0.05
    assert compare_psd(p, white) > 3.0  # a striped vs white frame fails the factor-of-2 target


def test_temporal_psd_of_white_noise_is_flat_and_parseval() -> None:
    cube = _cube(_only("tvh"), (16, 16), 512, seed=4)
    f, power = temporal_psd(cube)
    assert f[0] == 0.0 and f[-1] == pytest.approx(0.5)
    assert power.sum() == pytest.approx(float(cube.var(axis=0).mean()), rel=1e-6)
    body = power[1:-1]
    assert body.max() / body.min() < 2.0  # per-bin chi-square spread over 256 pixels
