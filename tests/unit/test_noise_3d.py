"""NVESD 3-D noise synthesiser (M4.3): axis conventions, temporal vs fixed terms, variance
closure and the row-mean variance identity -- the structure that makes thermal imagery thermal."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import RATIO_ORDER
from irsim.noise import FixedPattern, Sigmas7, synthesize_frame

BOSON_RATIOS = (0.02, 0.08, 0.15, 0.05, 0.05, 0.30, 1.0)  # t, v, h, tv, th, vh, tvh


def _only(name: str, value: float = 1.0) -> Sigmas7:
    return Sigmas7(**{k: (value if k == name else 0.0) for k in RATIO_ORDER})


def _cube(sigmas: Sigmas7, shape: tuple[int, int], n_frames: int, seed: int = 5) -> np.ndarray:
    fixed = FixedPattern.generate(shape, sigmas, seed)
    return np.stack([synthesize_frame(shape, sigmas, fixed, seed, f) for f in range(n_frames)])


def test_sigmas_from_ratios_and_closure() -> None:
    s = Sigmas7.from_ratios(2.0, BOSON_RATIOS)
    assert s.as_vector() == tuple(2.0 * r for r in BOSON_RATIOS)
    assert s.total == pytest.approx(2.0 * np.sqrt(sum(r * r for r in BOSON_RATIOS)))
    with pytest.raises(ValueError):
        Sigmas7(-1, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        Sigmas7.from_ratios(1.0, (1.0, 2.0))


def test_axes_not_swapped_and_fixed_terms_persist() -> None:
    """Only σ_H: every column is constant down its height and identical across frames; only σ_V:
    every row constant. Fails if the axes are swapped or fixed patterns are regenerated per
    frame."""
    h_cube = _cube(_only("h"), (16, 24), 5)
    assert np.all(h_cube == h_cube[:, :1, :]), "H must be constant along v"
    assert np.all(h_cube == h_cube[:1]), "H is fixed across frames"
    assert np.std(h_cube[0, 0, :]) > 0.5, "and it varies along h"
    v_cube = _cube(_only("v"), (16, 24), 5)
    assert np.all(v_cube == v_cube[:, :, :1]) and np.all(v_cube == v_cube[:1])
    vh_cube = _cube(_only("vh"), (16, 24), 5)
    assert np.all(vh_cube == vh_cube[:1]) and np.std(vh_cube[0]) > 0.5


def test_temporal_column_and_row_noise_change_per_frame() -> None:
    """Only σ_TH: columns constant within a frame; column vectors uncorrelated between frames."""
    th_cube = _cube(_only("th"), (16, 64), 50)
    assert np.all(th_cube == th_cube[:, :1, :])
    cols = th_cube[:, 0, :]  # (frames, columns)
    r = np.corrcoef(cols)
    off = r[~np.eye(50, dtype=bool)]
    assert np.max(np.abs(off)) < 0.5 and np.mean(np.abs(off)) < 0.15, "TH must not be fixed"
    tv_cube = _cube(_only("tv"), (64, 16), 50)
    assert np.all(tv_cube == tv_cube[:, :, :1])
    assert np.mean(np.abs(np.corrcoef(tv_cube[:, :, 0])[~np.eye(50, dtype=bool)])) < 0.15
    t_cube = _cube(_only("t"), (8, 8), 200)
    assert np.all(t_cube == t_cube[:, :1, :1]) and np.std(t_cube[:, 0, 0]) == pytest.approx(
        1.0, abs=0.15
    )
    tvh_cube = _cube(_only("tvh"), (32, 32), 4)
    assert not np.array_equal(tvh_cube[0], tvh_cube[1])


def test_variance_closure_boson_ratios() -> None:
    """Pooled variance over (t, v, h) of 400 frames of 64×64 equals Σσ² = 1.1243 within 3 %
    (fails if a component is missing, double-added or scaled by ratio instead of ratio²)."""
    sigmas = Sigmas7.from_ratios(1.0, BOSON_RATIOS)
    cube = _cube(sigmas, (64, 64), 400).astype(np.float64)
    expected = sum(r * r for r in BOSON_RATIOS)
    assert expected == pytest.approx(1.1243, abs=5e-5)
    assert abs(cube.var() / expected - 1.0) < 0.03, cube.var()


def test_row_mean_variance_identity() -> None:
    """Var over (t, v) of per-frame row means = σ_T² + σ_V² + σ_TV² + (σ_TH² + σ_VH² + σ_TVH²)/W
    within 10 %: the analytic link between the directional structure and the ratios."""
    sigmas = Sigmas7.from_ratios(1.0, BOSON_RATIOS)
    w = 64
    cube = _cube(sigmas, (64, w), 400, seed=9).astype(np.float64)
    row_means = cube.mean(axis=2)  # (t, v)
    t, v, h, tv, th, vh, tvh = sigmas.as_vector()
    expected = t * t + v * v + tv * tv + (th * th + vh * vh + tvh * tvh) / w
    assert abs(row_means.var() / expected - 1.0) < 0.10, (row_means.var(), expected)


def test_output_is_float32_finite_and_reproducible() -> None:
    sigmas = Sigmas7.from_ratios(3.0e-3, BOSON_RATIOS)
    fixed = FixedPattern.generate((512, 640), sigmas, 21)
    frame = synthesize_frame((512, 640), sigmas, fixed, 21, 7)
    assert frame.dtype == np.float32 and frame.shape == (512, 640) and np.all(np.isfinite(frame))
    assert np.array_equal(frame, synthesize_frame((512, 640), sigmas, fixed, 21, 7))
    assert not np.array_equal(frame, synthesize_frame((512, 640), sigmas, fixed, 21, 8))
    assert not np.array_equal(fixed.vh, FixedPattern.generate((512, 640), sigmas, 22).vh)
    with pytest.raises(ValueError, match="shape"):
        synthesize_frame((8, 8), sigmas, fixed, 21, 0)
