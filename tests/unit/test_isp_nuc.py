"""Two-point NUC (M5.6, §11.2): removes a linear per-pixel gain/offset non-uniformity exactly,
matches the spec formulas, is two-point (not a fit), and keeps the float32 path."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.isp import TwoPointNuc


def _signal(t: float) -> float:
    return 100.0 * (t - 273.15) + 20000.0  # a linear ideal response in DN


def test_removes_linear_fpn_to_1e_6() -> None:
    rng = np.random.default_rng(4)
    g = rng.uniform(0.9, 1.1, size=(64, 64))
    o = rng.uniform(-500.0, 500.0, size=(64, 64))
    frame = lambda t: (g * _signal(t) + o).astype(np.float32)  # noqa: E731
    nuc = TwoPointNuc.calibrate(frame(293.0), frame(313.0))
    corrected = nuc.apply(frame(303.0)).astype(np.float64)
    assert corrected.std() / corrected.mean() < 1e-6
    assert nuc.gain.dtype == np.float32 and nuc.offset.dtype == np.float32


def test_gain_offset_match_spec_formula_and_level_convention() -> None:
    rng = np.random.default_rng(5)
    lo = (1000.0 + rng.normal(0, 50, (8, 8))).astype(np.float32)
    hi = (3000.0 + rng.normal(0, 50, (8, 8))).astype(np.float32)
    nuc = TwoPointNuc.calibrate(lo, hi)
    g = (hi.astype(np.float64).mean() - lo.astype(np.float64).mean()) / (
        hi.astype(np.float64) - lo.astype(np.float64)
    )
    np.testing.assert_allclose(nuc.gain.astype(np.float64), g, rtol=1e-6)
    np.testing.assert_array_equal(nuc.offset, lo)
    assert np.all(nuc.apply(lo) == 0.0), "the corrected cold blackbody is 0 (ADR 0021 level)"
    assert nuc.apply(hi).astype(np.float64).mean() == pytest.approx(
        float(hi.mean() - lo.mean()), rel=1e-6
    )


def test_two_point_not_a_fit_quadratic_residual() -> None:
    """A quadratic response leaves the analytic residual at a third temperature."""
    a = 0.02
    resp = lambda t: 100.0 * (t - 273.15) + a * (t - 273.15) ** 2 + 20000.0  # noqa: E731
    frame = lambda t: np.full((4, 4), resp(t), np.float32)  # noqa: E731
    t_lo, t_hi, t_mid = 293.0, 313.0, 303.0
    nuc = TwoPointNuc.calibrate(frame(t_lo), frame(t_hi))
    corrected = float(nuc.apply(frame(t_mid))[0, 0])
    # the two-point line through (t_lo, 0) and (t_hi, resp_hi - resp_lo)
    linear_pred = (resp(t_hi) - resp(t_lo)) * (t_mid - t_lo) / (t_hi - t_lo)
    actual = resp(t_mid) - resp(t_lo)
    residual_expected = actual - linear_pred  # = -a * (t_mid - t_lo) * (t_hi - t_mid)
    assert residual_expected == pytest.approx(-a * 10.0 * 10.0, rel=1e-9)
    assert corrected - linear_pred == pytest.approx(residual_expected, rel=1e-2)


def test_guards_and_identity() -> None:
    with pytest.raises(TypeError, match="float16"):
        TwoPointNuc.calibrate(np.zeros((2, 2), np.float16), np.ones((2, 2), np.float16))
    with pytest.raises(ValueError):
        TwoPointNuc.calibrate(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))
    ident = TwoPointNuc.identity((2, 3))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(ident.apply(x), x)
    with pytest.raises(ValueError, match="shape"):
        ident.apply(np.zeros((3, 2), np.float32))
