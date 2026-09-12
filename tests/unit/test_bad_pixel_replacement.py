"""Bad-pixel replacement (M9.5b): the stencil's three signatures, and the footprint it leaves.

§10.4 asks for the defect *and* its replacement, because "the replacement artefact is what a
detector actually sees". The tests here pin what that artefact is:

* on a linear ramp an isolated defect is replaced exactly (well under a millikelvin), so smooth
  scene content -- most of a thermal image -- takes no radiometric bias;
* on white noise the replaced pixel's variance is sigma^2/4, which copying one neighbour
  (sigma^2) and an 8-neighbour mean (sigma^2/8) both fail -- and both alternatives are measured
  here so the assertion is shown to discriminate;
* the replaced pixel's local Laplacian is well under half its neighbours', which *is* the
  detectable footprint;
* 2x2 and 3x3 clusters fill, with no stuck floor or ceiling value leaking through.

docs/physics-model.md §10.4, §11.1. ADR 0055.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.isp import replace_bad_pixels
from irsim.noise import (
    DefectKind,
    DefectState,
    active_defect_mask,
    generate_map,
)


def _ramp(shape: tuple[int, int], slope_row: float = 5.0, slope_col: float = 1.0) -> np.ndarray:
    r = np.arange(shape[0])[:, None] * slope_row
    c = np.arange(shape[1])[None, :] * slope_col
    return (r + c + 300.0).astype(np.float64)


def _isolated_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for y, x in points:
        m[y, x] = True
    return m


# -- exactness on smooth content -------------------------------------------------------------


def test_an_isolated_defect_on_a_ramp_is_replaced_exactly() -> None:
    """< 1 mK, i.e. far below an LWIR NETD: the stencil is unbiased on a linear field."""
    shape = (16, 20)
    frame = _ramp(shape)
    mask = _isolated_mask(shape, [(5, 7), (9, 3), (12, 15)])
    out = replace_bad_pixels(frame, mask)
    assert np.max(np.abs(out[mask] - frame[mask])) < 1e-3


@pytest.mark.parametrize(
    ("slope_row", "slope_col"), [(5.0, 0.0), (0.0, 3.0), (2.0, -4.0), (-1.5, 0.25)]
)
def test_exactness_holds_for_any_linear_field(slope_row: float, slope_col: float) -> None:
    shape = (14, 14)
    frame = _ramp(shape, slope_row, slope_col)
    mask = _isolated_mask(shape, [(6, 6), (3, 9)])
    out = replace_bad_pixels(frame, mask)
    assert np.max(np.abs(out[mask] - frame[mask])) < 1e-9


def test_untouched_pixels_are_bit_identical_and_the_input_is_not_modified() -> None:
    shape = (12, 12)
    frame = _ramp(shape)
    before = frame.copy()
    mask = _isolated_mask(shape, [(4, 4)])
    out = replace_bad_pixels(frame, mask)
    assert np.array_equal(frame, before)
    assert np.array_equal(out[~mask], frame[~mask])


# -- the variance signature ------------------------------------------------------------------


def test_replaced_variance_is_a_quarter_and_the_alternatives_are_not() -> None:
    """sigma^2/4 is the 4-neighbour mean's signature; copy-one and 8-neighbour give sigma^2 and
    sigma^2/8. All three are measured on the same field, so the test shows what it excludes."""
    rng = np.random.default_rng(4)
    shape = (256, 256)
    sigma = 1.0
    frame = rng.standard_normal(shape) * sigma

    # A sparse, well-separated set of isolated defects: every one keeps four valid neighbours.
    mask = np.zeros(shape, dtype=bool)
    mask[4::8, 4::8] = True

    replaced = replace_bad_pixels(frame, mask)[mask]
    assert float(replaced.var()) == pytest.approx(sigma**2 / 4.0, rel=0.10)

    ys, xs = np.nonzero(mask)
    copy_one = frame[ys - 1, xs]
    eight = (
        frame[ys - 1, xs]
        + frame[ys + 1, xs]
        + frame[ys, xs - 1]
        + frame[ys, xs + 1]
        + frame[ys - 1, xs - 1]
        + frame[ys - 1, xs + 1]
        + frame[ys + 1, xs - 1]
        + frame[ys + 1, xs + 1]
    ) / 8.0
    assert float(copy_one.var()) == pytest.approx(sigma**2, rel=0.10)
    assert float(eight.var()) == pytest.approx(sigma**2 / 8.0, rel=0.15)
    # and neither is within reach of the 4-neighbour answer
    assert float(copy_one.var()) != pytest.approx(sigma**2 / 4.0, rel=0.10)
    assert float(eight.var()) != pytest.approx(sigma**2 / 4.0, rel=0.10)


# -- the detectable footprint ----------------------------------------------------------------


def test_the_replaced_pixel_has_a_suppressed_laplacian() -> None:
    """§10.4's "detectable smoothed footprint": the patch is locally too smooth.

    This is the artefact a perception stack actually sees, and the one a simulator that skips
    replacement -- or skips defects entirely -- fails to present.
    """
    rng = np.random.default_rng(9)
    shape = (256, 256)
    frame = rng.standard_normal(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[4::8, 4::8] = True
    out = replace_bad_pixels(frame, mask)

    def laplacian(a: np.ndarray) -> np.ndarray:
        lap = -4.0 * a.copy()
        lap[1:-1, 1:-1] = (
            a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1]
        )
        return lap

    lap = laplacian(out)
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    at_defect = np.abs(lap[mask & interior])
    elsewhere = np.abs(lap[(~mask) & interior])
    assert float(at_defect.mean()) < 0.5 * float(elsewhere.mean())


# -- clusters --------------------------------------------------------------------------------


@pytest.mark.parametrize("size", [2, 3, 4])
def test_square_clusters_are_filled_and_no_stuck_value_leaks(size: int) -> None:
    """A 3x3's centre has no valid neighbour on the first pass; iteration is what fills it."""
    shape = (24, 24)
    frame = _ramp(shape)
    stuck = 99999.0  # a floor/ceiling sentinel that must not survive
    mask = np.zeros(shape, dtype=bool)
    y0 = x0 = 10
    mask[y0 : y0 + size, x0 : x0 + size] = True
    frame = frame.copy()
    frame[mask] = stuck

    out = replace_bad_pixels(frame, mask)
    assert np.all(np.isfinite(out))
    assert not np.any(out[mask] == stuck)
    # Every filled value lies inside the range of the surrounding good data, so nothing extreme
    # has leaked out of the cluster.
    ring = ~mask
    lo, hi = float(out[ring].min()), float(out[ring].max())
    assert np.all((out[mask] >= lo) & (out[mask] <= hi))


def test_a_cluster_fills_from_the_outside_in() -> None:
    """The physical content of the iteration: the centre of a 3x3 is the last thing determined.

    Stated as a pass count, because a partial fill is an error rather than an output: one pass
    cannot reach the centre (it has no valid neighbour at all), two can. An isolated defect needs
    only one, which is what makes the iteration a cluster feature rather than a general cost.
    """
    shape = (16, 16)
    frame = _ramp(shape)
    cluster = np.zeros(shape, dtype=bool)
    cluster[7:10, 7:10] = True

    with pytest.raises(ValueError, match="still unfilled after"):
        replace_bad_pixels(frame, cluster, max_passes=1)
    two_passes = replace_bad_pixels(frame, cluster, max_passes=2)
    assert np.allclose(two_passes, replace_bad_pixels(frame, cluster))

    isolated = _isolated_mask(shape, [(8, 8)])
    assert np.allclose(
        replace_bad_pixels(frame, isolated, max_passes=1),
        replace_bad_pixels(frame, isolated),
    )


def test_the_result_does_not_depend_on_scan_order() -> None:
    """Passes are synchronous: a pass reads only what was valid when it began.

    Transposing the problem must transpose the answer. If the stencil consumed values written
    during the same pass, a 2x2 would fill differently row-major than column-major and goldens
    would not reproduce.
    """
    shape = (20, 20)
    frame = _ramp(shape, slope_row=3.0, slope_col=7.0)
    mask = np.zeros(shape, dtype=bool)
    mask[6:8, 6:8] = True
    mask[12, 3] = True
    a = replace_bad_pixels(frame, mask)
    b = replace_bad_pixels(frame.T.copy(), mask.T.copy()).T
    assert np.allclose(a, b, atol=1e-12)


def test_an_oversized_cluster_raises_rather_than_leaving_a_defect() -> None:
    """An unreplaced stuck value silently entering the NUC is worse than a loud failure."""
    shape = (16, 16)
    frame = _ramp(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[2:14, 2:14] = True
    with pytest.raises(ValueError, match="still unfilled after"):
        replace_bad_pixels(frame, mask, max_passes=2)


def test_a_fully_masked_array_raises() -> None:
    with pytest.raises(ValueError, match="nothing to interpolate from"):
        replace_bad_pixels(_ramp((8, 8)), np.ones((8, 8), dtype=bool))


def test_an_empty_mask_is_a_copy() -> None:
    frame = _ramp((8, 8))
    out = replace_bad_pixels(frame, np.zeros((8, 8), dtype=bool))
    assert np.array_equal(out, frame) and out is not frame


# -- dtype and the chain ---------------------------------------------------------------------


def test_an_integer_dn_plane_stays_integer_and_rounds_once() -> None:
    """§11.1 puts replacement on the raw DN plane, so uint16 in must give uint16 out."""
    shape = (16, 16)
    frame = (_ramp(shape) * 100.0).astype(np.uint16)
    mask = _isolated_mask(shape, [(5, 5), (9, 11)])
    out = replace_bad_pixels(frame, mask)
    assert out.dtype == np.uint16
    assert np.array_equal(out[~mask], frame[~mask])
    # exact on the ramp, to within the single rounding
    expected = (frame[4, 5].astype(np.int64) + frame[6, 5] + frame[5, 4] + frame[5, 6]) / 4.0
    assert abs(float(out[5, 5]) - expected) <= 0.5


def test_it_replaces_exactly_the_defects_active_this_frame() -> None:
    """Per-frame mask: an intermittent pixel is replaced only while it is actually bad."""
    shape = (128, 160)
    repo = pathlib.Path(__file__).resolve().parents[2]
    boson = yaml.safe_load((repo / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    noise = SensorConfig.model_validate(copy.deepcopy(boson)).sensor.noise

    bad_map = generate_map(shape, noise, 12345)
    all_off = DefectState(bad=np.zeros(shape, dtype=bool))
    all_on = DefectState(bad=bad_map.stateful_mask.copy())

    static = bad_map.mask_of(DefectKind.DEAD) | bad_map.mask_of(DefectKind.HOT)
    assert np.array_equal(active_defect_mask(bad_map, all_off), static)
    assert np.array_equal(active_defect_mask(bad_map, all_on), static | bad_map.stateful_mask)

    frame = _ramp(shape)
    out = replace_bad_pixels(frame, active_defect_mask(bad_map, all_off))
    # The intermittent pixels were not active, so they came through untouched.
    quiet = bad_map.stateful_mask & ~static
    assert np.array_equal(out[quiet], frame[quiet])
