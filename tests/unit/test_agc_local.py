"""ROI-weighted and locally-adaptive AGC (M9.10, §11.3).

Both variants exist because the global operators have one failure mode between them: **one hot
object sets the stretch for every pixel in the frame**. That is not a defect to be fixed -- a real
core does it and §15 Tier 5 requires the simulator to do it too -- so the global operators stay the
default and these are alternatives a config asks for by name.

Two of the tests here are identities rather than tolerances, and they are what make the additions
safe: uniform weights must reproduce the global operator **bit for bit**, and a single tile must
reproduce it bit for bit as well. An approximation in either would mean the new code path had
quietly become a second, slightly different AGC.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from irsim.config.sensor import IspSpec
from irsim.isp.agc import (
    CONSTANT_FRAME_LEVEL,
    DEFAULT_TILES,
    agc_linear,
    agc_plateau,
    agc_plateau_local,
    histogram_dn,
    plateau_lut,
    tile_bounds,
)
from irsim.isp.display import run_display_branch

SHAPE = (256, 320)
PLATEAU = 0.01


def noisy_frame(seed: int = 5, mean: float = 20000.0, sigma: float = 700.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, sigma, SHAPE), 0, 65535).astype(np.uint16)


def plume_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A clean frame, the same frame with an exhaust plume, and the background mask.

    The plume is a broad Gaussian rather than a saturated block on purpose: a block lands in one
    DN bin, which plateau equalisation clips away, and the global operator then survives it. A
    plume spread over many bins is what actually collapses a global stretch, and is what a real
    exhaust looks like.
    """
    rng = np.random.default_rng(5)
    base = rng.normal(20000.0, 700.0, SHAPE)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    blob = np.exp(-(((yy - 128) / 48.0) ** 2 + ((xx - 160) / 60.0) ** 2))
    clean = np.clip(base, 0, 65535).astype(np.uint16)
    plume = np.clip(base + 45000.0 * blob, 0, 65535).astype(np.uint16)
    return clean, plume, blob < 0.02


# --------------------------------------------------------------------------------------------
# The two identities
# --------------------------------------------------------------------------------------------


def test_uniform_weights_reproduce_the_global_operator_bit_for_bit() -> None:
    """Weighting every pixel by 1.0 must change nothing, in either operator.

    Not "to within a tolerance": the weighted histogram sums exact ones, so every percentile
    search and every CDF comparison lands on the same bin. If this were approximate, a config
    that set uniform weights would be running a subtly different camera.
    """
    frame = noisy_frame()
    ones = np.ones(SHAPE)
    assert np.array_equal(
        agc_linear(frame, 0.02, 0.98), agc_linear(frame, 0.02, 0.98, weights=ones)
    )
    assert np.array_equal(agc_plateau(frame, PLATEAU), agc_plateau(frame, PLATEAU, weights=ones))


def test_a_single_tile_is_the_global_operator_bit_for_bit() -> None:
    """One tile blends one mapping with weight 1.0, and multiplying by one changes no bits."""
    frame = noisy_frame()
    assert np.array_equal(agc_plateau(frame, PLATEAU), agc_plateau_local(frame, PLATEAU, (1, 1)))


def test_scaling_every_weight_changes_nothing() -> None:
    """The histogram is normalised by its own sum, so weights carry a ratio and not a unit."""
    frame = noisy_frame()
    rng = np.random.default_rng(11)
    w = rng.uniform(0.1, 1.0, SHAPE)
    a = agc_plateau(frame, PLATEAU, weights=w)
    b = agc_plateau(frame, PLATEAU, weights=w * 37.0)
    assert np.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------------------------
# ROI weighting
# --------------------------------------------------------------------------------------------


def test_an_roi_moves_the_stretch_onto_the_region_that_matters() -> None:
    """Weighting a dim corner stretches *it*, at the cost of the bright rest of the frame."""
    frame = np.full(SHAPE, 40000, dtype=np.uint16)
    frame[:64, :64] = np.linspace(1000, 1400, 64, dtype=np.uint16)[None, :]
    roi = np.zeros(SHAPE)
    roi[:64, :64] = 1.0

    globally = agc_linear(frame, 0.02, 0.98)
    weighted = agc_linear(frame, 0.02, 0.98, weights=roi)
    corner = (slice(None, 64), slice(None, 64))
    assert float(weighted[corner].std()) > 20.0 * float(globally[corner].std())
    # and the rest of the frame pays for it by clipping to white
    assert float(weighted[128:, 128:].min()) == pytest.approx(1.0)


def test_a_zero_weight_region_does_not_set_the_stretch() -> None:
    """Pixels weighted zero are still displayed -- they just get no vote on how."""
    frame = noisy_frame()
    frame[:32] = 65535  # a saturated band that would drag a global percentile
    w = np.ones(SHAPE)
    w[:32] = 0.0
    weighted = agc_linear(frame, 0.02, 0.98, weights=w)
    assert float(weighted[:32].min()) == pytest.approx(1.0)  # displayed, and clipped white
    assert float(weighted[32:].std()) > float(agc_linear(frame, 0.02, 0.98)[32:].std())


@pytest.mark.parametrize(
    "weights",
    [np.zeros(SHAPE), np.full(SHAPE, -1.0), np.ones((4, 4)), np.full(SHAPE, np.nan)],
)
def test_impossible_weights_are_refused(weights: np.ndarray) -> None:
    with pytest.raises(ValueError):
        agc_plateau(noisy_frame(), PLATEAU, weights=weights)


# --------------------------------------------------------------------------------------------
# The local operator
# --------------------------------------------------------------------------------------------


def test_local_keeps_the_background_readable_under_an_exhaust() -> None:
    """The measurement M9.10 exists for, and the answer to "why not just use the global one".

    Measured on this fixture: a global plateau stretch keeps 61 % of the background's contrast
    once the plume is in frame and a global *linear* stretch keeps 8 % -- the collapse that makes
    an airframe read as one flat white shape. The local operator keeps 93 %.
    """
    clean, plume, bg = plume_frame()
    reference = float(agc_plateau(clean, PLATEAU)[bg].std())
    global_std = float(agc_plateau(plume, PLATEAU)[bg].std())
    local_std = float(agc_plateau_local(plume, PLATEAU, (8, 8))[bg].std())

    assert local_std >= 0.80 * reference
    assert global_std < 0.70 * reference  # the global operator genuinely loses it
    assert local_std > 1.4 * global_std


def test_the_global_linear_collapse_is_the_worst_of_the_three() -> None:
    """Context for the number above: linear percentile clipping fares far worse than plateau."""
    clean, plume, bg = plume_frame()
    ref = float(agc_linear(clean, 0.02, 0.98)[bg].std())
    assert float(agc_linear(plume, 0.02, 0.98)[bg].std()) < 0.15 * ref


def test_blending_removes_the_tile_seam_entirely() -> None:
    """A tile boundary must be no more of a step than any other column.

    The guard is the unblended version built here: assigning each pixel its own tile's mapping and
    nothing else puts a **255-code** jump at the boundaries of a ramp -- a full black-to-white
    step -- where the bilinear blend leaves the boundary indistinguishable from the interior.
    """
    width = SHAPE[1]
    tiles = 8
    ramp = np.broadcast_to(np.linspace(0, 65535, width), SHAPE).astype(np.uint16)
    cols = tile_bounds(width, tiles)
    rows = tile_bounds(SHAPE[0], tiles)

    blocky = np.zeros(SHAPE, dtype=np.float64)
    for i in range(tiles):
        for j in range(tiles):
            patch = ramp[rows[i] : rows[i + 1], cols[j] : cols[j + 1]].astype(np.float64)
            lut = plateau_lut(histogram_dn(patch, 16), PLATEAU)
            blocky[rows[i] : rows[i + 1], cols[j] : cols[j + 1]] = (
                CONSTANT_FRAME_LEVEL if lut is None else lut[np.floor(patch).astype(np.int64)]
            )

    interior = np.setdiff1d(np.arange(1, width), cols[1:-1])

    def jumps(y: np.ndarray) -> tuple[int, int]:
        q = np.round(np.asarray(y) * 255).astype(int)
        d = np.abs(np.diff(q, axis=1))
        return int(d[:, cols[1:-1] - 1].max()), int(d[:, interior - 1].max())

    blended_bound, blended_inner = jumps(agc_plateau_local(ramp, PLATEAU, (tiles, tiles)))
    blocky_bound, blocky_inner = jumps(blocky)

    assert blended_bound <= blended_inner + 2  # M9.10: no seam, within 2 LSB8
    assert blocky_bound > 200 and blocky_inner == blended_inner  # the guard: it could have failed


def test_a_flat_tile_displays_mid_grey_like_a_flat_frame() -> None:
    """A tile with no dynamic range takes the same answer the global operator gives a flat frame."""
    frame = noisy_frame()
    frame[:32, :40] = 30000  # one tile, perfectly uniform
    out = agc_plateau_local(frame, PLATEAU, (8, 8))
    assert np.all(np.isfinite(out))
    # the interior of that tile is pulled towards mid-grey rather than to an arbitrary end
    assert abs(float(out[:16, :20].mean()) - CONSTANT_FRAME_LEVEL) < 0.35


def test_output_stays_in_range_and_float32() -> None:
    out = agc_plateau_local(noisy_frame(), PLATEAU, DEFAULT_TILES)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


@pytest.mark.parametrize("tiles", [(1, 1), (2, 3), (4, 4), (8, 8), (7, 5)])
def test_any_tiling_covers_the_frame(tiles: tuple[int, int]) -> None:
    """Tile counts that do not divide the frame must still tile it: no ragged last row."""
    out = agc_plateau_local(noisy_frame(), PLATEAU, tiles)
    assert out.shape == SHAPE
    assert np.all(np.isfinite(out))
    rows = tile_bounds(SHAPE[0], tiles[0])
    assert rows[0] == 0 and rows[-1] == SHAPE[0]
    assert int(np.diff(rows).max() - np.diff(rows).min()) <= 1


@pytest.mark.parametrize("tiles", [(0, 4), (4, 0), (1000, 4)])
def test_impossible_tilings_are_refused(tiles: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        agc_plateau_local(noisy_frame(), PLATEAU, tiles)


def test_the_local_operator_refuses_float16_like_the_others() -> None:
    with pytest.raises(TypeError):
        agc_plateau_local(np.zeros(SHAPE, dtype=np.float16), PLATEAU)


# --------------------------------------------------------------------------------------------
# Through the config and the display branch
# --------------------------------------------------------------------------------------------


def _isp(mode: str, tiles: tuple[int, int] = (8, 8)) -> IspSpec:
    return IspSpec(
        agc=mode,  # type: ignore[arg-type]
        plateau=PLATEAU,
        clip_percentiles=(0.02, 0.98),
        gamma=1.0,
        dde_gain=0.0,
        polarity="white_hot",
        palette="gray",
        agc_tiles=tiles,
    )


def test_a_scene_can_ask_for_the_local_operator_by_name() -> None:
    """`agc: plateau_local` reaches the display branch, and one tile still means the global one."""
    frame = noisy_frame()
    local = run_display_branch(frame, _isp("plateau_local"), 16)
    one_tile = run_display_branch(frame, _isp("plateau_local", (1, 1)), 16)
    global_ = run_display_branch(frame, _isp("plateau_equalization"), 16)

    assert np.array_equal(one_tile.y, global_.y)
    assert not np.array_equal(local.y, global_.y)
    assert local.display8.shape == global_.display8.shape


def test_the_tiling_is_part_of_the_isp_hash() -> None:
    """Two cameras that tile differently are two cameras, and the hash has to say so."""
    a = run_display_branch(noisy_frame(), _isp("plateau_local", (4, 4)), 16)
    b = run_display_branch(noisy_frame(), _isp("plateau_local", (8, 8)), 16)
    assert a.isp_hash != b.isp_hash


def test_the_global_default_is_unchanged_by_any_of_this() -> None:
    """Adding a mode must not move the camera every existing config already describes."""
    frame = noisy_frame()
    assert np.array_equal(
        run_display_branch(frame, _isp("plateau_equalization"), 16).y,
        agc_plateau(frame, PLATEAU, 16),
    )


@pytest.mark.parametrize("tiles", [(0, 8), (8, -1)])
def test_a_config_with_an_impossible_tiling_is_refused(tiles: tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        _isp("plateau_local", tiles)
