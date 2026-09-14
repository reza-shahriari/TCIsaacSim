"""Flat-region finding and per-pixel temporal noise (ME.2b, ADR 0023).

The finder's job is to refuse: a window with a cloud, a horizon or a drone in it must not be
offered to a noise analyser, and a window with nothing but sensor noise in it -- striping
included -- must be. Every test here is built so that it would fail if the finder confused scene
structure with noise in either direction, which is the only way it can be wrong.

Thresholds are in units of the window's own noise, so each test scales noise and structure
together and asserts the verdict, not a DN value.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import RATIO_ORDER
from irsim.noise import FixedPattern, Sigmas7, synthesize_frame
from irsim.validation import (
    FlatRegion,
    decompose_3d,
    estimate_floors,
    find_flat_regions,
    representative_frame,
    robust_noise_scale,
    temporal_noise,
    temporal_std_map,
)

BOSON_RATIOS = (0.02, 0.08, 0.15, 0.05, 0.05, 0.30, 1.0)
SIZE = 32


def _noise_frame(sigma: float, seed: int = 4, shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    return np.random.default_rng(seed).normal(100.0, sigma, size=shape)


def _covers(region: FlatRegion, rows: slice, cols: slice) -> bool:
    """Whether a region's window intersects a rectangle."""
    r_rows, r_cols = region.slices
    return not (
        r_rows.stop <= rows.start
        or r_rows.start >= rows.stop
        or r_cols.stop <= cols.start
        or r_cols.start >= cols.stop
    )


def test_robust_noise_scale_recovers_sigma_through_a_ramp_and_striping() -> None:
    """The ruler everything else is measured against: a gradient and column noise must not move it.

    A plain standard deviation would read 11 DN on the ramped frame instead of 2. The MAD of the
    first differences about their median cancels the gradient exactly, and taking the smaller of
    the two axes keeps a column pattern (which enters only the horizontal differences) out.
    """
    sigma = 2.0
    frame = _noise_frame(sigma)
    assert robust_noise_scale(frame) == pytest.approx(sigma, rel=0.03)

    ramp = (
        np.linspace(0.0, 40.0, frame.shape[1])[None, :]
        + np.linspace(0.0, 40.0, frame.shape[0])[:, None]
    )
    assert (frame + ramp).std() > 5.0 * sigma  # the ramp dominates a naive estimate
    assert robust_noise_scale(frame + ramp) == pytest.approx(sigma, rel=0.03)

    columns = np.random.default_rng(9).normal(0.0, 1.5, size=frame.shape[1])[None, :]
    assert robust_noise_scale(frame + columns) == pytest.approx(sigma, rel=0.05)


def test_a_frame_of_pure_noise_is_flat_everywhere() -> None:
    regions = find_flat_regions(_noise_frame(2.0), size=SIZE, limit=6)
    assert len(regions) == 6
    for region in regions:
        assert region.structure_ratio < 0.15, region
        assert region.noise == pytest.approx(2.0, rel=0.10)
    for i, first in enumerate(regions):
        for second in regions[i + 1 :]:
            assert not first.overlaps(second)


def test_a_cloud_and_a_horizon_are_refused() -> None:
    """No accepted window may contain a pixel where the scene departs from flat by one sigma."""
    sigma = 2.0
    frame = _noise_frame(sigma)
    rows, cols = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    cloud = 5.0 * sigma * np.exp(-((rows - 30.0) ** 2 + (cols - 40.0) ** 2) / (2.0 * 8.0**2))
    horizon = np.where(rows >= 100, 6.0 * sigma, 0.0)
    scene = cloud + horizon

    regions = find_flat_regions(frame + scene, size=SIZE, limit=8)
    assert regions, "a frame with clear sky in it must still yield windows"
    for region in regions:
        window = scene[region.slices[0], region.slices[1]]
        assert window.max() - window.min() < sigma, (region, window.ptp())


def test_column_striping_is_still_flat() -> None:
    """Striping is the thing being measured; a finder that rejected it would hide column noise.

    The frame carries the Boson's fixed-pattern ratios at a realistic level. A column pattern
    survives block averaging in one direction, so it scores some structure -- the test pins that
    it stays well inside the default threshold, which is what keeps sigma_H measurable.
    """
    sigmas = Sigmas7(**dict(zip(RATIO_ORDER, [1.5 * r for r in BOSON_RATIOS], strict=True)))
    fixed = FixedPattern.generate((128, 128), sigmas, 7)
    frame = synthesize_frame((128, 128), sigmas, fixed, 7, 0) + 100.0

    regions = find_flat_regions(frame, size=SIZE, limit=4)
    assert len(regions) == 4
    assert max(r.structure_ratio for r in regions) < 0.25


def test_a_point_target_takes_its_window_out() -> None:
    """Both a resolved target and a single-pixel one, and the second only the outlier rule can do.

    Four pixels at 20 sigma leave enough excess structure to be refused on that alone. One pixel at
    8 sigma does not -- averaged into a 4x4 block it is a quarter of the block noise -- so it is
    refused only because it sits outside the outlier band, which is exactly the sub-pixel target
    the MS.6 path renders and the case a variance-based finder would wave through.
    """
    sigma = 2.0
    resolved = _noise_frame(sigma)
    resolved[64:66, 70:72] += 20.0 * sigma
    accepted = find_flat_regions(resolved, size=SIZE, limit=16)
    assert accepted, "the rest of the frame is still flat"
    assert not any(_covers(r, slice(64, 66), slice(70, 72)) for r in accepted)

    point = _noise_frame(sigma)
    point[64, 70] += 8.0 * sigma
    assert not any(
        _covers(r, slice(64, 65), slice(70, 71))
        for r in find_flat_regions(point, size=SIZE, limit=16)
    )
    blind = find_flat_regions(point, size=SIZE, limit=16, outlier_sigma=12.0)
    assert any(_covers(r, slice(64, 65), slice(70, 71)) for r in blind)


def test_the_verdict_is_scale_free() -> None:
    """Doubling the DN scale of a clip must not change which windows are usable."""
    sigma = 2.0
    frame = _noise_frame(sigma)
    rows = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]][0]
    frame = frame + 0.05 * sigma * rows

    plain = find_flat_regions(frame, size=SIZE, limit=8)
    scaled = find_flat_regions(4.0 * frame + 50.0, size=SIZE, limit=8)
    assert [(r.row, r.col) for r in plain] == [(r.row, r.col) for r in scaled]
    for a, b in zip(plain, scaled, strict=True):
        assert b.structure_ratio == pytest.approx(a.structure_ratio, rel=1e-9)
        assert b.noise == pytest.approx(4.0 * a.noise, rel=1e-9)


def test_the_temporal_median_deletes_a_moving_target() -> None:
    """A drone crossing the window is in one frame of many, so the median never sees it."""
    rng = np.random.default_rng(2)
    cube = rng.normal(100.0, 2.0, size=(24, 64, 64))
    for index in range(cube.shape[0]):
        col = 2 + 2 * index
        cube[index, 30:33, col : col + 3] += 40.0

    assert representative_frame(cube).max() < 100.0 + 6.0 * 2.0
    on_the_clip = find_flat_regions(cube, size=SIZE, limit=4)
    assert any(_covers(r, slice(30, 33), slice(2, 52)) for r in on_the_clip)
    on_one_frame = find_flat_regions(cube[12], size=SIZE, limit=16)
    assert not any(_covers(r, slice(30, 33), slice(26, 29)) for r in on_one_frame)


def test_per_pixel_temporal_variance_equals_the_temporal_half_of_the_decomposition() -> None:
    """An independent route to the same numbers: mean per-pixel variance == T + TV + TH + TVH.

    The fixed terms (V, H, VH) do not vary with t and must cancel exactly in a per-pixel variance.
    If they leaked in, this would come out high by sigma_VH^2 -- 9 % here -- so the test is a real
    check on the decomposition's operator set, not a restatement of it.
    """
    sigma_tvh = 1.5
    sigmas = Sigmas7(**dict(zip(RATIO_ORDER, [sigma_tvh * r for r in BOSON_RATIOS], strict=True)))
    fixed = FixedPattern.generate((48, 48), sigmas, 3)
    cube = np.stack([synthesize_frame((48, 48), sigmas, fixed, 3, f) for f in range(120)])

    measured = temporal_noise(cube)
    decomposition = decompose_3d(cube)
    temporal = sum(decomposition.raw_variances[name] for name in ("t", "tv", "th", "tvh"))
    floors = estimate_floors(cube.shape, sigmas.as_vector())
    tolerance = 3.0 * float(np.sqrt(sum(floors[name] ** 2 for name in ("t", "tv", "th", "tvh"))))

    assert abs(measured.mean_variance - temporal) <= tolerance
    fixed_variance = sum(decomposition.raw_variances[name] for name in ("v", "h", "vh"))
    assert fixed_variance > 5.0 * tolerance, "the fixed terms must be big enough for this to bite"


def test_a_flickering_pixel_shows_up_in_the_temporal_map() -> None:
    rng = np.random.default_rng(6)
    cube = rng.normal(100.0, 1.0, size=(64, 32, 32))
    cube[:, 10, 20] = 100.0 + rng.normal(0.0, 12.0, size=64)

    per_pixel = temporal_std_map(cube)
    assert per_pixel[10, 20] > 8.0
    summary = temporal_noise(cube)
    assert summary.median == pytest.approx(1.0, rel=0.10)
    assert summary.outlier_fraction == pytest.approx(1.0 / 1024.0, abs=1e-9)


def test_float16_is_refused_everywhere() -> None:
    half = np.zeros((8, 16, 16), dtype=np.float16)
    for call in (
        lambda: representative_frame(half),
        lambda: temporal_std_map(half),
        lambda: robust_noise_scale(half[0]),
    ):
        with pytest.raises(TypeError, match="float16"):
            call()


def test_window_geometry_is_checked() -> None:
    frame = _noise_frame(1.0, shape=(40, 40))
    with pytest.raises(ValueError, match="multiple of smooth"):
        find_flat_regions(frame, size=30, smooth=4)
    with pytest.raises(ValueError, match="does not fit"):
        find_flat_regions(frame, size=64)
