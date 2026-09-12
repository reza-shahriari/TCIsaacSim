"""Bad-pixel map and defect injection (M9.5a): the spatial statistics and the RTS statistics.

§10.4's instruction is "clustered slightly (use a Poisson cluster process, not uniform)", and
clustering is the part that matters downstream: a cluster is what defeats the 4-neighbour
replacement of M9.5b, so a uniform map understates the artefact a perception stack actually sees.
The tests here therefore check the *process*, not the plumbing:

* the realised count matches the configured fraction within 10 %;
* the mean nearest-neighbour distance is far below the uniform-Poisson expectation, with a uniform
  map run through the same estimator as a control, so the assertion is shown to discriminate;
* stuck pixels are bit-identical across 100 frames at the floor and the ceiling;
* the flickering/blinking chain has the configured stationary occupancy and *geometric* dwell,
  which is what makes random telegraph noise RTS rather than a merely noisy pixel;
* the map is deterministic in the sensor seed -- a camera's defects belong to the camera.

docs/physics-model.md §10.4, §11.1. ADR 0055.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml
from scipy import stats
from scipy.spatial import cKDTree

from irsim.config.sensor import SensorConfig
from irsim.noise import (
    BadPixelMap,
    DefectKind,
    DefectState,
    advance_state,
    apply_defects,
    generate_map,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 12345
DN_MAX = 65535


def _noise(**over: Any):  # type: ignore[no-untyped-def]
    d = copy.deepcopy(BOSON)
    d["sensor"]["noise"].update(over)
    return SensorConfig.model_validate(d).sensor.noise


def _mean_nn(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    pts = np.stack([ys, xs], axis=1).astype(np.float64)
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(d[:, 1].mean())


def _uniform_poisson_nn(count: int, shape: tuple[int, int]) -> float:
    """E[nearest-neighbour distance] = 1/(2 sqrt(density)) for a uniform 2-D Poisson process."""
    density = count / (shape[0] * shape[1])
    return 0.5 / np.sqrt(density)


def _ks_statistic(sample: np.ndarray, p: float) -> float:
    """Two-sided KS distance between a sample and Geometric(p) on {1, 2, ...}."""
    x = np.sort(np.asarray(sample))
    n = x.size
    cdf = stats.geom.cdf(x, p)
    d_plus = float(np.max(np.arange(1, n + 1) / n - cdf))
    d_minus = float(np.max(cdf - np.arange(0, n) / n))
    return max(d_plus, d_minus)


def _ks_pvalue_geometric(sample: np.ndarray, p: float, n_mc: int = 399) -> float:
    """KS p-value against Geometric(p), calibrated by Monte Carlo.

    The *analytic* KS p-value assumes a continuous distribution. Applied to integer dwell times it
    reports p ~ 1e-174 for a perfectly geometric sample, because the empirical CDF is a step
    function and the statistic picks up the step at x = 1 (of height p) rather than any
    misfit. Calibrating the null by simulating geometric samples of the same size makes the test
    valid for discrete data -- and it still discriminates, as the companion test below shows.
    """
    observed = _ks_statistic(sample, p)
    rng = np.random.default_rng(20260912)
    null = np.array([_ks_statistic(rng.geometric(p, size=sample.size), p) for _ in range(n_mc)])
    return float((1 + np.count_nonzero(null >= observed)) / (1 + n_mc))


# -- the count -------------------------------------------------------------------------------


def test_count_matches_the_configured_fraction_within_10_percent() -> None:
    """1024^2 at 0.0015 -> 1573 expected. Realised runs under it: colliding offspring merge."""
    shape = (1024, 1024)
    noise = _noise()
    assert noise.bad_pixel_fraction == 0.0015
    got = generate_map(shape, noise, SEED).count
    expected = noise.bad_pixel_fraction * shape[0] * shape[1]
    assert expected == pytest.approx(1573, abs=1)
    assert got == pytest.approx(expected, rel=0.10)
    # The shortfall is the price of clustering, and it is a shortfall, never an excess.
    assert got <= expected


@pytest.mark.parametrize("fraction", [0.0005, 0.001, 0.005])
def test_the_count_scales_with_the_configured_fraction(fraction: float) -> None:
    shape = (512, 512)
    got = generate_map(shape, _noise(bad_pixel_fraction=fraction), SEED).count
    assert got == pytest.approx(fraction * shape[0] * shape[1], rel=0.12)


def test_a_zero_fraction_gives_a_clean_array() -> None:
    m = generate_map((64, 64), _noise(bad_pixel_fraction=0.0), SEED)
    assert m.count == 0 and not m.mask.any()


# -- the clustering --------------------------------------------------------------------------


def test_defects_are_clustered_not_uniform() -> None:
    """Mean NN distance below 0.6x the uniform-Poisson expectation (the roadmap's bar)."""
    shape = (1024, 1024)
    m = generate_map(shape, _noise(), SEED)
    nn = _mean_nn(m.mask)
    expected = _uniform_poisson_nn(m.count, shape)
    assert nn < 0.6 * expected


def test_a_uniform_map_would_fail_the_clustering_test() -> None:
    """The control: the estimator agrees with theory on a genuinely uniform map.

    Without this, the clustering assertion above could be passing because the estimator is wrong
    rather than because the process clusters.
    """
    shape = (1024, 1024)
    count = 1479
    rng = np.random.default_rng(3)
    flat = rng.choice(shape[0] * shape[1], size=count, replace=False)
    mask = np.zeros(shape, dtype=bool)
    mask.reshape(-1)[flat] = True
    nn = _mean_nn(mask)
    expected = _uniform_poisson_nn(count, shape)
    assert nn == pytest.approx(expected, rel=0.10)  # theory holds
    assert nn > 0.6 * expected  # and such a map fails the clustering bar


def test_clusters_of_two_and_more_actually_occur() -> None:
    """The reason clustering matters: adjacent defects are what M9.5b's 4-neighbour stencil must
    iterate over. A uniform map at this density essentially never produces them."""
    m = generate_map((512, 512), _noise(), SEED)
    mask = m.mask
    neighbours = np.zeros_like(mask, dtype=np.int32)
    neighbours[1:, :] += mask[:-1, :]
    neighbours[:-1, :] += mask[1:, :]
    neighbours[:, 1:] += mask[:, :-1]
    neighbours[:, :-1] += mask[:, 1:]
    touching = int(np.count_nonzero(mask & (neighbours > 0)))
    assert touching > 0.10 * m.count


def test_larger_lambda_makes_tighter_clusters() -> None:
    """lambda controls cluster size, so it must move the nearest-neighbour statistic."""
    shape = (512, 512)
    loose = generate_map(shape, _noise(bad_pixel_cluster_lambda=0.2), SEED)
    tight = generate_map(shape, _noise(bad_pixel_cluster_lambda=6.0), SEED)
    assert _mean_nn(tight.mask) < _mean_nn(loose.mask)


# -- the class mix ---------------------------------------------------------------------------


def test_the_type_mix_is_reproduced() -> None:
    m = generate_map((2048, 2048), _noise(), SEED)
    counts = m.counts()
    total = m.count
    mix = _noise().bad_pixel_type_mix
    for kind, want in (
        (DefectKind.DEAD, mix.dead),
        (DefectKind.HOT, mix.hot),
        (DefectKind.FLICKERING, mix.flickering),
        (DefectKind.BLINKING, mix.blinking),
    ):
        assert counts[kind] / total == pytest.approx(want, abs=0.03)


# -- determinism -----------------------------------------------------------------------------


def test_the_map_is_deterministic_per_sensor_seed() -> None:
    a = generate_map((256, 256), _noise(), SEED)
    b = generate_map((256, 256), _noise(), SEED)
    c = generate_map((256, 256), _noise(), SEED + 1)
    assert np.array_equal(a.kind, b.kind)
    assert not np.array_equal(a.kind, c.kind)


# -- injection: the static classes -----------------------------------------------------------


def _scene(shape: tuple[int, int]) -> np.ndarray:
    """A mid-scale ramp, so a stuck pixel is unambiguous against its neighbours."""
    rows, cols = shape
    ramp = np.linspace(8000, 40000, cols, dtype=np.float64)
    return np.tile(ramp, (rows, 1)).astype(np.uint16)


def test_stuck_pixels_are_bit_identical_across_100_frames() -> None:
    """Dead at the floor and hot at the ceiling, every frame, whatever the scene does."""
    shape = (128, 160)
    noise = _noise()
    m = generate_map(shape, noise, SEED)
    state = DefectState.initial(m, noise, SEED)
    dead, hot = m.mask_of(DefectKind.DEAD), m.mask_of(DefectKind.HOT)
    assert dead.any() and hot.any()

    for frame in range(100):
        scene = (_scene(shape).astype(np.int64) + 50 * frame).clip(0, DN_MAX).astype(np.uint16)
        out = apply_defects(scene, m, state, DN_MAX, noise.bad_pixel_rts_amplitude_dn)
        assert np.all(out[dead] == 0)
        assert np.all(out[hot] == DN_MAX)
        state = advance_state(state, m, noise, SEED, frame + 1)


def test_good_pixels_are_untouched_and_the_input_is_not_mutated() -> None:
    shape = (64, 80)
    noise = _noise()
    m = generate_map(shape, noise, SEED)
    state = DefectState.initial(m, noise, SEED)
    scene = _scene(shape)
    before = scene.copy()
    out = apply_defects(scene, m, state, DN_MAX, noise.bad_pixel_rts_amplitude_dn)
    assert np.array_equal(scene, before)
    good = ~m.mask
    assert np.array_equal(out[good], scene[good])


def test_a_flickering_pixel_still_responds_to_the_scene() -> None:
    """The class that survives a map built from one calibration frame: it is offset, not stuck."""
    shape = (96, 96)
    noise = _noise(bad_pixel_type_mix={"dead": 0.0, "hot": 0.0, "flickering": 1.0, "blinking": 0.0})
    m = generate_map(shape, noise, SEED)
    flick = m.mask_of(DefectKind.FLICKERING)
    assert flick.any()
    state = DefectState(bad=np.asarray(flick))  # force every one into its bad state

    low = np.full(shape, 10000, dtype=np.uint16)
    high = np.full(shape, 20000, dtype=np.uint16)
    a = apply_defects(low, m, state, DN_MAX, noise.bad_pixel_rts_amplitude_dn)
    b = apply_defects(high, m, state, DN_MAX, noise.bad_pixel_rts_amplitude_dn)
    # Offset, not pinned: the 10000 DN scene step comes through unchanged.
    assert np.all(b[flick].astype(np.int64) - a[flick].astype(np.int64) == 10000)
    assert np.all(a[flick] != low[flick])


def test_a_blinking_pixel_is_stuck_only_while_bad() -> None:
    shape = (96, 96)
    noise = _noise(bad_pixel_type_mix={"dead": 0.0, "hot": 0.0, "flickering": 0.0, "blinking": 1.0})
    m = generate_map(shape, noise, SEED)
    blink = m.mask_of(DefectKind.BLINKING)
    scene = _scene(shape)

    bad = apply_defects(scene, m, DefectState(bad=np.asarray(blink)), DN_MAX, 0.0)
    good = apply_defects(scene, m, DefectState(bad=np.zeros(shape, dtype=bool)), DN_MAX, 0.0)
    assert np.all(bad[blink] == 0)
    assert np.array_equal(good[blink], scene[blink])


def test_injection_refuses_a_float_plane() -> None:
    """§11.1 puts bad-pixel handling on the raw DN plane; a float plane is the wrong stage."""
    shape = (16, 16)
    noise = _noise()
    m = generate_map(shape, noise, SEED)
    state = DefectState.initial(m, noise, SEED)
    with pytest.raises(TypeError, match="raw DN plane"):
        apply_defects(np.zeros(shape, dtype=np.float32), m, state, DN_MAX, 0.0)  # type: ignore[arg-type]


def test_injection_refuses_a_mismatched_shape() -> None:
    noise = _noise()
    m = generate_map((16, 16), noise, SEED)
    state = DefectState.initial(m, noise, SEED)
    with pytest.raises(ValueError, match="!= bad-pixel map shape"):
        apply_defects(np.zeros((8, 8), dtype=np.uint16), m, state, DN_MAX, 0.0)


# -- the RTS statistics ----------------------------------------------------------------------


def _run_states(n_frames: int, shape: tuple[int, int], noise, seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    m = generate_map(shape, noise, seed)
    stateful = m.stateful_mask
    state = DefectState.initial(m, noise, seed)
    hist = np.zeros((n_frames, int(stateful.sum())), dtype=bool)
    for f in range(n_frames):
        hist[f] = state.bad[stateful]
        state = advance_state(state, m, noise, seed, f + 1)
    return hist


def test_rts_occupancy_matches_the_configuration_within_5_percent() -> None:
    noise = _noise(bad_pixel_rts_occupancy=0.3, bad_pixel_rts_dwell_frames=8.0)
    hist = _run_states(4000, (128, 160), noise, SEED)
    assert float(hist.mean()) == pytest.approx(noise.bad_pixel_rts_occupancy, rel=0.05)


def test_rts_starts_stationary_with_no_burn_in() -> None:
    """The first frame already has the stationary occupancy, so a short clip is not biased."""
    noise = _noise()
    # A large array so the first frame carries enough stateful pixels to measure an occupancy:
    # at 0.0015 with 30 % of defects stateful, 1024^2 gives ~450 and a standard error of ~0.02.
    hist = _run_states(2, (1024, 1024), noise, SEED)
    assert float(hist[0].mean()) == pytest.approx(noise.bad_pixel_rts_occupancy, abs=0.06)


def test_rts_dwell_times_are_geometric() -> None:
    """The defining statistic of random telegraph noise (KS p > 0.01).

    A pixel whose state were redrawn independently every frame would have a mean dwell of about
    1/(1-occupancy) frames and would fail this comfortably; it is the memory in the chain that
    produces the slow winking RTS is recognised by.
    """
    dwell_frames = 8.0
    noise = _noise(bad_pixel_rts_occupancy=0.3, bad_pixel_rts_dwell_frames=dwell_frames)
    hist = _run_states(3000, (512, 512), noise, SEED)

    runs: list[int] = []
    for col in range(hist.shape[1]):
        series = hist[:, col]
        edges = np.flatnonzero(np.diff(series.astype(np.int8)))
        if edges.size < 3:
            continue
        # Complete runs only: the first and last are censored by the window.
        starts = edges[:-1] + 1
        lengths = np.diff(edges)
        runs.extend(
            int(length) for start, length in zip(starts, lengths, strict=True) if series[start]
        )
    assert len(runs) > 500

    p = 1.0 / dwell_frames
    runs_arr = np.asarray(runs)
    assert float(runs_arr.mean()) == pytest.approx(dwell_frames, rel=0.10)
    assert _ks_pvalue_geometric(runs_arr, p) > 0.01


def test_the_dwell_test_rejects_a_memoryless_pixel() -> None:
    """The control: a pixel redrawn independently each frame must fail the dwell test.

    Its runs are geometric with p = 1 - occupancy (mean 1.43 frames at occupancy 0.3), not with
    p = 1/dwell, so this is what the assertion above is actually distinguishing -- the memory in
    the chain, which is what produces the slow winking RTS is recognised by.
    """
    occupancy, dwell_frames = 0.3, 8.0
    rng = np.random.default_rng(5)
    series = rng.random((3000, 120)) < occupancy
    runs: list[int] = []
    for col in range(series.shape[1]):
        s_col = series[:, col]
        edges = np.flatnonzero(np.diff(s_col.astype(np.int8)))
        if edges.size < 3:
            continue
        runs.extend(
            int(length)
            for start, length in zip(edges[:-1] + 1, np.diff(edges), strict=True)
            if s_col[start]
        )
    runs_arr = np.asarray(runs)
    assert float(runs_arr.mean()) < 2.0  # nowhere near 8 frames
    assert _ks_pvalue_geometric(runs_arr, 1.0 / dwell_frames) <= 0.01


def test_an_impossible_occupancy_and_dwell_pair_is_refused() -> None:
    """occupancy/dwell fix the switch probabilities; a pair implying p > 1 is not a chain."""
    noise = _noise(bad_pixel_rts_occupancy=0.95, bad_pixel_rts_dwell_frames=1.02)
    m = generate_map((16, 16), noise, SEED)
    state = DefectState.initial(m, noise, SEED)
    with pytest.raises(ValueError, match="switch probability"):
        advance_state(state, m, noise, SEED, 1)


def test_only_the_stateful_classes_ever_enter_the_bad_state() -> None:
    """Dead and hot pixels are pinned by the map, not by the chain; good pixels never appear."""
    noise = _noise()
    shape = (128, 128)
    m = generate_map(shape, noise, SEED)
    stateful = m.mask_of(DefectKind.FLICKERING) | m.mask_of(DefectKind.BLINKING)
    state = DefectState.initial(m, noise, SEED)
    for f in range(50):
        assert not np.any(state.bad & ~stateful)
        state = advance_state(state, m, noise, SEED, f + 1)


def test_the_map_rejects_an_out_of_range_kind() -> None:
    with pytest.raises(ValueError, match="outside DefectKind"):
        BadPixelMap(kind=np.full((4, 4), 9, dtype=np.uint8))
