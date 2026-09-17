"""The floor an 8-bit, lossily-coded public set puts under every noise statistic (ME.2b, ADR 0023).

The negative control the roadmap asks for is the first test: a cube whose true temporal sigma is
0.3 of one code must either be recovered within the stated bias or be flagged codec-limited. It is
both -- Sheppard's correction lands within 0.5 % and the flag fires anyway, because at 1.04 times
the floor the correction happening to work is not evidence that it can be trusted.

The codec round trips need ffmpeg and are skipped without it. They are the calibration behind
ADR 0023's claim that a lossy set's noise statistic is a lower bound: the lossless case proves the
harness itself is transparent, and the lossy case shows a component that was measurable before the
encoder is below the floor after it.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import RATIO_ORDER
from irsim.noise import FixedPattern, Sigmas7, synthesize_frame
from irsim.validation import (
    blockiness,
    codec_floor,
    decompose_3d,
    flag_codec_limited,
    quantisation_floor,
    quantiser_step,
    sheppard_correct,
)
from irsim_eval.transcode import h264_round_trip
from irsim_eval.video import ffmpeg_available

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

BOSON_RATIOS = (0.02, 0.08, 0.15, 0.05, 0.05, 0.30, 1.0)
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")


def _dithered_cube(sigma: float, frames: int = 300, seed: int = 17) -> np.ndarray:
    """Gaussian noise on sub-code offsets, rounded to 8-bit codes -- one pure TVH component.

    The offsets span exactly one code, so the quantiser is dithered (which is the condition
    Sheppard's correction needs) without adding a fixed pattern of its own that would show up in
    the decomposition as something to flag.
    """
    rng = np.random.default_rng(seed)
    base = 120.0 + rng.uniform(0.0, 1.0, size=(48, 48))
    return np.rint(base[None] + rng.normal(0.0, sigma, size=(frames, 48, 48))).astype(np.uint8)


def _boson_cube(
    sigma_tvh: float, frames: int = 150, shape: tuple[int, int] = (64, 64)
) -> np.ndarray:
    sigmas = Sigmas7(**dict(zip(RATIO_ORDER, [sigma_tvh * r for r in BOSON_RATIOS], strict=True)))
    fixed = FixedPattern.generate(shape, sigmas, 11)
    cube = np.stack([synthesize_frame(shape, sigmas, fixed, 11, f) for f in range(frames)]) + 120.0
    return np.clip(np.rint(cube), 0, 255).astype(np.uint8)


def test_a_sub_code_sigma_is_corrected_and_flagged() -> None:
    """The roadmap's quantised negative control, both halves of it."""
    cube = _dithered_cube(0.3)
    floor = codec_floor(cube)
    flagged = flag_codec_limited(decompose_3d(cube), floor)

    assert floor.step == 1
    assert floor.sigma_floor == pytest.approx(0.2887, abs=1e-4)
    # raw: the quantiser adds its own 1/12 of a code squared, so 0.3 reads ~0.41 -- +38 %
    assert flagged.decomposition.tvh == pytest.approx(0.4155, abs=0.02)
    assert flagged.corrected["tvh"] == pytest.approx(0.3, rel=0.05)
    assert flagged.limited["tvh"], "0.3 codes is 1.04 floors; it may not be reported as measured"


def test_a_sigma_well_above_the_floor_is_measured_not_flagged() -> None:
    cube = _dithered_cube(1.0)
    flagged = flag_codec_limited(decompose_3d(cube), codec_floor(cube))
    assert flagged.corrected["tvh"] == pytest.approx(1.0, rel=0.01)
    assert not flagged.limited["tvh"]
    assert flagged.measurable() == ("tvh",)


def test_sheppard_is_the_stated_bias_and_says_so_where_it_fails() -> None:
    """The correction is exact arithmetic on the variance; its *validity* is the measured part."""
    assert sheppard_correct(np.hypot(2.0, quantisation_floor(1.0)), 1.0) == pytest.approx(2.0)
    assert sheppard_correct(0.2, 1.0) == 0.0  # below the floor there is nothing left to recover
    assert quantisation_floor(4.0) == pytest.approx(4.0 / np.sqrt(12.0))
    with pytest.raises(ValueError, match="positive"):
        quantisation_floor(0.0)


def test_the_lattice_step_is_read_off_the_data() -> None:
    rng = np.random.default_rng(5)
    codes = np.rint(rng.normal(128.0, 6.0, size=(20, 32, 32)))
    assert quantiser_step(codes) == 1
    for step in (3, 4, 7):
        assert quantiser_step(step * np.rint(codes / step)) == step
    assert quantiser_step(np.full((4, 8, 8), 42.0)) == 1  # undetermined, not "any step fits"
    assert codec_floor(np.full((4, 8, 8), 42, dtype=np.uint8)).determinate is False
    with pytest.raises(ValueError, match="integer lattice"):
        quantiser_step(np.array([1.0, 1.5, 2.0]))


def test_blockiness_sees_an_eight_pixel_grid_and_nothing_else() -> None:
    """A block-coded frame, an uncoded one, and a smooth texture that is not on the grid."""
    rng = np.random.default_rng(0)
    flat = rng.normal(120.0, 1.5, size=(40, 96, 96))
    assert blockiness(flat).z < 3.0
    assert not blockiness(flat).blocking

    tile = np.indices((12, 12)).sum(axis=0) % 2
    blocks = np.repeat(np.repeat(np.where(tile, 1.0, -1.0), 8, 0), 8, 1)
    coded = blockiness(flat + blocks[None])
    assert coded.blocking and coded.z > 15.0
    assert coded.ratio_h > 1.0 and coded.ratio_v > 1.0

    wave = 3.0 * np.sin(2.0 * np.pi * np.arange(96) / 13.0)
    assert not blockiness(flat + wave[None, None, :]).blocking


def test_eight_bits_makes_the_small_boson_terms_unmeasurable() -> None:
    """What a 3-D decomposition of a public clip is *allowed* to claim, at two noise levels.

    At sigma_TVH = 1.5 codes -- a plausible stretch for a Y16 stream mapped into 8 bits -- every
    component except the temporal white one sits within two floors of 0.289 codes, so the striping
    ratios this project cares about cannot be read off such a clip at all. Four times noisier and
    the column and fixed terms come back. The flag has to track the data, not the dataset.
    """
    quiet = _boson_cube(1.5)
    assert flag_codec_limited(decompose_3d(quiet), codec_floor(quiet)).measurable() == ("tvh",)

    loud = _boson_cube(6.0)
    assert flag_codec_limited(decompose_3d(loud), codec_floor(loud)).measurable() == (
        "h",
        "vh",
        "tvh",
    )


@needs_ffmpeg
def test_a_lossless_round_trip_is_bit_exact() -> None:
    """Without the full-range flags this drifts by a code, and every lossy number below would be
    measuring the harness instead of the encoder."""
    cube = _boson_cube(6.0, frames=24, shape=(64, 64))
    assert np.array_equal(h264_round_trip(cube, fps=60.0, crf=0), cube)


@needs_ffmpeg
@pytest.mark.slow
def test_a_lossy_stream_moves_a_measured_component_below_the_floor() -> None:
    """The calibration behind "a noise statistic from a lossy set is a lower bound".

    CRF 18 is a *high quality* setting, and it still removes 96 % of the temporal noise from a
    flat-sky cube at a plausible 1.5-code level: sensor noise on flat sky is expensive to code and
    invisible to a viewer, which is exactly what x264 was built to exploit. A cube whose TVH term
    was measurable before the encoder has no measurable component at all after it. Asserted as
    factors rather than values so it holds across x264 versions and presets.
    """
    cube = _boson_cube(1.5, frames=48, shape=(64, 64))
    before = flag_codec_limited(decompose_3d(cube), codec_floor(cube))
    assert not before.limited["tvh"]

    coded = h264_round_trip(cube, fps=60.0, crf=18)
    after = flag_codec_limited(decompose_3d(coded), codec_floor(coded))
    assert after.decomposition.tvh < 0.25 * before.decomposition.tvh
    assert after.measurable() == (), "the codec set this floor, not the sensor"
    assert after.floor.levels_used < 8, "and it flattened the clip to a handful of codes"


def test_the_round_trip_harness_refuses_what_it_cannot_do() -> None:
    cube = np.zeros((4, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="exactly one"):
        h264_round_trip(cube, crf=18, bitrate_kbps=500.0)
    with pytest.raises(ValueError, match="even dimensions"):
        h264_round_trip(np.zeros((4, 9, 8), dtype=np.uint8), crf=18)
    with pytest.raises(TypeError, match="8-bit"):
        h264_round_trip(cube.astype(np.float32), crf=18)
