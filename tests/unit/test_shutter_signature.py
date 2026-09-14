"""Reading the shutter off the frames (ME.3a, ADR 0068).

Two things a public 8-bit clip can still say about the sensor: when its shutter closed, and what
grew between those events. Each test here is built so that the obvious wrong implementation fails
it -- a freeze detector that counts repeated frames without asking whether the fixed pattern
changed calls a stalled recorder an FFC, and a growth fit that puts the factor of two in the
amplitude instead of the variance reports twice the true correlation time and looks fine.

The generators are the project's own `FfcController` and `ou_step`, written for the physics and
not for this test.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import NucSpec
from irsim.isp.ffc import FfcController
from irsim.noise.drift import ou_step
from irsim.validation import (
    MIN_INTERVAL_CLIP_S,
    find_freezes,
    fit_pattern_growth,
    freeze_intervals_s,
    line_means,
    pattern_energy,
)
from irsim_eval.transcode import h264_round_trip
from irsim_eval.video import ffmpeg_available

FPS = 60.0
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")


def _nuc(interval_s: float = 180.0, freeze_ms: float = 700.0) -> NucSpec:
    return NucSpec(
        mode="shuttered",
        ffc_interval_s=interval_s,
        ffc_freeze_ms=freeze_ms,
        residual_gain_ppm_per_k=0.0,
        residual_offset_mk_per_k=0.0,
    )


def _held_clip(
    n_frames: int,
    *,
    interval_s: float = 180.0,
    shape: tuple[int, int] = (12, 12),
    new_pattern_at_ffc: bool = True,
    temporal_sigma: float = 1.0,
    seed: int = 3,
) -> tuple[np.ndarray, list[int]]:
    """A clip whose freezes follow the real controller's schedule, stamped with NumPy.

    The controller decides *when* (``fires_on``) and how long (``freeze_frames``); the frames are
    then stamped directly rather than pushed through ``process`` one at a time, because the Boson's
    schedule needs 21 600 frames to show two events and a per-frame Python loop over that is
    seconds of unit test for no extra information. ``test_the_controller_and_the_detector_agree``
    closes that gap on a short clip by running the real ``process`` loop.
    """
    controller = FfcController(nuc=_nuc(interval_s), fps=FPS)
    rng = np.random.default_rng(seed)
    pattern = rng.normal(0.0, 3.0, size=shape)
    cube = 120.0 + rng.normal(0.0, temporal_sigma, size=(n_frames, *shape))
    cube += pattern
    fires = [i for i in range(n_frames) if controller.fires_on(i)]
    for first in fires:
        if new_pattern_at_ffc:
            fresh = rng.normal(0.0, 3.0, size=shape)
            cube[first + controller.freeze_frames :] += fresh - pattern
            pattern = fresh
        cube[first : first + controller.freeze_frames] = cube[first - 1]
    return cube, fires


def test_the_boson_schedule_is_recovered_exactly() -> None:
    """42-frame freezes every 10 800 frames: the numbers M9.7 pins, read back off the video."""
    n_frames = 21_700
    cube, fires = _held_clip(n_frames)
    assert fires == [10_800, 21_600]

    freezes = find_freezes(cube, fps=FPS)
    assert [f.first_frame for f in freezes] == fires
    assert [f.frames for f in freezes] == [42, 42]
    assert freezes[0].duration_s == pytest.approx(0.7, abs=1e-9)
    assert all(f.looks_like_ffc for f in freezes)

    intervals = freeze_intervals_s(freezes, fps=FPS, n_frames=n_frames)
    assert intervals.tolist() == [180.0]


def test_a_stalled_recorder_is_not_a_shutter() -> None:
    """The discriminator the row exists for: repeated frames alone are not an FFC.

    Same freezes, same lengths, but the fixed pattern continues unchanged across them -- a dropped
    chunk of recording rather than a shutter. A detector that only counts repeated frames reports
    an FFC here, and every interval it feeds into ME.5 is then a property of the disk.
    """
    cube, fires = _held_clip(1500, interval_s=6.0, new_pattern_at_ffc=False)
    stall = find_freezes(cube, fps=FPS)
    assert [f.frames for f in stall] == [42] * len(fires)
    assert max(f.pattern_change for f in stall) < 2.5
    assert not any(f.looks_like_ffc for f in stall)

    shutter = find_freezes(_held_clip(1500, interval_s=6.0)[0], fps=FPS)
    assert min(f.pattern_change for f in shutter) > 5.0
    assert all(f.looks_like_ffc for f in shutter)


def test_the_controller_and_the_detector_agree_frame_for_frame() -> None:
    """The real `process` loop in the loop, so the stamped clips above are not marking their own
    homework: what the camera emits is what the detector reads back.

    The clip runs past the last freeze on purpose: a run that is still going when the video ends is
    reported at the length seen, and a test that ended mid-freeze would be asserting the clip
    length rather than the camera's.
    """
    controller = FfcController(nuc=_nuc(2.0), fps=FPS)
    rng = np.random.default_rng(8)
    emitted = []
    pattern = rng.normal(0.0, 3.0, size=(16, 16))
    for index in range(450):
        if controller.fires_on(index):
            pattern = rng.normal(0.0, 3.0, size=(16, 16))
        frame = rng.normal(120.0, 1.0, size=(16, 16)) + pattern
        emitted.append(controller.process(frame, index)[0])

    freezes = find_freezes(np.stack(emitted), fps=FPS)
    assert [f.first_frame for f in freezes] == [120, 240, 360]
    assert [f.frames for f in freezes] == [controller.freeze_frames] * 3


def test_a_ten_second_clip_may_not_report_an_interval() -> None:
    """Halmstad's clips are 10 s and the Boson's schedule is 180 s (ME.1a's index says both)."""
    cube, _ = _held_clip(600, interval_s=3.0)
    freezes = find_freezes(cube, fps=FPS)
    assert len(freezes) >= 2, "the freezes themselves are perfectly measurable"
    with pytest.raises(ValueError, match="cannot show an FFC interval"):
        freeze_intervals_s(freezes, fps=FPS, n_frames=600)
    assert MIN_INTERVAL_CLIP_S == 180.0


@needs_ffmpeg
@pytest.mark.slow
def test_freeze_length_survives_a_codec_round_trip_and_refuses_when_it_cannot() -> None:
    """Both halves of what a codec does to this statistic.

    A held frame costs an encoder almost nothing, so with the noise a real core has (4 codes here)
    the runs survive **exactly** through CRF 18 -- start frame and length both. But the still test
    is a ratio to the clip's own median gap, and a codec that has removed the noise leaves no
    median gap to compare against: at 1 code and CRF 23 x264 flattens the whole clip and the
    detector refuses, where a detector using a fixed threshold would report one 600-frame freeze.
    At CRF 18 the same quiet clip is worse than useless -- it reports a freeze at frame 288 that
    the camera never had. This is ME.2b's codec floor again, for a different statistic.
    """
    codes = np.clip(
        np.rint(_held_clip(600, interval_s=3.0, shape=(64, 64), temporal_sigma=4.0)[0]), 0, 255
    )
    noisy = codes.astype(np.uint8)
    freezes = find_freezes(h264_round_trip(noisy, fps=FPS, crf=18), fps=FPS)
    assert [(f.first_frame, f.frames) for f in freezes] == [(180, 42), (360, 42), (540, 42)]

    quiet = np.clip(
        np.rint(_held_clip(600, interval_s=3.0, shape=(64, 64), temporal_sigma=1.0)[0]), 0, 255
    ).astype(np.uint8)
    with pytest.raises(ValueError, match="does not change from frame to frame"):
        find_freezes(h264_round_trip(quiet, fps=FPS, crf=23), fps=FPS)


def test_the_residual_grows_from_zero_and_the_factor_of_two_is_in_the_variance() -> None:
    """An OU column pattern recalibrated to zero at each shutter, recovered to better than 15 %.

    This is the fit that is easy to get wrong by a factor of two: the pattern's *amplitude*
    approaches its stationary value as 1 - e^(-t/tau) and its *variance* as 1 - e^(-2t/tau). The
    test asserts the true tau and explicitly rejects 2 tau, which is what fitting the variance with
    the amplitude's law returns.
    """
    tau_s, fps, columns = 30.0, 6.0, 64
    n_frames = int(300.0 * fps)  # a five-minute sequence
    resets = [0, int(180.0 * fps)]
    rng = np.random.default_rng(4)

    series = np.zeros((n_frames, columns))
    state = np.zeros(columns, dtype=np.float32)
    for index in range(n_frames):
        if index in resets:
            state = np.zeros(columns, dtype=np.float32)  # the shutter recalibrates the residual
        else:
            state = ou_step(state, sigma=2.0, dt_s=1.0 / fps, tau_s=tau_s, rng=rng)
        series[index] = state + rng.normal(0.0, 0.3, size=columns)

    growth = fit_pattern_growth(pattern_energy(series), fps=fps, resets=resets)
    assert growth.grows
    assert growth.n_segments == 2
    assert growth.tau_s == pytest.approx(tau_s, rel=0.15)
    assert abs(growth.tau_s - 2.0 * tau_s) > 0.3 * tau_s, "the factor of two is in the variance"
    assert growth.sigma_inf == pytest.approx(2.0, rel=0.15)
    assert growth.floor == pytest.approx(0.3, rel=0.4)


def test_a_pattern_that_never_recalibrates_shows_no_growth() -> None:
    """The control: a stationary pattern with no shutter has a floor and nothing to fit."""
    rng = np.random.default_rng(5)
    fixed = rng.normal(0.0, 2.0, size=64)
    series = fixed[None, :] + rng.normal(0.0, 0.3, size=(600, 64))
    growth = fit_pattern_growth(pattern_energy(series), fps=6.0, resets=[0])
    assert growth.sigma_inf < 0.2 * growth.floor
    assert growth.floor == pytest.approx(2.0, rel=0.15)


def test_line_means_average_the_white_noise_down_and_keep_the_stripes() -> None:
    rng = np.random.default_rng(7)
    white = rng.normal(120.0, 4.0, size=(40, 64, 64))
    lines = line_means(white)
    assert pattern_energy(lines.column).mean() == pytest.approx(4.0 / np.sqrt(64), rel=0.10)

    stripes = white + rng.normal(0.0, 3.0, size=64)[None, None, :]
    assert pattern_energy(line_means(stripes).column).mean() == pytest.approx(3.0, rel=0.15)
    assert pattern_energy(line_means(stripes).row).mean() == pytest.approx(0.5, rel=0.15)


def test_inputs_are_checked() -> None:
    cube = np.zeros((8, 4, 4), dtype=np.float16)
    with pytest.raises(TypeError, match="float16"):
        find_freezes(cube, fps=FPS)
    with pytest.raises(ValueError, match="does not change from frame to frame"):
        find_freezes(np.ones((8, 4, 4)), fps=FPS)
    with pytest.raises(ValueError, match="none were given"):
        fit_pattern_growth(np.ones(10), fps=FPS, resets=[])
