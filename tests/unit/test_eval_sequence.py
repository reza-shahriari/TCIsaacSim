"""The sequence reader and the static-clip gate (ME.1b).

Two things are under test and they fail in different ways. The reader fails loudly -- a missing
frame or a widened dtype raises. The **classifier** fails silently, and that is why most of this
file is about it: a panned clip that gets called static produces a per-pixel temporal standard
deviation that is larger than the sensor's, perfectly smooth, and indistinguishable from a noisier
camera. The two controls the roadmap names -- a translated clip and a jittered one -- are the
whole point, so they are built with the *same per-frame step size* and must come out opposite.

docs/physics-model.md §10.2, §15 T4; ADR 0003.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim_eval.data import Box, Sequence, read_sequence, write_sequence
from irsim_eval.motion import (
    STATIC_DISPLACEMENT_PX,
    classify_clip,
    cumulative_shifts,
    estimate_shift,
)

SIZE = 64
FRAMES = 24
#: The pan's step, and roughly the jitter's too: what separates the controls is accumulation,
#: not amplitude.
STEP_PX = 0.35
#: Per-frame jitter sigma. Chosen so that 3 sigma stays under a pixel -- a mount that wobbles by
#: more than a whole pixel is not "static" in the sense that matters, since scene content has then
#: left the detector pixel it started in and a per-pixel temporal statistic stops being about that
#: pixel. Its frame-to-frame step (2 sigma in 2-D) is comparable to the pan's, which is the point.
JITTER_PX = 0.22


def textured_frame(rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """A frame with structure at several scales, so a correlation peak is well defined.

    Smooth rather than white: phase correlation on pure noise is trivially easy and would not
    resemble sky, which is what these clips are mostly made of.
    """
    coarse = rng.normal(size=(size // 8, size // 8))
    upsampled = np.kron(coarse, np.ones((8, 8)))
    fine = 0.25 * rng.normal(size=(size, size))
    field = upsampled + fine
    field -= field.min()
    field /= field.max() + 1e-12
    return (40.0 + 170.0 * field).astype(np.uint8)


def shifted(base: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel translation by Fourier shift; these frames wrap, so it is exact."""
    rows, cols = base.shape
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.rfftfreq(cols)[None, :]
    spectrum = np.fft.rfft2(base.astype(np.float64))
    phase = np.exp(-2j * np.pi * (kx * dx + ky * dy))
    out = np.fft.irfft2(spectrum * phase, s=(rows, cols))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260913)


@pytest.fixture
def translated_clip() -> np.ndarray:
    """A slow pan: every step the same direction, so displacement accumulates.

    Seeded independently of the other fixtures. Sharing one generator would make each clip depend
    on which tests asked for it first, so a test that requested both would silently get different
    data from a test that requested one -- which is exactly how this pair stopped comparing like
    with like the first time.
    """
    base = textured_frame(np.random.default_rng(20260913))
    return np.stack([shifted(base, STEP_PX * i, 0.0) for i in range(FRAMES)])


@pytest.fixture
def jittered_clip() -> np.ndarray:
    """A shaken mount: comparable steps, zero mean, going nowhere."""
    generator = np.random.default_rng(20260913)
    base = textured_frame(generator)
    offsets = generator.normal(scale=JITTER_PX, size=(FRAMES, 2))
    offsets -= offsets.mean(axis=0)  # zero-mean by construction: no net drift
    offsets[0] = 0.0
    return np.stack([shifted(base, float(dx), float(dy)) for dx, dy in offsets])


# --- the reader ---------------------------------------------------------------------------------


def test_a_synthetic_sequence_round_trips_through_disk(
    tmp_path: pathlib.Path, rng: np.random.Generator
) -> None:
    """Frames, boxes and per-frame attributes all come back exactly as they went in."""
    images = np.stack([textured_frame(rng) for _ in range(5)])
    boxes = [(Box(10.0, 12.0, 4.0, 3.0, "drone"),) for _ in range(5)]
    attributes = [{"occluded": i % 2 == 0, "range_m": 100.0 + i} for i in range(5)]
    original = Sequence.from_arrays(
        images, name="clip_0001", boxes=boxes, attributes=attributes, source_dataset="synthetic"
    )

    write_sequence(tmp_path / "clip", original)
    reloaded = read_sequence(tmp_path / "clip")

    assert reloaded.name == "clip_0001"
    assert reloaded.source_dataset == "synthetic"
    assert reloaded.shape == (SIZE, SIZE)
    assert len(reloaded) == 5
    assert np.array_equal(reloaded.images(), images)
    for i, frame in enumerate(reloaded):
        assert frame.index == i
        assert frame.image.dtype == np.uint8
        assert frame.boxes == boxes[i]
        assert frame.attributes["range_m"] == 100.0 + i
        assert frame.attributes["occluded"] is (i % 2 == 0)


def test_boxes_are_readable_without_decoding_any_frame(
    tmp_path: pathlib.Path, rng: np.random.Generator
) -> None:
    """A size histogram over a whole set must not pay for the pixels it never looks at."""
    images = np.stack([textured_frame(rng) for _ in range(3)])
    boxes = [(Box(1.0, 2.0, 3.0, 4.0, "drone"),) for _ in range(3)]
    write_sequence(tmp_path / "clip", Sequence.from_arrays(images, boxes=boxes))
    reloaded = read_sequence(tmp_path / "clip")
    for i in range(3):
        assert reloaded.boxes_at(i)[0].label == "drone"


def test_the_box_convention_is_the_project_convention() -> None:
    """Top-left plus size; a box on exactly the first pixel has its centre at (0.5, 0.5).

    Pinned because half a pixel of disagreement here is invisible in a picture and changes every
    size-versus-range number derived from these annotations.
    """
    box = Box(0.0, 0.0, 1.0, 1.0)
    assert box.centre == (0.5, 0.5)
    assert box.area_px == 1.0
    assert box.extent_px == 1.0
    assert Box(4.0, 6.0, 8.0, 2.0).centre == (8.0, 7.0)
    with pytest.raises(ValueError, match="positive size"):
        Box(0.0, 0.0, 0.0, 3.0)


def test_frames_that_are_not_eight_bit_are_refused(rng: np.random.Generator) -> None:
    """These sets are 8-bit; a widened dtype would hide the floor under every statistic."""
    images = np.stack([textured_frame(rng) for _ in range(2)]).astype(np.float32)
    with pytest.raises(TypeError, match="8-bit"):
        Sequence.from_arrays(images)


def test_a_frame_whose_shape_contradicts_the_index_is_an_error(
    tmp_path: pathlib.Path, rng: np.random.Generator
) -> None:
    """Silently reshaping would misaddress every pixel a later analyser touched."""
    images = np.stack([textured_frame(rng) for _ in range(2)])
    write_sequence(tmp_path / "clip", Sequence.from_arrays(images))
    np.save(tmp_path / "clip" / "frames" / "000001.npy", np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="the index says"):
        read_sequence(tmp_path / "clip")[1]


def test_an_unknown_schema_version_is_refused(
    tmp_path: pathlib.Path, rng: np.random.Generator
) -> None:
    images = np.stack([textured_frame(rng) for _ in range(2)])
    write_sequence(tmp_path / "clip", Sequence.from_arrays(images))
    index = tmp_path / "clip" / "sequence.json"
    data = json.loads(index.read_text())
    data["schema_version"] = 99
    index.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version"):
        read_sequence(tmp_path / "clip")


def test_frames_are_zero_padded_so_a_directory_sorts_in_order(
    tmp_path: pathlib.Path, rng: np.random.Generator
) -> None:
    """frame 9 before frame 10 in every tool that lists a directory."""
    images = np.stack([textured_frame(rng) for _ in range(12)])
    write_sequence(tmp_path / "clip", Sequence.from_arrays(images))
    names = sorted(p.name for p in (tmp_path / "clip" / "frames").iterdir())
    assert names == [f"{i:06d}.npy" for i in range(12)]


# --- the static-clip gate -------------------------------------------------------------------


def test_phase_correlation_recovers_a_known_sub_pixel_shift(rng: np.random.Generator) -> None:
    """The estimator itself, against shifts it was given, to a tenth of a pixel."""
    base = textured_frame(rng)
    for dx, dy in [(0.0, 0.0), (2.0, 0.0), (0.0, -3.0), (1.5, 2.5), (-0.4, 0.6)]:
        measured = estimate_shift(base, shifted(base, dx, dy))
        assert measured[0] == pytest.approx(dx, abs=0.1), f"dx for ({dx}, {dy})"
        assert measured[1] == pytest.approx(dy, abs=0.1), f"dy for ({dx}, {dy})"


def test_a_translated_clip_is_flagged_as_moving(translated_clip: np.ndarray) -> None:
    """The roadmap's first control. 0.35 px a frame for 24 frames is 8 px of pan."""
    verdict = classify_clip(translated_clip)
    assert not verdict.is_static
    assert verdict.max_displacement_px == pytest.approx(STEP_PX * (FRAMES - 1), abs=0.3)
    assert "measure the scene crossing the detector" in verdict.reason


def test_a_jittered_clip_is_flagged_as_static(jittered_clip: np.ndarray) -> None:
    """The roadmap's second control: the same step size, but it goes nowhere."""
    verdict = classify_clip(jittered_clip)
    assert verdict.is_static, verdict.reason
    assert verdict.max_displacement_px < STATIC_DISPLACEMENT_PX
    assert "describe the sensor" in verdict.reason


def test_the_two_controls_take_the_same_size_steps(
    translated_clip: np.ndarray, jittered_clip: np.ndarray
) -> None:
    """Without this the pair proves nothing: it would just be 'big motion versus small motion'.

    Both clips move by about the same amount between consecutive frames. What separates them is
    only whether those steps accumulate, which is exactly the distinction the classifier is for
    and the one a per-frame motion magnitude cannot make.
    """
    moving = classify_clip(translated_clip)
    still = classify_clip(jittered_clip)
    assert still.rms_step_px == pytest.approx(moving.rms_step_px, rel=2.0)
    assert still.rms_step_px > 0.2, "the jittered control must actually be moving frame to frame"
    assert still.max_displacement_px < STATIC_DISPLACEMENT_PX
    assert moving.max_displacement_px > 10.0 * still.max_displacement_px


def test_drift_ratio_separates_a_pan_from_a_shake(
    translated_clip: np.ndarray, jittered_clip: np.ndarray
) -> None:
    """The number that explains a verdict: near 1 the steps agree, near 0 they cancel."""
    assert classify_clip(translated_clip).drift_ratio > 5.0
    assert classify_clip(jittered_clip).drift_ratio < 2.0


def test_a_still_clip_with_only_noise_is_static(rng: np.random.Generator) -> None:
    """The degenerate case that matters most: a tripod. Noise must not read as motion."""
    base = textured_frame(rng)
    clip = np.stack(
        [
            np.clip(base.astype(np.int16) + rng.integers(-3, 4, base.shape), 0, 255).astype(
                np.uint8
            )
            for _ in range(FRAMES)
        ]
    )
    verdict = classify_clip(clip)
    assert verdict.is_static
    assert verdict.max_displacement_px < 0.5


def test_displacement_is_measured_from_the_first_frame_not_accumulated(
    jittered_clip: np.ndarray,
) -> None:
    """Summing per-frame estimates would sum their errors and random-walk past the threshold.

    Measured against frame zero, a long jittery clip stays bounded however long it runs.
    """
    shifts = cumulative_shifts(jittered_clip)
    assert shifts.shape == (FRAMES, 2)
    assert np.array_equal(shifts[0], [0.0, 0.0])
    assert float(np.hypot(shifts[:, 0], shifts[:, 1]).max()) < STATIC_DISPLACEMENT_PX


def test_the_threshold_is_the_detector_pixel(translated_clip: np.ndarray) -> None:
    """One pixel is the line because below it scene content has not left its own pixel."""
    assert STATIC_DISPLACEMENT_PX == 1.0
    generous = classify_clip(translated_clip, max_displacement_px=100.0)
    assert generous.is_static, "the threshold is what decides, and it is a parameter"


def test_a_clip_too_short_to_have_motion_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        cumulative_shifts(np.zeros((1, SIZE, SIZE), np.uint8))
