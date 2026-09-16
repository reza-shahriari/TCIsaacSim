"""IG.5 — the motion AOV does not reach the G-buffer, and its convention is never guessed.

Three places in this repository say the motion plane is omitted on this build. `gbuffer_isaac`'s
own channel table says "no motion AOV transports motion"; `irsim.optics.motion` opens by saying the
`motion_px` plane "had to be left empty", which is the reason the analytic tracker exists at all;
and ADR 0014's addendum has the measurement — `motion_vectors` sitting at a ~6e-5 floor after a
**180 px** displacement.

The code delivered it anyway. `_reject_reason` applies its all-zero test only to *required*
channels, and 6e-5 is not zero in any case, so the plane reached `RawAovs.motion`, was scaled by a
**defaulted** `motion_convention="pixels"`, and arrived in the G-buffer where `irsim.optics.stage`
ran the smear path on it.

Numerically that is a no-op. The hazard is the convention: nothing has ever checked the sign,
Replicator's documentation gives both signs opposite to this project's contract, and a plane that
is noise today is a plane that is backwards the day a build starts filling it in. So the channel is
not attached, and the conversion refuses to guess.

docs/physics-model.md §13.3, §9.2; ADR 0014 addendum; roadmap IG.5, IG.6.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim_isaac.pipeline.gbuffer_isaac import (
    AOV_NAMES,
    UNVERIFIED_CHANNELS,
    AovReader,
    RawAovs,
    geometry_planes,
    motion_px_per_frame,
)

CAMERA_POSITION = (0.0, 0.0, 0.0)


def _aovs(shape=(8, 8), *, motion: bool) -> RawAovs:
    distance = np.full(shape, 12.0, dtype=np.float32)
    normal = np.zeros((*shape, 3), dtype=np.float32)
    normal[:, :, 1] = 1.0
    position = np.zeros((*shape, 3), dtype=np.float32)
    position[:, :, 2] = -12.0
    return RawAovs(
        distance_m=distance,
        normal=normal,
        position=position,
        instance_id=np.ones(shape, dtype=np.uint32),
        motion=np.full((*shape, 2), 6e-5, dtype=np.float32) if motion else None,
    )


# --- the channel is not attached ------------------------------------------------------------


def test_motion_is_the_unverified_channel() -> None:
    assert "motion" in UNVERIFIED_CHANNELS
    # Still in the survey list: a probe has to be able to ask what it returns, which is the only
    # way it could ever stop being unverified.
    assert "motion" in AOV_NAMES


def test_a_reader_does_not_attach_an_unverified_channel_by_default() -> None:
    reader = AovReader("/Render/rp", required=())
    assert "motion" not in reader.names
    assert reader.skipped == ("motion",)
    # Everything the physics does use is untouched.
    for channel in ("distance", "position", "normal", "instance"):
        assert channel in reader.names


def test_a_caller_can_still_ask_for_it_deliberately() -> None:
    """What `scripts/probe_isaac_geometry.py` does — surveying is the point of a probe."""
    reader = AovReader("/Render/rp", required=(), unverified=("motion",))
    assert "motion" in reader.names
    assert reader.skipped == ()


def test_requiring_an_unverified_channel_is_refused() -> None:
    """Otherwise the reader would drop it and then raise 'produced no data', blaming the build."""
    with pytest.raises(ValueError, match="cannot be required while unverified"):
        AovReader("/Render/rp", required=("motion",))


def test_only_a_real_unverified_channel_can_be_opted_into() -> None:
    with pytest.raises(ValueError, match="are not unverified channels"):
        AovReader("/Render/rp", required=(), unverified=("normal",))


# --- the convention is never guessed --------------------------------------------------------


def test_a_motion_plane_without_a_convention_is_refused() -> None:
    """The defect: `motion_convention` used to default to "pixels" and nothing had checked it."""
    with pytest.raises(ValueError, match="no motion_convention"):
        geometry_planes(_aovs(motion=True), camera_position=CAMERA_POSITION)


def test_a_named_convention_still_works() -> None:
    planes = geometry_planes(
        _aovs(motion=True), camera_position=CAMERA_POSITION, motion_convention="pixels"
    )
    assert planes.motion_px is not None
    assert planes.motion_px.dtype == np.float32


def test_no_motion_aov_means_no_plane_and_no_complaint() -> None:
    """The shipped path. `motion_px` is optional in M0.6, so its absence is not an error."""
    planes = geometry_planes(_aovs(motion=False), camera_position=CAMERA_POSITION)
    assert planes.motion_px is None


# --- why the guess mattered -----------------------------------------------------------------


def test_the_three_conventions_disagree_by_far_more_than_a_rounding() -> None:
    """A defaulted convention is not a small error: it is the resolution, and the sign of y.

    This is what made the default dangerous rather than merely untidy — had a build started
    filling the AOV in, the smear would have run at 128x the true rate, or backwards.
    """
    plane = np.full((4, 4, 2), 0.01, dtype=np.float32)
    shape = (256, 256)
    as_pixels = motion_px_per_frame(plane, convention="pixels", shape=shape)
    as_ndc = motion_px_per_frame(plane, convention="ndc", shape=shape)
    as_uv = motion_px_per_frame(plane, convention="uv", shape=shape)

    assert as_pixels[..., 0].max() == pytest.approx(0.01)
    assert as_ndc[..., 0].max() == pytest.approx(1.28)  # x128
    assert as_uv[..., 0].max() == pytest.approx(2.56)  # x256
    # And ndc points y the other way, so the smear runs in the opposite direction.
    assert np.sign(as_ndc[..., 1]).max() == -np.sign(as_uv[..., 1]).max()


def test_the_measured_floor_is_not_zero_so_an_all_zero_guard_would_never_have_caught_it() -> None:
    """Why the fix is 'do not attach' rather than 'reject an empty plane'.

    ADR 0014's addendum measured `motion_vectors` at a ~6e-5 floor after a 180 px displacement.
    `_reject_reason` rejects an all-zero buffer, but only on required channels — and this is not
    all-zero, so widening that test to every channel would still have let this through.
    """
    floor = np.full((8, 8, 2), 6e-5, dtype=np.float32)
    assert np.any(floor != 0.0)
    converted = motion_px_per_frame(floor, convention="pixels", shape=(8, 8))
    # Six hundredths of a milli-pixel per frame, presented as a velocity. Not zero, not motion.
    assert 0.0 < float(np.abs(converted).max()) < 1e-3
