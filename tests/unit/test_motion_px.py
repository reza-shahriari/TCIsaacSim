"""Engine-free tests for synthesised image-plane motion (roadmap M10.1b, §13.3, §9.2).

The renderer does not transport motion on this build (ADR 0014 addendum), so ``motion_px`` is
computed from rigid-body transforms instead. That makes it *exact* rather than approximate, which
raises the bar these tests should hold it to: a displacement of 3 px/frame is not "about 3", it is
3 to floating-point, uniformly across the frame, and a static scene is not "small", it is zero.

The three cases the roadmap names are each a different way for the arithmetic to be wrong:

* a **moving object** with a static camera catches the scale and the sign;
* a **static scene** catches a stray transform being applied twice or not at all;
* an **object and camera moving together** catches the two halves being composed in the wrong
  order -- which is invisible in the first two cases, because there the camera is identity.

A fourth case is here because the tracked-target stages depend on it: the **background** is at
infinity, so it does not translate with the camera, only rotate. If that were wrong the sky behind
a tracked aircraft would smear by the target's translation instead of the mount's rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.config.sensor import DistortionSpec
from irsim.optics.motion import BACKGROUND_OBJECT_ID, image_plane_motion, transform_points
from irsim.optics.projection import Intrinsics

SIZE = 64
FOCAL_PX = 500.0
DEPTH_M = 10.0
IDENTITY = np.eye(4)


def intrinsics() -> Intrinsics:
    return Intrinsics(
        fx_px=FOCAL_PX, fy_px=FOCAL_PX, cx_px=SIZE / 2, cy_px=SIZE / 2, width=SIZE, height=SIZE
    )


def pinhole() -> DistortionSpec:
    return DistortionSpec(model="brown_conrady", coeffs=[0.0] * 5)


def frontal_slab() -> np.ndarray:
    """Per-pixel USD camera-space positions of a plane at ``DEPTH_M``, filling the frame."""
    offsets = (np.arange(SIZE) + 0.5 - SIZE / 2) / FOCAL_PX * DEPTH_M
    x, y = np.meshgrid(offsets, -offsets)
    return np.stack([x, y, np.full_like(x, -DEPTH_M)], axis=-1)


def translation(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    """A USD (row-vector) 4x4 translation: the offset lives in the bottom row."""
    m = np.eye(4)
    m[3, :3] = (dx, dy, dz)
    return m


def yaw(degrees: float) -> np.ndarray:
    """Rotation about +Y in the row-vector convention."""
    a = math.radians(degrees)
    m = np.eye(4)
    m[0, 0], m[0, 2] = math.cos(a), -math.sin(a)
    m[2, 0], m[2, 2] = math.sin(a), math.cos(a)
    return m


def motion_of(
    object_prev: np.ndarray,
    object_cur: np.ndarray = IDENTITY,
    camera_prev: np.ndarray = IDENTITY,
    camera_cur: np.ndarray = IDENTITY,
    *,
    background: bool = False,
) -> np.ndarray:
    ids = np.full((SIZE, SIZE), BACKGROUND_OBJECT_ID if background else 1, dtype=np.int64)
    return image_plane_motion(
        frontal_slab(),
        ids,
        np.stack([IDENTITY, object_prev]),
        np.stack([IDENTITY, object_cur]),
        camera_prev,
        camera_cur,
        intrinsics(),
        pinhole(),
    )


def metres_for(pixels: float) -> float:
    """The world displacement at ``DEPTH_M`` that projects to ``pixels`` on this lens."""
    return pixels * DEPTH_M / FOCAL_PX


# --- the three roadmap cases ------------------------------------------------------------------


@pytest.mark.parametrize("pixels", [3.0, 0.0, -2.0, 11.5])
def test_a_translating_object_reports_its_own_pixel_velocity(pixels: float) -> None:
    """A bar at 3 px/frame reports 3.0 -- exactly, and identically at every pixel.

    Uniformity is the part worth checking. A frontal plane translating parallel to the focal plane
    moves every one of its points by the same number of pixels, so any spread across the frame
    would mean the projection is being applied inconsistently -- which an average would hide.
    """
    motion = motion_of(translation(dx=-metres_for(pixels)))
    assert float(motion[..., 0].mean()) == pytest.approx(pixels, abs=1e-9)
    assert float(motion[..., 0].std()) == pytest.approx(0.0, abs=1e-9)
    assert float(np.abs(motion[..., 1]).max()) == pytest.approx(0.0, abs=1e-9)


def test_the_y_axis_points_down_as_the_gbuffer_contract_says() -> None:
    """An object moving *up* in the world gives a negative v: image y increases downwards.

    The G-buffer's ``motion_px`` is (vx, vy) with y down (tests/conftest.py). A smear kernel that
    took this the other way up would blur along the mirror direction and still look like motion
    blur, which is why the sign gets its own test rather than riding along with the magnitude.
    """
    motion = motion_of(translation(dy=-metres_for(4.0)))
    assert float(motion[..., 1].mean()) == pytest.approx(-4.0, abs=1e-9)
    assert float(np.abs(motion[..., 0]).max()) == pytest.approx(0.0, abs=1e-9)


def test_a_static_scene_reports_exactly_zero() -> None:
    """Not "small": zero. Both projections are of the same point through the same pose."""
    motion = motion_of(IDENTITY)
    assert float(np.abs(motion).max()) == 0.0


def test_an_object_and_camera_moving_together_report_zero() -> None:
    """A tracked target is stationary on the focal plane however fast it crosses the sky.

    This is the case that catches the two transform pairs being composed in the wrong order --
    invisible in the other tests, where the camera is identity and order cannot matter. It is also
    the reason a tracking mount takes smear off the target and puts it on the background.
    """
    offset = metres_for(9.0)
    motion = motion_of(translation(dx=-offset), camera_prev=translation(dx=-offset))
    assert float(np.abs(motion).max()) < 1e-9


def test_the_object_and_camera_case_is_not_trivially_zero() -> None:
    """Guard on the test above: each half alone must produce a large, opposite displacement.

    Without this, a function that returned zeros unconditionally would pass the tracking test.
    """
    offset = metres_for(9.0)
    object_only = motion_of(translation(dx=-offset))
    camera_only = motion_of(IDENTITY, camera_prev=translation(dx=-offset))
    assert float(object_only[..., 0].mean()) == pytest.approx(9.0, abs=1e-9)
    assert float(camera_only[..., 0].mean()) == pytest.approx(-9.0, abs=1e-9)


# --- the background ---------------------------------------------------------------------------


def test_the_background_does_not_translate_with_the_camera() -> None:
    """Sky is at infinity: a camera that slides sideways does not move it at all.

    Parallax is a function of depth, and the background has none. Treating it as a point at the
    G-buffer's finite depth would smear the sky by the camera's translation, which for a sensor
    on a moving vehicle would be badly wrong and for a tracking pedestal is simply not what
    happens.
    """
    motion = motion_of(IDENTITY, camera_prev=translation(dx=-metres_for(9.0)), background=True)
    assert float(np.abs(motion).max()) < 1e-9


def test_the_background_does_rotate_with_the_camera() -> None:
    """A yawing mount sweeps the sky across the frame by about f tan(delta). This is the smear a
    tracked target's background actually gets.

    The tolerance is 0.1 %, not machine precision, and the reason is worth stating: a pixel's
    displacement under a yaw is f(tan(alpha + delta) - tan(alpha)) at its own field angle alpha,
    and with an even sensor width no pixel centre lands exactly on the axis -- the nearest is half
    a pixel off. Asserting the exact off-axis form here would be re-deriving the implementation in
    the test, which proves nothing; 0.1 % of 4.4 px is 0.004 px and still pins the scale.
    """
    angle_deg = 0.5
    motion = motion_of(IDENTITY, camera_prev=yaw(angle_deg), background=True)
    centre = motion[SIZE // 2, SIZE // 2, 0]
    expected = FOCAL_PX * math.tan(math.radians(angle_deg))
    assert abs(float(centre)) == pytest.approx(expected, rel=1e-3)

    # A yaw is *not* a pure horizontal shift, and it would be wrong to assert that it is. Under a
    # perspective projection v = f y / -z, and yawing changes z, so an off-axis point with y != 0
    # moves vertically too by about v (x/f) delta -- 0.019 px at this frame's corner, which is
    # exactly what comes out. What the test can say is that the cross-term stays a small fraction
    # of the shift it accompanies; swapped axes or a transposed rotation would put them level.
    horizontal = float(np.abs(motion[..., 0]).max())
    vertical = float(np.abs(motion[..., 1]).max())
    assert vertical < 0.01 * horizontal, f"vy {vertical:.4f} against vx {horizontal:.4f}"


def test_a_static_camera_leaves_the_background_still() -> None:
    motion = motion_of(IDENTITY, background=True)
    assert float(np.abs(motion).max()) == 0.0


# --- contract ---------------------------------------------------------------------------------


def test_the_plane_matches_the_gbuffer_contract() -> None:
    """(H, W, 2) float32 -- what `irsim.config.gbuffer` will accept as ``motion_px``."""
    motion = motion_of(translation(dx=metres_for(2.0)))
    assert motion.shape == (SIZE, SIZE, 2)
    assert motion.dtype == np.float32
    assert np.all(np.isfinite(motion))


def test_a_padded_position_aov_is_accepted() -> None:
    """The renderer hands positions over as (H, W, 4) with a padded fourth component.

    Measured, not assumed: the in-sim test failed on exactly this before the core learned to take
    the first three, which is the rule the rest of the Isaac adapter already applies to every
    float AOV. The padding must not change the answer.
    """
    points = frontal_slab()
    padded = np.concatenate([points, np.ones((SIZE, SIZE, 1))], axis=-1)
    ids = np.ones((SIZE, SIZE), dtype=np.int64)
    stacks_prev = np.stack([IDENTITY, translation(dx=-metres_for(3.0))])
    stacks_cur = np.stack([IDENTITY, IDENTITY])
    args = (stacks_prev, stacks_cur, IDENTITY, IDENTITY, intrinsics(), pinhole())
    plain = image_plane_motion(points, ids, *args)
    with_pad = image_plane_motion(padded, ids, *args)
    assert np.array_equal(plain, with_pad)
    assert float(with_pad[..., 0].mean()) == pytest.approx(3.0, abs=1e-9)


def test_it_refuses_mismatched_or_wrongly_typed_inputs() -> None:
    stack = np.stack([IDENTITY, IDENTITY])
    with pytest.raises(ValueError, match=r"\(H, W, >=3\)"):
        image_plane_motion(
            np.zeros((SIZE, SIZE, 2)),
            np.zeros((SIZE, SIZE), dtype=np.int64),
            stack,
            stack,
            IDENTITY,
            IDENTITY,
            intrinsics(),
            pinhole(),
        )
    with pytest.raises(TypeError, match="integer plane"):
        image_plane_motion(
            frontal_slab(),
            np.zeros((SIZE, SIZE), dtype=np.float32),
            stack,
            stack,
            IDENTITY,
            IDENTITY,
            intrinsics(),
            pinhole(),
        )
    with pytest.raises(ValueError, match="does not match"):
        image_plane_motion(
            frontal_slab(),
            np.zeros((SIZE + 1, SIZE), dtype=np.int64),
            stack,
            stack,
            IDENTITY,
            IDENTITY,
            intrinsics(),
            pinhole(),
        )


def test_transform_points_uses_the_row_vector_convention() -> None:
    """USD matrices multiply on the left of a row vector; the column convention transposes them.

    A transposed rotation is still a rotation, so the mistake produces motion that is smooth,
    plausible and wrong. Pinned here because it is the one convention this module cannot infer.
    """
    point = np.array([[[1.0, 2.0, 3.0]]])
    assert np.allclose(transform_points(point, translation(dx=5.0)), [[[6.0, 2.0, 3.0]]])
    # +90 degrees about +Y carries +X onto -Z in this convention.
    turned = transform_points(np.array([[[1.0, 0.0, 0.0]]]), yaw(90.0))
    assert np.allclose(turned, [[[0.0, 0.0, -1.0]]], atol=1e-12)
