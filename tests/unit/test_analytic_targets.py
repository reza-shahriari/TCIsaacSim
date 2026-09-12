"""The analytic-target bridge's geometry (MS.6 in sim): world space back to the camera.

A point target is injected at a pixel position, so the whole thing turns on getting one
transform right, and that transform sits between two opposite conventions: USD matrices are
row-vector (``p_world = p_camera @ R``) while ``ray_directions`` applies its rotation as
``vec @ rot.T``. A transpose in the wrong place rotates the target by *twice* the camera tilt in
the wrong direction and puts it somewhere plausible in the frame, which is why this is pinned
here, engine-free, rather than discovered in a render.

docs/physics-model.md §8.3; ADR 0071 (the handover at one pixel), ADR 0014 addendum (the frame).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim_isaac.aerial_demo import DEMO_TARGETS, DemoTarget, camera_space_position
from irsim_isaac.pipeline.ir_camera import world_to_camera

TILT_DEG = 8.0


def usd_rotate_x(degrees: float) -> np.ndarray:
    """USD's row-vector rotation about +X: the matrix such that ``p_world = p_camera @ R``.

    Written out rather than imported so that the test states the convention independently of the
    code under test; the diagnostic that measured it on the real stage read exactly this.
    """
    a = math.radians(degrees)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), math.sin(a)], [0.0, -math.sin(a), math.cos(a)]]
    )


def camera_to_world_of(degrees: float) -> np.ndarray:
    """What ``IrCamera`` stores: the transpose of the upper-left 3x3 of the USD matrix."""
    return usd_rotate_x(degrees).T


def test_world_to_camera_inverts_the_usd_rotation() -> None:
    """p_camera -> p_world -> p_camera returns the original point, to 1e-12 m."""
    rot = usd_rotate_x(TILT_DEG)
    for point in [(0.0, 0.0, -100.0), (12.0, -3.5, -480.0), (-1.0, 40.0, -2500.0)]:
        p_cam = np.array(point)
        p_world = p_cam @ rot
        back = world_to_camera(p_world, (0.0, 0.0, 0.0), camera_to_world_of(TILT_DEG))
        assert np.allclose(back, p_cam, atol=1e-12)


def test_a_target_placed_on_the_boresight_comes_back_on_the_boresight() -> None:
    """The stage places targets by field angle; the bridge must recover exactly those angles.

    A sign error in the transpose would rotate by twice the tilt in the wrong direction -- here,
    putting a boresight target 16 degrees below it, which is still inside the frame and still
    looks like a target.
    """
    rot = usd_rotate_x(TILT_DEG)
    for az_deg, el_deg in [(0.0, 0.0), (-6.0, 3.0), (6.5, 4.0), (-9.0, 9.0)]:
        p_cam = np.array(camera_space_position(500.0, az_deg, el_deg))
        back = world_to_camera(p_cam @ rot, (0.0, 0.0, 0.0), camera_to_world_of(TILT_DEG))
        assert math.degrees(math.asin(back[1] / 500.0)) == pytest.approx(el_deg, abs=1e-9)
        assert math.degrees(math.atan2(back[0], -back[2])) == pytest.approx(az_deg, abs=1e-9)


def test_the_camera_translation_is_subtracted_before_the_rotation() -> None:
    """A camera off the origin: the offset is a world-space translation, not a camera-space one."""
    rot = usd_rotate_x(TILT_DEG)
    origin = np.array([10.0, 5.0, -2.0])
    p_cam = np.array([3.0, 4.0, -250.0])
    p_world = p_cam @ rot + origin
    back = world_to_camera(p_world, origin, camera_to_world_of(TILT_DEG))
    assert np.allclose(back, p_cam, atol=1e-12)


# --- the handover at one pixel -----------------------------------------------------------------


def demo_target(name: str) -> DemoTarget:
    entry = next(t for t in DEMO_TARGETS if t[0] == name)
    return DemoTarget(
        name, f"/World/Targets/{name}", entry[1], entry[2], entry[3], entry[4], entry[5], entry[6]
    )


def test_the_split_at_one_pixel_is_where_the_demo_stage_says_it_is() -> None:
    """The Boson's 0.857 mrad pixel puts three of the six demo targets below one pixel.

    Pinned because the split decides which code path renders a target, and the two paths are
    mutually exclusive: a target that moved across the boundary without the stage noticing would
    be either drawn twice or not at all.
    """
    ifov_mrad = 1e3 * 0.012 / 14.0
    assert ifov_mrad == pytest.approx(0.857, abs=0.001)
    subpixel = {t[0] for t in DEMO_TARGETS if demo_target(t[0]).subpixel_at(ifov_mrad)}
    assert subpixel == {"drone_mid", "drone_far", "motor_pod"}


def test_angular_size_falls_as_one_over_range() -> None:
    """0.35 m at 500 m is 0.7 mrad, at 1500 m 0.233: the geometry behind the whole problem."""
    assert demo_target("drone_mid").angular_size_mrad() == pytest.approx(0.7, abs=1e-9)
    assert demo_target("drone_far").angular_size_mrad() == pytest.approx(0.2333, abs=1e-4)


def test_a_resolved_target_is_never_handed_to_the_analytic_path() -> None:
    """The aircraft spans 5.6 px: injecting it would be a point source where a shape belongs."""
    ifov_mrad = 1e3 * 0.012 / 14.0
    aircraft = demo_target("aircraft")
    assert aircraft.pixels_across(ifov_mrad) > 5.0
    assert not aircraft.subpixel_at(ifov_mrad)
