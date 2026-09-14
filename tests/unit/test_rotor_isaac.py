"""Rotor discs on a stage (ADR 0081): prim transform -> veil, and the occlusion that keeps a
blade off the motor bell it is bolted to.

Engine-free despite living in ``irsim_isaac``: nothing here imports ``omni`` or ``pxr``, exactly as
``motion_isaac`` takes its stage as an argument. The test that earns its keep is the last one --
occlusion by the disc *plane* rather than by the disc centre's range, which for a 75-degree tilt at
20 m is a difference of two thirds of a metre across the ellipse.
"""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.optics.projection import Intrinsics
from irsim.optics.rotor import RotorDisc
from irsim_isaac.pipeline.rotor_isaac import (
    RotorMount,
    build_rotor_veils,
    camera_ray_directions,
    disc_pose_world,
    occlusion_mask,
)
from irsim_isaac.quadrotor import HEAVY_LIFT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

DISC = RotorDisc()
MOUNT = RotorMount(disc=DISC, offset_m=(0.64, 0.16, -0.64))
#: An identity camera at the origin: USD looks down -Z, so a target in front sits at negative z.
CAMERA_TO_WORLD = np.eye(3)
CAMERA_POSITION = np.zeros(3)

#: The quadrotor stage's own geometry: 20 m away, 15 degrees up. Written first with the aircraft
#: level with the camera, which puts the rotor axis square to the line of sight -- the disc is then
#: exactly edge-on, its ellipse has no area and no ray meets its plane. That is a real degenerate
#: case, but it is not this scene's, and testing against it measured nothing.
DEMO_RANGE_M = 20.0
DEMO_ELEVATION_DEG = 15.0
DEMO_POSITION = (
    0.0,
    DEMO_RANGE_M * math.sin(math.radians(DEMO_ELEVATION_DEG)),
    -DEMO_RANGE_M * math.cos(math.radians(DEMO_ELEVATION_DEG)),
)


def _sensor(width: int = 64, height: int = 64):  # type: ignore[no-untyped-def]
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    return SensorConfig.model_validate(d).sensor


def _usd_xform(translate=(0.0, 0.0, 0.0), pitch_deg: float = 0.0) -> np.ndarray:
    """A USD row-vector 4x4: rotation about +X by ``pitch_deg``, then the translation."""
    c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    m = np.eye(4)
    m[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    m[3, :3] = translate
    return m


# --------------------------------------------------------------------------------------------
# From the prim's transform to the disc
# --------------------------------------------------------------------------------------------


def test_an_identity_transform_leaves_the_mount_where_it_was_authored() -> None:
    centre, axis = disc_pose_world(np.eye(4), MOUNT)
    assert centre == pytest.approx(MOUNT.offset_m)
    assert axis == pytest.approx((0.0, 1.0, 0.0))


def test_translating_the_airframe_moves_the_disc_and_not_its_axis() -> None:
    centre, axis = disc_pose_world(_usd_xform(translate=(3.0, -5.0, 40.0)), MOUNT)
    assert centre == pytest.approx((0.64 + 3.0, 0.16 - 5.0, -0.64 + 40.0))
    assert axis == pytest.approx((0.0, 1.0, 0.0))


@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 25.0, -15.0])
def test_pitching_the_airframe_tilts_the_disc_by_the_same_angle(pitch_deg: float) -> None:
    """A multirotor accelerates by tilting its rotor plane, so this is the whole attitude story."""
    _, axis = disc_pose_world(_usd_xform(pitch_deg=pitch_deg), MOUNT)
    assert math.degrees(math.acos(np.clip(axis[1], -1.0, 1.0))) == pytest.approx(
        abs(pitch_deg), abs=1e-9
    )


def test_a_collapsed_transform_is_refused() -> None:
    m = np.eye(4)
    m[:3, :3] = 0.0
    with pytest.raises(ValueError):
        disc_pose_world(m, MOUNT)
    with pytest.raises(ValueError):
        disc_pose_world(np.eye(3), MOUNT)
    with pytest.raises(ValueError):
        RotorMount(disc=DISC, offset_m=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 0.0))


# --------------------------------------------------------------------------------------------
# Ray directions
# --------------------------------------------------------------------------------------------


def test_rays_are_unit_and_the_centre_ray_is_the_optical_axis() -> None:
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    assert rays.shape == (64, 64, 3)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    # cx/cy sit on a pixel corner, so the four pixels around it straddle the axis symmetrically
    quad = rays[31:33, 31:33]
    assert quad[..., 0].sum() == pytest.approx(0.0, abs=1e-12)
    assert quad[..., 1].sum() == pytest.approx(0.0, abs=1e-12)
    assert np.all(rays[..., 2] > 0.0)  # OpenCV: +Z is forward


def test_the_corner_ray_matches_the_field_angle() -> None:
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    corner = rays[0, 0]
    expected = math.atan2(math.hypot(31.5, 31.5), 500.0)
    assert math.acos(float(corner[2])) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------------------------
# Occlusion
# --------------------------------------------------------------------------------------------


def _flat_scene(depth: float, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.full(shape, depth, dtype=np.float64)


def test_a_nearer_surface_occludes_and_a_farther_one_does_not() -> None:
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    centre, axis = np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0])
    assert occlusion_mask(_flat_scene(10.0), rays, centre, axis).all()
    assert not occlusion_mask(_flat_scene(30.0), rays, centre, axis).any()


@pytest.mark.parametrize("sky_depth", [0.0, np.inf])
def test_sky_never_occludes(sky_depth: float) -> None:
    """Sky carries ``distance_m = 0`` in the G-buffer, not infinity, and zero is the dangerous one.

    :mod:`irsim.config.gbuffer` stores the sentinel as zero deliberately, so a consumer that
    ignores ``sky_mask`` still sees tau = 1. Read as a distance it is a surface *at the camera*,
    nearer than anything, and the first version of ``occlusion_mask`` therefore marked **every
    pixel of the frame** occluded and erased all four discs from the render. This test originally
    used ``inf`` -- what the AOV reports rather than what the adapter stores -- and passed.
    """
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    depth = _flat_scene(sky_depth)
    centre, axis = np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0])
    assert not occlusion_mask(depth, rays, centre, axis).any()
    sky = np.ones(depth.shape, dtype=bool)
    assert not occlusion_mask(depth, rays, centre, axis, sky_mask=sky).any()


def test_the_sky_mask_wins_over_a_stale_distance() -> None:
    """An explicit mask is the documented signal, so it overrides whatever the depth plane says."""
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    depth = _flat_scene(5.0)  # would occlude a 20 m disc everywhere
    centre, axis = np.array([0.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0])
    assert occlusion_mask(depth, rays, centre, axis).all()
    sky = np.zeros(depth.shape, dtype=bool)
    sky[:32] = True
    mask = occlusion_mask(depth, rays, centre, axis, sky_mask=sky)
    assert not mask[:32].any() and mask[32:].all()


def test_occlusion_follows_the_plane_and_not_the_centre_range() -> None:
    """The test that justifies intersecting the plane rather than comparing with one range.

    A disc tilted 75 degrees at 20 m spans two thirds of a metre in depth across its own ellipse.
    Put a surface at exactly 20 m and the centre-range shortcut says nothing is occluded anywhere,
    while the truth is that half the disc is behind it. At this stage's scale that half is about
    18 pixels of arm drawn on the wrong side of the aircraft.
    """
    intr = Intrinsics(fx_px=1166.0, fy_px=1166.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    tilt = math.radians(75.0)
    centre = np.array([0.0, 0.0, 20.0])
    axis = np.array([0.0, math.sin(tilt), math.cos(tilt)])

    mask = occlusion_mask(_flat_scene(20.0), rays, centre, axis)
    assert mask.any() and not mask.all()  # the shortcut would give all-False
    # the occluded half is the one whose rays reach the plane beyond 20 m, split about the centre
    assert mask[:32].all() != mask[32:].all()
    assert 0.4 < mask.mean() < 0.6


def test_a_mismatched_depth_plane_is_refused() -> None:
    intr = Intrinsics(fx_px=500.0, fy_px=500.0, cx_px=32.0, cy_px=32.0, width=64, height=64)
    rays = camera_ray_directions(intr)
    with pytest.raises(ValueError):
        occlusion_mask(_flat_scene(10.0, (32, 32)), rays, np.zeros(3) + 1.0, np.array([0, 0, 1.0]))


# --------------------------------------------------------------------------------------------
# The four discs of a quadrotor
# --------------------------------------------------------------------------------------------


def test_the_rotors_sit_above_their_motors_and_clear_each_other() -> None:
    """Real geometry, not a plausible one: discs above the bells, and no two overlapping."""
    spec = HEAVY_LIFT
    mounts = spec.rotor_mounts()
    assert len(mounts) == 4
    motors = {p.name: p for p in spec.parts() if p.name.startswith("motor_")}
    assert len(motors) == 4
    top_of_bell = 0.5 * spec.body_m[1] + spec.motor_height_m
    for index, mount in enumerate(mounts):
        motor = motors[f"motor_{index}"]
        assert mount.offset_m[0] == pytest.approx(motor.centre_m[0])
        assert mount.offset_m[2] == pytest.approx(motor.centre_m[2])
        assert mount.offset_m[1] == pytest.approx(top_of_bell + spec.rotor_clearance_m)
    centres = np.array([m.offset_m for m in mounts])
    for i in range(4):
        for j in range(i + 1, 4):
            gap = float(np.linalg.norm(centres[i] - centres[j]))
            assert gap > 2.0 * spec.rotor.radius_m, "adjacent discs overlap"


def test_every_rotor_reaches_the_frame_as_a_veil() -> None:
    sensor = _sensor(256, 256)
    mounts = HEAVY_LIFT.rotor_mounts(3000.0)
    veils = build_rotor_veils(
        mounts,
        _usd_xform(translate=DEMO_POSITION),
        CAMERA_POSITION,
        CAMERA_TO_WORLD,
        sensor,
        supersample=1,
        blade_radiance=10.0,
        integration_s=1.0 / 60.0,
    )
    assert len(veils) == 4
    for veil in veils:
        assert veil.range_m == pytest.approx(20.0, abs=1.0)
        assert veil.ellipse.semi_major_px > 5.0
        # 3000 rpm through a whole 60 Hz frame: 1.67 blade spacings, the annulus regime
        assert veil.swept_rad == pytest.approx(2.0 * math.pi * 50.0 / 60.0)


def test_a_shorter_integration_moves_every_disc_into_the_arc_regime() -> None:
    sensor = _sensor(256, 256)
    common = dict(
        local_to_world=_usd_xform(translate=DEMO_POSITION),
        camera_position=CAMERA_POSITION,
        camera_to_world=CAMERA_TO_WORLD,
        sensor=sensor,
        supersample=1,
        blade_radiance=10.0,
    )
    bolo = build_rotor_veils(HEAVY_LIFT.rotor_mounts(), integration_s=1.0 / 60.0, **common)
    cooled = build_rotor_veils(HEAVY_LIFT.rotor_mounts(), integration_s=0.002, **common)
    spacing = 2.0 * math.pi / DISC.blades
    assert all(v.swept_rad > spacing for v in bolo)
    assert all(v.swept_rad < 0.25 * spacing for v in cooled)


def test_discs_behind_the_camera_are_dropped() -> None:
    veils = build_rotor_veils(
        HEAVY_LIFT.rotor_mounts(),
        _usd_xform(translate=(0.0, 0.0, 20.0)),  # USD looks down -Z, so +Z is behind
        CAMERA_POSITION,
        CAMERA_TO_WORLD,
        _sensor(256, 256),
        supersample=1,
        blade_radiance=10.0,
        integration_s=1.0 / 60.0,
    )
    assert veils == []


def _demo_veils(depth: np.ndarray | None = None):  # type: ignore[no-untyped-def]
    sensor = _sensor(256, 256)
    return build_rotor_veils(
        HEAVY_LIFT.rotor_mounts(),
        _usd_xform(translate=DEMO_POSITION),
        CAMERA_POSITION,
        CAMERA_TO_WORLD,
        sensor,
        supersample=1,
        blade_radiance=10.0,
        integration_s=1.0 / 60.0,
        distance_m=depth,
    )


def test_the_demo_geometry_sees_the_discs_well_off_edge_on() -> None:
    """15 degrees of camera elevation against a level airframe is a 75-degree disc tilt."""
    for veil in _demo_veils():
        assert math.degrees(veil.ellipse.tilt_rad) == pytest.approx(75.0, abs=2.0)
        assert veil.ellipse.semi_minor_px > 2.0


def test_a_surface_across_the_whole_frame_occludes_every_disc() -> None:
    intr = Intrinsics.from_sensor(_sensor(256, 256), 1)
    near = np.full((intr.height, intr.width), 5.0)  # in front of a 20 m aircraft, everywhere
    for veil in _demo_veils(near):
        assert veil.occluded is not None
        cx, cy = veil.ellipse.centre_px
        assert bool(veil.occluded[int(cy), int(cx)])


def test_a_sky_only_frame_leaves_every_disc_unveiled() -> None:
    """The real demo case: the quadrotor is the only geometry, so the discs are over sky.

    With the G-buffer's zero sentinel this is exactly the configuration that failed in sim while
    every unit test passed.
    """
    intr = Intrinsics.from_sensor(_sensor(256, 256), 1)
    sky_only = np.zeros((intr.height, intr.width))
    for veil in _demo_veils(sky_only):
        assert veil.occluded is not None and not veil.occluded.any()


def test_nothing_in_front_leaves_every_disc_unveiled_where_it_is_drawn() -> None:
    """Checked over the ellipse, which is the only place the mask is consulted.

    Across the rest of the frame the mask is meaningless and allowed to be: the disc *plane* is
    infinite, so a ray far off the disc meets it kilometres away and any ordinary surface is
    nearer. Asserting over the whole frame was the first version of this test, and it failed on
    arithmetic that was right.
    """
    intr = Intrinsics.from_sensor(_sensor(256, 256), 1)
    far = np.full((intr.height, intr.width), 500.0)
    for veil in _demo_veils(far):
        assert veil.occluded is not None
        cx, cy = veil.ellipse.centre_px
        reach = int(veil.ellipse.semi_major_px)
        y0, y1 = int(cy) - reach, int(cy) + reach + 1
        x0, x1 = int(cx) - reach, int(cx) + reach + 1
        assert not veil.occluded[max(0, y0) : y1, max(0, x0) : x1].any()


def test_omitting_the_depth_plane_leaves_no_mask() -> None:
    assert all(v.occluded is None for v in _demo_veils())
