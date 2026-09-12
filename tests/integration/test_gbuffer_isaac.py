"""Roadmap M10.1: the geometry AOVs, measured against an analytically known scene.

The adapter's arithmetic is covered engine-free in ``tests/unit/test_gbuffer_isaac.py``. What only
a renderer can answer is here: which annotators this build actually delivers, what frame and units
they use, and whether the numbers agree with geometry the test authored itself. Every expectation
comes from ``irsim_isaac.geometry_probe``'s closed forms, never from a previous render, so these
fail on a renderer change instead of blessing it.

ADR 0014 measured the transport channels on an unlit, static, front-parallel ramp with the camera
at the world origin, and left four questions open, each of which has a test below:

1. does a normals AOV deliver on a *lit* scene, and in which frame;
2. is the position AOV world- or camera-space (indistinguishable at the origin);
3. does ambient occlusion deliver, and is the V_s geometric correction right without it;
4. does a motion AOV deliver, and in what units.

Answers, measured by ``scripts/probe_isaac_geometry.py``: ``normals`` (float32, full resolution,
world space) -- **not** ``PtWorldNormal``, which is fp16, half resolution and all zero; the
position AOV is world space; no ambient-occlusion AOV delivers, so V_s is the unoccluded
geometric form; and no motion AOV transports motion at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

RESOLUTION = 384
RAY_LENGTH_TOL_M = 0.01
COS_THETA_TOL = 0.01
SKY_VIEW_TOL = 0.05
#: Above this, ``motion_vectors`` is carrying something; it sits at ~6e-5 on this build.
MOTION_NOISE_FLOOR_PX = 1e-3


@pytest.fixture(scope="module")
def scene(simulation_app: Any) -> Any:
    del simulation_app
    from irsim_isaac.geometry_probe import build_geometry_scene

    built = build_geometry_scene(resolution=RESOLUTION)
    assert built.errors == {}, built.errors
    return built


@pytest.fixture(scope="module")
def frame(scene: Any) -> Any:
    """One settled frame: the raw AOVs, the assembled planes, and what the probe resolved."""
    import omni.replicator.core as rep

    from irsim_isaac.geometry_probe import (
        configure_renderer,
        detect_position_frame,
        translate_bar,
    )
    from irsim_isaac.pipeline.gbuffer_isaac import AovReader, geometry_planes

    settings = configure_renderer()
    rp = rep.create.render_product(scene.camera_path, (RESOLUTION, RESOLUTION))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    reader = AovReader(rp_path, device="cpu", expected_shape=(RESOLUTION, RESOLUTION)).attach(
        settle_frames=24
    )

    aovs = reader.read()
    position_frame = detect_position_frame(scene, np.asarray(aovs.position))
    planes = geometry_planes(
        aovs,
        up_axis=scene.up_axis,
        position_frame=position_frame["verdict"],
        camera_position=scene.camera_position,
    )

    # A second frame with the bar displaced by exactly 3 px of image motion.
    translate_bar(scene, 1)
    reader.step(frames=1)
    moved = reader.read()

    yield {
        "settings": settings,
        "reader": reader,
        "aovs": aovs,
        "planes": planes,
        "position_frame": position_frame,
        "moved": moved,
    }
    translate_bar(scene, 0)
    reader.detach()


def _at(scene: Any, planes: Any, name: str) -> tuple[int, int]:
    col, row = scene.pixel_of(scene.targets[name].centre)
    assert 0 <= col < planes.shape[1] and 0 <= row < planes.shape[0], (name, col, row)
    return col, row


# --- what this build delivers ------------------------------------------------------------------


def test_required_channels_deliver(frame: Any) -> None:
    """Distance, position and normals all arrive; the two that scale radiometry are float32.

    The normals AOV is float16 on this build (ADR 0014 addendum) and that is accepted: it feeds a
    cosine with 0.01 of tolerance, where fp16 costs ~1e-3. Distance is a different matter -- it
    scales optical depth and the 1/R^2 falloff -- so half precision there stays an error.
    """
    reader, aovs = frame["reader"], frame["aovs"]
    for channel in ("distance", "position", "normal"):
        assert channel in reader.resolved, (channel, reader.failures)
    assert np.asarray(aovs.distance_m).dtype == np.float32, aovs.distance_m.dtype
    assert np.asarray(aovs.position).dtype == np.float32, aovs.position.dtype
    assert np.issubdtype(np.asarray(aovs.normal).dtype, np.floating), aovs.normal.dtype


def test_aovs_arrive_at_the_requested_resolution(frame: Any) -> None:
    """AA must be off: DLSS renders internally at half and every pixel lookup then misaddresses."""
    assert frame["settings"]["/rtx/post/aa/op"] == 0, frame["settings"]
    assert np.asarray(frame["aovs"].distance_m).shape == (RESOLUTION, RESOLUTION)
    assert frame["planes"].shape == (RESOLUTION, RESOLUTION)


def test_position_aov_frame_is_unambiguous(frame: Any) -> None:
    """The camera is off the world origin, so one frame fits and the other misses by |C| = 5.6 m."""
    verdict = frame["position_frame"]
    best = min(verdict["err_world_m"], verdict["err_camera_m"])
    worst = max(verdict["err_world_m"], verdict["err_camera_m"])
    assert best < 0.05, verdict
    assert worst > 1.0, verdict  # the two hypotheses really are separated by this scene


# --- the numbers -------------------------------------------------------------------------------


def test_tilted_quad_ray_length(scene: Any, frame: Any) -> None:
    """DistanceToCameraSD is a ray length; z-depth would miss by 9 % on this 24-degree quad."""
    planes = frame["planes"]
    col, row = _at(scene, planes, "tilted_quad")
    expected = scene.targets["tilted_quad"].distance_m
    got = float(planes.distance_m[row, col])
    assert abs(got - expected) < RAY_LENGTH_TOL_M, (got, expected)
    z_depth = expected * abs(
        (scene.camera_position[2] - scene.targets["tilted_quad"].centre[2]) / expected
    )
    assert abs(z_depth - expected) > 10 * RAY_LENGTH_TOL_M  # the two are genuinely distinguishable


def test_sphere_cos_theta_matches_the_closed_form(scene: Any, frame: Any) -> None:
    """cos(theta) across the disc, where the optical-axis shortcut would show a > 0.05 bias."""
    from irsim_isaac.geometry_probe import sphere_cos_theta

    planes = frame["planes"]
    col, row = _at(scene, planes, "sphere")
    sphere = scene.targets["sphere"]
    depth = scene.camera_position[2] - sphere.centre[2]
    span = int(0.6 * scene.focal_px * (sphere.radius_m or 0.0) / depth)
    cols, rows = np.meshgrid(
        np.arange(col - span, col + span + 1), np.arange(row - span, row + span + 1)
    )
    expected = sphere_cos_theta(scene, cols, rows)
    got = planes.normal_dot_view[rows, cols].astype(np.float64)
    on_sphere = np.isfinite(expected) & (~planes.sky_mask[rows, cols])
    assert on_sphere.sum() > 50, on_sphere.sum()
    err = np.abs(got[on_sphere] - expected[on_sphere])
    assert float(np.median(err)) < COS_THETA_TOL, float(np.median(err))
    assert np.ptp(expected[on_sphere]) > 0.3, "the sampled patch must span a range of angles"


@pytest.mark.parametrize(
    ("name", "expected"), [("plate_up", 1.0), ("plate_vertical", 0.5), ("plate_under", 0.0)]
)
def test_plate_sky_view_factors(scene: Any, frame: Any, name: str, expected: float) -> None:
    """V_s = occlusion (1 + n.up)/2 (ADR 0045): raw occlusion would read ~1 for all three."""
    planes = frame["planes"]
    col, row = _at(scene, planes, name)
    got = float(planes.sky_view_factor[row, col])
    assert not bool(planes.sky_mask[row, col]), f"{name} did not render at {(col, row)}"
    assert abs(got - expected) < SKY_VIEW_TOL, (name, got, expected)


def test_normals_are_world_space_not_camera_space(scene: Any, frame: Any) -> None:
    """Pitch the camera: a world normal keeps n.up = 1 on the up-facing plate, a camera one drops.

    With an axis-aligned camera the two conventions are numerically identical, so this is the only
    measurement in the suite that can tell them apart -- and the answer decides whether every
    angular emissivity and sky-view factor downstream is computed about the right axis.
    """
    from irsim_isaac.geometry_probe import pitch_camera

    reader = frame["reader"]
    pitch_deg = 20.0
    pitch_camera(scene, -pitch_deg)  # look down, keeping the up-facing plate in frame
    try:
        reader.step(frames=6)
        normal = np.asarray(reader.read().normal, dtype=np.float64)[:, :, :3]
    finally:
        pitch_camera(scene, 0.0)
        reader.step(frames=2)

    norm = np.linalg.norm(normal, axis=2)
    valid = norm > 0.5
    assert valid.any(), "no normals survived the pitch"
    n_up = np.abs(normal[:, :, 1] / np.where(valid, norm, 1.0))[valid]
    best = float(n_up.max())
    camera_space_bound = float(np.cos(np.radians(pitch_deg))) + 0.01
    assert best > 0.99, f"max |n.up| = {best:.4f} after a {pitch_deg} deg pitch"
    assert best > camera_space_bound, (
        f"max |n.up| = {best:.4f} is consistent with camera-space normals "
        f"(cos {pitch_deg} deg = {camera_space_bound - 0.01:.4f}), not world-space"
    )


def test_motion_aov_does_not_transport_motion_on_this_build(scene: Any, frame: Any) -> None:
    """The negative result behind the ADR 0014 addendum: reopen it when this test fails.

    ``Motion2d`` returns no data and ``motion_vectors`` attaches as a full-resolution float32
    plane that stays at a ~6e-5 constant floor however far the geometry moves -- measured here at
    180 px of displacement, sixty times the 3 px the M10.1 verification asks for. That is the
    "zero-filled output with status ok" hazard ADR 0014 names on the SPG side, so it is pinned as
    a fact rather than left to be rediscovered: ``motion_px`` stays absent from the G-buffer (it
    is optional in the M0.6 contract) and the bolometer-smear work that needs it is tracked
    separately.
    """
    from irsim_isaac.geometry_probe import translate_bar

    moved = frame["moved"]
    if moved.motion is None:
        return  # the channel vanished entirely: also a change worth noticing, but not a failure
    reader = frame["reader"]
    translate_bar(scene, 60)  # 180 px of image motion
    try:
        reader.step(frames=1)
        far = np.asarray(reader.read().motion, dtype=np.float64)
    finally:
        translate_bar(scene, 0)
        reader.step(frames=1)

    magnitude = float(np.hypot(far[:, :, 0], far[:, :, 1]).max())
    assert magnitude < MOTION_NOISE_FLOOR_PX, (
        f"motion_vectors reported {magnitude:.4g} after a 180 px displacement: the channel now "
        "carries motion. Reopen the ADR 0014 addendum, work out its units, and wire motion_px."
    )


# --- the M0.6 contract ---------------------------------------------------------------------------


def test_assembled_gbuffer_satisfies_the_m0_6_schema(scene: Any, frame: Any) -> None:
    """The adapter's output is a valid G-buffer: right dtypes, finite, sky handled."""
    from irsim.config.gbuffer import GBuffer
    from irsim_isaac.pipeline.gbuffer_isaac import to_gbuffer

    planes = frame["planes"]
    shape = planes.shape
    gbuf = to_gbuffer(
        planes,
        temperature_k=np.full(shape, 295.0, dtype=np.float32),
        material_id=np.ones(shape, dtype=np.int32),
        sky_temperature_k=np.float32(233.0),
    )
    assert isinstance(gbuf, GBuffer)
    assert gbuf.temperature_k.dtype == np.float32
    assert gbuf.distance_m.dtype == np.float32
    assert gbuf.material_id.dtype == np.int32
    assert np.isfinite(gbuf.distance_m).all()
    assert gbuf.sky_view_factor.min() >= 0.0 and gbuf.sky_view_factor.max() <= 1.0
    assert gbuf.normal_dot_view.min() >= 0.0 and gbuf.normal_dot_view.max() <= 1.0


def test_background_pixels_become_the_sky_mask(scene: Any, frame: Any) -> None:
    """Background pixels are masked and left at distance 0: a naive consumer still sees tau = 1."""
    planes = frame["planes"]
    assert planes.sky_mask.any(), "the scene must leave some background visible"
    assert not planes.sky_mask.all()
    assert np.all(planes.distance_m[planes.sky_mask] == 0.0)
    corner = (0, 0)
    assert bool(planes.sky_mask[corner]), "the frame corner should be background"
