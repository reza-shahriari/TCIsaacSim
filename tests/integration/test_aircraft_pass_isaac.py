"""In-sim aircraft phenomenology: the nozzle appears when the aspect turns (ADR 0075).

The headline claim of the aircraft stage cannot be checked engine-free. That the nozzles sit aft
of and inside their nacelles is geometry and `tests/unit/test_aircraft_pass.py` asserts it; that
the *renderer* therefore hides them from the front and shows them from the rear is a statement
about ray tracing, and only a render can make it. The instance-id plane settles it exactly -- every
pixel carries the integer id of the prim it hit, so "how much nozzle is visible" is a count, not an
estimate.

The second claim here is about the mount. `IrCamera` caches the camera pose at `open()` because
every pixel's ray direction is built from it, so a camera that is re-aimed between frames and not
refreshed keeps casting rays from where it used to be -- the geometry AOVs follow the prim while
the elevations and the sky temperature stay behind. That failure produces a completely normal
looking frame, which is why it is worth an explicit test.

docs/physics-model.md §6.6 (by extension), ADR 0075
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_YAML = REPO / "configs" / "scenes" / "aircraft_pass_clear_noon.yaml"

WIDTH, HEIGHT = 320, 256
FOCAL_MM = 25.0  # a longer lens than the demo's, so the aircraft is well resolved at 320 px
SUPERSAMPLE = 2

#: How much more nozzle a rear aspect must show than a head-on one. The geometry gives roughly an
#: order of magnitude; asserting a factor of four leaves room for the sampling of a 2 px object
#: without weakening the claim into nothing.
NOZZLE_ASPECT_RATIO = 4.0


def _build(tophat_lwir_lut: Any, track: Any) -> Any:
    from irsim.config.sensor import SensorConfig
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.scene import Scene
    from irsim_isaac.aircraft_pass import build_aircraft_pass
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    raw = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    raw["sensor"]["fpa"].update(width=WIDTH, height=HEIGHT)
    raw["sensor"]["optics"]["focal_length_mm"] = FOCAL_MM
    raw["sensor"]["optics"]["supersample_factor"] = SUPERSAMPLE
    sensor = SensorConfig.model_validate(raw)

    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    stage = build_aircraft_pass(track=track)
    assert stage.errors == {}, stage.errors

    table = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    pipeline = PipelineConfig.from_sensor(
        sensor, table, tophat_lwir_lut, sky=scene.sky_models["lwir"], atmosphere=scene.layered
    )
    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=stage.prim_to_target,
        resolutions=resolutions,
        camera_path=stage.camera_path,
        strict_materials=False,
        strict_thermal_nodes=False,
        frame_period_s=1.0 / 30.0,
    ).open(settle_frames=16)
    camera.stage_info = stage  # type: ignore[attr-defined]
    return camera


def _fly_to(camera: Any, t_s: float) -> dict[str, Any]:
    """Put the aircraft where the track says at ``t_s``, aim the mount at it, render."""
    import omni.usd
    from pxr import Gf, UsdGeom

    from irsim_isaac.aircraft_pass import look_at_quaternion

    stage = camera.stage_info
    sample = stage.track.sample(t_s)
    usd_stage = omni.usd.get_context().get_stage()
    aircraft = UsdGeom.Xformable(usd_stage.GetPrimAtPath(stage.aircraft_path))
    aircraft.GetOrderedXformOps()[0].Set(Gf.Vec3d(*(float(v) for v in sample.position_m)))
    aim = look_at_quaternion(sample.position_m)
    camera_ops = UsdGeom.Xformable(usd_stage.GetPrimAtPath(stage.camera_path)).GetOrderedXformOps()
    camera_ops[0].Set(Gf.Quatf(float(aim[0]), Gf.Vec3f(*(float(v) for v in aim[1:]))))
    camera.refresh_pose()

    outputs = camera.get_outputs(rt_subframes=4)
    last = camera.last_frame
    counts: dict[str, int] = {}
    for ident, path in last.labels.items():
        node = stage.prim_to_target.get(str(path))
        if node is not None:
            counts[node] = counts.get(node, 0) + int((last.instance_id == ident).sum())
    return {
        "sample": sample,
        "counts": counts,
        "t_app": np.asarray(outputs.apparent_t, dtype=np.float64),
        "frame": last,
    }


@pytest.fixture(scope="module")
def pass_frames(simulation_app: Any, tophat_lwir_lut: Any) -> Any:
    """One head-on frame and one tail-on frame of the same aircraft at the same range."""
    del simulation_app
    from irsim_isaac.aircraft_pass import PassTrack

    track = PassTrack(speed_m_s=150.0, altitude_m=150.0, offset_m=200.0, cpa_time_s=5.0)
    camera = _build(tophat_lwir_lut, track)
    try:
        # Symmetric about closest approach, so the two frames are at the *same range* and differ
        # only in aspect. Comparing a near frame with a far one would confound the two.
        yield {"head": _fly_to(camera, 1.0), "tail": _fly_to(camera, 9.0), "camera": camera}
    finally:
        camera.close()


def test_the_two_frames_differ_only_in_aspect(pass_frames: Any) -> None:
    """Guard on the comparison itself: same range, mirrored aspect about the beam."""
    head, tail = pass_frames["head"]["sample"], pass_frames["tail"]["sample"]
    assert head.range_m == pytest.approx(tail.range_m, rel=1e-9)
    assert head.aspect_deg < 60.0 < 120.0 < tail.aspect_deg
    assert head.aspect_deg + tail.aspect_deg == pytest.approx(180.0, abs=1e-6)


def test_the_nozzle_is_hidden_head_on_and_exposed_tail_on(pass_frames: Any) -> None:
    """The renderer's own occlusion, counted off the instance-id plane.

    This is the aircraft stage's reason to exist. An aircraft's measured infrared signature varies
    by a large factor around the clock because the hot part is behind the airframe from the front,
    and a simulator that showed the nozzle from every aspect would over-predict head-on detection
    range -- the aspect that matters most for an approaching threat.
    """
    head = pass_frames["head"]["counts"].get("nozzle", 0)
    tail = pass_frames["tail"]["counts"].get("nozzle", 0)
    assert tail > 0, "the nozzle must be visible from behind"
    assert tail > NOZZLE_ASPECT_RATIO * max(head, 1), (
        f"nozzle is {head} px head-on and {tail} px tail-on; the nacelle should be hiding it"
    )


def test_the_airframe_itself_is_equally_visible_from_both_aspects(pass_frames: Any) -> None:
    """The skin area barely changes, so the nozzle's swing is occlusion and not just projection.

    Without this the previous test would also pass on an aircraft that simply presented more of
    everything from behind, which would say nothing about the engine being hidden.
    """
    head = pass_frames["head"]["counts"].get("skin", 0)
    tail = pass_frames["tail"]["counts"].get("skin", 0)
    assert head > 50 and tail > 50
    assert 0.5 < tail / head < 2.0, f"skin area {head} vs {tail} px should be comparable"


def test_the_hottest_pixel_is_hotter_from_behind(pass_frames: Any) -> None:
    """The signature the sensor actually reports rises with aspect, not just the pixel count."""
    head = float(pass_frames["head"]["t_app"].max())
    tail = float(pass_frames["tail"]["t_app"].max())
    assert tail > head + 10.0, f"peak apparent temperature {head:.1f} K head-on, {tail:.1f} K tail"


def test_refreshing_the_pose_moves_the_sky_with_the_mount(pass_frames: Any) -> None:
    """Re-aiming the pedestal must move the ray elevations, or the sky is left behind.

    `IrCamera` caches the pose that every pixel's ray direction is built from. Without
    `refresh_pose` the geometry AOVs follow the re-aimed prim while the elevations, the sky
    temperature and every view cosine describe the previous aim -- a plausible frame of two
    different cameras. The two frames here are aimed 20 degrees apart in elevation, so the
    boresight elevation must differ by about that much.
    """
    camera = pass_frames["camera"]
    del camera
    elevations = []
    for key in ("head", "tail"):
        frame = pass_frames[key]["frame"]
        el = np.degrees(np.asarray(frame.elevation_rad, dtype=np.float64))
        elevations.append(float(el[el.shape[0] // 2, el.shape[1] // 2]))
    head_el, tail_el = elevations
    # Symmetric instants about closest approach are at the same true elevation, so these two must
    # *agree* -- and they only can if the pose was refreshed, because the mount swept 20 degrees
    # up and back down between them.
    assert head_el == pytest.approx(tail_el, abs=1.0), (
        f"boresight elevation {head_el:.2f} vs {tail_el:.2f} deg at mirrored instants"
    )
    expected = np.degrees(
        np.arcsin(
            pass_frames["head"]["sample"].position_m[1] / pass_frames["head"]["sample"].range_m
        )
    )
    assert head_el == pytest.approx(float(expected), abs=0.5), (
        "the boresight should sit on the target's true elevation"
    )
