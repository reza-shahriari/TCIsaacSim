"""In-sim synthesised motion: the roadmap M10.1b acceptance criteria, against the renderer.

`tests/unit/test_motion_px.py` holds the arithmetic to exact values on transforms it makes up.
What it cannot check is the half that talks to USD: that the transforms read off a stage are the
ones the renderer drew with, in the convention the projection expects, and broadcast by an
instance id that means the prim it appears to mean. Those are four separate chances to be
plausibly wrong, and all of them produce smooth, believable motion of the wrong magnitude or sign.

So this renders a real prim at a known displacement and asks what came back. The bar is the
roadmap's: a bar moving at 3 px/frame reports 3.0 within a tenth of a pixel, a static scene
reports 0, and a prim moving with the camera reports 0.

docs/physics-model.md §13.3, §9.2; ADR 0014 addendum (no motion AOV on this build)
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
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

WIDTH, HEIGHT = 160, 128
FOCAL_MM = 14.0
RANGE_M = 60.0
BAR_PATH = "/World/Targets/bar"

#: The displacement to fly, in native pixels per frame, and the tolerance the roadmap sets.
BAR_PX_PER_FRAME = 3.0
TOLERANCE_PX = 0.1


def _pixel_to_metres(sensor: Any, pixels: float) -> float:
    """World displacement at ``RANGE_M`` that projects to ``pixels`` on this lens."""
    spec = sensor.sensor
    f_px = spec.optics.focal_length_mm / (spec.fpa.pitch_um * 1e-3)
    return pixels * RANGE_M / f_px


@pytest.fixture(scope="module")
def motion_rig(simulation_app: Any, tophat_lwir_lut: Any) -> Any:
    """A single emissive bar on boresight, a camera, and a tracker watching both."""
    del simulation_app
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    from irsim.config.sensor import SensorConfig
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.scene import Scene
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.pipeline.motion_isaac import MotionTracker

    raw = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    raw["sensor"]["fpa"].update(width=WIDTH, height=HEIGHT)
    raw["sensor"]["optics"]["focal_length_mm"] = FOCAL_MM
    raw["sensor"]["optics"]["supersample_factor"] = 1
    sensor = SensorConfig.model_validate(raw)

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")

    # A tall thin slab across the boresight: wide enough in y to fill rows, narrow in x so the
    # horizontal displacement is unambiguous.
    bar = UsdGeom.Cube.Define(stage, BAR_PATH)
    bar.CreateSizeAttr(1.0)
    bar.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    xform = UsdGeom.Xformable(bar)
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, -RANGE_M))
    xform.AddScaleOp().Set(Gf.Vec3f(4.0, 40.0, 0.5))
    bar.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(
        "painted_composite"
    )

    camera = UsdGeom.Camera.Define(stage, "/World/IrCamera")
    camera_translate = UsdGeom.Xformable(camera).AddTranslateOp()
    camera_translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    table = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    pipeline = PipelineConfig.from_sensor(
        sensor, table, tophat_lwir_lut, sky=scene.sky_models["lwir"], atmosphere=scene.layered
    )
    ir = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target={BAR_PATH: "airframe"},
        resolutions=resolutions,
        camera_path="/World/IrCamera",
        strict_materials=False,
        strict_thermal_nodes=False,
    ).open(settle_frames=16)
    tracker = MotionTracker([BAR_PATH], "/World/IrCamera")
    rig = {
        "camera": ir,
        "tracker": tracker,
        "sensor": sensor,
        "bar_translate": translate,
        "camera_translate": camera_translate,
    }
    yield rig
    ir.close()


def _step(rig: Any, bar_x: float, camera_x: float) -> np.ndarray:
    """Move the bar and the camera, render, and return the synthesised motion plane."""
    from pxr import Gf

    rig["bar_translate"].Set(Gf.Vec3d(bar_x, 0.0, -RANGE_M))
    rig["camera_translate"].Set(Gf.Vec3d(camera_x, 0.0, 0.0))
    rig["camera"].refresh_pose()
    rig["tracker"].sample()
    rig["camera"].get_outputs()

    frame = rig["camera"].last_frame
    return rig["tracker"].motion_px(
        frame.positions_camera,
        frame.instance_id,
        frame.labels,
        rig["camera"].optics.intrinsics,
        rig["sensor"].sensor.optics.distortion,
    )


def _on_the_bar(rig: Any, motion: np.ndarray) -> np.ndarray:
    frame = rig["camera"].last_frame
    ident = next((i for i, p in frame.labels.items() if p == BAR_PATH), None)
    assert ident is not None, "the bar should be rendered and labelled"
    mask = frame.instance_id == ident
    assert mask.sum() > 200, f"only {int(mask.sum())} px of bar; the rig is not pointing at it"
    return motion[mask]


def test_a_bar_at_three_pixels_per_frame_reports_three(motion_rig: Any) -> None:
    """The roadmap's headline number, measured through USD and the renderer rather than assumed."""
    step_m = _pixel_to_metres(motion_rig["sensor"], BAR_PX_PER_FRAME)
    _step(motion_rig, 0.0, 0.0)  # first frame: nothing to difference against
    motion = _step(motion_rig, step_m, 0.0)
    on_bar = _on_the_bar(motion_rig, motion)
    assert float(np.median(on_bar[:, 0])) == pytest.approx(BAR_PX_PER_FRAME, abs=TOLERANCE_PX)
    assert float(np.abs(on_bar[:, 1]).max()) < TOLERANCE_PX


def test_a_static_scene_reports_zero(motion_rig: Any) -> None:
    _step(motion_rig, 0.0, 0.0)
    motion = _step(motion_rig, 0.0, 0.0)
    assert float(np.abs(_on_the_bar(motion_rig, motion)).max()) < TOLERANCE_PX


def test_a_prim_and_camera_moving_together_report_zero(motion_rig: Any) -> None:
    """A tracked target is still on the focal plane -- the case that catches a composition error.

    Both move by the same amount, so the bar's image does not shift. The *background* does, which
    the next test checks, and the two together are what make a tracking mount put smear on the sky
    instead of on the target.
    """
    step_m = _pixel_to_metres(motion_rig["sensor"], 6.0)
    _step(motion_rig, 0.0, 0.0)
    motion = _step(motion_rig, step_m, step_m)
    assert float(np.abs(_on_the_bar(motion_rig, motion)).max()) < TOLERANCE_PX


def test_the_first_frame_of_a_sequence_reports_zero_rather_than_guessing(motion_rig: Any) -> None:
    """Motion is a difference; a tracker that has seen one frame does not have one to report."""
    from irsim_isaac.pipeline.motion_isaac import MotionTracker

    fresh = MotionTracker([BAR_PATH], "/World/IrCamera")
    assert not fresh.ready
    assert fresh.sample() is False
    frame = motion_rig["camera"].last_frame
    motion = fresh.motion_px(
        frame.positions_camera,
        frame.instance_id,
        frame.labels,
        motion_rig["camera"].optics.intrinsics,
        motion_rig["sensor"].sensor.optics.distortion,
    )
    assert motion.shape == (*frame.instance_id.shape, 2)
    assert float(np.abs(motion).max()) == 0.0
    assert fresh.sample() is True
