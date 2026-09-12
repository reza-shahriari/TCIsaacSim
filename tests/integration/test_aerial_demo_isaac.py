"""Roadmap M10.19: in-sim aerial phenomenology on the demo stage.

Every claim here is about the *background*, because on a sky-target camera the background is most
of the image and is the thing the detector has to separate a target from. It is also where the
errors hide: a sky that is smooth, monotonic and completely wrong looks exactly like a sky.

The sharpest check is where the **horizon** falls. It is not free parameters -- with the camera
tilted by a known angle and a known focal length, elevation zero lands at
``cy + f_px * tan(tilt)`` and nowhere else. That single number caught the bug this stage was
built to find: ``Camera3dPositionSD`` returns positions in **camera** space, not world space as
ADR 0014 recorded, which only becomes visible once the camera is rotated. Read as world, the
frame centre's ray has zero elevation, so the horizon sits in the middle of the picture, the upper
half of the sky is painted with the ground temperature, and every ``normal_dot_view`` and
sky-view factor tilts with it -- a perfectly plausible frame.

docs/physics-model.md §15 T3, §5.3; ADR 0003 (aerial first), ADR 0044/MS.2 (the sky profile),
ADR 0060 (the thermal bridge), ADR 0014 addendum (the position frame).
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

WIDTH, HEIGHT = 320, 256
#: An 8 mm lens on this reduced format, not the Boson's 14 mm. The whole demo has to fit: with a
#: 14 mm lens a 256-row frame subtends +/-6.3 deg, so an 8 deg tilt puts the horizon 36 rows below
#: the bottom of the picture and the aircraft above the top of it. At 8 mm the half-field is
#: 10.9 deg, which holds the horizon, the sky gradient and every target at once.
FOCAL_MM = 8.0
SUPERSAMPLE = 2
TILT_DEG = 8.0

#: The horizon's predicted row follows from the tilt and the focal length alone; 3 px of slack
#: covers the softening from the PSF and the box filter, not any freedom in the geometry.
HORIZON_TOLERANCE_PX = 3.0


@pytest.fixture(scope="module")
def demo_camera(simulation_app: Any, tophat_lwir_lut: Any) -> Any:
    """`IrCamera` on the aerial demo stage, at a small format so the render is quick."""
    del simulation_app
    from irsim.config.sensor import SensorConfig
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.scene import Scene
    from irsim_isaac.aerial_demo import build_aerial_demo
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    raw = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    raw["sensor"]["fpa"].update(width=WIDTH, height=HEIGHT)
    raw["sensor"]["optics"]["focal_length_mm"] = FOCAL_MM
    raw["sensor"]["optics"]["supersample_factor"] = SUPERSAMPLE
    sensor = SensorConfig.model_validate(raw)

    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    demo = build_aerial_demo(camera_tilt_deg=TILT_DEG)
    assert demo.errors == {}, demo.errors

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
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        camera_path=demo.camera_path,
        strict_materials=False,
    ).open(settle_frames=16)
    camera.demo = demo  # type: ignore[attr-defined]
    yield camera
    camera.close()


@pytest.fixture(scope="module")
def frame(demo_camera: Any) -> Any:
    """One rendered frame, with a row profile taken over **background** pixels only.

    A plain row mean would be a statement about the targets as much as the sky: a warm drone
    lifts its own row and the profile stops being monotonic where the target is, which says
    nothing about the sky model. The instance-id plane already separates the two exactly (id 0
    is "hit nothing"), so the profile is the median of the background pixels in each row --
    median rather than mean so that a stray edge pixel cannot move it either.
    """
    out = demo_camera.get_outputs()
    assert out.apparent_t is not None
    t = np.asarray(out.apparent_t, dtype=np.float64)
    last = demo_camera.last_frame
    k = demo_camera.config.supersample
    background = last.instance_id == 0
    if k > 1:
        h, w = background.shape[0] // k, background.shape[1] // k
        background = background[: h * k, : w * k].reshape(h, k, w, k).all(axis=(1, 3))
    profile = np.array(
        [
            np.median(t[row, background[row]]) if background[row].any() else np.nan
            for row in range(t.shape[0])
        ]
    )
    assert np.all(np.isfinite(profile)), "every row should contain some background"
    return {"t": t, "profile": profile, "outputs": out, "frame": last, "background": background}


def horizon_row() -> float:
    """Where elevation zero lands: cy + f_px tan(tilt), in native pixels."""
    f_px = FOCAL_MM / 0.012
    return HEIGHT / 2.0 + f_px * math.tan(math.radians(TILT_DEG))


# --- the background -------------------------------------------------------------------------


def test_every_target_resolves_to_a_material(demo_camera: Any) -> None:
    """The stage authors `thermal:material` on each prim, so nothing should be UNMAPPED."""
    planes = demo_camera.planes()
    del planes
    last = demo_camera.last_frame
    assert last is not None
    assert not last.unmapped.any(), "a demo target has no material mapping"


def elevation_profile(frame: Any, demo_camera: Any) -> np.ndarray:
    """Per-native-row ray elevation in degrees, down the middle column of the G-buffer."""
    el = np.degrees(np.asarray(frame["frame"].elevation_rad, dtype=np.float64))
    k = demo_camera.config.supersample
    return el[::k, el.shape[1] // 2] if k > 1 else el[:, el.shape[1] // 2]


def test_the_boresight_points_where_the_camera_was_tilted(frame: Any, demo_camera: Any) -> None:
    """The ray at the frame centre must rise 8 degrees, because that is how the camera was posed.

    This is the assertion that would have caught the position-frame bug on its own.
    ``Camera3dPositionSD`` is in **camera** space, so reading it as world leaves the boresight
    flat at 0 degrees no matter how the camera is posed -- and every derived quantity, the sky
    temperature, the sky-view factor, the view cosine, quietly describes a camera pointing
    somewhere else.
    """
    el = elevation_profile(frame, demo_camera)
    assert float(el[HEIGHT // 2]) == pytest.approx(TILT_DEG, abs=0.05)


def test_the_horizon_falls_where_the_geometry_puts_it(frame: Any, demo_camera: Any) -> None:
    """Elevation zero at cy + f_px tan(tilt), to within a pixel. No free parameters."""
    el = elevation_profile(frame, demo_camera)
    measured = float(np.argmin(np.abs(el)))
    expected = horizon_row()
    assert expected < HEIGHT, "the stage should put the horizon inside the frame"
    assert abs(measured - expected) < HORIZON_TOLERANCE_PX, (
        f"elevation crosses zero at row {measured}, geometry says {expected:.1f}"
    )
    assert float(el[0]) > float(el[-1]), "elevation must fall down the frame"


def test_the_horizon_is_not_a_visible_edge(frame: Any) -> None:
    """A grazing clear sky converges on the ground temperature, so nothing marks the horizon.

    Worth asserting because it is counter-intuitive and because it invalidates the obvious way to
    find the horizon in an image. The sky's path length grows without bound toward zero elevation,
    so its emissivity approaches one at roughly the air temperature -- which is what the ground is
    at too. The step across the horizon is therefore small next to the sky gradient over the frame,
    and an algorithm that looked for the strongest edge would find a place in the sky instead.
    """
    row = int(horizon_row())
    profile = frame["profile"]
    step = abs(float(profile[row + 6]) - float(profile[row - 6]))
    gradient_span = float(profile[row - 8]) - float(profile[0])
    assert step < 0.5 * gradient_span, (
        f"horizon step {step:.2f} K against a {gradient_span:.2f} K sky gradient"
    )


def test_the_sky_gets_colder_with_elevation_all_the_way_up(frame: Any) -> None:
    """MS.2 in-sim: apparent sky temperature falls monotonically as the ray rises.

    Monotone over every row above the horizon, not merely colder at the top -- a frame whose
    upper half had been overwritten with the ground temperature would still be "colder at the
    top" and would fail this.
    """
    sky = frame["profile"][: int(horizon_row()) - 8]
    assert len(sky) > 50
    steps = np.diff(sky)
    assert np.all(steps > 0.0), "the sky profile must warm monotonically toward the horizon"
    assert sky[0] < sky[-1] - 20.0, f"only {sky[-1] - sky[0]:.1f} K across the sky gradient"


def test_the_ground_below_the_horizon_is_one_temperature(frame: Any) -> None:
    """Phase 1 has no ground thermal model: below the horizon is a single T_ground (ADR 0060)."""
    ground = frame["profile"][int(horizon_row()) + 8 :]
    assert len(ground) > 20
    assert float(np.ptp(ground)) < 0.5, f"ground varies by {float(np.ptp(ground)):.2f} K"
    sky_mean = float(frame["profile"][: int(horizon_row()) - 8].mean())
    assert float(ground.mean()) > sky_mean


def test_the_sky_is_colder_than_the_air_and_the_ground_is_not(frame: Any, demo_camera: Any) -> None:
    """A clear LWIR sky reads far below ambient, the ground reads at it: the scene's whole sign.

    If these two ever came out the same way round, a target warmer than ambient would silhouette
    against the wrong background and every detection statistic downstream would be measuring
    something else.
    """
    weather = demo_camera.scene.weather_at(demo_camera.t_rel_s)
    t_air = float(weather.t_air_k)
    sky_top = float(frame["profile"][:16].mean())
    ground = float(frame["profile"][int(horizon_row()) + 8 :].mean())
    assert sky_top < t_air - 25.0, f"sky {sky_top:.1f} K against air {t_air:.1f} K"
    assert abs(ground - t_air) < 15.0, f"ground {ground:.1f} K against air {t_air:.1f} K"


# --- the targets ----------------------------------------------------------------------------


def target_patch(frame: Any, demo_camera: Any, name: str) -> np.ndarray | None:
    """Apparent temperature over the pixels of one target prim, or None if it is not resolved."""
    last = frame["frame"]
    path = demo_camera.demo.targets[name].prim_path
    ident = next((i for i, p in last.labels.items() if p == path), None)
    if ident is None:
        return None
    k = demo_camera.config.supersample
    mask = last.instance_id == ident
    if k > 1:
        h, w = mask.shape[0] // k, mask.shape[1] // k
        mask = mask[: h * k, : w * k].reshape(h, k, w, k).any(axis=(1, 3))
    return frame["t"][mask] if mask.any() else None


def test_the_resolved_targets_stand_out_against_the_sky(frame: Any, demo_camera: Any) -> None:
    """A 296 K airframe against a 250-270 K sky is strong positive contrast, and must read so."""
    seen = 0
    for name in ("drone_near", "aircraft"):
        patch = target_patch(frame, demo_camera, name)
        if patch is None or patch.size < 4:
            continue
        seen += 1
        local_sky = float(frame["profile"][: int(horizon_row()) - 8].mean())
        assert float(np.median(patch)) > local_sky + 5.0, (
            f"{name}: {float(np.median(patch)):.1f} K against sky {local_sky:.1f} K"
        )
    assert seen == 2, "both resolved targets should be visible in the frame"


def test_a_sub_pixel_target_is_diluted_rather_than_missing(frame: Any, demo_camera: Any) -> None:
    """0.27 px of drone spread over a whole pixel loses most of its contrast -- why MS.6 exists.

    The renderer samples geometry; it does not know that a target smaller than a pixel should
    contribute its own radiance weighted by the fill fraction. Whatever the renderer does give,
    the sub-pixel target must read weaker than the resolved one at a quarter of the range, and
    that gap is the thing the analytic point-target path (MS.6) exists to replace.
    """
    ifov_mrad = 1e3 * 0.012 / FOCAL_MM  # 1.5 mrad on this lens
    near = demo_camera.demo.targets["drone_near"]
    far = demo_camera.demo.targets["drone_far"]
    assert not near.subpixel_at(ifov_mrad) and far.subpixel_at(ifov_mrad)

    near_patch = target_patch(frame, demo_camera, "drone_near")
    far_patch = target_patch(frame, demo_camera, "drone_far")
    assert near_patch is not None and near_patch.size >= 4
    sky = float(frame["profile"][: int(horizon_row()) - 8].mean())
    near_contrast = float(np.median(near_patch)) - sky
    far_contrast = 0.0 if far_patch is None else float(np.median(far_patch)) - sky
    assert near_contrast > far_contrast, (
        f"sub-pixel target at {far.range_m:.0f} m reads {far_contrast:.2f} K over sky, "
        f"resolved target at {near.range_m:.0f} m reads {near_contrast:.2f} K"
    )


def test_the_four_outputs_come_back_at_the_declared_dtypes(frame: Any) -> None:
    """§12.2 on a real scene rather than a test stage."""
    out = frame["outputs"]
    assert out.radiance is not None and out.radiance.dtype == np.float32
    assert out.apparent_t is not None and out.apparent_t.dtype == np.float32
    assert out.dn16 is not None and out.dn16.dtype == np.uint16
    assert out.display8 is not None and out.display8.dtype == np.uint8
    assert out.display8.shape == (HEIGHT, WIDTH, 4)
    assert np.all(np.isfinite(out.apparent_t))
