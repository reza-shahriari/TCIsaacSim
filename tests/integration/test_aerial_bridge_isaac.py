"""Roadmap M10.18: the thermal bridge closed against real rendered instance ids and rays.

The bridge's arithmetic is covered engine-free in ``tests/unit/test_aerial_bridge.py``. What needs
a renderer is the closure: that the ids the renderer emits, routed through ``idToLabels`` and the
per-prim solver map, put each authored temperature back on exactly the pixels of its own prim, and
that background pixels take the MS.2 sky profile evaluated from their own ray direction.

The stage is ``material_probe``'s five quads, reused rather than duplicated -- M10.18 needs several
separately identifiable prims and a visible background, which is exactly what that scene provides.
The prim names are material-flavoured; here they are simply five distinct thermal facets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

RESOLUTION = 256
READBACK_TOL_K = 0.010
SKY_TOL_K = 0.5

#: One prescribed temperature per quad, spanning the range an aerial scene actually spans, and
#: including the 317.25 K the M10.18 verification names.
TARGET_K = {
    "Body": 317.25,
    "Window": 268.40,
    "Road": 301.00,
    "Trim": 295.55,
    "Mystery": 330.10,
}


@pytest.fixture(scope="module")
def scene(simulation_app: Any) -> Any:
    del simulation_app
    from irsim_isaac.material_probe import build_material_scene

    built = build_material_scene(resolution=RESOLUTION)
    assert built.errors == {}, built.errors
    return built


@pytest.fixture(scope="module")
def irsim_scene(tophat_lwir_lut: Any) -> Any:
    """An irsim Scene whose targets are the five quads, each pinned at its own temperature."""
    import pathlib

    from irsim.config.scene import SceneConfig, TargetSpec, load_scene_config
    from irsim.scene import Scene

    repo = pathlib.Path(__file__).resolve().parents[2]
    config = load_scene_config(repo / "configs" / "scenes" / "sky_target_clear_day.yaml")
    targets = [
        TargetSpec(
            name=name.lower(),
            solver="prescribed",
            schedule_s=[0.0, 7200.0],
            schedule_k=[t, t],
        )
        for name, t in TARGET_K.items()
    ]
    spec = config.scene.model_copy(update={"targets": targets})
    return Scene.from_config(
        SceneConfig(schema_version=config.schema_version, scene=spec), {"lwir": tophat_lwir_lut}
    )


@pytest.fixture(scope="module")
def frame(scene: Any) -> Any:
    """One settled render: instance ids, their label table, and the per-pixel ray elevation."""
    import omni.replicator.core as rep

    from irsim_isaac.geometry_probe import configure_renderer
    from irsim_isaac.pipeline.aerial_bridge import elevation_from_rays
    from irsim_isaac.pipeline.gbuffer_isaac import AovReader, ray_directions
    from irsim_isaac.pipeline.material_ids import labels_from_payload

    configure_renderer()
    rp = rep.create.render_product(scene.camera_path, (RESOLUTION, RESOLUTION))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    reader = AovReader(
        rp_path,
        device="cpu",
        required=("instance", "position"),
        expected_shape=(RESOLUTION, RESOLUTION),
    ).attach(settle_frames=24)
    aovs = reader.read()

    rays = ray_directions(aovs.position, frame="world", camera_position=scene.camera_position)
    yield {
        "aovs": aovs,
        "labels": {
            str(k): v for k, v in labels_from_payload(aovs.device_handles["instance"]).items()
        },
        "elevation": elevation_from_rays(rays, up=(0.0, 1.0, 0.0)),
        "sky_mask": np.asarray(aovs.instance_id) == 0,
    }
    reader.detach()


@pytest.fixture(scope="module")
def bridge(irsim_scene: Any, scene: Any, frame: Any) -> Any:
    from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge

    prim_to_target = {t.prim_path: name.lower() for name, t in scene.targets.items()}
    built = AerialThermalBridge(irsim_scene, prim_to_target, band="lwir")
    built.advance_to(2.0)
    return built


# --- the closure ---------------------------------------------------------------------------------


def test_each_prim_reads_back_its_own_authored_temperature(
    scene: Any, frame: Any, bridge: Any
) -> None:
    """317.25 K authored on one quad must come back as 317.25 K on that quad's pixels only."""
    plane = bridge.temperature_plane(
        frame["aovs"].instance_id,
        frame["labels"],
        sky_mask=frame["sky_mask"],
        elevation_rad=frame["elevation"],
    )
    assert plane.dtype == np.float32
    for name, expected in TARGET_K.items():
        rows, cols = scene.patch_of(name, half_px=3)
        patch = plane[rows, cols].astype(np.float64)
        assert np.max(np.abs(patch - expected)) < READBACK_TOL_K, (
            name,
            float(patch.mean()),
            expected,
        )


def test_the_five_prims_are_five_distinct_temperatures(scene: Any, frame: Any, bridge: Any) -> None:
    """A collapsed id channel would give every quad the same temperature and still look fine."""
    plane = bridge.temperature_plane(
        frame["aovs"].instance_id, frame["labels"], sky_mask=frame["sky_mask"]
    )
    seen = {name: float(plane[scene.patch_of(name, 2)].mean()) for name in TARGET_K}
    assert len({round(v, 3) for v in seen.values()}) == len(TARGET_K), seen


def test_a_rendered_prim_with_no_thermal_node_raises(
    irsim_scene: Any, scene: Any, frame: Any
) -> None:
    """Mapping only some prims must fail loudly, naming the ones left without a temperature."""
    from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge

    partial = {scene.targets["Body"].prim_path: "body"}
    bridge = AerialThermalBridge(irsim_scene, partial, band="lwir")
    with pytest.raises(KeyError, match="no thermal node"):
        bridge.facet_table(frame["labels"])


# --- the sky -------------------------------------------------------------------------------------


def test_sky_pixels_follow_the_ms2_elevation_profile(frame: Any, bridge: Any) -> None:
    """Background pixels take T_sky of their own ray, not one number for the whole frame."""
    plane = bridge.temperature_plane(
        frame["aovs"].instance_id,
        frame["labels"],
        sky_mask=frame["sky_mask"],
        elevation_rad=frame["elevation"],
    )
    sky = frame["sky_mask"]
    assert sky.sum() > 1000, "the scene must leave a real background"

    expected = bridge.background_temperature_k(frame["elevation"])
    assert np.max(np.abs(plane[sky].astype(np.float64) - expected[sky])) < SKY_TOL_K


def test_the_sky_is_colder_higher_up(frame: Any, bridge: Any) -> None:
    """A clear sky cools with elevation; a constant sky would pass the per-pixel test trivially."""
    sky = frame["sky_mask"]
    elevation = frame["elevation"]
    plane = bridge.temperature_plane(
        frame["aovs"].instance_id,
        frame["labels"],
        sky_mask=sky,
        elevation_rad=elevation,
    ).astype(np.float64)

    low = sky & (elevation < np.deg2rad(-10.0))
    high = sky & (elevation > np.deg2rad(10.0))
    if low.sum() < 50 or high.sum() < 50:
        pytest.skip("the field does not span enough elevation on this scene")
    assert plane[high].mean() < plane[low].mean(), (plane[high].mean(), plane[low].mean())


def test_geometry_pixels_are_not_overwritten_by_the_sky(
    scene: Any, frame: Any, bridge: Any
) -> None:
    plane = bridge.temperature_plane(
        frame["aovs"].instance_id,
        frame["labels"],
        sky_mask=frame["sky_mask"],
        elevation_rad=frame["elevation"],
    )
    rows, cols = scene.patch_of("Mystery", half_px=3)
    assert abs(float(plane[rows, cols].mean()) - TARGET_K["Mystery"]) < READBACK_TOL_K


# --- one weather object --------------------------------------------------------------------------


def test_a_sky_model_on_different_weather_is_refused(
    irsim_scene: Any, scene: Any, tophat_lwir_lut: Any
) -> None:
    """CLAUDE.md #6 holds through the engine path too, not only in the engine-free tests."""
    import pathlib

    from irsim.config.scene import SceneConfig, TargetSpec, load_scene_config
    from irsim.scene import Scene
    from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge

    repo = pathlib.Path(__file__).resolve().parents[2]
    config = load_scene_config(repo / "configs" / "scenes" / "sky_target_clear_day.yaml")
    targets = [
        TargetSpec(name=n.lower(), solver="prescribed", schedule_s=[0.0, 7200.0], schedule_k=[t, t])
        for n, t in TARGET_K.items()
    ]
    other = Scene.from_config(
        SceneConfig(
            schema_version=config.schema_version,
            scene=config.scene.model_copy(update={"targets": targets}),
        ),
        {"lwir": tophat_lwir_lut},
    )
    assert other.weather is not irsim_scene.weather
    prim_to_target = {t.prim_path: name.lower() for name, t in scene.targets.items()}
    with pytest.raises(ValueError, match="different WeatherSeries"):
        AerialThermalBridge(irsim_scene, prim_to_target, sky=other.sky_models["lwir"])
