"""IG.1 -- ``IrCamera``'s three strictness guards are three flags, and one cannot silence another.

A frame can be wrong in three unrelated ways: a prim the renderer drew that the label table does
not name, a prim with no thermal node, and a pixel on a field-backed prim that falls outside every
one of its patches. Until IG.1 all three were behind ``strict_materials``, and all six render
scripts passed it ``False`` to get past the first two -- so the third, the one whose own docstring
calls the fallback "a seam that looks like physics", was off in every frame this project has
produced.

The tests drive the real :meth:`IrCamera.planes` over a synthetic frame rather than asserting on
the attributes, because the defect being guarded against is a *routing* one: three flags that are
all stored and then all read from the same place would pass an attribute test and fail here.

docs/physics-model.md §13.1; ADR 0047 (UNMAPPED takes eps = 1), ADR 0087 (the field), roadmap IG.1.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from irsim.materials.mapping import Resolution
from irsim.pipeline.core import PipelineConfig
from irsim.scene import Scene
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField
from irsim_isaac.pipeline.gbuffer_isaac import RawAovs
from irsim_isaac.pipeline.ir_camera import IrCamera
from irsim_isaac.pipeline.point_bridge import SurfaceBinding

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

ROAD_PATH = "/World/road"
ROAD_ID = 3
#: A prim the renderer drew whose id never reached ``idToLabels`` -- the ``strict_materials`` case.
STRANGER_ID = 9

EX = np.array([1.0, 0.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})


def _patch(n_u: int = 6, n_v: int = 6, d: float = 0.5) -> PlanarPatch:
    return PlanarPatch(
        origin_m=np.zeros(3),
        u_axis=EX,
        v_axis=EZ,
        n_u=n_u,
        n_v=n_v,
        du_m=d,
        dv_m=d,
        thickness_m=0.2,
    )


def _frozen_field(patch: PlanarPatch, lo: float = 290.0, hi: float = 320.0) -> PlanarThermalField:
    """A field whose cells hold a known ramp: the capacity is huge, so the initial state stands."""
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, 1e9),
        emissivity=np.full(patch.n_cells, 0.95),
        solar_absorptivity=np.full(patch.n_cells, 0.9),
    )
    return PlanarThermalField(
        patch,
        props,
        lambda _t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0),
        0.0,
        np.linspace(lo, hi, patch.n_cells),
    )


class _FakeReader:
    """Stands in for ``AovReader``: ``planes(step=False)`` only ever calls ``read``."""

    def __init__(self, aovs: RawAovs) -> None:
        self._aovs = aovs

    def read(self) -> RawAovs:
        return self._aovs


def _aovs(patch: PlanarPatch, *, stranger: bool = False, shift_m: float = 0.0) -> RawAovs:
    """One prim filling the frame, one pixel per cell, positions in world space.

    ``shift_m`` slides the surface off its patch -- the geometry is where it always was and the
    patch was authored too small, which is the failure ``strict_patch_coverage`` exists to catch.
    """
    centres = patch.cell_centres().reshape(*patch.shape, 3) + np.array([shift_m, 0.0, 0.0])
    shape = patch.shape
    ids = np.full(shape, ROAD_ID, dtype=np.uint32)
    if stranger:
        ids[0, 0] = STRANGER_ID
    normal = np.broadcast_to(np.array([0.0, 1.0, 0.0], dtype=np.float32), (*shape, 3)).copy()
    return RawAovs(
        distance_m=np.full(shape, 5.0, dtype=np.float32),
        normal=normal,
        position=centres.astype(np.float32),
        instance_id=ids,
        device_handles={"instance": {"info": {"idToLabels": {"0": "BACKGROUND", "3": ROAD_PATH}}}},
    )


def _camera(
    sensor: Any,
    scene: Any,
    materials: Any,
    lut: Any,
    *,
    bind_field: PlanarThermalField | None = None,
    thermal_node: bool = True,
    **strictness: bool,
) -> IrCamera:
    camera = IrCamera(
        sensor,
        scene,
        pipeline=PipelineConfig.from_sensor(sensor, materials, lut),
        prim_to_target={ROAD_PATH: "airframe"} if thermal_node else {},
        resolutions=[
            Resolution(path=ROAD_PATH, material="aluminium", material_id=1, rule="override")
        ],
        position_frame="world",
        up_axis="Y",
        surface_fields=[SurfaceBinding(ROAD_PATH, bind_field)] if bind_field is not None else (),
        **strictness,
    )
    camera._up_axis = "Y"
    # What `open()` would have read off the prim. World frame, so the position AOV passes
    # through untouched and only the ray directions use this.
    camera._camera_position = np.array([1.25, 5.0, 1.25])
    return camera


def _render(camera: IrCamera, aovs: RawAovs) -> Any:
    camera._reader = _FakeReader(aovs)  # type: ignore[assignment]
    return camera.planes(step=False)


# --- each guard fires on its own failure, and only on its own ---------------------------------


def test_an_unlabelled_prim_trips_only_the_material_guard(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, aerial_materials, tophat_lwir_lut
) -> None:
    patch = _patch()
    aovs = _aovs(patch, stranger=True)

    with pytest.raises(KeyError, match="not in idToLabels"):
        _render(_camera(aerial_sensor, scene, aerial_materials, tophat_lwir_lut), aovs)

    # The other two guards have nothing to say about it: with only the material one relaxed, the
    # frame assembles. If the flags were still one, this would raise for the wrong reason.
    planes = _render(
        _camera(aerial_sensor, scene, aerial_materials, tophat_lwir_lut, strict_materials=False),
        aovs,
    )
    assert planes["temperature_k"].shape == patch.shape


def test_a_prim_with_no_thermal_node_trips_only_the_node_guard(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, aerial_materials, tophat_lwir_lut
) -> None:
    aovs = _aovs(_patch())

    with pytest.raises(KeyError, match="no thermal node"):
        _render(
            _camera(aerial_sensor, scene, aerial_materials, tophat_lwir_lut, thermal_node=False),
            aovs,
        )

    planes = _render(
        _camera(
            aerial_sensor,
            scene,
            aerial_materials,
            tophat_lwir_lut,
            thermal_node=False,
            strict_thermal_nodes=False,
        ),
        aovs,
    )
    # `fill_k` is 0 K on purpose: loudly wrong beats plausibly wrong (ADR 0047's sibling rule).
    assert float(planes["temperature_k"].max()) == 0.0


def test_a_pixel_outside_every_patch_trips_only_the_coverage_guard(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, aerial_materials, tophat_lwir_lut
) -> None:
    patch = _patch()
    # A metre off: every pixel of the last two columns leaves the patch entirely.
    aovs = _aovs(patch, shift_m=1.0)

    with pytest.raises(ValueError, match="fall outside every patch"):
        _render(
            _camera(
                aerial_sensor,
                scene,
                aerial_materials,
                tophat_lwir_lut,
                bind_field=_frozen_field(patch),
            ),
            aovs,
        )

    planes = _render(
        _camera(
            aerial_sensor,
            scene,
            aerial_materials,
            tophat_lwir_lut,
            bind_field=_frozen_field(patch),
            strict_patch_coverage=False,
        ),
        aovs,
    )
    assert np.isfinite(planes["temperature_k"]).all()


# --- the regression IG.1 exists for ------------------------------------------------------------


def test_relaxing_the_other_two_leaves_patch_coverage_on(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, aerial_materials, tophat_lwir_lut
) -> None:
    """Exactly what the six render scripts now ask for, against exactly the frame that was silent.

    Before IG.1 this call was spelled ``strict_materials=False`` and bought silence on all three.
    """
    patch = _patch()
    with pytest.raises(ValueError, match="fall outside every patch"):
        _render(
            _camera(
                aerial_sensor,
                scene,
                aerial_materials,
                tophat_lwir_lut,
                bind_field=_frozen_field(patch),
                strict_materials=False,
                strict_thermal_nodes=False,
            ),
            _aovs(patch, stranger=True, shift_m=1.0),
        )


def test_a_covered_surface_still_reads_its_own_cells(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, aerial_materials, tophat_lwir_lut
) -> None:
    """The guard must not be passing by refusing to sample: the field still reaches the plane."""
    patch = _patch()
    field = _frozen_field(patch)
    camera = _camera(
        aerial_sensor,
        scene,
        aerial_materials,
        tophat_lwir_lut,
        bind_field=field,
        strict_materials=False,
        strict_thermal_nodes=False,
    )
    planes = _render(camera, _aovs(patch))
    got = np.asarray(planes["temperature_k"], dtype=np.float64)
    # The field's own state at the **absolute** clock the camera ran it to. Sampling the oracle at
    # t = 0 instead misses by a few mK: `t0_s` is hours into the weather axis and even a 1e9
    # J/m^2K slab radiates a little over that span (M10.3's trap, and it is a real difference).
    at = scene.t0_s + camera.t_rel_s
    expected = np.asarray(field.temperature_at(at), dtype=np.float64).reshape(patch.shape)
    # Bilinear sampling is exact at cell centres, so the only slack is the float32 the plane and
    # the position AOV are carried in -- 3e-5 K at 320 K, two orders under the 50 mK NETD.
    assert np.max(np.abs(got - expected)) < 1e-4
    # And it is a gradient, not one value per prim -- the defect ADR 0087 exists to fix.
    assert got.max() - got.min() > 25.0
