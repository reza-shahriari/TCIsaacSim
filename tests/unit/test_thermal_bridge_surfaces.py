"""M10.3: a rendered prim can take its temperature from the §12.3 thermal solver.

Phase 1 (M10.18) mapped a prim to one of the scene's *target solvers* -- a scripted drone motor, a
relaxing airframe. Phase 2 lets a prim map to a §12.3 **thermal surface** instead, so a rendered
roof is the one M6.12's energy balance solved rather than a number somebody typed beside it.

Both kinds live in one map, because "a thing with a temperature" is all the G-buffer cares about
and a stage should be able to mix a solved facet with a scripted drone without sorting them. What
the tests below pin is the part that can go wrong silently: the two kinds use **different time
bases**, and a name that is both kinds is refused rather than resolved by precedence.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.scene import Scene
from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge

REPO = pathlib.Path(__file__).resolve().parents[2]
FACET_SCENE = REPO / "configs" / "scenes" / "thermal_facet_scene.yaml"
BOSON = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


@pytest.fixture(scope="module")
def facet_scene():  # type: ignore[no-untyped-def]
    from irsim.config.loader import load_sensor_config

    sensor = load_sensor_config(BOSON)
    lut = BandLUT.build(load_spectral_response(sensor.sensor.band.spectral_response), n=1501)
    return Scene.from_file(FACET_SCENE, {"lwir": lut})


def test_the_scene_has_solved_surfaces_to_map_to(facet_scene) -> None:  # type: ignore[no-untyped-def]
    assert facet_scene.thermal is not None
    assert len(facet_scene.thermal_surfaces) >= 5
    assert "roof_black" in facet_scene.thermal_surfaces


def test_a_prim_can_take_its_temperature_from_the_thermal_solver(facet_scene) -> None:  # type: ignore[no-untyped-def]
    """The step itself: a prim mapped to a surface reads M6.12's energy balance, to the kelvin."""
    bridge = AerialThermalBridge(facet_scene, {"/World/Roof": "roof_black"}, band="lwir")
    assert bridge.surface_names == {"roof_black"}
    temperatures = bridge.temperatures()
    assert "roof_black" in temperatures
    expected = facet_scene.surface_temperature_k("roof_black", facet_scene.t0_s)
    assert temperatures["roof_black"] == pytest.approx(expected, abs=1e-6)


def test_the_surface_lookup_uses_absolute_time_and_the_targets_relative(facet_scene) -> None:  # type: ignore[no-untyped-def]
    """The trap this step introduces, pinned.

    The tick bracket runs on time *relative* to the scene start; `ThermalField.temperature_at`
    takes weather-axis *absolute* time. Use the wrong one and the render shows a different hour of
    the day with no other symptom -- on a diurnal scene, hours of error in the surface temperature.
    """
    bridge = AerialThermalBridge(facet_scene, {"/World/Roof": "roof_black"}, band="lwir")
    bridge.advance_to(3600.0)
    got = bridge.temperatures()["roof_black"]
    assert got == pytest.approx(
        facet_scene.surface_temperature_k("roof_black", facet_scene.t0_s + 3600.0), abs=1e-6
    )
    # ...and the relative-time answer is not merely different, it does not exist: this scene
    # starts a day into its weather file, so a relative lookup falls before the field's own start
    # and raises. That is luck rather than a guarantee -- on a scene whose `t0_s` is zero the same
    # mistake would return a plausible number from the wrong hour with no symptom at all, which is
    # why the convention is written into `temperatures`' docstring and pinned here.
    assert facet_scene.t0_s > 0.0
    with pytest.raises(ValueError, match="before the field's start"):
        facet_scene.surface_temperature_k("roof_black", 3600.0)
    # The field had to be *advanced* to be readable there: it refuses a query past its last tick
    # rather than solving on demand, so a bridge that only read would have raised.
    assert facet_scene.thermal.latest_t_s >= facet_scene.t0_s + 3600.0


def test_a_name_that_is_both_kinds_is_refused_rather_than_resolved(facet_scene) -> None:  # type: ignore[no-untyped-def]
    """Whichever kind won a precedence rule, the other would be silently ignored."""
    from irsim.thermal.aerial import airframe_solver

    targets = dict(facet_scene.targets)
    targets["roof_black"] = airframe_solver(facet_scene.weather, offset_k=2.0)
    clashing = type(facet_scene)(
        spec=facet_scene.spec,
        weather=facet_scene.weather,
        grey_atmosphere=facet_scene.grey_atmosphere,
        targets=targets,
        t0_s=facet_scene.t0_s,
        layered=facet_scene.layered,
        environment=facet_scene.environment,
        sky_models=facet_scene.sky_models,
        thermal=facet_scene.thermal,
        thermal_surfaces=facet_scene.thermal_surfaces,
    )
    with pytest.raises(ValueError, match="ambiguous thermal names"):
        AerialThermalBridge(clashing, {"/World/Roof": "roof_black"}, band="lwir")


def test_an_unmapped_name_names_both_kinds_in_the_error(facet_scene) -> None:
    """The error has to say what *is* available, or a typo costs a debugging session."""
    with pytest.raises(ValueError, match="targets .* and thermal surfaces"):
        AerialThermalBridge(facet_scene, {"/World/X": "no_such_thing"}, band="lwir")


def test_a_surface_contributes_no_tick_interpolation_error(facet_scene) -> None:
    """Surfaces are not ticked: the ThermalField is integrated once over the whole scene and read
    by interpolation, so this bridge adds nothing to their error budget."""
    bridge = AerialThermalBridge(facet_scene, {"/World/Roof": "roof_black"}, band="lwir")
    assert bridge.tick_error_k("roof_black") == 0.0


def test_the_facet_table_carries_a_surface_temperature_into_the_plane(facet_scene) -> None:
    """End to end: the number reaches `temperature_k`, which is the only thing stage 1 reads."""
    bridge = AerialThermalBridge(facet_scene, {"/World/Roof": "roof_black"}, band="lwir")
    ids = np.array([[1, 1], [1, 1]], dtype=np.int32)
    plane = bridge.temperature_plane(ids, {1: "/World/Roof"}, sky_mask=np.zeros((2, 2), bool))
    assert plane.dtype == np.float32
    assert float(plane[0, 0]) == pytest.approx(
        facet_scene.surface_temperature_k("roof_black", facet_scene.t0_s), abs=1e-3
    )
