"""``IrCamera``'s engine-free half (M10.9a-ii): the sensor config → USD camera mapping.

Everything here runs without Isaac Sim, which is the point: the config→USD mapping is where a
camera silently becomes a different camera, and catching that should not cost a 35 s Kit boot.
The in-sim closure -- that the renderer honours what was written -- is
``tests/integration/test_ir_camera_isaac.py``.

docs/physics-model.md §8.3, §8.4; ADR 0015 addendum.
"""

from __future__ import annotations

import pathlib

import pytest

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import DistortionSpec
from irsim.optics.projection import Intrinsics
from irsim.pipeline.core import PipelineConfig
from irsim.scene import Scene
from irsim_isaac.pipeline.ir_camera import (
    DISTORTION_SCHEMA,
    IrCamera,
    camera_optics,
    distortion_attributes,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

#: The full attribute set of ``OmniLensDistortionOpenCvPinholeAPI`` (measured; see
#: ``scripts/probe_isaac_camera.py``). The mapping must write **all** of it.
PINHOLE_ATTRS = {
    "fx",
    "fy",
    "cx",
    "cy",
    "imageSize",
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "p1",
    "p2",
    "s1",
    "s2",
    "s3",
    "s4",
}


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_sensor_config(BOSON_YAML)


# --- the camera prim ------------------------------------------------------------------------


def test_aperture_is_the_physical_sensor_size(boson) -> None:  # type: ignore[no-untyped-def]
    """640 × 12 µm = 7.68 mm across, 512 × 12 µm = 6.144 mm down."""
    optics = camera_optics(boson.sensor)
    assert optics.horizontal_aperture_mm == pytest.approx(7.68, abs=1e-12)
    assert optics.vertical_aperture_mm == pytest.approx(6.144, abs=1e-12)
    assert optics.focal_length_mm == pytest.approx(14.0)


def test_focal_length_in_pixels_follows_from_the_usd_aperture_ratio(boson) -> None:  # type: ignore[no-untyped-def]
    """USD derives f_px from focalLength/horizontalAperture × width; it must equal f/pitch.

    This is the identity that ties the prim to the radiometry. If the aperture were left at USD's
    35 mm default, the same stage would image a 3.6× wider field while every field angle, cos⁴
    weight and atmospheric path length in the core continued to describe the narrow one.
    """
    optics = camera_optics(boson.sensor)
    width = optics.resolution[0]
    implied = optics.focal_length_mm / optics.horizontal_aperture_mm * width
    assert implied == pytest.approx(optics.intrinsics.fx_px, rel=1e-12)
    assert optics.intrinsics.fx_px == pytest.approx(14.0 / 0.012 * optics.supersample, rel=1e-12)


def test_render_product_is_the_supersampled_grid(boson) -> None:  # type: ignore[no-untyped-def]
    """The renderer is asked for k × native (§8.3), the grid ``run_frame`` demands."""
    k = boson.sensor.optics.supersample_factor
    optics = camera_optics(boson.sensor)
    assert optics.resolution == (640 * k, 512 * k)
    assert optics.supersample == k


def test_supersampling_does_not_change_the_aperture(boson) -> None:  # type: ignore[no-untyped-def]
    """Subdividing pixels is not a different lens: the physical aperture is unchanged."""
    one = camera_optics(boson.sensor, 1)
    four = camera_optics(boson.sensor, 4)
    assert four.horizontal_aperture_mm == pytest.approx(one.horizontal_aperture_mm)
    assert four.focal_length_mm == pytest.approx(one.focal_length_mm)
    assert four.intrinsics.fx_px == pytest.approx(4.0 * one.intrinsics.fx_px)


# --- the distortion schema ------------------------------------------------------------------


def test_brown_conrady_maps_to_the_opencv_pinhole_schema(boson) -> None:  # type: ignore[no-untyped-def]
    schema, _ = distortion_attributes(
        boson.sensor.optics.distortion, camera_optics(boson.sensor).intrinsics
    )
    assert schema == "OmniLensDistortionOpenCvPinholeAPI"
    assert DISTORTION_SCHEMA["kannala_brandt"] == "OmniLensDistortionOpenCvFisheyeAPI"


def test_every_attribute_of_the_schema_is_written(boson) -> None:  # type: ignore[no-untyped-def]
    """All 17, not just the coefficients -- an unwritten attribute is someone else's camera.

    Measured on 6.1.0-rc.26: the schema defaults are fx = 900, cx = 1024,
    imageSize = (2048, 1024). Writing only the five Brown–Conrady terms would leave a lens for a
    2048 × 1024 sensor on a 640 × 512 camera, which renders without complaint.
    """
    intr = camera_optics(boson.sensor).intrinsics
    _, attrs = distortion_attributes(boson.sensor.optics.distortion, intr)
    prefix = "omni:lensdistortion:opencvPinhole:"
    assert all(name.startswith(prefix) for name in attrs)
    assert {name[len(prefix) :] for name in attrs} == PINHOLE_ATTRS


def test_written_intrinsics_are_the_ones_the_projection_oracle_uses(boson) -> None:  # type: ignore[no-untyped-def]
    """The schema carries its own fx/cx/imageSize; they must be the oracle's, else the two
    disagree."""
    intr = camera_optics(boson.sensor).intrinsics
    _, attrs = distortion_attributes(boson.sensor.optics.distortion, intr)
    prefix = "omni:lensdistortion:opencvPinhole:"
    assert attrs[prefix + "fx"] == pytest.approx(intr.fx_px)
    assert attrs[prefix + "fy"] == pytest.approx(intr.fy_px)
    assert attrs[prefix + "cx"] == pytest.approx(intr.cx_px)
    assert attrs[prefix + "cy"] == pytest.approx(intr.cy_px)
    assert attrs[prefix + "imageSize"] == (intr.width, intr.height)


def test_brown_conrady_coefficients_are_written_positionally() -> None:
    """[k1, k2, p1, p2, k3] land on those names; k4..s4 are written as explicit zeros."""
    intr = Intrinsics(1000.0, 1000.0, 320.0, 256.0, 640, 512)
    spec = DistortionSpec(model="brown_conrady", coeffs=[-0.28, 0.09, 0.001, -0.002, 0.004])
    _, attrs = distortion_attributes(spec, intr)
    prefix = "omni:lensdistortion:opencvPinhole:"
    assert attrs[prefix + "k1"] == pytest.approx(-0.28)
    assert attrs[prefix + "k2"] == pytest.approx(0.09)
    assert attrs[prefix + "p1"] == pytest.approx(0.001)
    assert attrs[prefix + "p2"] == pytest.approx(-0.002)
    assert attrs[prefix + "k3"] == pytest.approx(0.004)
    assert all(attrs[prefix + n] == 0.0 for n in ("k4", "k5", "k6", "s1", "s2", "s3", "s4"))


def test_fisheye_writes_all_four_coefficients_over_a_non_zero_default() -> None:
    """The fisheye schema defaults k1 to 0.00245, so a zero-distortion fisheye must say so."""
    intr = Intrinsics(1000.0, 1000.0, 320.0, 256.0, 640, 512)
    spec = DistortionSpec(model="kannala_brandt", coeffs=[0.0, 0.0, 0.0, 0.0])
    schema, attrs = distortion_attributes(spec, intr)
    prefix = "omni:lensdistortion:opencvFisheye:"
    assert schema == "OmniLensDistortionOpenCvFisheyeAPI"
    assert {name[len(prefix) :] for name in attrs} == {
        "fx",
        "fy",
        "cx",
        "cy",
        "imageSize",
        "k1",
        "k2",
        "k3",
        "k4",
    }
    assert attrs[prefix + "k1"] == 0.0


def test_ftheta_is_refused_by_the_usd_mapping_too() -> None:
    """Refused in both places, for the same reason: the convention is unmeasured (ADR 0015)."""
    intr = Intrinsics(1000.0, 1000.0, 320.0, 256.0, 640, 512)
    with pytest.raises(NotImplementedError, match="not verified"):
        distortion_attributes(DistortionSpec(model="ftheta", coeffs=[0.0, 500.0]), intr)


# --- construction guards --------------------------------------------------------------------


@pytest.fixture(scope="module")
def scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})


def test_a_pipeline_built_for_another_sensor_is_refused(  # type: ignore[no-untyped-def]
    boson, scene, tophat_lwir_lut, aerial_materials, aerial_sensor
) -> None:
    """The config and the camera must describe one camera; two would differ only in the numbers."""
    pipeline = PipelineConfig.from_sensor(aerial_sensor, aerial_materials, tophat_lwir_lut)
    with pytest.raises(ValueError, match="different SensorConfig"):
        IrCamera(
            boson,
            scene,
            pipeline=pipeline,
            prim_to_target={},
            resolutions=[],
        )


def test_an_atmosphere_on_another_weather_series_is_refused(  # type: ignore[no-untyped-def]
    scene, tophat_lwir_lut, aerial_materials, aerial_sensor, aerial_sky
) -> None:
    """CLAUDE.md #6: a camera cannot run one weather in the sky and another in the solvers."""
    pipeline = PipelineConfig.from_sensor(
        aerial_sensor, aerial_materials, tophat_lwir_lut, sky=aerial_sky
    )
    assert aerial_sky.weather is not scene.weather
    with pytest.raises(ValueError, match="different WeatherSeries"):
        IrCamera(
            aerial_sensor,
            scene,
            pipeline=pipeline,
            prim_to_target={},
            resolutions=[],
        )


def test_the_camera_clock_advances_one_frame_period(  # type: ignore[no-untyped-def]
    aerial_sensor, scene, tophat_lwir_lut, aerial_materials
) -> None:
    """60 Hz means 16.67 ms per capture -- the base the FFC schedule and the drift run on."""
    pipeline = PipelineConfig.from_sensor(aerial_sensor, aerial_materials, tophat_lwir_lut)
    camera = IrCamera(
        aerial_sensor,
        scene,
        pipeline=pipeline,
        prim_to_target={},
        resolutions=[],
    )
    assert camera.frame_period_s == pytest.approx(1.0 / 60.0)
    assert camera.t_rel_s == 0.0
    with pytest.raises(RuntimeError, match="open"):
        camera.planes()
