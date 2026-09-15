"""M10.22: the reflective bands' illumination reaches the Isaac render path.

`irsim_isaac.pipeline.illumination_isaac` is engine glue that imports no engine module -- it takes
arrays and a Scene and returns planes -- so it is tested here, on the plain interpreter, and not
behind `@pytest.mark.isaac`.

**The failure these tests exist to prevent.** M11.3 and M11.4 built the reflected-sunlight and
night-sky terms and M11.2 built the bundle that carries them, and nothing in the engine glue ever
wrote the planes `run_frame` reads them out of. Every Isaac render this project produced was
emission only: right for LWIR, where reflected sunlight is 0.35 % of the signal, and **black** for
SWIR or NIR, where it is essentially the whole signal. The first test is that arithmetic, run
through the real pipeline stage, so "black without it" is a measurement and not a claim.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.materials import MaterialLibrary, MaterialTable
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.scene import Scene
from irsim_isaac.pipeline.illumination_isaac import (
    SceneIllumination,
    sun_cos_incidence,
    sun_direction_stage,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE = REPO / "configs" / "scenes" / "quad_flight_clear_noon.yaml"
SENSORS = {
    "nir": REPO / "configs/sensors/example_nir_si_1280.yaml",
    "swir": REPO / "configs/sensors/example_swir_ingaas_640.yaml",
    "mwir": REPO / "configs/sensors/example_mwir_insb_640.yaml",
    "lwir": REPO / "configs/sensors/flir_boson_640_lwir.yaml",
}
COARSE_N = 2001


@pytest.fixture(scope="module")
def sensors():  # type: ignore[no-untyped-def]
    return {name: load_sensor_config(path) for name, path in SENSORS.items()}


@pytest.fixture(scope="module")
def luts(sensors):  # type: ignore[no-untyped-def]
    return {
        name: BandLUT.build(load_spectral_response(cfg.sensor.band.spectral_response), n=COARSE_N)
        for name, cfg in sensors.items()
    }


def _scene(band: str, lut: BandLUT, quantity: str):  # type: ignore[no-untyped-def]
    return Scene.from_file(SCENE, {band: lut}, quantity=quantity)


def _quantity(cfg) -> str:  # type: ignore[no-untyped-def]
    return "lb" if cfg.sensor.fpa.type == "bolometer" else "lb_q"


# --- the geometry ------------------------------------------------------------------------------


def test_the_sun_direction_matches_the_distant_light_the_renderer_uses() -> None:
    """The one vector both halves must agree on.

    `irsim_isaac.stage.add_sky_dome` aims the USD DistantLight along this same expression. If the
    two were computed separately the shadows in the companion visible frame would fall away from
    the sun the radiometry used, and the error would only show up when somebody overlaid them.
    """
    for elevation, azimuth, heading in [(61.0, 180.0, 0.0), (10.0, 90.0, 35.0), (0.0, 0.0, 0.0)]:
        s = sun_direction_stage(elevation, azimuth, heading)
        assert np.linalg.norm(s) == pytest.approx(1.0, abs=1e-12)
        assert s[1] == pytest.approx(math.sin(math.radians(elevation)), abs=1e-12)
        az = math.radians(azimuth - heading)
        el = math.radians(elevation)
        assert s[0] == pytest.approx(math.sin(az) * math.cos(el), abs=1e-12)
        assert s[2] == pytest.approx(-math.cos(az) * math.cos(el), abs=1e-12)
    # Straight up is +Y, and a sun on the -Z horizon points along -Z.
    assert sun_direction_stage(90.0, 0.0, 0.0) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert sun_direction_stage(0.0, 0.0, 0.0) == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_a_surface_facing_away_from_the_sun_receives_nothing_rather_than_negative_light() -> None:
    """Clamping at zero is not cosmetic: an unclamped cosine would *subtract* sunlight."""
    up = np.zeros((2, 3, 3))
    up[..., 1] = 1.0
    down = -up
    noon = sun_direction_stage(90.0, 0.0, 0.0)
    assert np.allclose(sun_cos_incidence(up, noon), 1.0)
    assert np.allclose(sun_cos_incidence(down, noon), 0.0)
    slant = sun_direction_stage(30.0, 0.0, 0.0)
    assert np.allclose(sun_cos_incidence(up, slant), math.sin(math.radians(30.0)))
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        sun_cos_incidence(np.zeros((4, 4)), noon)


# --- the gate ----------------------------------------------------------------------------------


def test_an_emissive_band_gets_no_bundle_at_all(sensors, luts) -> None:  # type: ignore[no-untyped-def]
    """`None`, not an empty bundle: an LWIR render must take the same code path it always did and
    come out bit-identical, which is only guaranteed if nothing new is attached to it."""
    cfg = sensors["lwir"]
    scene = _scene("lwir", luts["lwir"], _quantity(cfg))
    assert SceneIllumination.for_camera(cfg, scene, _quantity(cfg)) is None


@pytest.mark.parametrize("band", ["nir", "swir", "mwir"])
def test_a_reflective_band_gets_a_solar_plane_that_follows_the_sun(band, sensors, luts) -> None:  # type: ignore[no-untyped-def]
    cfg = sensors[band]
    quantity = _quantity(cfg)
    illumination = SceneIllumination.for_camera(cfg, _scene(band, luts[band], quantity), quantity)
    assert illumination is not None and illumination.solar is not None

    up = np.zeros((4, 4, 3))
    up[..., 1] = 1.0
    noon = illumination.planes(up, sun_elevation_deg=61.0, sun_azimuth_deg=180.0)
    low = illumination.planes(up, sun_elevation_deg=8.0, sun_azimuth_deg=180.0)
    night = illumination.planes(up, sun_elevation_deg=-6.0, sun_azimuth_deg=180.0)

    assert np.all(noon["l_sun"] > 0.0)
    # Two mechanisms, and both must be in it: the cosine *and* the slant path. Low sun is dimmer
    # than the cosine alone predicts, because the beam has been through more atmosphere.
    cosine_only = (
        float(noon["l_sun"][0, 0]) * math.sin(math.radians(8.0)) / math.sin(math.radians(61.0))
    )
    assert float(low["l_sun"][0, 0]) < cosine_only
    # Below the horizon there is no direct beam at all -- not a small one.
    assert np.all(night["l_sun"] == 0.0)


def test_sky_pixels_reflect_nothing(sensors, luts) -> None:  # type: ignore[no-untyped-def]
    """A sky pixel's radiance is the sky model and the atmosphere. Leaving an incident term on it
    would have stage 1 reflect sunlight off the sky."""
    cfg = sensors["nir"]
    quantity = _quantity(cfg)
    illumination = SceneIllumination.for_camera(cfg, _scene("nir", luts["nir"], quantity), quantity)
    assert illumination is not None
    up = np.zeros((4, 4, 3))
    up[..., 1] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[0] = True
    planes = illumination.planes(up, sun_elevation_deg=61.0, sun_azimuth_deg=180.0, sky_mask=mask)
    assert np.all(planes["l_sun"][0] == 0.0)
    assert np.all(planes["l_sun"][1:] > 0.0)
    with pytest.raises(ValueError, match="sky_mask"):
        illumination.planes(
            up, sun_elevation_deg=61.0, sun_azimuth_deg=180.0, sky_mask=np.zeros((2, 2), bool)
        )


# --- the reason the module exists ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("band", "low", "high"),
    [
        # Dry asphalt at 300 K under a 61-degree sun, measured. NIR is the extreme: a render
        # without the bundle was fifteen orders of magnitude too dark. MWIR is the interesting
        # one -- reflected sunlight adds 17 % on top of self-emission, which is exactly what
        # `regime: mixed` is for, and is bracketed tightly because a band that is *nearly* all
        # emission is where a wrong solar term hides best.
        ("nir", 1e14, 1e16),
        ("swir", 1e7, 1e8),
        ("mwir", 1.10, 1.25),
    ],
)
def test_without_the_bundle_a_reflective_band_renders_essentially_nothing(
    band, low, high, sensors, luts
) -> None:  # type: ignore[no-untyped-def]
    """The measurement behind this whole step, run through the real stage-1 kernel.

    A 300 K surface in a reflective band emits almost nothing. With the sun attached it reflects
    a great deal, and the ratio is how wrong every Isaac render of these bands was before M10.22.
    """
    cfg = sensors[band]
    quantity = _quantity(cfg)
    scene = _scene(band, luts[band], quantity)
    illumination = SceneIllumination.for_camera(cfg, scene, quantity)
    assert illumination is not None

    table = MaterialTable.from_library(MaterialLibrary.load(), cfg.sensor.band.band_id)
    material_id = np.full((4, 4), table.id_for("asphalt_dry"), dtype=np.int32)
    temperature = np.full((4, 4), 300.0, dtype=np.float32)
    up = np.zeros((4, 4, 3))
    up[..., 1] = 1.0
    planes = illumination.planes(up, sun_elevation_deg=61.0, sun_azimuth_deg=180.0)

    from irsim.pipeline.illumination import Illumination

    emission_only = band_radiance(temperature, material_id, table, luts[band], quantity)
    lit = band_radiance(
        temperature,
        material_id,
        table,
        luts[band],
        quantity,
        illumination=Illumination.for_regime(
            cfg.sensor.band.regime, quantity, l_sun=planes["l_sun"]
        ),
    )
    ratio = float(np.mean(lit)) / max(float(np.mean(emission_only)), 1e-300)
    assert low < ratio < high, f"{band}: sunlit/emission-only = {ratio:.3e}"


def test_an_emissive_band_is_bit_identical_with_and_without_a_solar_plane(sensors, luts) -> None:  # type: ignore[no-untyped-def]
    """§5.2's gate, end to end: LWIR renders were correct before this step and must stay exactly
    so. Bit-identical, not close -- the gate drops the term to None rather than adding a zero."""
    cfg = sensors["lwir"]
    quantity = _quantity(cfg)
    table = MaterialTable.from_library(MaterialLibrary.load(), cfg.sensor.band.band_id)
    material_id = np.full((4, 4), table.id_for("asphalt_dry"), dtype=np.int32)
    temperature = np.full((4, 4), 300.0, dtype=np.float32)

    from irsim.pipeline.illumination import Illumination

    plain = band_radiance(temperature, material_id, table, luts["lwir"], quantity)
    with_sun = band_radiance(
        temperature,
        material_id,
        table,
        luts["lwir"],
        quantity,
        illumination=Illumination.for_regime(
            cfg.sensor.band.regime, quantity, l_sun=np.full((4, 4), 500.0)
        ),
    )
    assert np.array_equal(plain, with_sun)
