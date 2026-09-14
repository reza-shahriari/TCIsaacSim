"""Structured cloud in the rendered background (MS.3/ADR 0070 into the M10.18 bridge).

`SkyModel.radiance` has always carried cloud, but as its **expectation**: the uniform blend
(1 - c eps) L_clear + c eps L_base, which is the mean over a structured field and is exactly right
for a LUT or a tilt integral. What a rendered frame needs is the field itself, because against a
sky background the dominant false alarm is not sensor noise, it is a cloud edge -- and an edge has
no mean.

So the property that makes this an upgrade rather than a different model is that **the mean is
preserved**: averaged over the whole sky the structured field returns the uniform blend it
replaced. If that failed, every existing LUT, tilt integral and reflected term would silently
disagree with the rendered background.

The second property is that the field is fixed to the **sky**. A field fixed to the image plane is
equally stable frame to frame and travels with the sensor, so a slewing mount never sweeps across
cloud and a tracked target never crosses one -- which removes the very clutter the field is for.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.cloud import generate_sky_cloud
from irsim.config.loader import load_sensor_config
from irsim.radiometry.lut_files import load_band_lut_for_config
from irsim.scene import Scene
from irsim_isaac.pipeline.aerial_bridge import (
    AerialThermalBridge,
    azimuth_from_rays,
    elevation_from_rays,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "quad_flight_clear_noon.yaml"
SENSOR_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

#: The structured field's mean must agree with the uniform blend to better than this, over the
#: whole sky. It is not exact: coverage is clumped and the clear radiance varies with elevation,
#: so which elevations a realisation happens to cover moves the mean slightly. 1 % is far inside
#: the ~0.2 % observed across seeds and far outside anything a wrong model would achieve.
MEAN_PRESERVATION_TOL = 0.01


@pytest.fixture(scope="module")
def scene() -> Scene:
    sensor = load_sensor_config(SENSOR_YAML)
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    return Scene.from_file(SCENE_YAML, {"lwir": lut})


def whole_sky(n_el: int = 180, n_az: int = 720) -> tuple[np.ndarray, np.ndarray]:
    el = np.radians((np.arange(n_el) + 0.5) * 90.0 / n_el)[:, None].repeat(n_az, 1)
    az = np.radians((np.arange(n_az) + 0.5) * 360.0 / n_az)[None, :].repeat(n_el, 0)
    return el, az


# --- the field --------------------------------------------------------------------------------


def test_the_whole_sky_coverage_is_the_weather_cloud_fraction(scene: Scene) -> None:
    """Exactly c of the sky, from the shared WeatherSeries (CLAUDE.md #6), not a tuned number."""
    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    cloud = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed=11)
    el, az = whole_sky()
    assert float(cloud.sample(el, az).mean()) == pytest.approx(fraction, abs=1e-6)


def test_one_elevation_ring_is_not_the_sky_average(scene: Scene) -> None:
    """Cloud clumps: a single ring departs from c, which is the point of having structure at all.

    A field whose every ring carried exactly c would be a haze, not cloud, and would never give a
    detector an edge to false-alarm on.
    """
    fraction = float(scene.weather.at(scene.t0_s).cloud_fraction)
    cloud = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed=11)
    rings = [
        float(cloud.sample(np.radians(np.full(720, e)), np.radians(np.arange(720) * 0.5)).mean())
        for e in (10.0, 25.0, 40.0, 60.0)
    ]
    assert max(rings) > min(rings) + 0.01, f"rings {rings} are suspiciously uniform"


def test_the_structured_field_preserves_the_uniform_blend_it_replaces(scene: Scene) -> None:
    """Averaged over the sky, the field returns the expectation every other consumer uses.

    This is the compatibility property. `SkyModel.radiance` feeds the elevation LUT, the tilt
    integral and ADR 0045's reflected term; if the rendered background's mean drifted from it, the
    same sky would be two different skies depending on which code path asked.
    """
    sky = scene.sky_models["lwir"]
    t = scene.t0_s
    fraction = float(scene.weather.at(t).cloud_fraction)
    el, az = whole_sky()
    uniform = sky.radiance(t, el.ravel())
    for seed in (1, 2, 3, 11):
        coverage = generate_sky_cloud(scene.environment.clouds.beta, fraction, seed).sample(el, az)
        structured = sky.radiance_field(t, el.ravel(), coverage.ravel())
        relative = abs(float(structured.mean() - uniform.mean()) / float(uniform.mean()))
        assert relative < MEAN_PRESERVATION_TOL, f"seed {seed}: mean drifted by {relative:.2%}"


def test_covered_pixels_read_warmer_than_clear_ones(scene: Scene) -> None:
    """Cloud base sits far above a cold clear zenith, so cloud is a *positive* contrast feature.

    Worth pinning: a drone is also warmer than the clear sky, so cloud and target have the same
    polarity and cloud edge is a genuine confuser rather than something a sign test removes.
    """
    sky = scene.sky_models["lwir"]
    t = scene.t0_s
    el = np.radians(np.full(4000, 45.0))
    covered = np.ones(4000, dtype=bool)
    clear = np.zeros(4000, dtype=bool)
    warm = sky.apparent_temperature_field(t, el, covered)
    cold = sky.apparent_temperature_field(t, el, clear)
    assert float(warm.mean()) > float(cold.mean()) + 10.0


# --- the bridge -------------------------------------------------------------------------------


def bridge(scene: Scene, **kwargs: object) -> AerialThermalBridge:
    return AerialThermalBridge(scene, {}, band="lwir", **kwargs)  # type: ignore[arg-type]


def test_without_a_seed_the_background_is_unchanged(scene: Scene) -> None:
    """Bit-identical to the uniform blend: every existing frame and golden stays exactly as it was.

    The structured path is opt-in for this reason. Turning it on by default would have moved every
    committed background by a few kelvin with no commit saying so.
    """
    el, az = whole_sky(40, 80)
    plain = bridge(scene)
    assert plain.cloud is None
    assert np.array_equal(
        plain.background_temperature_k(el, az), plain.background_temperature_k(el, None)
    )


def test_a_seed_puts_structure_in_the_background(scene: Scene) -> None:
    el, az = whole_sky(60, 120)
    plain = bridge(scene).background_temperature_k(el, az)
    seeded = bridge(scene, cloud_seed=11).background_temperature_k(el, az)
    assert not np.allclose(plain, seeded)
    assert float(seeded.max()) > float(plain.max()), "cloud should add a warm tail"


def test_the_field_is_fixed_to_the_sky_not_to_the_frame(scene: Scene) -> None:
    """The same world direction gives the same cloud however the camera happens to be pointed.

    Sampled here as two overlapping windows of the sky: where they overlap they must agree. An
    image-plane field would instead give whatever its own pixel grid said, so the overlap would
    disagree and a slewing mount would drag its clouds along with it.
    """
    cloudy = bridge(scene, cloud_seed=11)
    el = np.radians(np.linspace(10.0, 50.0, 64))[:, None].repeat(64, 1)
    # Two windows offset by a whole number of *samples*, so the overlapping columns ask about
    # identical world directions. Offsetting by an arbitrary angle instead would compare
    # neighbouring directions and fail on the field's own resolution rather than on its frame.
    step_deg = 0.5
    columns = np.arange(64) * step_deg
    az_a = np.radians(columns)[None, :].repeat(64, 0)
    az_b = np.radians(columns + 32 * step_deg)[None, :].repeat(64, 0)
    first = cloudy.background_temperature_k(el, az_a)
    second = cloudy.background_temperature_k(el, az_b)
    assert np.allclose(first[:, 32:], second[:, :32], atol=1e-9)
    # ...and the windows are not trivially identical: they overlap by half, not wholly.
    assert not np.allclose(first, second, atol=1e-9)


def test_cloud_without_a_sky_model_is_refused(scene: Scene) -> None:
    """Both terms of a covered pixel come from the sky model; a seed alone cannot make cloud."""
    with pytest.raises(ValueError, match="cloud needs a sky model"):
        AerialThermalBridge(scene, {}, cloud_seed=11)


def test_cloud_without_azimuth_falls_back_rather_than_banding(scene: Scene) -> None:
    """Sampling the field by elevation alone would stripe the sky horizontally. It does not.

    A caller that has not got azimuth gets the uniform blend, which is merely less detailed. The
    failure this avoids is the one that looks deliberate: horizontal bands across the sky that a
    reader would take for a real atmospheric layer.
    """
    el, az = whole_sky(40, 80)
    cloudy = bridge(scene, cloud_seed=11)
    no_azimuth = cloudy.background_temperature_k(el, None)
    plain = bridge(scene).background_temperature_k(el, None)
    assert np.array_equal(no_azimuth, plain)


# --- the azimuth helper -----------------------------------------------------------------------


def test_azimuth_from_rays_is_measured_from_forward_about_up() -> None:
    rays = np.array([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]])
    azimuth = np.degrees(azimuth_from_rays(rays))
    assert np.allclose(azimuth, [0.0, 90.0, 180.0, 270.0], atol=1e-9)


def test_azimuth_ignores_the_component_along_up() -> None:
    """A ray climbing in elevation keeps its bearing, which is what makes the sampling stable."""
    level = np.array([[[1.0, 0.0, -1.0]]]) / math.sqrt(2.0)
    steep = np.array([[[1.0, 8.0, -1.0]]]) / math.sqrt(66.0)
    assert float(azimuth_from_rays(level)[0, 0]) == pytest.approx(
        float(azimuth_from_rays(steep)[0, 0]), abs=1e-9
    )


def test_azimuth_and_elevation_agree_on_their_conventions() -> None:
    """Straight up is elevation 90; the azimuth there is degenerate but must stay finite."""
    up = np.array([[[0.0, 1.0, 0.0]]])
    assert float(np.degrees(elevation_from_rays(up)[0, 0])) == pytest.approx(90.0)
    assert np.all(np.isfinite(azimuth_from_rays(up)))


def test_azimuth_refuses_a_forward_parallel_to_up() -> None:
    with pytest.raises(ValueError, match="forward must not be parallel"):
        azimuth_from_rays(np.zeros((1, 1, 3)), up=(0.0, 1.0, 0.0), forward=(0.0, 1.0, 0.0))
