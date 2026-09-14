"""Engine-free tests for the companion frame's environment dome (ADR 0073).

Nothing here is infrared physics and none of it feeds the sensor chain, so the bar is different
from the rest of the suite: these tests exist to stop the *visible* frame from lying about the
scene. A daylight model that is smooth, plausible and pointed the wrong way is the failure that
matters -- it would put the sun on the wrong side of the picture and quietly contradict the sun
direction the thermal solver is using, which is exactly the sort of disagreement the companion
frame was added to make visible.

The checks that carry weight are therefore the ones tied to something outside the model: the sun
lands where NOAA's azimuth says it does; the Perez shape function normalises exactly to the zenith
value it is scaled by; the turbidity comes out of a *column* ratio and not a ground-level one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim_isaac.visible_sky import (
    DOME_EXPOSURE_TARGET,
    DOME_POLE_AXIS,
    TURBIDITY_RANGE,
    DomeSpec,
    dome_intensity,
    environment_map,
    latlong_directions,
    luminance_coefficients,
    perez,
    sky_luminance_xyy,
    stage_direction,
    turbidity_from_visibility,
    xyy_to_linear_rgb,
    zenith_chromaticity,
    zenith_luminance_cd_m2,
)

#: The visible Rayleigh coefficient of configs/atmospheres/us_standard_clear.yaml.
RAYLEIGH_PER_M = 1.2e-5
#: Clear-day visibility of the phase-1 weather file.
CLEAR_VISIBILITY_M = 23000.0

#: Where the demo's sun actually is: 48.1 N, 11.6 E, 2024-06-21T04:00Z (NOAA, irsim.thermal.solar).
DEMO_SUN_EL_DEG = 5.62
DEMO_SUN_AZ_DEG = 60.74


def demo_spec(**overrides: float) -> DomeSpec:
    kwargs: dict[str, float] = {
        "sun_elevation_deg": DEMO_SUN_EL_DEG,
        "sun_azimuth_deg": DEMO_SUN_AZ_DEG,
        "turbidity": turbidity_from_visibility(CLEAR_VISIBILITY_M, RAYLEIGH_PER_M),
        "dni_w_m2": 86.0,
        "dhi_w_m2": 11.1,
        "visibility_m": CLEAR_VISIBILITY_M,
    }
    kwargs.update(overrides)
    return DomeSpec(**kwargs)  # type: ignore[arg-type]


# --- turbidity ------------------------------------------------------------------------------


def test_a_clear_day_gives_a_clear_sky_turbidity() -> None:
    """23 km visibility is a clear day, which is T ~ 3 -- not the T ~ 14 of a ground-level ratio.

    This is the test that pins the definition. Turbidity is the ratio of *column* optical depths,
    and visibility is a ground-level extinction coefficient; dividing one by the other without
    giving the molecular and aerosol layers their own scale heights gives 3.912 / (V gamma_R) =
    14.2 here, which is dense haze, and would render a clear June morning as a white sky.
    """
    ground_ratio = 3.912 / (CLEAR_VISIBILITY_M * RAYLEIGH_PER_M)
    assert ground_ratio > 12.0, "the mistake this test exists to catch should be large"
    t = turbidity_from_visibility(CLEAR_VISIBILITY_M, RAYLEIGH_PER_M)
    assert 2.5 < t < 3.5, f"a clear 23 km day should be T ~ 3, got {t:.2f}"


def test_turbidity_falls_as_the_air_clears() -> None:
    """Monotone in visibility, and bounded by the range Preetham's fit was regressed on."""
    values = [turbidity_from_visibility(v, RAYLEIGH_PER_M) for v in (2e3, 1e4, 2.3e4, 8e4, 1e6)]
    assert all(a >= b for a, b in zip(values, values[1:], strict=False)), values
    assert values[0] == pytest.approx(TURBIDITY_RANGE[1]), "thick haze should hit the upper clamp"
    assert values[-1] == pytest.approx(TURBIDITY_RANGE[0]), "an infinitely clear sky hits the floor"


def test_turbidity_refuses_nonsense_inputs() -> None:
    with pytest.raises(ValueError):
        turbidity_from_visibility(0.0, RAYLEIGH_PER_M)
    with pytest.raises(ValueError):
        turbidity_from_visibility(CLEAR_VISIBILITY_M, 0.0)


# --- the Preetham distribution --------------------------------------------------------------


def test_zenith_luminance_is_a_real_daylight_number() -> None:
    """A clear sky with the sun overhead has a zenith luminance of order 10-20 kcd/m2.

    Independent of the code: the value is a measurable property of the real sky, and an error in
    the sign or the units of the Preetham zenith equation moves it by orders of magnitude.
    """
    overhead = zenith_luminance_cd_m2(2.2, 0.0)
    assert 1.0e4 < overhead < 2.5e4, overhead
    # Lower sun, dimmer zenith; and a hazier sky at the same sun angle is brighter at the zenith
    # because the aerosol scatters light out of the beam and into every other direction.
    assert zenith_luminance_cd_m2(2.2, math.radians(75.0)) < overhead
    assert zenith_luminance_cd_m2(6.0, math.radians(60.0)) > zenith_luminance_cd_m2(
        2.2, math.radians(60.0)
    )


def test_the_distribution_normalises_exactly_to_its_zenith_value() -> None:
    """At the zenith the scaled Perez shape must return the zenith luminance and chromaticity.

    The model is a *shape* function normalised by its own value at the zenith and multiplied by an
    absolute zenith quantity. Getting the normalisation direction wrong still produces a smooth
    sky with the right colours -- just at the wrong absolute level -- so this equality is the only
    thing that holds the two halves together.
    """
    for turbidity in (2.0, 3.0, 6.0):
        for sun_zenith in (0.0, math.radians(45.0), math.radians(80.0)):
            luminance, cx, cy = sky_luminance_xyy(0.0, sun_zenith, turbidity, sun_zenith)
            expect_x, expect_y = zenith_chromaticity(turbidity, sun_zenith)
            assert float(luminance) == pytest.approx(
                zenith_luminance_cd_m2(turbidity, sun_zenith), rel=1e-12
            )
            assert float(cx) == pytest.approx(expect_x, rel=1e-12)
            assert float(cy) == pytest.approx(expect_y, rel=1e-12)


def test_the_gradient_term_brightens_toward_the_horizon() -> None:
    """At a fixed angle from the sun, the clear sky gets brighter as the ray drops. Every step.

    The Perez form is separable -- F = f(theta) g(gamma) -- so holding gamma fixed isolates
    (1 + A e^(B/cos theta)) exactly. A and B are both negative across the useful turbidity range,
    which makes that factor rise monotonically from the zenith to the horizon, roughly doubling.
    A sky that darkened toward the horizon would look like dusk at noon and would invert the
    contrast of anything low in the frame.
    """
    theta = np.radians(np.arange(0.0, 90.0, 2.0))
    ratios = []
    for turbidity in (2.0, 3.0, 6.0):
        values = perez(theta, math.radians(80.0), luminance_coefficients(turbidity))
        assert np.all(np.diff(values) > 0.0), turbidity
        ratios.append(float(values[-1] / values[0]))
    # ...and the gradient *flattens* as the air thickens, because multiple scattering fills the
    # zenith in. A hazy sky that still had a clear sky's horizon-to-zenith contrast would be the
    # tell that turbidity had been dropped somewhere between the weather and the map.
    assert ratios[0] > ratios[1] > ratios[2] > 1.05, ratios


def test_the_sky_is_brightest_at_the_sun() -> None:
    """The Perez circumsolar term has to peak at gamma = 0 and fall away from it."""
    gamma = np.radians(np.array([0.0, 5.0, 20.0, 60.0, 120.0]))
    values = perez(math.radians(45.0), gamma, luminance_coefficients(3.0))
    assert np.all(np.diff(values) < 0.0), values


# --- the texture layout --------------------------------------------------------------------


def test_the_texture_layout_is_the_one_measured_off_the_renderer() -> None:
    """Row 0 of the map points along the stage's +Z and the azimuth runs from +X toward -Y.

    This is a *renderer* convention, not a choice, and getting it wrong is silent: the sky still
    renders, it is simply the wrong part of the sky. The first version of this module built the
    map in elevation and azimuth, which on this build put the whole camera field inside one
    texture pole and filled the companion frame with ground.
    ``tests/integration/test_aerial_demo_isaac.py`` re-measures it against the renderer; here we
    only hold the generator to it.
    """
    height = 64
    d = latlong_directions(height)
    assert d.shape == (height, 2 * height, 3)
    assert np.allclose(np.linalg.norm(d, axis=-1), 1.0)
    # Row 0 hugs the pole, the last row the opposite pole.
    assert np.allclose(d[0].mean(axis=0), DOME_POLE_AXIS, atol=0.05)
    assert np.allclose(d[-1].mean(axis=0), -np.asarray(DOME_POLE_AXIS), atol=0.05)
    # On the equator, column 0 is +X and the azimuth advances toward -Y.
    equator = d[height // 2]
    assert np.allclose(equator[0], (1.0, 0.0, 0.0), atol=0.05)
    assert np.allclose(equator[equator.shape[0] // 4], (0.0, -1.0, 0.0), atol=0.05)


def test_stage_direction_is_the_convention_the_targets_use() -> None:
    """Azimuth off -Z toward +X, elevation toward +Y: the frame `camera_space_position` uses."""
    assert np.allclose(stage_direction(0.0, 0.0), (0.0, 0.0, -1.0), atol=1e-12)
    assert np.allclose(stage_direction(90.0, 0.0), (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(stage_direction(0.0, 90.0), (1.0, 0.0, 0.0), atol=1e-12)


# --- the map --------------------------------------------------------------------------------


def test_the_map_is_float32_finite_and_non_negative() -> None:
    """CLAUDE.md #2 applies to every buffer this repo writes, radiometric or not."""
    m = environment_map(demo_spec(), height=64)
    assert m.dtype == np.float32
    assert m.shape == (64, 128, 3)
    assert np.all(np.isfinite(m))
    assert np.all(m >= 0.0)


def test_the_map_rejects_an_odd_or_tiny_height() -> None:
    """An odd height would put the horizon *inside* a row that is neither sky nor ground."""
    for bad in (0, 7, 65):
        with pytest.raises(ValueError):
            environment_map(demo_spec(), height=bad)


@pytest.mark.parametrize("sun_az", [0.0, 60.74, 180.0, 285.0])
@pytest.mark.parametrize("heading", [0.0, 90.0])
def test_the_sun_lands_where_noaa_says_it_does(sun_az: float, heading: float) -> None:
    """The brightest texel of the map points at the sun, to within a texel.

    This is the assertion with an outside reference: the azimuth comes from
    :func:`irsim.thermal.solar.sun_position`, and ``heading_deg`` is the compass bearing of the
    stage's -Z axis, so the sun's direction is fixed by geometry and not by anything in this
    module. A sign error in either would still give a perfectly convincing sky with the glow on
    the wrong side of the frame -- and would contradict the sun direction the thermal solver is
    loading the targets with. Compared as a *vector* so the test says nothing about the texture
    layout, which is the renderer's business and is checked separately.
    """
    height = 180
    spec = demo_spec(sun_elevation_deg=25.0, sun_azimuth_deg=sun_az, heading_deg=heading)
    image = environment_map(spec, height=height)
    brightest = latlong_directions(height).reshape(-1, 3)[int(np.argmax(image.sum(-1)))]
    expected = stage_direction(25.0, sun_az - heading)
    separation_deg = math.degrees(math.acos(float(np.clip(brightest @ expected, -1.0, 1.0))))
    assert separation_deg < 2.0 * 180.0 / height, f"{separation_deg:.2f} deg off the sun"


def test_the_zenith_is_blue_and_the_low_sun_is_warm() -> None:
    """Rayleigh blue overhead, reddened light near a 5 degree sun: the colour signature of dawn.

    Chromaticity is the half of the model that carries no energy and is therefore the half that
    can be silently dropped -- a map built from the luminance alone is a convincing grey sky.
    """
    height = 180
    image = environment_map(demo_spec(), height=height)
    direction = latlong_directions(height)
    zenith = image[direction[..., 1] > 0.995].mean(axis=0)
    assert zenith[2] > 1.3 * zenith[0], f"the zenith should be blue, got {zenith}"
    sun = stage_direction(DEMO_SUN_EL_DEG, DEMO_SUN_AZ_DEG)
    near_sun = image[(direction @ sun) > math.cos(math.radians(4.0))].mean(axis=0)
    assert near_sun[0] > 1.5 * near_sun[2], f"a 5 degree sun should be red, got {near_sun}"


def test_below_the_horizon_is_terrain_and_not_sky() -> None:
    """The lower hemisphere is a lit Lambertian ground: darker than the sky, and the albedo's hue.

    Without it the dome's lower half is the sky's own colour and the visible frame has no horizon
    at all, which is the state this module replaced.
    """
    height = 180
    spec = demo_spec(camera_height_m=2.0)
    image = environment_map(spec, height=height)
    up = latlong_directions(height)[..., 1]
    sky = float(image[(up > 0.0) & (up < 0.02)].mean())
    ground = float(image[(up < 0.0) & (up > -0.02)].mean())
    assert ground < 0.5 * sky, f"ground {ground:.0f} against sky {sky:.0f}"
    nadir = image[up < -0.995].mean(axis=0)
    albedo = np.asarray(spec.ground_albedo, dtype=np.float64)
    assert np.allclose(nadir / nadir.sum(), albedo / albedo.sum(), atol=1e-3)


def test_haze_pulls_the_far_ground_toward_the_sky() -> None:
    """Poorer visibility makes the ground just below the horizon read closer to the horizon sky.

    Koschmieder again, this time as aerial perspective: the terrain a ray meets just below the
    horizon is kilometres away, so at 5 km visibility it is mostly airlight and at 40 km mostly
    ground. A dome without this has a painted line where the horizon is.
    """
    height = 360
    up = latlong_directions(height)[..., 1]
    just_above = (up > 0.0) & (up < 0.01)
    just_below = (up < 0.0) & (up > -0.01)
    sky = float(environment_map(demo_spec(), height=height)[just_above].mean())
    clear = float(environment_map(demo_spec(visibility_m=40e3), height=height)[just_below].mean())
    murky = float(environment_map(demo_spec(visibility_m=5e3), height=height)[just_below].mean())
    assert abs(murky - sky) < abs(clear - sky)


def test_night_is_dark_and_still_finite() -> None:
    """Preetham is a daylight fit; below its validity the map fades instead of going negative.

    A negative luminance clips to black, which looks like a working night sky right up until
    someone asks why the ground is also black.
    """
    night = environment_map(demo_spec(sun_elevation_deg=-10.0, dni_w_m2=0.0, dhi_w_m2=0.0), 64)
    day = environment_map(demo_spec(sun_elevation_deg=40.0), 64)
    assert np.all(np.isfinite(night)) and np.all(night >= 0.0)
    assert float(night.mean()) < 0.05 * float(day.mean())
    assert float(night.mean()) > 0.0, "a night sky is dim, not absent"


def test_xyy_to_rgb_keeps_a_neutral_neutral() -> None:
    """The D65 white point must come back as equal linear R, G and B."""
    rgb = xyy_to_linear_rgb(100.0, 0.3127, 0.3290)
    assert rgb.shape == (3,)
    assert float(rgb.max() / rgb.min()) == pytest.approx(1.0, abs=2e-3)


# --- the exposure ---------------------------------------------------------------------------


def test_the_exposure_puts_every_daytime_sky_at_the_same_level() -> None:
    """intensity x median sky luminance is the target, whatever the hour: it is an auto-exposure.

    Deliberate, and the reason the visible frame carries no absolute brightness information. The
    test is here so that nobody later "fixes" the exposure into something scene-dependent and
    silently turns the companion frame into a photometric claim.
    """
    for elevation in (10.0, 30.0, 60.0):
        spec = demo_spec(sun_elevation_deg=elevation)
        image = environment_map(spec, height=64)
        up = latlong_directions(64)[..., 1] > 0.0
        median = float(np.median(image[up]))
        assert dome_intensity(spec, image) * median == pytest.approx(DOME_EXPOSURE_TARGET, rel=1e-6)


def test_night_is_not_exposed_up_to_look_like_noon() -> None:
    """The twilight fade is divided back out of the reference, so a dark scene renders dark."""
    night = demo_spec(sun_elevation_deg=-8.0, dni_w_m2=0.0, dhi_w_m2=0.0)
    day = demo_spec(sun_elevation_deg=40.0)
    night_map = environment_map(night, 64)
    day_map = environment_map(day, 64)
    up = latlong_directions(64)[..., 1] > 0.0
    night_level = dome_intensity(night, night_map) * float(np.median(night_map[up]))
    day_level = dome_intensity(day, day_map) * float(np.median(day_map[up]))
    assert night_level < 0.05 * day_level
