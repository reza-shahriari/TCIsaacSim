"""M6.13 — Tier 3 diurnal phenomenology on the facet scene (§6.3, §15 T3, §16.4 step 6).

§6.3 gives this bench its acceptance test in one sentence: *"If your simulator does not reproduce a
contrast collapse near sunrise/sunset, your temperature model is not running. Make 'does the scene
go flat at 0600 and 1900?' an acceptance test."* So that is measured first, on the **scene-wide
spread** rather than on a chosen pair — a pair's crossing time depends on which pair you choose by
a couple of hours, and the operational claim is about the picture.

Half of it reproduces spectacularly and half does not, and both are recorded below.

The facet scene is deliberately made of pairs that differ in exactly one thing — sunlit vs shaded
asphalt, wet vs dry soil, black vs white paint, a parked hood vs one at 100 km/h — so that when a
difference appears there is only one thing it can be attributed to.

docs/physics-model.md §6.3, §15 T3, §16.4 step 6; ADR 0036, ADR 0037
"""

from __future__ import annotations

import pathlib
from datetime import timedelta

import numpy as np
import pytest

from irsim.config.scene import SceneConfig, load_scene_config
from irsim.scene import Scene
from irsim.thermal.solar import sun_position_utc
from irsim.thermal.weather import WEATHER_FIELDS, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "configs" / "scenes" / "thermal_facet_scene.yaml"
#: The scene's own site. Solar noon lands near 10:00 UTC, so local = UTC + 2 (M6.12).
LOCAL_OFFSET_H = 2.0
SUNRISE_UTC_H = 2.33
SUNSET_UTC_H = 17.73
SAMPLE_MINUTES = 5


def _profile(scene: Scene) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(hours UTC from t₀, temperatures (H, N) in kelvin, surface names) over one day."""
    scene.thermal.advance_to(scene.t0_s + 24 * 3600.0)
    hours = np.arange(0.0, 24.0, SAMPLE_MINUTES / 60.0)
    temps = np.array(
        [scene.thermal.temperature_at(scene.t0_s + h * 3600.0) for h in hours], dtype=np.float64
    )
    return hours, temps, list(scene.thermal_surfaces)


@pytest.fixture(scope="module")
def baseline() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return _profile(Scene.from_file(SCENE_PATH))


def _at(
    hours: np.ndarray, temps: np.ndarray, names: list[str], name: str, hour_utc: float
) -> float:
    return float(np.interp(hour_utc, hours, temps[:, names.index(name)]))


def _rebuilt(**overrides: object) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The same scene under a modified weather series, built in memory.

    Each override is a callable applied to one weather column, so a variant differs from the
    baseline in exactly the column named and nothing else -- including the epoch, the time grid
    and every other column, which is what makes the comparison attributable.
    """
    config = load_scene_config(SCENE_PATH)
    original = Scene.from_file(SCENE_PATH).weather
    columns = {
        field: np.asarray(overrides[field](original.columns()[field]))  # type: ignore[operator]
        if field in overrides
        else np.asarray(original.columns()[field])
        for field in WEATHER_FIELDS
    }
    weather = WeatherSeries.from_arrays(original.epoch_utc, original.time_s, **columns)
    scene = Scene.from_config(config, weather_override=weather)
    return _profile(scene)


# ---------------------------------------------------------------------------------------------
# §6.3's own acceptance test: does the scene go flat?
# ---------------------------------------------------------------------------------------------


def _spread(hours: np.ndarray, temps: np.ndarray, names: list[str]) -> np.ndarray:
    """Scene-wide standard deviation across surfaces, excluding the two "same surface, one knob"
    duplicates whose whole purpose is to differ."""
    keep = [i for i, n in enumerate(names) if n not in ("hood_moving", "asphalt_shade")]
    return np.asarray(temps[:, keep].std(axis=1))


def test_the_scene_goes_flat_at_dawn(baseline) -> None:  # type: ignore[no-untyped-def]
    """From **12.7 K** of spread at midday to **0.88 K** an hour after sunrise: a 93 % collapse.

    This is §6.3's acceptance test and the most operationally important thing in the model. It is
    measured on the scene-wide spread rather than on a chosen pair, because the claim is about
    what the picture does.
    """
    hours, temps, names = baseline
    spread = _spread(hours, temps, names)
    dawn = (hours > SUNRISE_UTC_H - 1.5) & (hours < SUNRISE_UTC_H + 1.5)
    assert float(spread[dawn].min()) < 1.5
    assert float(spread[dawn].min()) == pytest.approx(0.88, abs=0.2)
    assert float(spread.max()) > 10.0
    assert float(spread[dawn].min()) / float(spread.max()) < 0.15


def test_the_dusk_flat_is_not_reproduced_and_that_is_recorded(baseline) -> None:  # type: ignore[no-untyped-def]
    """⚠️ §6.3 asks for a collapse at **both** ends. This scene has one.

    At sunset the spread is still **5.5 K** and it decays monotonically through the night to the
    dawn minimum. The reason is visible in the per-pair crossings: each pair of surfaces does
    cross, but they cross at different times spread over three hours, so the *ensemble* never
    passes through a common point. A dusk collapse needs a facet set whose members share a solar
    absorptivity as well as differing in mass; this one deliberately does not, because its pairs
    exist to isolate other variables.

    Recorded rather than engineered away: making the scene produce a dusk flat by choosing its
    materials for that would be fitting the fixture to the acceptance test.
    """
    hours, temps, names = baseline
    spread = _spread(hours, temps, names)
    at_sunset = float(np.interp(SUNSET_UTC_H, hours, spread))
    assert at_sunset == pytest.approx(5.5, abs=1.0)
    assert at_sunset > 3.0 * float(spread.min())
    # every pair does cross, somewhere between one and three hours from sunset
    offsets = []
    for a, b in (
        ("asphalt_sun", "concrete"),
        ("soil_dry", "concrete"),
        ("asphalt_sun", "soil_wet"),
    ):
        diff = temps[:, names.index(a)] - temps[:, names.index(b)]
        crossings = hours[np.where(np.diff(np.sign(diff)) != 0)[0]]
        evening = [c for c in crossings if c > 12.0]
        assert evening, (a, b)
        offsets.append(abs(float(evening[0]) - SUNSET_UTC_H) * 60.0)
    assert all(60.0 < o < 200.0 for o in offsets), offsets


def test_the_morning_crossover_lands_near_sunrise(baseline) -> None:  # type: ignore[no-untyped-def]
    """The half that does work at the pair level: a fast, light surface crosses a slow one within
    about half an hour of sunrise."""
    hours, temps, names = baseline
    diff = temps[:, names.index("concrete")] - temps[:, names.index("steel_panel")]
    crossings = hours[np.where(np.diff(np.sign(diff)) != 0)[0]]
    morning = [c for c in crossings if c < 8.0]
    assert morning, crossings
    assert abs(float(morning[0]) - SUNRISE_UTC_H) * 60.0 < 60.0


# ---------------------------------------------------------------------------------------------
# the diurnal curve itself
# ---------------------------------------------------------------------------------------------


def test_sunlit_asphalt_peaks_in_the_early_afternoon(baseline) -> None:  # type: ignore[no-untyped-def]
    hours, temps, names = baseline
    column = temps[:, names.index("asphalt_sun")]
    peak_local = (float(hours[int(np.argmax(column))]) + LOCAL_OFFSET_H) % 24.0
    assert 13.0 <= peak_local <= 16.0, peak_local
    assert float(column.max()) - 273.15 == pytest.approx(60.1, abs=2.0)


def test_the_swing_exceeds_the_roadmaps_band_and_the_reason_is_the_back_boundary(baseline) -> None:  # type: ignore[no-untyped-def]
    """⚠️ Measured **47.1 K** against the roadmap's 15–40 K, and the model says why.

    These facets are **single nodes with an adiabatic back** (M6.9's solver), so the day's heat has
    nowhere to go but back out of the surface. Real ground conducts into depth, which is exactly
    what M6.8's two-node solver with a finite R₂d and T_deep is for — and using it here would pull
    the swing down into the quoted band. The single-node number is therefore an **upper bound**,
    not a disagreement with the spec, and it is left as one because a scene that wants the bound
    tightened should say what is underneath its ground rather than have this fixture guess.
    """
    hours, temps, names = baseline
    column = temps[:, names.index("asphalt_sun")]
    swing = float(column.max() - column.min())
    assert swing == pytest.approx(47.1, abs=3.0)
    assert swing > 15.0
    # the shaded twin, which never sees the beam, sits inside the quoted band
    shaded = temps[:, names.index("asphalt_shade")]
    assert 15.0 <= float(shaded.max() - shaded.min()) <= 40.0


def test_the_surface_falls_below_the_air_at_night(baseline) -> None:  # type: ignore[no-untyped-def]
    """Radiative cooling to a clear sky, which is half the swing and the reason a clear night
    reads colder than the thermometer says."""
    scene = Scene.from_file(SCENE_PATH)
    hours, temps, names = baseline
    coldest = int(np.argmin(temps[:, names.index("asphalt_sun")]))
    t_air = float(np.asarray(scene.weather.at(scene.t0_s + float(hours[coldest]) * 3600.0).t_air_k))
    assert float(temps[coldest, names.index("asphalt_sun")]) < t_air - 1.0


# ---------------------------------------------------------------------------------------------
# the pairs: one variable each
# ---------------------------------------------------------------------------------------------


def test_shade_is_far_colder_by_day_and_indistinguishable_at_night(baseline) -> None:  # type: ignore[no-untyped-def]
    """The same material, the same hour, differing only in the direct beam."""
    hours, temps, names = baseline
    noon = _at(hours, temps, names, "asphalt_sun", 12.0) - _at(
        hours, temps, names, "asphalt_shade", 12.0
    )
    night = _at(hours, temps, names, "asphalt_sun", 0.0) - _at(
        hours, temps, names, "asphalt_shade", 0.0
    )
    assert noon > 3.0
    assert abs(night) < 1.0


def test_wet_soil_is_colder_than_dry_in_the_afternoon(baseline) -> None:  # type: ignore[no-untyped-def]
    """Three times the areal heat capacity, and it shows in both directions."""
    hours, temps, names = baseline
    afternoon = _at(hours, temps, names, "soil_dry", 12.0) - _at(
        hours, temps, names, "soil_wet", 12.0
    )
    assert afternoon > 2.0
    # and warmer at dawn, which is the other half of "more mass"
    assert _at(hours, temps, names, "soil_wet", 3.0) > _at(hours, temps, names, "soil_dry", 3.0)


def test_a_moving_hood_is_far_colder_than_a_parked_one(baseline) -> None:  # type: ignore[no-untyped-def]
    """100 km/h against parked: forced convection and nothing else differs. **26 K** at 14:00."""
    hours, temps, names = baseline
    difference = _at(hours, temps, names, "hood_still", 12.0) - _at(
        hours, temps, names, "hood_moving", 12.0
    )
    assert difference > 3.0
    assert difference == pytest.approx(26.4, abs=4.0)


def test_black_and_white_paint_are_identical_at_night_and_nothing_alike_at_noon(baseline) -> None:  # type: ignore[no-untyped-def]
    """Same ε, opposite α_sol — which is why a white car and a black car look the same in LWIR
    at 03:00 and 29 K apart at 13:00."""
    hours, temps, names = baseline
    assert (
        abs(
            _at(hours, temps, names, "roof_black", 1.0)
            - _at(hours, temps, names, "roof_white", 1.0)
        )
        < 0.2
    )
    assert (
        _at(hours, temps, names, "roof_black", 11.0) - _at(hours, temps, names, "roof_white", 11.0)
    ) > 20.0


# ---------------------------------------------------------------------------------------------
# weather sensitivity
# ---------------------------------------------------------------------------------------------


def test_overcast_cuts_both_the_swing_and_the_night_contrast(baseline) -> None:  # type: ignore[no-untyped-def]
    """Cloud removes the beam by day and the cold sky by night, so it flattens both ends."""
    hours, temps, names = baseline
    cloudy_hours, cloudy, cloudy_names = _rebuilt(
        cloud_fraction=lambda c: np.full_like(c, 0.95),
        dni_w_m2=lambda d: d * 0.15,
        dhi_w_m2=lambda d: d * 1.6,
    )
    clear_swing = float(np.ptp(temps[:, names.index("asphalt_sun")]))
    cloudy_swing = float(np.ptp(cloudy[:, cloudy_names.index("asphalt_sun")]))
    assert cloudy_swing < 0.70 * clear_swing, (clear_swing, cloudy_swing)

    night = hours < SUNRISE_UTC_H
    clear_night = float(_spread(hours, temps, names)[night].mean())
    cloudy_night = float(_spread(cloudy_hours, cloudy, cloudy_names)[night].mean())
    assert cloudy_night < 0.70 * clear_night, (clear_night, cloudy_night)


def test_doubling_the_wind_lowers_the_peak(baseline) -> None:  # type: ignore[no-untyped-def]
    """2.5 → 5 m/s. Forced convection is the only term that moves, and it moves the peak by more
    than a kelvin — which is why a scene that ignores wind gets its hottest surfaces wrong."""
    hours, temps, names = baseline
    windy_hours, windy, windy_names = _rebuilt(wind_speed_m_s=lambda w: w * 2.0)
    calm_peak = float(temps[:, names.index("asphalt_sun")].max())
    windy_peak = float(windy[:, windy_names.index("asphalt_sun")].max())
    assert calm_peak - windy_peak > 1.0, (calm_peak, windy_peak)


def test_the_bench_runs_on_a_sixty_second_tick() -> None:
    """The scene's own tick, so this bench is measuring what a render would see."""
    config = load_scene_config(SCENE_PATH)
    assert isinstance(config, SceneConfig)
    assert config.scene.thermal is not None
    assert config.scene.thermal.tick_s == 60.0


def test_the_sun_times_this_module_asserts_against_are_the_scene_s_own() -> None:
    """The sunrise and sunset constants above are derived, not typed — if the site or the start
    date moves, this fails rather than the phenomenology quietly drifting past them."""
    scene = Scene.from_file(SCENE_PATH)
    hours = np.arange(0.0, 24.0, 1.0 / 60.0)
    elevation = np.array(
        [
            sun_position_utc(
                scene.spec.site.latitude_deg,
                scene.spec.site.longitude_deg,
                scene.weather.epoch_utc + timedelta(seconds=scene.t0_s + h * 3600.0),
            ).elevation_deg
            for h in hours
        ]
    )
    up = elevation > 0.0
    assert float(hours[int(np.argmax(up))]) == pytest.approx(SUNRISE_UTC_H, abs=0.1)
    assert float(hours[len(up) - 1 - int(np.argmax(up[::-1]))]) == pytest.approx(
        SUNSET_UTC_H, abs=0.1
    )
