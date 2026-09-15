"""M6.12 — the `thermal:` block, and the trap of pairing a site with the wrong weather file.

A scene that names a weather file and a site is making a claim those two have to agree about:
**where noon is**. `clear_midlat_summer_48h` declares "local = UTC+2" in its own header, so its DNI
half-sine and its T_air sinusoid are keyed to a solar noon near 10:00 UTC. Pairing it with a
Californian site does not raise — `solar_loading` multiplies the file's DNI by a cos θ that peaks
ten hours later and the product peaks somewhere in between. The scene renders, the diurnal curve
still looks like a diurnal curve, and every time-of-day statement about it is wrong. This module
pins the agreement rather than trusting it.

docs/physics-model.md §12.3 thermal block, §12.2; ADR 0032, ADR 0037, ADR 0043
"""

from __future__ import annotations

import copy
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest
import yaml

from irsim.config.scene import (
    MIN_SCENE_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    SceneConfig,
    load_scene_config,
)
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_PATH = REPO / "configs" / "scenes" / "thermal_facet_scene.yaml"


@pytest.fixture(scope="module")
def scene() -> Scene:
    return Scene.from_file(SCENE_PATH)


def _raw() -> dict:
    return yaml.safe_load(SCENE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# the config
# ---------------------------------------------------------------------------------------------


def test_the_committed_facet_scene_validates_and_builds(scene: Scene) -> None:
    assert scene.thermal is not None
    assert scene.thermal_surfaces == (
        "asphalt_sun",
        "asphalt_shade",
        "asphalt_wet",
        "concrete",
        "roof_black",
        "roof_white",
        "glass",
    )


def test_one_weather_object_reaches_every_consumer(scene: Scene) -> None:
    """CLAUDE.md #6, including the new one. The forcing model carries `.weather` precisely so the
    `Scene`'s existing guard can see it — a forcing model quietly holding a second series is the
    failure that guard is for."""
    assert scene.thermal.forcing_at.weather is scene.atmosphere.weather
    assert scene.thermal.forcing_at.weather is scene.weather
    assert "thermal" in scene.consumers


def test_a_naive_start_datetime_is_rejected() -> None:
    raw = _raw()
    raw["scene"]["start_utc"] = "2024-06-22T00:00:00"
    with pytest.raises(ValueError, match="UTC offset"):
        SceneConfig.model_validate(raw)


def test_duplicate_surface_names_are_rejected() -> None:
    raw = _raw()
    raw["scene"]["thermal"]["surfaces"].append(dict(raw["scene"]["thermal"]["surfaces"][0]))
    with pytest.raises(ValueError, match="surface names must be unique"):
        SceneConfig.model_validate(raw)


def test_a_surface_cannot_author_an_emissivity() -> None:
    """ADR 0043's rule, enforced by the schema forbidding the field rather than ignoring it."""
    raw = _raw()
    raw["scene"]["thermal"]["surfaces"][0]["emissivity"] = 0.5
    with pytest.raises(ValueError, match="[Ee]xtra"):
        SceneConfig.model_validate(raw)


def test_older_scene_documents_still_load() -> None:
    """v5 added `thermal:` as an *optional* field, so every v4 document is a valid v5 document.

    A range is the honest representation of a backwards-compatible change; refusing a v4 scene
    would be refusing it for a change that cannot affect it — and would break every scene config
    in the repository for an addition none of them use.
    """
    assert MIN_SCENE_SCHEMA_VERSION < SCENE_SCHEMA_VERSION
    for path in sorted((REPO / "configs" / "scenes").glob("*.yaml")):
        config = load_scene_config(path)
        assert MIN_SCENE_SCHEMA_VERSION <= config.schema_version <= SCENE_SCHEMA_VERSION

    raw = _raw()
    raw["schema_version"] = SCENE_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="readable range"):
        SceneConfig.model_validate(raw)
    raw["schema_version"] = MIN_SCENE_SCHEMA_VERSION - 1
    with pytest.raises(ValueError, match="readable range"):
        SceneConfig.model_validate(raw)


def test_a_scene_without_a_thermal_block_is_unchanged() -> None:
    """Every scene written before M6.12 keeps its phase-1 prescribed and Newton solvers."""
    aerial = Scene.from_file(REPO / "configs" / "scenes" / "sky_target_clear_day.yaml")
    assert aerial.thermal is None and aerial.thermal_surfaces == ()
    assert aerial.targets, "the phase-1 target solvers must still be there"
    with pytest.raises(ValueError, match="no thermal block"):
        aerial.surface_temperature_k("anything", aerial.t0_s)


# ---------------------------------------------------------------------------------------------
# the site and the weather file must agree about noon
# ---------------------------------------------------------------------------------------------


def test_the_site_agrees_with_the_weather_file_about_where_noon_is(scene: Scene) -> None:
    """⚠️ The trap. The file's header says "local = UTC+2"; the site is at +30°E, so solar noon
    lands near 10:00 UTC and the file's irradiance and the scene's sun geometry peak together.

    Checked by looking at the **sunlit-minus-shaded** difference, which isolates the direct beam:
    it must peak within a couple of hours of the file's own DNI maximum. Under a mismatched site
    the two peak ten hours apart and this fails.
    """
    scene.thermal.advance_to(scene.t0_s + 24 * 3600.0)
    sun_index = scene.thermal_surfaces.index("asphalt_sun")
    shade_index = scene.thermal_surfaces.index("asphalt_shade")
    hours = np.arange(0.0, 24.0, 0.25)
    direct = np.array(
        [
            float(scene.thermal.temperature_at(scene.t0_s + h * 3600.0)[sun_index])
            - float(scene.thermal.temperature_at(scene.t0_s + h * 3600.0)[shade_index])
            for h in hours
        ]
    )
    beam_peak_hour = float(hours[int(np.argmax(direct))])

    weather = scene.weather
    times = np.asarray(weather.time_s)
    window = (times >= scene.t0_s) & (times < scene.t0_s + 24 * 3600.0)
    dni = np.array([float(weather.at(float(t)).dni_w_m2) for t in times[window]])
    dni_peak_hour = float((times[window][int(np.argmax(dni))] - scene.t0_s) / 3600.0)
    # the surface lags the forcing by its own time constant, so this is "a few hours", not "equal"
    assert abs(beam_peak_hour - dni_peak_hour) < 5.0, (beam_peak_hour, dni_peak_hour)


def test_a_mismatched_site_shifts_the_peak_by_most_of_a_day() -> None:
    """The counter-test: the same weather file at a Californian site, which does not raise."""
    raw = _raw()
    raw["scene"]["site"]["longitude_deg"] = -122.08
    raw["scene"]["site"]["latitude_deg"] = 37.42
    mismatched = Scene.from_config(SceneConfig.model_validate(raw))
    mismatched.thermal.advance_to(mismatched.t0_s + 24 * 3600.0)
    sun_index = mismatched.thermal_surfaces.index("asphalt_sun")
    shade_index = mismatched.thermal_surfaces.index("asphalt_shade")
    hours = np.arange(0.0, 24.0, 0.25)

    def beam_peak(field, t0):  # type: ignore[no-untyped-def]
        direct = [
            float(field.temperature_at(t0 + h * 3600.0)[sun_index])
            - float(field.temperature_at(t0 + h * 3600.0)[shade_index])
            for h in hours
        ]
        return float(hours[int(np.argmax(direct))])

    aligned = Scene.from_file(SCENE_PATH)
    aligned.thermal.advance_to(aligned.t0_s + 24 * 3600.0)
    shift = abs(
        beam_peak(mismatched.thermal, mismatched.t0_s) - beam_peak(aligned.thermal, aligned.t0_s)
    )
    assert shift > 3.0, f"a site ten hours of longitude away shifted the beam peak by {shift} h"


# ---------------------------------------------------------------------------------------------
# the field behaves like M6.11's
# ---------------------------------------------------------------------------------------------


def test_the_scene_field_starts_spun_up_rather_than_transient(scene: Scene) -> None:
    """The spin-up ends at t₀, so the first frame is the state the weather implies — not the air
    temperature, and not a transient decaying through the first few hours of the clip."""
    t_air = float(np.asarray(scene.weather.at(scene.t0_s).t_air_k))
    first = scene.thermal.temperature_at(scene.t0_s)
    assert np.any(np.abs(first.astype(np.float64) - t_air) > 1.0)
    # the slow surfaces are *above* the air at midnight, which is what "still warm at midnight" is
    concrete = scene.surface_temperature_k("concrete", scene.t0_s)
    assert concrete > t_air + 1.0


def test_the_query_is_const_and_named(scene: Scene) -> None:
    scene.thermal.advance_to(scene.t0_s + 60.0)
    before = scene.thermal.state_hash()
    for _ in range(500):
        scene.surface_temperature_k("roof_black", scene.t0_s + 30.0)
    assert scene.thermal.state_hash() == before
    with pytest.raises(KeyError, match="unknown surface"):
        scene.surface_temperature_k("bonnet", scene.t0_s)


def test_the_scene_is_deterministic() -> None:
    a, b = Scene.from_file(SCENE_PATH), Scene.from_file(SCENE_PATH)
    a.thermal.advance_to(a.t0_s + 3600.0)
    b.thermal.advance_to(b.t0_s + 3600.0)
    assert a.thermal.state_hash() == b.thermal.state_hash()
    assert isinstance(a.spec.start_utc, datetime)
    assert a.spec.start_utc.tzinfo is not None
    assert a.spec.start_utc.astimezone(timezone.utc) == b.spec.start_utc
    assert copy.copy(a.thermal_surfaces) == b.thermal_surfaces
