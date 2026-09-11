"""Scene builder (M6.17): the weather is loaded once and the same object reaches every
consumer; no consumer constructor takes a file; two weather objects cannot coexist; the T_air
used for path radiance equals the T_air used for convection at every timestamp."""

from __future__ import annotations

import inspect
import pathlib

import pytest

import irsim.scene as scene_mod
from irsim.atmosphere import Atmosphere, load_atmosphere_preset
from irsim.config.scene import SceneConfig, load_scene_config
from irsim.radiometry.lut import BandLUT
from irsim.scene import Scene
from irsim.thermal import NewtonCoolingSolver, PrescribedSolver, load_weather_csv

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"


def test_no_consumer_constructor_takes_a_file() -> None:
    for cls in (Atmosphere, NewtonCoolingSolver, PrescribedSolver, Scene):
        params = inspect.signature(cls.__init__).parameters
        for p in params:
            assert "file" not in p and "path" not in p, f"{cls.__name__}({p})"


def test_weather_loaded_once_and_shared(
    tophat_lwir_lut: BandLUT, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[pathlib.Path] = []
    real = scene_mod.load_weather_csv

    def counting(path):  # type: ignore[no-untyped-def]
        calls.append(pathlib.Path(path))
        return real(path)

    monkeypatch.setattr(scene_mod, "load_weather_csv", counting)
    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    assert len(calls) == 1 and calls[0].name == "clear_midlat_summer_48h.csv"
    assert scene.atmosphere.weather is scene.weather
    for name, solver in scene.targets.items():
        w = getattr(solver, "weather", None)
        assert w is None or w is scene.weather, name
    assert scene.targets["airframe"].weather is scene.weather  # type: ignore[attr-defined]
    assert scene.t0_s == 4 * 3600.0 and set(scene.consumers) == {
        "atmosphere",
        "target:airframe",
        "target:engine",
    }


def test_path_radiance_and_convection_see_the_same_t_air(tophat_lwir_lut: BandLUT) -> None:
    scene = Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})
    airframe = scene.targets["airframe"]
    assert isinstance(airframe, NewtonCoolingSolver)
    for h in range(0, 44):
        t_rel = h * 3600.0
        t_air_path = scene.atmosphere.state(scene.t0_s + t_rel).t_air_k
        t_air_conv = airframe.ambient_at(scene.t0_s + t_rel)
        assert abs(t_air_path - t_air_conv) < 1e-9
        assert abs(t_air_path - scene.weather_at(t_rel).t_air_k) < 1e-9
    temps = scene.advance_targets(0.0, 300.0)
    assert temps["engine"] == 340.0, "prescribed node exact at its schedule node"
    assert 296.15 > temps["airframe"] > scene.weather_at(150.0).t_air_k, "relaxing toward T_air"


def test_two_weather_objects_cannot_coexist(tophat_lwir_lut: BandLUT) -> None:
    cfg = load_scene_config(SCENE_YAML)
    scene = Scene.from_config(cfg, {"lwir": tophat_lwir_lut})
    other = load_weather_csv(REPO / "data" / "weather" / "clear_midlat_summer_48h.csv")
    assert other == scene.weather, "equal content ..."
    assert other is not scene.weather, "... but a different object is still refused"
    with pytest.raises(ValueError, match="different WeatherSeries"):
        Scene(
            spec=scene.spec,
            weather=scene.weather,
            atmosphere=Atmosphere(load_atmosphere_preset("haze"), other),
            targets=scene.targets,
            t0_s=scene.t0_s,
        )
    with pytest.raises(ValueError, match="different WeatherSeries"):
        Scene(
            spec=scene.spec,
            weather=scene.weather,
            atmosphere=scene.atmosphere,
            targets={"x": NewtonCoolingSolver(300.0, 10.0, other)},
            t0_s=scene.t0_s,
        )
    with pytest.raises(ValueError, match="different WeatherSeries"):
        Scene(
            spec=scene.spec,
            weather=scene.weather,
            atmosphere=scene.atmosphere,
            targets=scene.targets,
            t0_s=scene.t0_s,
            extra_consumers={"sky": type("Sky", (), {"weather": other})()},
        )
    with pytest.raises(ValueError, match="outside"):
        Scene(
            spec=scene.spec,
            weather=scene.weather,
            atmosphere=scene.atmosphere,
            targets={},
            t0_s=-1.0,
        )


def test_scene_config_guards() -> None:
    raw = {
        "schema_version": 1,
        "scene": {
            "name": "x",
            "weather_file": "weather/clear_midlat_summer_48h.csv",
            "atmosphere_preset": "haze",
            "site": {"latitude_deg": 0.0, "longitude_deg": 0.0},
            "start_utc": "2024-06-21T04:00:00Z",
            "targets": [{"name": "a", "solver": "newton", "t0_k": 300.0, "tau_s": 10.0}],
        },
    }
    SceneConfig.model_validate(raw)
    bad = {**raw, "scene": {**raw["scene"], "start_utc": "2024-06-21T04:00:00"}}
    with pytest.raises(ValueError, match="UTC offset"):
        SceneConfig.model_validate(bad)
    bad = {
        **raw,
        "scene": {**raw["scene"], "targets": [{"name": "a", "solver": "newton", "t0_k": 300.0}]},
    }
    with pytest.raises(ValueError, match="needs t0_k and tau_s"):
        SceneConfig.model_validate(bad)
    bad = {
        **raw,
        "scene": {
            **raw["scene"],
            "targets": [
                {
                    "name": "a",
                    "solver": "prescribed",
                    "schedule_s": [0.0, 0.0],
                    "schedule_k": [300.0, 310.0],
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        SceneConfig.model_validate(bad)
    bad = {**raw, "scene": {**raw["scene"], "targets": raw["scene"]["targets"] * 2}}
    with pytest.raises(ValueError, match="unique"):
        SceneConfig.model_validate(bad)
    bad = {**raw, "scene": {**raw["scene"], "air_temperature_k": 300.0}}
    with pytest.raises(ValueError):
        SceneConfig.model_validate(bad)
