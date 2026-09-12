"""Reflected environment term (M7.13): the isothermal enclosure identity for every emissivity and
sky-view factor, the sky-view-factor formula against a cosine-weighted Monte Carlo, the cold-roof
drop under a clear sky vs overcast, bare aluminium colder than paint, and the Scene plumbing."""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.config.environment import load_environment_preset
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.environment import environment_radiance, sky_view_factor
from irsim.radiometry.lut import BandLUT
from irsim.scene import Scene
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
EPS = {1: 0.05, 2: 0.5, 3: 0.9, 4: 1.0}


def _weather(t_air: float, cloud: float, rh: float = 0.3) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, rh, 1.0, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _sky(lut: BandLUT, t_air: float, cloud: float, env: str = "clear_dry") -> SkyModel:
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(t_air, cloud), {"lwir": lut}
    )
    return SkyModel(atm, load_environment_preset(env), "lwir", lut)


def _config(lut: BandLUT, sky: SkyModel | None, materials: MaterialTable) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=16, height=8)
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        materials,
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
        sky=sky,
    )


def _gbuffer(t_k: float) -> dict[str, np.ndarray]:
    shape = (8, 16)
    ids = np.ones(shape, np.int32)
    ids[:, 4:8] = 2
    ids[:, 8:12] = 3
    ids[:, 12:] = 4
    v_s = np.ones(shape, np.float32)
    v_s[2:4] = 0.5
    v_s[4:6] = 0.25
    v_s[6:8] = 0.0
    return {
        "temperature_k": np.full(shape, t_k, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": ids,
        "sky_view_factor": v_s,
    }


def test_isothermal_enclosure_reads_the_surface_temperature(tophat_lwir_lut: BandLUT) -> None:
    """T_s = T_sky = T_ground: T_app = T_s within 1 mK for eps in {0.05, 0.5, 0.9, 1} and V_s in
    {0, 0.25, 0.5, 1} -- the reflected term closes.

    The enclosure temperature is *derived from the sky model* rather than assumed to be T_air. A
    fully overcast sky is tilt-independent (the blend is the cloud term alone), which is what makes
    a single enclosure temperature constructible at all; its value is whatever the cloud model says
    it is, so this identity survives MS.3 changing the cloud base. The ground is pinned to the same
    temperature with ground.mode 'fixed' -- under ground.mode 'air' the ground would sit at T_air
    and the enclosure would not be isothermal.
    """
    t_air = 293.15
    overcast = _sky(tophat_lwir_lut, t_air, cloud=1.0)
    l_sky = float(overcast.effective_radiance(0.0, 0.0))
    for tilt_deg in (0.0, 45.0, 90.0, 180.0):
        assert float(overcast.effective_radiance(0.0, math.radians(tilt_deg))) == pytest.approx(
            l_sky, rel=1e-12
        ), "a fully overcast sky must be tilt-independent for the enclosure to be constructible"
    t_enclosure = float(tophat_lwir_lut.apparent_temperature(np.asarray(l_sky))[()])

    raw = yaml.safe_load((REPO / "configs" / "environments" / "clear_dry.yaml").read_text())
    raw["environment"]["ground"] = {"mode": "fixed", "fixed_temperature_k": t_enclosure}
    from irsim.config.environment import EnvironmentConfig

    sky = SkyModel(
        overcast.atmosphere,
        EnvironmentConfig.model_validate(raw).environment,
        "lwir",
        tophat_lwir_lut,
    )
    cfg = _config(tophat_lwir_lut, sky, MaterialTable.from_mapping(EPS))
    out = run_frame(_gbuffer(t_enclosure), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert out.apparent_t is not None
    err_mk = np.abs(out.apparent_t.astype(np.float64) - t_enclosure) * 1e3
    assert err_mk.max() < 1.0, err_mk.max()


def test_sky_view_factor_formula_and_monte_carlo() -> None:
    assert sky_view_factor(1.0).item() == 1.0 and sky_view_factor(0.0).item() == 0.5
    assert sky_view_factor(-1.0).item() == 0.0 and sky_view_factor(0.0, 0.4).item() == 0.2
    rng = np.random.default_rng(7)
    n = 400_000
    # cosine-weighted directions about the normal: (sqrt(u1) cos, sqrt(u1) sin, sqrt(1 - u1))
    u1, u2 = rng.random(n), rng.random(n)
    r = np.sqrt(u1)
    local = np.stack(
        [r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2), np.sqrt(1.0 - u1)], axis=1
    )
    for tilt_deg in (0.0, 30.0, 60.0, 90.0, 135.0):
        beta = np.radians(tilt_deg)
        # rotate the local frame so the normal is (sin beta, 0, cos beta)
        up_z = local[:, 2] * np.cos(beta) - local[:, 0] * np.sin(beta)
        fraction_sky = float(np.mean(up_z > 0.0))
        assert abs(fraction_sky - sky_view_factor(np.cos(beta)).item()) < 1e-3, tilt_deg
    with pytest.raises(ValueError):
        sky_view_factor(1.5)
    with pytest.raises(ValueError):
        sky_view_factor(0.5, 1.5)


def test_cold_roof_drop_under_a_clear_sky(tophat_lwir_lut: BandLUT) -> None:
    """A horizontal eps = 0.9 panel at T_air reads colder under a clear sky than under cloud, by
    the amount the reflected term predicts.

    Both references are computed from each sky's *own* effective radiance -- the overcast panel is
    not assumed to read exactly T_air, because the cloud deck radiates at its base temperature, not
    at the surface air temperature (MS.3). What this test owns is the reflected term, so it asserts
    the pipeline reproduces L = L_B(T_air) - (1 - eps)(L_B(T_air) - L_sky,eff) for each sky, the
    drop between them, and the phenomenology: overcast flattens the image, a mirror exaggerates it.
    """
    t_air = 288.15
    clear = _sky(tophat_lwir_lut, t_air, cloud=0.0)
    overcast = _sky(tophat_lwir_lut, t_air, cloud=1.0)
    materials = MaterialTable.from_mapping({1: 0.9, 2: 0.09})
    g = _gbuffer(t_air)
    g["material_id"][:] = 1
    g["material_id"][:, 8:] = 2
    g["sky_view_factor"][:] = 1.0
    res = {}
    for name, sky in (("clear", clear), ("overcast", overcast)):
        cfg = _config(tophat_lwir_lut, sky, materials)
        out = run_frame(g, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
        assert out.apparent_t is not None
        res[name] = out.apparent_t.astype(np.float64)
    lb_air = float(tophat_lwir_lut.lookup(np.float64(t_air))[()])
    dlb = float(tophat_lwir_lut.lookup(np.float64(t_air), "dlb_dt")[()])
    l_eff = {
        n: float(s.effective_radiance(0.0, 0.0))
        for n, s in (("clear", clear), ("overcast", overcast))
    }

    def predicted(name: str) -> float:
        """T_app of an eps = 0.9 panel at T_air seeing only this sky (V_s = 1)."""
        lb = lb_air - (1 - 0.9) * (lb_air - l_eff[name])
        return float(tophat_lwir_lut.apparent_temperature(np.asarray(lb))[()])

    # the pipeline reproduces the reflected term for each sky, not merely their difference
    for name in ("clear", "overcast"):
        assert float(res[name][0, 0]) == pytest.approx(predicted(name), abs=1e-3), name

    drop_paint = float(res["clear"][0, 0] - res["overcast"][0, 0])
    exact = predicted("clear") - predicted("overcast")
    linear = -(1 - 0.9) * (l_eff["overcast"] - l_eff["clear"]) / dlb
    assert drop_paint < 0.0 and abs(drop_paint - exact) < 0.05, (drop_paint, exact)
    assert abs(drop_paint - linear) < 0.2, "the linearised form carries a ~0.1 K second-order term"
    # overcast flattens the image: the panel sits closer to its own temperature than under clear sky
    assert abs(float(res["overcast"][0, 0]) - t_air) < abs(float(res["clear"][0, 0]) - t_air)
    assert l_eff["overcast"] > l_eff["clear"], "cloud is warmer than the clear column"
    assert float(res["clear"][0, 0] - res["clear"][0, 8]) > 10.0, "aluminium mirrors the cold sky"


def test_environment_radiance_ground_modes_and_guards(tophat_lwir_lut: BandLUT) -> None:
    t_air = 290.0
    sky = _sky(tophat_lwir_lut, t_air, cloud=0.0)
    v = np.array([[0.0, 0.5, 1.0]], np.float32)
    l_env = environment_radiance(sky, tophat_lwir_lut, 0.0, v)
    assert l_env.dtype == np.float32
    lb_air = float(tophat_lwir_lut.lookup(np.float64(t_air))[()])
    assert l_env[0, 0] == pytest.approx(lb_air, rel=1e-6), "ground at T_air for V_s = 0"
    assert l_env[0, 2] == pytest.approx(float(sky.effective_radiance(0.0, 0.0)), rel=1e-6)
    assert l_env[0, 0] > l_env[0, 1] > l_env[0, 2]
    raw = yaml.safe_load((REPO / "configs" / "environments" / "clear_dry.yaml").read_text())
    raw["environment"]["ground"] = {"mode": "fixed", "fixed_temperature_k": 300.0}
    from irsim.config.environment import EnvironmentConfig

    fixed = SkyModel(
        sky.atmosphere, EnvironmentConfig.model_validate(raw).environment, "lwir", tophat_lwir_lut
    )
    assert environment_radiance(fixed, tophat_lwir_lut, 0.0, np.zeros((1, 1)))[
        0, 0
    ] == pytest.approx(float(tophat_lwir_lut.lookup(np.float64(300.0))[()]), rel=1e-6)
    with pytest.raises(ValueError):
        environment_radiance(sky, tophat_lwir_lut, 0.0, np.array([[1.5]]))
    raw["environment"]["ground"] = {"mode": "solver"}
    solver_sky = SkyModel(
        sky.atmosphere, EnvironmentConfig.model_validate(raw).environment, "lwir", tophat_lwir_lut
    )
    with pytest.raises(ValueError, match="M6.12"):
        _config(tophat_lwir_lut, solver_sky, MaterialTable.constant(0.9))


def test_scene_builds_sky_models_on_the_same_weather(tophat_lwir_lut: BandLUT) -> None:
    scene = Scene.from_file(
        REPO / "configs" / "scenes" / "sky_target_clear_day.yaml", {"lwir": tophat_lwir_lut}
    )
    assert scene.spec.environment_preset == "clear_dry" and scene.layered is not None
    sky = scene.sky_models["lwir"]
    assert sky.weather is scene.weather and scene.layered.weather is scene.weather
    assert set(scene.consumers) >= {"atmosphere", "layered", "sky:lwir"}
    cfg = _config(tophat_lwir_lut, sky, MaterialTable.constant(0.9))
    assert cfg.sky is sky
    t_sky = float(sky.apparent_temperature_k(scene.t0_s, np.radians(30.0)))
    assert t_sky < scene.weather_at(0.0).t_air_k - 20.0
