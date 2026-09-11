"""Stage 2 on the G-buffer (M8.6): the §3.3 known answer through the whole chain, isothermal
invariance at 10 m and 5 km, vectorised == scalar loop, sky pixels bit-identical, float16
refused, the constant-tau fallback, and identity without an Atmosphere."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.atmosphere import Atmosphere, load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.atmosphere import apply_atmosphere_gbuffer, atmosphere_stage
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import band_radiance_tophat
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
T_AIR = 290.0


@pytest.fixture(scope="module")
def lut_8_12(tmp_path_factory: pytest.TempPathFactory) -> BandLUT:
    p = tmp_path_factory.mktemp("lut") / "tophat_8_12.csv"
    p.write_text("# exact top-hat\n8.0,1.0\n12.0,1.0\n")
    return BandLUT.build(load_spectral_response(p))


@pytest.fixture(scope="module")
def atmosphere() -> Atmosphere:
    w = WeatherSeries.constant(WeatherSample(T_AIR, 0.5, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0)
    return Atmosphere(load_atmosphere_preset("us_standard_clear"), w)


def _config(lut: BandLUT, atmosphere: Atmosphere | None, **kw: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=8.0, lambda_max_um=12.0)
    d["sensor"]["fpa"].update(width=16, height=8)
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        psf_enabled=False,
        atmosphere=atmosphere,
        **kw,
    )


def _gbuffer(
    t_k: float, distance_m: float, shape: tuple[int, int] = (8, 16)
) -> dict[str, np.ndarray]:
    return {
        "temperature_k": np.full(shape, t_k, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, distance_m, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.ones(shape, np.float32),
    }


def _bisect_apparent(target_l: float) -> float:
    a, b = 250.0, 350.0
    for _ in range(80):
        m = 0.5 * (a + b)
        if band_radiance_tophat(8.0, 12.0, m) < target_l:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def test_known_answer_through_the_whole_chain(lut_8_12: BandLUT, atmosphere: Atmosphere) -> None:
    """310 K blackbody, T_air 290 K, tau = 0.8, 8-12 um -> T_app = 306.303 K within 1 mK
    (roadmap M8.6), cross-checked by bisection on the closed-form band radiance."""
    gamma = atmosphere.gamma("lwir", 0.0)
    d = -math.log(0.8) / gamma
    cfg = _config(lut_8_12, atmosphere)
    state = PipelineState(housing_temp_k=cfg.t_housing_cal_k, t_s=0.0)
    out = run_frame(_gbuffer(310.0, d), cfg, state)
    assert out.apparent_t is not None
    expect = _bisect_apparent(
        0.8 * band_radiance_tophat(8.0, 12.0, 310.0) + 0.2 * band_radiance_tophat(8.0, 12.0, T_AIR)
    )
    assert expect == pytest.approx(306.303, abs=1e-3)
    err_mk = np.max(np.abs(out.apparent_t.astype(np.float64) - expect)) * 1e3
    assert err_mk < 1.0, f"{err_mk:.3f} mK"


@pytest.mark.parametrize("distance", [10.0, 5000.0])
def test_isothermal_gbuffer_reads_t_air(
    lut_8_12: BandLUT, atmosphere: Atmosphere, distance: float
) -> None:
    cfg = _config(lut_8_12, atmosphere)
    out = run_frame(
        _gbuffer(T_AIR, distance), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
    )
    assert out.apparent_t is not None
    assert np.max(np.abs(out.apparent_t.astype(np.float64) - T_AIR)) * 1e3 < 1.0


def test_vectorised_equals_scalar_loop_and_float16_refused() -> None:
    rng = np.random.default_rng(11)
    lb = rng.uniform(10.0, 60.0, (6, 7))
    d = rng.uniform(0.0, 3000.0, (6, 7))
    gamma, l_air = 4e-4, 30.0
    out = apply_atmosphere_gbuffer(lb, d, gamma, l_air)
    for i in range(6):
        for j in range(7):
            tau = math.exp(-gamma * d[i, j])
            assert out[i, j] == pytest.approx(tau * lb[i, j] + (1 - tau) * l_air, rel=1e-12)
    with pytest.raises(TypeError, match="float16"):
        apply_atmosphere_gbuffer(lb.astype(np.float16), d, gamma, l_air)
    with pytest.raises(ValueError, match="shape"):
        apply_atmosphere_gbuffer(lb, d[:, :3], gamma, l_air)


def test_sky_pixels_bit_identical(lut_8_12: BandLUT, atmosphere: Atmosphere) -> None:
    lb = np.random.default_rng(2).uniform(10.0, 60.0, (8, 16)).astype(np.float32)
    d = np.full((8, 16), 500.0, np.float32)
    sky = np.zeros((8, 16), bool)
    sky[:4] = True
    out = apply_atmosphere_gbuffer(lb, d, 4e-4, 30.0, sky_mask=sky)
    assert (
        out.dtype == np.float32
        and np.array_equal(out[:4], lb[:4])
        and not np.array_equal(out[4:], lb[4:])
    )
    out2 = apply_atmosphere_gbuffer(lb, d, 4e-4, 30.0, sky_mask=sky, tau_override=0.5)
    assert np.array_equal(out2[:4], lb[:4])
    with pytest.raises(ValueError, match="sky_mask"):
        apply_atmosphere_gbuffer(lb, d, 4e-4, 30.0, sky_mask=sky.astype(np.uint8))
    # through run_frame: the sky half of the frame is unchanged by switching the atmosphere on
    g = _gbuffer(310.0, 800.0)
    g["temperature_k"][:4] = 240.0  # apparent sky temperature, already "through the atmosphere"
    g["material_id"][:4] = 0
    g["distance_m"][:4] = 0.0
    g["sky_mask"] = sky
    with_atm = run_frame(g, _config(lut_8_12, atmosphere), PipelineState(housing_temp_k=300.0))
    without = run_frame(g, _config(lut_8_12, None), PipelineState(housing_temp_k=300.0))
    assert with_atm.apparent_t is not None and without.apparent_t is not None
    assert np.array_equal(with_atm.apparent_t[:4], without.apparent_t[:4])
    assert np.all(with_atm.apparent_t[4:] < without.apparent_t[4:]), "geometry cooled toward T_air"


def test_tau_override_is_distance_independent(lut_8_12: BandLUT, atmosphere: Atmosphere) -> None:
    cfg = _config(lut_8_12, atmosphere, tau_override=0.8)
    a = run_frame(_gbuffer(310.0, 10.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    b = run_frame(_gbuffer(310.0, 5000.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert a.apparent_t is not None and b.apparent_t is not None
    assert np.array_equal(a.apparent_t, b.apparent_t)
    assert abs(float(a.apparent_t[4, 8]) - 306.303) < 2e-3
    with pytest.raises(ValueError, match="needs an Atmosphere"):
        _config(lut_8_12, None, tau_override=0.8)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        _config(lut_8_12, atmosphere, tau_override=1.5)


def test_stage_function_and_identity_without_atmosphere(
    lut_8_12: BandLUT, atmosphere: Atmosphere
) -> None:
    cfg = _config(lut_8_12, atmosphere)
    planes = {
        "radiance": np.full((8, 16), 40.0, np.float32),
        "distance_m": np.full((8, 16), 1000.0, np.float32),
    }
    out = atmosphere_stage(planes, cfg, PipelineState(t_s=0.0))["radiance"]
    l_air = float(lut_8_12.lookup(np.float64(T_AIR))[()])
    tau = math.exp(-atmosphere.gamma("lwir", 0.0) * 1000.0)
    np.testing.assert_allclose(out, tau * 40.0 + (1 - tau) * l_air, rtol=1e-6)
    none = _config(lut_8_12, None)
    assert none.atmosphere is None and none.tau_override is None
    assert np.array_equal(
        atmosphere_stage(planes, none, PipelineState())["radiance"], planes["radiance"]
    )
