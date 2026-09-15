"""Tier 2 NETD bench end to end (M4.9): two blackbodies through run_frame (band radiance →
optics → detector noise → correlated noise → ADC), measured in the DN domain; replay is
bit-identical, frame 500 can be reached directly, float16 is refused, sensors are independent."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.noise import Sigmas7
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT
from irsim.validation import (
    compare_absolute,
    decompose_3d,
    estimate_floors,
    load_measured_pairs,
    measured_netd_k,
    measured_path,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _config(lut: BandLUT, seed: int = 7, size: int = 64) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=size, height=size)
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d), MaterialTable.constant(1.0), lut=lut, sensor_seed=seed
    )


def _gbuffer(shape: tuple[int, int], t: float) -> dict[str, np.ndarray]:
    temp = np.full(shape, t, dtype=np.float32)
    return {
        "temperature_k": temp,
        "encoded_t": encode_temperature(temp),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.full(shape, 0.5, np.float32),
    }


def _stack(cfg: PipelineConfig, t: float, n: int, first_frame: int = 0) -> np.ndarray:
    state = PipelineState(frame_index=first_frame, housing_temp_k=cfg.t_housing_cal_k)
    g = _gbuffer(cfg.sensor.sensor.fpa_shape, t)
    frames = []
    for _ in range(n):
        out = run_frame(g, cfg, state)
        assert out.dn16 is not None
        frames.append(out.dn16.astype(np.float64))
    return np.stack(frames)


MEASURED_NETD = measured_path("netd", "flir_boson_640_lwir")


@pytest.mark.skipif(not MEASURED_NETD.is_file(), reason=f"no measured NETD at {MEASURED_NETD}")
def test_against_measured_netd(boson_lut: BandLUT) -> None:
    """When a bench file exists, NETD is compared **absolutely** -- no gain fit (M12.4).

    Millikelvin is a physical unit the simulator has to predict, not a range choice: a fit here
    would let the model be wrong by any factor and still pass, which is the single thing this
    measurement exists to rule out.
    """
    cfg = _config(boson_lut)
    temps, measured_mk = load_measured_pairs(MEASURED_NETD)
    simulated_mk = np.array(
        [measured_netd_k(_stack(cfg, t, 200), _stack(cfg, t + 3.0, 200), 3.0) * 1e3 for t in temps]
    )
    result = compare_absolute(simulated_mk, measured_mk, tolerance=0.15)
    assert result.passed, result.describe()


def test_dn_domain_netd_within_10_percent_of_anchor(boson_lut: BandLUT) -> None:
    cfg = _config(boson_lut)
    netd = measured_netd_k(_stack(cfg, 300.0, 200), _stack(cfg, 303.0, 200), 3.0)
    assert abs(netd / 0.050 - 1.0) < 0.10, f"{netd * 1e3:.1f} mK"


def test_dn_domain_3d_ratios(boson_lut: BandLUT) -> None:
    """The DN stack minus the ideal (noise-free) frame: on an un-NUC'd camera the cos⁴ vignetting
    gradient would otherwise read as fixed row/column pattern (it is one, physically -- the M9
    NUC stage removes it; here the noise-free frame is subtracted so the 3-D terms are isolated)."""
    cfg = _config(boson_lut)
    ideal = _config(boson_lut)
    object.__setattr__(ideal, "noise_enabled", False)
    cube = _stack(cfg, 300.0, 200) - _stack(ideal, 300.0, 1)[0]
    sigma_tvh = float(
        cfg.detector.response(
            np.ones(cfg.sensor.sensor.fpa_shape, np.float32) * 1e-9, 0, 7
        ).sigma_dn.mean()
    )
    injected = Sigmas7.from_ratios(sigma_tvh, cfg.sensor.sensor.noise.sigma_ratios())
    d = decompose_3d(cube)
    floors = estimate_floors(cube.shape, injected.as_vector())
    # quantisation adds a uniform 1/12 LSB^2 to the TVH variance: allow it explicitly
    for name, true_sigma in zip(
        ("t", "v", "h", "tv", "th", "vh", "tvh"), injected.as_vector(), strict=True
    ):
        extra = 1.0 / 12.0 if name == "tvh" else 0.0
        assert abs(d.raw_variances[name] - true_sigma**2 - extra) <= 3 * floors[name] + 1e-3, (
            name,
            d.raw_variances[name],
            true_sigma**2,
        )


def test_replay_and_direct_frame_access(boson_lut: BandLUT) -> None:
    a = _stack(_config(boson_lut, seed=3, size=16), 300.0, 4)
    b = _stack(_config(boson_lut, seed=3, size=16), 300.0, 4)
    assert np.array_equal(a, b)
    sequential = _stack(_config(boson_lut, seed=3, size=16), 300.0, 501)[-1]
    direct = _stack(_config(boson_lut, seed=3, size=16), 300.0, 1, first_frame=500)[0]
    assert np.array_equal(sequential, direct), "memoryless components: frame 500 needs no history"


def test_sensors_are_uncorrelated_and_float16_refused(boson_lut: BandLUT) -> None:
    a = _stack(_config(boson_lut, seed=1, size=32), 300.0, 20)
    b = _stack(_config(boson_lut, seed=2, size=32), 300.0, 20)
    r = np.corrcoef((a - a.mean()).ravel(), (b - b.mean()).ravel())[0, 1]
    assert abs(r) < 0.05
    cfg = _config(boson_lut, size=16)
    g = _gbuffer((16, 16), 300.0)
    g["temperature_k"] = g["temperature_k"].astype(np.float16)
    with pytest.raises(TypeError, match="float16"):
        run_frame(g, cfg, PipelineState())
