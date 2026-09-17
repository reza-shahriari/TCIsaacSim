"""§15 Tier 3 phenomenology through the whole pipeline (M5.7): a hot exhaust entering the frame
collapses the 8-bit contrast of a pedestrian against its background under linear AGC while the
DN16 contrast is untouched; the Boson's plateau equalisation keeps most of the background
contrast. Also the first-image reproducibility check."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
H, W = 96, 128


def _config(lut: BandLUT, **isp: Any) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=W, height=H)
    d["sensor"]["optics"]["supersample_factor"] = 1
    d["sensor"]["isp"].update(isp)
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d), MaterialTable.constant(1.0), lut=lut, sensor_seed=3
    )


def _scene(exhaust: bool) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    t = (300.0 + rng.normal(0.0, 0.5, (H, W))).astype(np.float32)  # background with texture
    ped = np.zeros((H, W), bool)
    ped[40:70, 50:60] = True
    t[ped] = 302.0
    hot = np.zeros((H, W), bool)
    if exhaust:
        hot[10:30, 90:120] = True
        t[hot] = 600.0
    planes = {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones((H, W), np.float32),
        "distance_m": np.full((H, W), 50.0, np.float32),
        "material_id": np.ones((H, W), np.int32),
        "sky_view_factor": np.full((H, W), 0.5, np.float32),
    }
    return planes, ped, hot


def _halo(hot: np.ndarray, margin: int = 12) -> np.ndarray:
    """The exhaust plus a margin: the optical PSF spreads a 600 K patch's halo over a few px."""
    out = hot.copy()
    ys, xs = np.nonzero(hot)
    if ys.size:
        out[
            max(0, ys.min() - margin) : ys.max() + margin + 1,
            max(0, xs.min() - margin) : xs.max() + margin + 1,
        ] = True
    return out


def _contrast(img: np.ndarray, ped: np.ndarray, hot: np.ndarray) -> float:
    """Pedestrian mean minus background mean, the background excluding the exhaust halo."""
    bg = ~ped & ~_halo(hot)
    return float(img[ped].mean() - img[bg].mean())


def test_hot_exhaust_collapses_linear_agc_contrast_but_not_dn16(boson_lut: BandLUT) -> None:
    cfg = _config(boson_lut, agc="linear", dde_gain=0.0)
    (p0, ped, _), (p1, _, hot1) = _scene(False), _scene(True)
    o0 = run_frame(p0, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    o1 = run_frame(p1, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert (
        o0.display8 is not None
        and o1.display8 is not None
        and o0.dn16 is not None
        and o1.dn16 is not None
    )
    c8_0 = _contrast(o0.display8[..., 0].astype(np.float64), ped, hot1)
    c8_1 = _contrast(o1.display8[..., 0].astype(np.float64), ped, hot1)
    c16_0 = _contrast(o0.dn16.astype(np.float64), ped, hot1)
    c16_1 = _contrast(o1.dn16.astype(np.float64), ped, hot1)
    assert c8_0 > 5.0, "the pedestrian must be visible without the exhaust"
    assert c8_1 < 0.5 * c8_0, (c8_0, c8_1)
    assert abs(c16_1 / c16_0 - 1.0) < 0.05, "DN16 contrast is unchanged by the AGC"


def test_plateau_equalisation_retains_background_contrast(boson_lut: BandLUT) -> None:
    cfg = _config(boson_lut, agc="plateau_equalization", plateau=0.012, dde_gain=0.0)
    (p0, ped, _), (p1, _, hot1) = _scene(False), _scene(True)
    o0 = run_frame(p0, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    o1 = run_frame(p1, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert o0.display8 is not None and o1.display8 is not None
    bg = ~ped & ~_halo(hot1)
    std0 = o0.display8[..., 0][bg].astype(np.float64).std()
    std1 = o1.display8[..., 0][bg].astype(np.float64).std()
    assert std1 > 0.5 * std0, (std0, std1)


def test_frame_reproducibility_including_display(boson_lut: BandLUT) -> None:
    cfg = _config(boson_lut)
    planes, _, _ = _scene(True)
    a = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    b = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert a.display8 is not None and b.display8 is not None
    assert np.array_equal(a.display8, b.display8) and a.isp_hash == b.isp_hash
