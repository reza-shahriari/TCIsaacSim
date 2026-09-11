"""End-to-end golden frame (M5.7): the four outputs of run_frame on gbuffer_ramp and on a
hot-patch scene with the Boson configuration (noise on, sensor seed fixed), stored with the
config hash + NumPy version. Tolerances: radiance 1e-5 relative, apparent T 1 mK, DN16 +/-1,
display +/-1 code. Also writes outputs/first_image_*.png for the human look."""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
from golden_store import GoldenStore

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.io import write_png
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.encoding import encode_temperature
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
OUT_DIR = REPO / "outputs"


def _config(boson_lut: BandLUT) -> tuple[PipelineConfig, str]:
    cfg = load_sensor_config(BOSON_YAML)
    d = cfg.model_dump(mode="json")
    d["sensor"]["fpa"].update(width=256, height=256)
    d["sensor"]["optics"]["supersample_factor"] = 1
    small = SensorConfig.model_validate(d)
    key = hashlib.sha256(
        (config_hash(cfg) + "|256x256|numpy=" + np.__version__).encode()
    ).hexdigest()
    return PipelineConfig.from_sensor(
        small, MaterialTable.constant(1.0), lut=boson_lut, sensor_seed=77
    ), key


def _hot_patch_scene() -> dict[str, np.ndarray]:
    t = np.full((256, 256), 295.0, np.float32)
    t[100:160, 60:120] = 305.0
    t[30:60, 190:240] = 600.0
    return {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones((256, 256), np.float32),
        "distance_m": np.full((256, 256), 50.0, np.float32),
        "material_id": np.ones((256, 256), np.int32),
        "sky_view_factor": np.full((256, 256), 0.5, np.float32),
    }


def _check_outputs(
    golden: GoldenStore, name: str, planes: dict[str, np.ndarray], boson_lut: BandLUT
) -> None:
    cfg, key = _config(boson_lut)
    out = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert (
        out.radiance is not None
        and out.apparent_t is not None
        and out.dn16 is not None
        and out.display8 is not None
    )
    golden.check(f"{name}_radiance", out.radiance, config_hash=key, rtol=1e-5, units="W/m2/sr")
    golden.check(f"{name}_apparent_t", out.apparent_t, config_hash=key, atol=1e-3, units="K (1 mK)")
    golden.check(f"{name}_dn16", out.dn16, config_hash=key, atol=1.0, units="DN16")
    golden.check(
        f"{name}_display8", out.display8[..., 0], config_hash=key, atol=1.0, units="DN8 (R channel)"
    )
    OUT_DIR.mkdir(exist_ok=True)
    write_png(OUT_DIR / f"first_image_{name}.png", out.display8)


def test_golden_ramp(
    golden: GoldenStore, gbuffer_ramp: dict[str, np.ndarray], boson_lut: BandLUT
) -> None:
    _check_outputs(golden, "boson_ramp", gbuffer_ramp, boson_lut)


def test_golden_hot_patch(golden: GoldenStore, boson_lut: BandLUT) -> None:
    _check_outputs(golden, "boson_hot_patch", _hot_patch_scene(), boson_lut)
