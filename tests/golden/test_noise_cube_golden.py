"""Golden noise cube (M4.10): 8 frames of 64x64 pre-quantisation signal from the Boson detector
+ NoiseStage on a uniform 300 K flux, keyed on the config hash *and* the NumPy version (the
Box-Muller log/cos can move by an ulp between libms, so a NumPy change is reported STALE rather
than FAILING). Compared at 1e-4 DN, about 1e-6 of the noise sigma."""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
from golden_store import GoldenStore

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    MicrobolometerDetector,
    anchor_noise,
    fpa_params_from_config,
)
from irsim.noise import NoiseStage, measure_from_uniform_scene
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT
from irsim.validation import compare_psd, spatial_psd

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


def test_noise_cube_golden(golden: GoldenStore, boson_lut: BandLUT) -> None:
    cfg = load_sensor_config(BOSON_YAML)
    d = cfg.model_dump(mode="json")
    d["sensor"]["fpa"].update(width=64, height=64)
    small = SensorConfig.model_validate(d).sensor
    p = fpa_params_from_config(SensorConfig(sensor=small))
    assert isinstance(p, BolometerParams)

    def flux(t: float) -> np.ndarray:
        return np.full(
            small.fpa_shape,
            float(pixel_power(boson_lut.lookup(t), 1.0, 0.92, small.detector_active_area_m2)),
            np.float32,
        )

    det = MicrobolometerDetector(
        p,
        BolometerTransfer.from_power_range(
            float(flux(233.15)[0, 0]), float(flux(473.15)[0, 0]), 16
        ),
        anchor_noise(small, boson_lut),
    )
    stage = NoiseStage.from_sensor(small, sensor_seed=1234)
    cube, sigma_tvh = measure_from_uniform_scene(det, stage, flux(300.0), 8)
    key = hashlib.sha256((config_hash(cfg) + "|numpy=" + np.__version__).encode()).hexdigest()
    golden.check(
        "boson_noise_cube_8x64x64",
        cube - cube.mean(),
        config_hash=key,
        atol=1e-4,
        units="DN (signal, mean removed)",
    )
    # the cube's spatial PSD is not white: striping puts more than a few percent on the axis lines
    p_cube = spatial_psd(cube)
    assert p_cube.fraction_on_kv0() > 0.02 or p_cube.fraction_on_kh0() > 0.02
    assert compare_psd(p_cube, p_cube) == 1.0
    assert sigma_tvh > 0.0
