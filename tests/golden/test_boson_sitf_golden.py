"""Golden SITF of the committed Boson configuration (M4.8): noiseless floor DN for ten extended
blackbodies through the calibrated bolometer transfer, keyed on the config hash, 1 DN tolerance.
A regenerated response file or a band-block change reports STALE; a change in the transfer, the
LUT or the optics with the same inputs reports FAILING."""

from __future__ import annotations

import pathlib

import numpy as np
from golden_store import GoldenStore

from irsim.config.loader import config_hash, load_sensor_config
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    MicrobolometerDetector,
    anchor_noise,
    fpa_params_from_config,
)
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
TEMPS = np.linspace(253.0, 453.0, 10)


def _flux(sensor, lut: BandLUT, t: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    phi = float(
        pixel_power(
            lut.lookup(t),
            sensor.optics.f_number,
            sensor.optics.transmittance,
            sensor.detector_active_area_m2,
        )
    )
    return np.full((1, 1), phi, dtype=np.float32)


def test_boson_sitf_golden(golden: GoldenStore, boson_lut: BandLUT) -> None:
    cfg = load_sensor_config(BOSON_YAML)
    s = cfg.sensor
    p = fpa_params_from_config(cfg)
    assert isinstance(p, BolometerParams)
    lo, hi = float(_flux(s, boson_lut, 233.15)[0, 0]), float(_flux(s, boson_lut, 473.15)[0, 0])
    det = MicrobolometerDetector(
        p, BolometerTransfer.from_power_range(lo, hi, 16), anchor_noise(s, boson_lut)
    )
    noiseless = np.array(
        [float(det.noiseless_signal_dn(_flux(s, boson_lut, float(t)))[0, 0]) for t in TEMPS]
    )
    assert np.all(np.diff(noiseless) > 0)
    golden.check(
        "boson_sitf_dn",
        np.floor(noiseless).astype(np.uint16),
        config_hash=config_hash(cfg),
        atol=1.0,
        units="DN (noiseless floor DN at 253-453 K, 10 steps)",
    )
    noisy = np.array(
        [int(det.response(_flux(s, boson_lut, float(t)), 0, 0).dn[0, 0]) for t in TEMPS]
    )
    assert (
        np.abs(noisy.astype(np.int64) - np.floor(noiseless).astype(np.int64)).max()
        < 5 * det.sigma_signal_dn + 1
    )
