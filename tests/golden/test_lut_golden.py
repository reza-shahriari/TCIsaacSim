"""Golden slice of the LWIR Boson LUT (M1.11): the committed regression for the radiometry core.

Every 100th entry of Lb and dLb/dT (161 values each, 200-1000 K at 5 K) is stored with the band
hash of the committed Boson config + response. If the physics, the quadrature grid or the
response file changes, this says so: STALE when the inputs changed (band hash), FAILING when the
numbers moved with the same inputs. Tolerance is stated in millikelvin via rtol (see below).
"""

from __future__ import annotations

import pathlib

import numpy as np
from golden_store import GoldenStore

from irsim.config.loader import band_hash, load_sensor_config
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
STRIDE = 100

# 1 mK in radiance is dLb/dT * 1e-3; relative to Lb that is 2.0e-6 at 1000 K (its loosest) and
# 2.5e-5 at 200 K. rtol = 2e-6 therefore means "within 1 mK everywhere, and tighter when cold".
RTOL_LB_1MK = 2e-6
# dLb/dT: 1e-5 relative, ~50x float32 rounding, tolerant of last-ulp libm differences (ADR 0012).
RTOL_DLB = 1e-5


def test_lut_slice_matches_golden(golden: GoldenStore, boson_lut: BandLUT) -> None:
    cfg = load_sensor_config(BOSON_YAML)
    key = band_hash(cfg)
    assert boson_lut.n == 16001 and (boson_lut.n - 1) % STRIDE == 0
    golden.check(
        "boson_lwir_lb_slice",
        boson_lut.lb[::STRIDE],
        config_hash=key,
        rtol=RTOL_LB_1MK,
        units="W/m2/sr (rtol 2e-6 = 1 mK at 1000 K)",
    )
    golden.check(
        "boson_lwir_dlb_dt_slice",
        boson_lut.dlb_dt[::STRIDE],
        config_hash=key,
        rtol=RTOL_DLB,
        units="W/m2/sr/K",
    )


def test_slice_covers_the_grid(boson_lut: BandLUT) -> None:
    temps = boson_lut.temperatures_k[::STRIDE]
    assert temps[0] == 200.0 and temps[-1] == 1000.0 and len(temps) == 161
    assert np.all(np.diff(temps) == 5.0)
