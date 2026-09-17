"""The generated LWIR Boson LUT (M1.10, ADR 0013): the first configured band.

The response is an estimate, so these are plausibility bounds rather than known answers: the
effective wavelength must sit inside the band, Lb(300 K) must be within 10 % of the top-hat
closed form, and the 300 -> 373 K derivative ratio must bracket the [R9] NETD anchor.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.radiometry.constants import C_LIGHT, H_PLANCK
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.lut_files import build_band_lut_for_config, load_band_lut_for_config
from irsim.radiometry.planck import band_radiance_tophat

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


def test_make_luts_output_loads_for_the_repo_config(tmp_path: pathlib.Path) -> None:
    """What `make luts` writes for the committed YAML + CSV loads back through the config."""
    cfg = load_sensor_config(BOSON_YAML)
    lut, paths = build_band_lut_for_config(cfg, tmp_path)
    back = load_band_lut_for_config(cfg, tmp_path)
    assert np.array_equal(back.lb, lut.lb) and back.n == 16001
    assert paths.sidecar.name.endswith("_lut.json")


def test_effective_wavelength_at_300k_inside_the_band(boson_lut: BandLUT) -> None:
    """λ_eff = hc / (Lb / Lb_q): energy per photon averaged over the band, 9.5-11.5 µm for LWIR."""
    lb = float(boson_lut.lookup(300.0)[()])
    lb_q = float(boson_lut.lookup(300.0, "lb_q")[()])
    lam_eff_um = H_PLANCK * C_LIGHT / (lb / lb_q) * 1e6
    assert 9.5 < lam_eff_um < 11.5, lam_eff_um


def test_lb_300k_within_10_percent_of_tophat(boson_lut: BandLUT) -> None:
    lb = float(boson_lut.lookup(300.0)[()])
    tophat = band_radiance_tophat(7.5, 13.5, 300.0)
    assert tophat == pytest.approx(55.49, rel=1e-3)
    assert abs(lb / tophat - 1.0) < 0.10, f"{lb:.3f} vs {tophat:.3f}"


def test_derivative_ratio_brackets_the_netd_anchor(boson_lut: BandLUT) -> None:
    ratio = float(boson_lut.lookup(373.0, "dlb_dt")[()] / boson_lut.lookup(300.0, "dlb_dt")[()])
    assert 1.65 < ratio < 1.85, ratio
