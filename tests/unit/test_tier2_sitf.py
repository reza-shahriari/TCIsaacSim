"""Tier 2 SITF bench on the CPU reference (M3.12; docs/physics-model.md §15 Tier 2).

Ten extended blackbodies 253-453 K through the Boson configuration: DN strictly increasing and
smooth, linear in L_B(T) to well under an LSB, blackbody apparent temperature within 10 mK, and
the grey-body gap equal to the LUT prediction. Self-consistency only until a measured SITF CSV
exists under data/validation/ (ADR 0003); that comparison is skipped when the file is absent.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import DEFAULT_DATA_DIR, load_sensor_config
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig
from irsim.radiometry.lut import BandLUT
from irsim.validation import sitf

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
MEASURED_SITF = DEFAULT_DATA_DIR / "validation" / "sitf" / "flir_boson_640_lwir.csv"
TEMPS = np.linspace(253.0, 453.0, 10)


@pytest.fixture(scope="module")
def boson_config(boson_lut: BandLUT) -> PipelineConfig:
    sensor = load_sensor_config(BOSON_YAML)
    # native-resolution bench: the supersample factor only adds cost for a uniform field
    from irsim.config.sensor import SensorConfig

    d = sensor.model_dump(mode="json")
    d["sensor"]["optics"]["supersample_factor"] = 1
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.from_mapping({1: 1.0, 2: 0.9}),
        lut=boson_lut,
        noise_enabled=False,  # SITF characterises the ideal transfer; NETD is its own bench
    )


@pytest.fixture(scope="module")
def bench(boson_config: PipelineConfig):  # type: ignore[no-untyped-def]
    return sitf(boson_config, TEMPS)


def test_dn_strictly_increasing_and_smooth(bench) -> None:  # type: ignore[no-untyped-def]
    dn = bench.dn_mean
    assert np.all(np.diff(dn) > 0), dn
    # smooth: DN sampled at uniform L_B spacing has a vanishing second difference (pure
    # quantisation); the bench temperatures are uniform in T, so resample onto uniform L_B first
    lb_uniform = np.linspace(bench.band_radiance[0], bench.band_radiance[-1], 10)
    dn_uniform = np.interp(lb_uniform, bench.band_radiance, dn)
    rel_second = np.abs(np.diff(dn_uniform, 2)) / dn_uniform[1:-1]
    assert rel_second.max() < 1e-3, rel_second
    # noise-free: the radiometric branch divides cos⁴ out, so its spread is float32 rounding only;
    # the DN spread inside the 10 % centre ROI is the residual cos⁴ fall-off (< 0.3 %)
    assert np.all(bench.apparent_t_std * 1e3 < 1.0), bench.apparent_t_std
    assert np.all(bench.dn_std / bench.dn_mean < 3e-3), bench.dn_std / bench.dn_mean


def test_dn_linear_in_band_radiance_under_half_lsb(bench) -> None:  # type: ignore[no-untyped-def]
    a, b, rms = bench.linear_fit_residual_lsb()
    assert rms < 0.5, f"residual {rms:.3f} LSB rms (floor quantiser: 0.29 expected)"
    assert a > 0
    # and the un-quantised signal is linear to float32 precision
    a2, b2 = np.polyfit(bench.band_radiance, bench.signal_mean, 1)
    resid = bench.signal_mean - (a2 * bench.band_radiance + b2)
    assert np.max(np.abs(resid)) < 1e-2


def test_blackbody_apparent_temperature_under_10mK(bench) -> None:  # type: ignore[no-untyped-def]
    err_mk = np.abs(bench.apparent_t_mean - bench.temperatures_k) * 1e3
    assert err_mk.max() < 10.0, err_mk


def test_grey_body_gap_matches_lut_prediction(boson_config: PipelineConfig) -> None:
    grey = sitf(boson_config, [300.0], material_id=2)
    lut = boson_config.lut
    predicted = float(lut.apparent_temperature(np.float32(0.9 * float(lut.lookup(300.0)[()])))[()])
    assert abs(grey.apparent_t_mean[0] - predicted) * 1e3 < 20.0
    assert 300.0 - grey.apparent_t_mean[0] > 5.0, "a 0.9 grey body must read several kelvin low"


@pytest.mark.skipif(not MEASURED_SITF.is_file(), reason=f"no measured SITF at {MEASURED_SITF}")
def test_against_measured_sitf(bench) -> None:  # type: ignore[no-untyped-def]
    """When a measured (T_bb, DN) table exists, the simulated SITF shape must match it after a
    linear gain/offset fit (absolute DN is a range choice, ADR 0019)."""
    data = np.loadtxt(MEASURED_SITF, delimiter=",", comments="#")
    t_meas, dn_meas = data[:, 0], data[:, 1]
    sim = np.interp(t_meas, bench.temperatures_k, bench.dn_mean)
    a, b = np.polyfit(sim, dn_meas, 1)
    resid = dn_meas - (a * sim + b)
    assert np.sqrt(np.mean(resid**2)) < 0.01 * np.ptp(dn_meas)
