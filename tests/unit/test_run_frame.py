"""run_frame (M3.11): output dtypes and omit-not-zero flags, DN monotone over blackbodies,
apparent T equal to kinetic T for ε = 1, and the supersampled G-buffer contract."""

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

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _config(
    lut: BandLUT, width: int = 32, height: int = 16, supersample: int = 1, **outputs: Any
) -> PipelineConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"]["supersample_factor"] = supersample
    d["sensor"]["outputs"].update(outputs)
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(d), MaterialTable.constant(1.0), lut=lut, noise_enabled=False
    )


def _uniform_gbuffer(shape: tuple[int, int], t: float) -> dict[str, np.ndarray]:
    temp = np.full(shape, t, dtype=np.float32)
    return {
        "temperature_k": temp,
        "encoded_t": encode_temperature(temp),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.full(shape, 0.5, np.float32),
    }


def test_output_dtypes_and_state_advance(tophat_lwir_lut: BandLUT) -> None:
    cfg = _config(tophat_lwir_lut)
    state = PipelineState(housing_temp_k=300.0)
    out = run_frame(_uniform_gbuffer((16, 32), 300.0), cfg, state)
    assert (
        out.radiance is not None
        and out.radiance.dtype == np.float32
        and out.radiance.shape == (16, 32)
    )
    assert out.apparent_t is not None and out.apparent_t.dtype == np.float32
    assert out.dn16 is not None and out.dn16.dtype == np.uint16
    assert out.display8 is not None and out.display8.dtype == np.uint8
    assert out.display8.shape == (16, 32, 4) and out.isp_hash is not None
    assert out.signal_dn.dtype == np.float32 and out.flux.dtype == np.float32
    assert state.frame_index == 1


def test_flags_omit_outputs_never_zeros(tophat_lwir_lut: BandLUT) -> None:
    cfg = _config(
        tophat_lwir_lut,
        radiance_linear=False,
        apparent_temperature=False,
        dn_16=False,
        display_8=False,
    )
    out = run_frame(_uniform_gbuffer((16, 32), 300.0), cfg, PipelineState())
    assert (
        out.radiance is None
        and out.apparent_t is None
        and out.dn16 is None
        and out.display8 is None
    )
    only_t = _config(tophat_lwir_lut, radiance_linear=False)
    out2 = run_frame(_uniform_gbuffer((16, 32), 300.0), only_t, PipelineState())
    assert out2.radiance is None and out2.apparent_t is not None


def test_dn_strictly_increasing_over_blackbodies(tophat_lwir_lut: BandLUT) -> None:
    cfg = _config(tophat_lwir_lut)
    dns = []
    for t in np.linspace(250.0, 450.0, 11):
        out = run_frame(_uniform_gbuffer((16, 32), float(t)), cfg, PipelineState())
        assert out.dn16 is not None
        dns.append(int(out.dn16[8, 16]))
    assert all(b > a for a, b in zip(dns[:-1], dns[1:], strict=True)), dns
    assert dns[0] > 0 and dns[-1] < 65535


@pytest.mark.parametrize("t", [250.0, 300.0, 373.0, 450.0])
def test_apparent_t_equals_kinetic_for_blackbody(tophat_lwir_lut: BandLUT, t: float) -> None:
    """ε = 1, no atmosphere, housing at its calibration temperature: |T_app − T| < 1 mK."""
    cfg = _config(tophat_lwir_lut)
    out = run_frame(
        _uniform_gbuffer((16, 32), t), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
    )
    assert out.apparent_t is not None
    err_mk = np.max(np.abs(out.apparent_t.astype(np.float64) - t)) * 1e3
    assert err_mk < 1.0, f"{err_mk:.3f} mK"


def test_supersampled_gbuffer_required_and_step_edge_box_filtered(tophat_lwir_lut: BandLUT) -> None:
    cfg = _config(tophat_lwir_lut, width=8, height=4, supersample=4)
    with pytest.raises(ValueError, match="supersample"):
        run_frame(_uniform_gbuffer((4, 8), 300.0), cfg, PipelineState())
    g = _uniform_gbuffer((16, 32), 300.0)
    g["temperature_k"][:, 18:] = 320.0  # edge between native pixels 4 and 5, 2 sub-samples into 4
    g["encoded_t"] = encode_temperature(g["temperature_k"])
    out = run_frame(g, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert out.radiance is not None
    lb300, lb320 = (
        float(tophat_lwir_lut.lookup(300.0)[()]),
        float(tophat_lwir_lut.lookup(320.0)[()]),
    )
    assert out.radiance[0, 3] == pytest.approx(lb300, rel=1e-5)
    assert out.radiance[0, 4] == pytest.approx(0.5 * (lb300 + lb320), rel=1e-5), (
        "box-filtered radiance"
    )
    assert out.radiance[0, 5] == pytest.approx(lb320, rel=1e-5)


def test_photon_fpa_runs_the_same_chain(boson_lut: BandLUT) -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 32,
        "height": 16,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.7,
        "well_capacity_e": 1.0e7,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
    }
    d["sensor"]["optics"]["supersample_factor"] = 1
    # the LWIR LUT stands in for an MWIR one here: only the chain's plumbing is under test
    cfg = PipelineConfig.from_sensor(
        SensorConfig.model_validate(d),
        MaterialTable.constant(1.0),
        lut=boson_lut,
        noise_enabled=False,
    )
    assert cfg.quantity == "lb_q" and cfg.calibration is None
    out = run_frame(_uniform_gbuffer((16, 32), 300.0), cfg, PipelineState())
    assert out.dn16 is not None and out.dn16.dtype == np.uint16 and out.apparent_t is not None
    assert abs(float(out.apparent_t[8, 16]) - 300.0) < 0.01


def test_noise_on_by_default_and_reproducible(tophat_lwir_lut: BandLUT) -> None:
    """The default chain is noisy (M4.9): two frames differ, the same seed replays exactly, and
    the ideal chain is the noise-free limit."""
    import pathlib

    import yaml

    from irsim.config.sensor import SensorConfig

    repo = pathlib.Path(__file__).resolve().parents[2]
    d = yaml.safe_load((repo / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    d["sensor"]["fpa"].update(width=32, height=16)
    d["sensor"]["optics"]["supersample_factor"] = 1
    sensor = SensorConfig.model_validate(d)
    noisy = PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=tophat_lwir_lut, sensor_seed=5
    )
    assert noisy.noise_enabled
    g = _uniform_gbuffer((16, 32), 300.0)
    a = run_frame(g, noisy, PipelineState(housing_temp_k=noisy.t_housing_cal_k))
    b = run_frame(g, noisy, PipelineState(housing_temp_k=noisy.t_housing_cal_k))
    assert a.dn16 is not None and b.dn16 is not None and np.array_equal(a.dn16, b.dn16)
    state = PipelineState(housing_temp_k=noisy.t_housing_cal_k)
    f0, f1 = run_frame(g, noisy, state), run_frame(g, noisy, state)
    assert f0.dn16 is not None and f1.dn16 is not None and not np.array_equal(f0.dn16, f1.dn16)
    ideal = _config(tophat_lwir_lut)
    i0 = run_frame(g, ideal, PipelineState(housing_temp_k=ideal.t_housing_cal_k)).signal_dn
    i1 = run_frame(g, ideal, PipelineState(housing_temp_k=ideal.t_housing_cal_k)).signal_dn
    assert np.array_equal(i0, i1), (
        "the ideal chain is deterministic (only the cos^4 gradient varies)"
    )
    assert (a.signal_dn - i0).std() > 1.0, "the noisy chain adds noise on top of the ideal frame"
