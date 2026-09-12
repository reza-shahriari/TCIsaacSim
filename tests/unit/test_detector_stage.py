"""Stage 4 on the plane dict (M10.6's oracle): where the membrane lag sits and who owns its state.

`BolometerLowPass` itself is tested by M9.1; what is new here is the composition ADR 0052 fixes --
the lag acts on the *ideal* signal, the photon path has none at all, and the per-pixel state lives
in `PipelineState.buffers` so it has one owner and one reset path.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.detector.lowpass import alpha_for
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState
from irsim.pipeline.detector import IIR_STATE_KEY, bolometer_lag, detector_stage
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs/sensors/flir_boson_640_lwir.yaml").read_text())


def _config(lut: BandLUT, photon: bool = False) -> PipelineConfig:
    raw = copy.deepcopy(BOSON)
    raw["sensor"]["fpa"].update(width=8, height=8)
    if photon:
        raw["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
        raw["sensor"]["fpa"] = {
            "type": "photon",
            "width": 8,
            "height": 8,
            "pitch_um": 15.0,
            "fill_factor": 1.0,
            "frame_rate_hz": 30,
            "bit_depth": 14,
            "quantum_efficiency": 0.8,
            "well_capacity_e": 1.0e6,
            "integration_time_ms": 5.0,
            "dark_current_model": "arrhenius",
        }
    return PipelineConfig.from_sensor(
        SensorConfig.model_validate(raw), MaterialTable.from_mapping({1: 0.95}), lut=lut
    )


def _flux(config: PipelineConfig, scale: float) -> np.ndarray:
    """A uniform pixel power at ``scale`` of the way up the transfer's calibrated span."""
    transfer = config.detector.transfer
    span = config.fpa.dn_max / transfer.gain_dn_per_w
    return np.full(config.fpa.shape, transfer.offset_w + scale * span, np.float32)


def test_the_first_frame_adopts_its_input_rather_than_ramping_from_zero(
    tophat_lwir_lut: BandLUT,
) -> None:
    """A core that has been staring at the scene is in equilibrium with it. Starting at zero
    would put a frame-long ramp at the head of every sequence and every exported dataset."""
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    phi = _flux(config, 0.4)
    ideal = config.detector.noiseless_signal_dn(phi)
    out = detector_stage({"flux": phi}, config, state)["signal_dn"]
    assert np.allclose(out, ideal, rtol=0, atol=0)
    assert out.dtype == np.float32


def test_a_step_moves_by_alpha_on_the_frame_after_the_first(tophat_lwir_lut: BandLUT) -> None:
    config = _config(tophat_lwir_lut)
    alpha = alpha_for(config.fpa.frame_dt_s, config.fpa.thermal_time_constant_s)
    state = PipelineState()
    cold, hot = _flux(config, 0.2), _flux(config, 0.8)
    settled = detector_stage({"flux": cold}, config, state)["signal_dn"]
    first = detector_stage({"flux": hot}, config, state)["signal_dn"]
    target = config.detector.noiseless_signal_dn(hot)
    fraction = float(np.mean((first - settled) / (target - settled)))
    assert fraction == pytest.approx(alpha, rel=1e-5)


def test_the_state_lives_in_pipeline_buffers_as_float32(tophat_lwir_lut: BandLUT) -> None:
    """ADR 0052: one owner, one reset path, and float32 because it carries signal (#2)."""
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    detector_stage({"flux": _flux(config, 0.3)}, config, state)
    buffer = state.buffers[IIR_STATE_KEY]
    assert isinstance(buffer, np.ndarray)
    assert buffer.dtype == np.float32 and buffer.shape == config.fpa.shape


def test_dropping_the_buffer_is_a_genuine_cold_start(tophat_lwir_lut: BandLUT) -> None:
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    cold, hot = _flux(config, 0.2), _flux(config, 0.8)
    detector_stage({"flux": cold}, config, state)
    del state.buffers[IIR_STATE_KEY]
    out = detector_stage({"flux": hot}, config, state)["signal_dn"]
    assert np.allclose(out, config.detector.noiseless_signal_dn(hot)), "must re-adopt, not ramp"


def test_the_photon_path_is_memoryless_and_keeps_no_state(tophat_mwir_lut: BandLUT) -> None:
    """Cooled photon detectors do not smear on these timescales -- the §15 Tier 3 contrast with
    LWIR. A filter here would invent a lag the device does not have."""
    config = _config(tophat_mwir_lut, photon=True)
    state = PipelineState()
    low = np.full(config.fpa.shape, 1.0e8, np.float32)
    high = np.full(config.fpa.shape, 5.0e8, np.float32)
    detector_stage({"flux": low}, config, state)
    out = detector_stage({"flux": high}, config, state)["signal_dn"]
    assert np.allclose(out, config.detector.noiseless_signal_dn(high))
    assert IIR_STATE_KEY not in state.buffers


def test_bolometer_lag_refuses_a_photon_fpa(tophat_mwir_lut: BandLUT) -> None:
    config = _config(tophat_mwir_lut, photon=True)
    with pytest.raises(TypeError, match="memoryless"):
        bolometer_lag(np.ones(config.fpa.shape, np.float32), config, PipelineState())


def test_a_shape_change_mid_sequence_raises_rather_than_silently_resetting(
    tophat_lwir_lut: BandLUT,
) -> None:
    """A silent reset would re-adopt, so the lag would vanish for that frame and the sequence
    would look ideal exactly where the format changed."""
    config = _config(tophat_lwir_lut)
    state = PipelineState()
    detector_stage({"flux": _flux(config, 0.3)}, config, state)
    wrong: dict[str, Any] = {"flux": np.full((4, 4), float(_flux(config, 0.3)[0, 0]), np.float32)}
    with pytest.raises(ValueError, match="does not match the filter state"):
        detector_stage(wrong, config, state)
