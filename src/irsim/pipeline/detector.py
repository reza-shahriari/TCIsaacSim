"""Stage 4 on the plane dict: pixel power → the ideal detector signal, through the membrane lag.

    flux (W or photons/s)  →  S_ideal (DN units, float32)  →  [bolometer only] the τ_th IIR

The composition is the one ADR 0052 fixes: the membrane integrates *incident flux*, so the lag is
applied to the ideal signal and noise (stage 5, M9.8's wiring) is injected after it. Filtering the
noisy output instead would correlate σ_TVH across frames and quietly shrink the per-frame temporal
variance by α/(2 − α) — a factor of 0.68 at 60 Hz and τ_th = 10 ms — so the NETD anchor would stop
delivering the datasheet figure with nothing visible in any single frame.

Cooled photon detectors are memoryless on these timescales and take no filter at all, which is the
§15 Tier 3 phenomenology check that LWIR smears and cooled MWIR does not.

The per-pixel state is float32 (non-negotiable #2 — it carries signal) and lives in
``PipelineState.buffers`` under :data:`IIR_STATE_KEY`, not inside the detector, so detector
instances stay immutable and every cross-frame buffer has one owner and one reset path. The
arithmetic is `irsim.detector.lowpass.BolometerLowPass`'s, driven here rather than reimplemented.

M9.8 wires this into ``run_frame`` together with the rest of the M9 chain (FPA node, housing
drift, defects, NUC residual, FFC); this module is the stage itself, and the oracle the Warp twin
(M10.6) is compared against.

docs/physics-model.md §9.1, §9.2, §13.4 stage 4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.detector.lowpass import BolometerLowPass
from irsim.detector.params import BolometerParams
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = ["IIR_STATE_KEY", "bolometer_lag", "detector_stage", "DetectorStage"]

#: Where the bolometer's per-pixel IIR state lives in ``PipelineState.buffers`` (ADR 0052).
IIR_STATE_KEY = "bolometer_iir"


def bolometer_lag(
    signal_dn: NDArray[np.floating], config: PipelineConfig, state: PipelineState
) -> NDArray[np.float32]:
    """Advance the membrane IIR one frame, keeping the state in ``state.buffers`` (§9.2).

    The first frame adopts its input: a core that has been staring at the scene is already in
    thermal equilibrium with it, and starting from zero would put a frame-long ramp at the head of
    every sequence and every exported dataset. Drop the buffer to model a genuine cold start.
    """
    fpa = config.fpa
    if not isinstance(fpa, BolometerParams):
        raise TypeError("the membrane lag is a bolometer property; photon FPAs are memoryless")
    lag = BolometerLowPass(
        tau_s=fpa.thermal_time_constant_s, state=state.buffers.get(IIR_STATE_KEY)
    )
    out = lag.step(signal_dn, fpa.frame_dt_s)
    assert lag.state is not None  # step() always leaves one
    state.buffers[IIR_STATE_KEY] = lag.state
    return out


def detector_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-4 entry point: ``flux`` → ``signal_dn``, ideal (noiseless) and thermally lagged."""
    signal = config.detector.noiseless_signal_dn(np.asarray(planes["flux"]))
    if isinstance(config.fpa, BolometerParams):
        signal = bolometer_lag(signal, config, state)
    return {"signal_dn": signal}


class DetectorStage:
    name = "detector"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return detector_stage(planes, config, state)
