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


#: Where the previous frame's scene time is kept, so a time-lapse camera can be told the truth
#: about its own interval (ADR 0074 renders one frame per six seconds of scene time).
LAST_T_KEY = "bolometer_iir_t_s"


def lag_interval_s(config: PipelineConfig, state: PipelineState) -> float:
    """Seconds since the previous frame: elapsed scene time when it is known, else 1/frame_rate.

    **A policy, not a property of the membrane.** It is applied by `run_frame`, which owns the
    caller's clock, and *not* by :func:`bolometer_lag`, whose default stays the configured frame
    interval. That is deliberate: `detector_stage` is the CPU oracle the Warp twin is compared
    against (`EQUIVALENCE_STAGES["detector"]`), the twin takes `alpha_for(fpa.frame_dt_s, ...)`
    at kernel-launch time, and quietly making the shared primitive depend on `state.t_s` would
    have made the two paths disagree for any caller that advanced its clock.

    **The detector's frame rate is the wrong answer for a time-lapse.** A 10 ms membrane settles
    completely across a six-second capture interval, and driving it at 1/60 s instead leaves 19 %
    of the *previous capture* in the picture -- a ghost of a scene six seconds old. The fallback
    is for callers that never advance ``t_s`` at all, which is most unit tests and every
    single-frame use, and for them the two answers coincide anyway.
    """
    dt = float(config.fpa.frame_dt_s)
    previous = state.buffers.get(LAST_T_KEY)
    if previous is not None:
        elapsed = float(state.t_s) - float(previous)
        if elapsed > 0.0:
            dt = elapsed
    state.buffers[LAST_T_KEY] = float(state.t_s)
    return dt


def bolometer_lag(
    signal_dn: NDArray[np.floating],
    config: PipelineConfig,
    state: PipelineState,
    dt_s: float | None = None,
) -> NDArray[np.float32]:
    """Advance the membrane IIR one frame, keeping the state in ``state.buffers`` (§9.2).

    The first frame adopts its input: a core that has been staring at the scene is already in
    thermal equilibrium with it, and starting from zero would put a frame-long ramp at the head of
    every sequence and every exported dataset. Drop the buffer to model a genuine cold start.

    ``dt_s`` defaults to the configured frame interval, which is what the Warp twin is launched
    with; `run_frame` passes :func:`lag_interval_s` instead so a time-lapse is told the truth.
    """
    fpa = config.fpa
    if not isinstance(fpa, BolometerParams):
        raise TypeError("the membrane lag is a bolometer property; photon FPAs are memoryless")
    lag = BolometerLowPass(
        tau_s=fpa.thermal_time_constant_s, state=state.buffers.get(IIR_STATE_KEY)
    )
    out = lag.step(signal_dn, float(fpa.frame_dt_s) if dt_s is None else float(dt_s))
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
