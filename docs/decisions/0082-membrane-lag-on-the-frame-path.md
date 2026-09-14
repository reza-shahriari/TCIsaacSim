# ADR 0082 — The bolometer membrane lag, actually on the frame path

**Status:** Accepted
**Date:** 2026-09-14

## Context

M9.1 built `BolometerLowPass` — §9.2's per-pixel membrane IIR, the thing that gives an uncooled
LWIR camera its characteristic trail behind a moving target — and left the wiring to M9.8. ADR 0052
settled where its state lives and that it must run *before* noise. `irsim.pipeline.detector`
drives it. M9.8's roadmap title is **"wire IIR, FPA node, housing, drift, defects, NUC residual,
FFC into run_frame"**, and `irsim.pipeline.sensor_chain`'s own module docstring lists the division
of labour as *"stages 3–5 happen in `run_frame`: optics, **detector + membrane IIR**, 3-D noise,
ADC"*.

None of that was true. `run_frame`'s stage 4 is `_detector_signal`, which reaches the detector
through

    Detector.response(flux, frame_index, sensor_seed) -> DetectorFrame
    Detector.noiseless_signal_dn(flux) -> signal

— a protocol with **no `dt` and no state**, so it could not have carried a lag whatever it
intended. `irsim.detector.bolometer` says as much in its own header: *"Dynamic behaviour (the τ_th
IIR across frames, §9.2) is M9.1; this module is the static transfer."* The only caller of
`detector_stage`, and so the only thing that ever ran the IIR, was the Warp equivalence harness and
its unit tests.

So every frame *sequence* this simulator has produced — the quadrotor flight, the aircraft pass,
the maritime demo, every golden — came from a bolometer with a thermal time constant of **zero**.
A moving target left no trail. §15 Tier 3's "lateral motion smears LWIR, not cooled MWIR" was true
only in the within-frame half ADR 0077 supplied; the across-frame half, which is the half a viewer
actually recognises, was missing.

This is ADR 0077's failure mode a second time — a mechanism built, unit-tested and never called —
with one difference that made it harder to see: **the documentation asserted it was wired.** A
grep for `BolometerLowPass` finds a module that drives it and a roadmap row that claims to have
connected it, and both readings are wrong.

## Options considered

**1. Where the lag goes.**

  a. Inside `MicrobolometerDetector.response`. Would need the detector to own per-frame state,
     which ADR 0052 explicitly placed in `PipelineState.buffers` so that one object owns it and a
     reset clears it.
  b. In the M9 chain's "temporal filter" slot. Wrong physics: §11.1's temporal filter is an
     *image-processing* stage and is deliberately the identity (ADR 0058). The membrane is the
     detector's own response, §9.2, and belongs in stage 4.
  c. **In `_detector_signal`, between the static transfer and the noise** (chosen). This is where
     ADR 0052 says it goes and where `detector_stage` already had it, so the two CPU paths and the
     Warp twin now describe one thing.

**2. Getting the noise after the lag.**

  `response()` bundles the static transfer with the per-pixel noise, so there was no seam to insert
  at. `MicrobolometerDetector.frame_from_signal(signal_dn, …)` opens one, and `response` is now
  that composed with the transfer, so they cannot drift apart.

  **The seam is the bolometer's alone and is not on the `Detector` protocol.** A photon detector's
  noise is Poisson in electron space (CLAUDE.md #3); there is no signal-DN point at which noise
  could correctly be added, and it has no membrane to insert anyway. Giving the protocol a method
  only one implementer can honour would have invited exactly the wrong generalisation later.

**3. What interval drives it.**

  a. The configured frame rate, `1/frame_rate_hz`. Correct for a real-time camera and **wrong for
     a time-lapse**: ADR 0074 films the quadrotor at one frame per six seconds of scene time, a
     10 ms membrane settles completely across that, and driving it at 1/60 s instead leaves
     `e^(−dt/τ)` = **19 %** of a scene six seconds old in the picture. That ghost is a flattering
     error — it smooths precisely the change being filmed.
  b. **Elapsed scene time when the caller advances its clock, the frame rate otherwise** (chosen,
     `lag_interval_s`). `IrCamera` sets `state.t_s` before every `run_frame`, so both the
     time-lapse and the real-time stages get their true interval; a caller that never advances
     `t_s` — most unit tests, every single-frame use — falls back, and for them the two answers
     coincide.

  **This is a `run_frame` policy, not a property of the membrane**, and `bolometer_lag`'s default
  stays the configured interval. `detector_stage` is the CPU oracle the Warp twin is compared
  against, and the twin takes `alpha_for(fpa.frame_dt_s, …)` at kernel-launch time; making the
  shared primitive depend on `state.t_s` would have made the two disagree for any caller that
  advanced its clock. The policy is applied by the one function that owns the caller's clock.

## Decision

`run_frame`'s stage 4 lags a bolometer's signal before any noise is added, over the elapsed scene
interval, with the state in `PipelineState.buffers`. A photon detector takes the flux straight.

## Consequences

**Held to the closed form, not to "there is now a tail".** The step response is
`1 − e^(−k·dt/τ)` to 1e-6 in signal space, the first frame is α = 0.811124 of the step, and
successive residuals fall by exactly `e^(−dt/τ)` = 0.188876. A cooled photon detector settles in
one frame, which is the other half of the Tier 3 line.

**The tolerance on the tail is derived rather than chosen.** The signal plane is float32 and each
residual is a difference of numbers near 10 120 DN, so by the fourth term the residual is ~2 DN and
carries three significant figures: a flat `rel=1e-4` passes on the first two ratios and fails on
the fourth for no physical reason. The test scales its tolerance by float32's own precision on each
residual, and the first term — measured 1.1e-5 — stays tight.

**No golden moved.** `BolometerLowPass` adopts its first input, because a core staring at a scene
is already in equilibrium with it, so every single-frame reference is bit-identical. One test
changed: `test_through_run_frame_a_transparent_pixel_follows_what_is_behind_it` shared one
`PipelineState` across two frames and so read the second 81 % of the way through its settle —
measured 65.50 against the 80.75 it asserts, which is exactly α. It is a steady-state radiometric
identity and now takes a state per frame.

**What the rendered scenes do now.** The quadrotor time-lapse is unchanged: six seconds settles the
membrane completely, which is the right answer and the reason option 3(b) exists. The aircraft pass
runs real time at 30 Hz, where α = 0.964, so it gains a 3.6 % tail — small, and the first trail
this repository has ever rendered.

**What is still not modelled.** The membrane is linear and single-pole; a real bolometer's τ varies
with substrate temperature and bias, and §9.2 does not model that either. The lag is applied to the
whole frame with one τ, so a defective pixel with a different time constant is not represented.

## Revisit when

* ME.5's temporal PSD on flat sky says the §11.1 temporal filter is *not* the identity — then
  there are two temporal mechanisms on one path and ADR 0058's placement argument has to be redone
  with both.
* A scene needs a cold start rather than a settled one, at which point "the first frame adopts its
  input" becomes a choice a config makes rather than a constant.
* A detector arrives whose noise is neither Gaussian in DN nor Poisson in electrons, and the seam
  in option 2 stops being expressible.
