# 0057 — Where the FFC controller lives, and what the freeze does

- Status: accepted
- Date: 2026-09-12
- Step: M9.7 (`feat(isp): FFC controller — schedule, freeze, reset; shutterless and ideal modes`)
- Spec: docs/physics-model.md §11.2, §12.2 `nuc`, §15 Tier 3

## Context

Three questions had to be settled before the flat-field correction could be written, and the spec
answers none of them:

1. **Does the controller belong in `irsim.noise` or `irsim.isp`?** It resets noise objects, but it
   is triggered by a schedule and it modifies the emitted image.
2. **What does the freeze actually emit?** §11.2 says "the image freezes for 0.5–1 s" without
   saying what a frozen frame contains.
3. **What does `shutterless` mean quantitatively?** §11.2 says only "a slow-drift +
   scene-based-correction residual instead of periodic freezes".

## Decision

### The controller is ISP, and it owns ΔT_FPA

`irsim.isp.ffc.FfcController`. An FFC is a **camera behaviour** — a mechanism closes, a schedule
fires, the output stream changes — not a noise source. `irsim.noise` generates fields; `irsim.isp`
decides what the camera does with them and what it emits. Putting the controller in `noise` would
have made that package responsible for the output stream.

The consequence that matters more than the package name: **the controller owns ΔT_eff**, the
temperature offset M9.6's residual is evaluated at. The three modes differ almost entirely in that
one quantity, so putting it anywhere else would have scattered the definition of "mode" across
three files. `NucResidual` stays a pure function of the ΔT it is handed.

Objects the shutter recalibrates attach through a one-method `Resettable` protocol
(`reset(frame_index)`): M9.6's `NucResidual`, M9.4's `DriftingPattern`, and anything M9.8 adds.
The controller does not import them and does not know what they are.

`frame_index` is passed to `reset` and deliberately **ignored** by both current implementations:
epochs are counted by FFC, not by frame, so a sequence replayed at a different frame rate gets the
same sequence of patterns.

### The freeze holds the last good frame; it does not blank

A closed shutter means the FPA is looking at the shutter blade. A camera that emitted *those*
frames would show a flat field, and a perception stack would watch the scene vanish and reappear —
conspicuous, and not what real cores do. They hold the last good frame, and that is what makes the
artefact worth simulating: for 0.7 s the image is **stale** rather than obviously wrong, so
anything tracking through it sees motion stop dead and then jump. A tracker that survives a blank
may still fail on a hold.

`freeze_frames = round(freeze_s · fps)` — **round, not ceil**. 60 Hz × 0.7 s is exactly 42; 9 Hz ×
0.7 s is 6.3 and becomes 6. A freeze is a physical duration being sampled by the frame clock, and
at 9 Hz the partial frame at the end is more likely to complete than not. Ceil would systematically
lengthen every freeze at low frame rates, which is where the artefact is most visible.

Frame 0 never fires: the camera was calibrated when it was switched on, and firing at 0 would put a
freeze at the opening of every sequence, golden fixtures included.

### `shutterless` is a first-order lag, not an unbounded drift

A shutterless core runs a scene-based correction: it estimates the fixed pattern from
pixel-sized translations of the focal plane and subtracts it continuously [R32, R33]. The residual
therefore settles where the estimator's convergence balances the drift, rather than tracking
ΔT_FPA without limit. The simplest model with that property, and the one adopted here, is

    dΔT_eff/dt = dΔT/dt − ΔT_eff/τ,      τ = `nuc.shutterless_tau_s`

integrated exactly over each step. Under a constant drift rate *r* it approaches *r·τ* instead of
growing. At τ = 120 s and 0.05 K/s the residual at 540 s is 1.15× its 180 s value, against 3× for
an uncorrected core — comfortably inside the roadmap's ≤ 1.5× bar, and the test asserts both the
bound and the saturation value so that "bounded" is not satisfied by merely being slow.

`ideal` is ΔT_eff ≡ 0: no residual, no freeze, no reset. The control case every other mode is
measured against.

## Consequences

- `nuc.shutterless_tau_s` is new (sensor schema v8), defaults to 120 s, and is **ESTIMATED** — no
  published convergence time for a scene-based correction was available. It is unused outside
  `mode: shutterless`, so no existing config changes behaviour.
- Because the controller owns ΔT_eff, a caller that computes the residual without going through it
  gets whatever ΔT it invents. M9.8 wires the single path; until then that coupling is a
  convention, not an enforcement.
- The first frame of a sequence cannot be held (there is nothing yet to hold), so an FFC landing
  there passes the live frame through rather than inventing one. In practice `fires_on` excludes
  frame 0, so this only arises in a bench that drives the controller directly.
- A held frame is copied, not aliased. A caller that reuses its input buffer — which the pipeline
  does — would otherwise see the held frame change underneath it.

## Revisit when

ME.3's ISP-behaviour extractor measures a real FFC interval, freeze duration or between-FFC growth
that this schedule cannot reproduce; or a shutterless core's measured residual turns out to grow
with a shape other than a first-order lag (a 1/f-like slow component would need a different model,
not a different τ).
