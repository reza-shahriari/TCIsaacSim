# 0054 — The fixed-pattern drift model, and which components drift

- **Status:** Accepted
- Date: 2026-09-12
- Step: M9.4 (`feat(noise): mean-reverting (OU) drift of fixed-pattern components`)
- Spec: docs/physics-model.md §10.3 ("pattern breathing"), §10.2, §8.2

## Context

§10.3 says the fixed-pattern noise of a real focal plane is not fixed: between corrections the
pattern *breathes*. It does not say what stochastic process produces that, and it does not say
which of the seven 3-D components take part. Both are ours to choose, and both choices are the kind
that look harmless for a hundred frames and wrong after ten thousand.

Two further mechanisms in the M9 chain also move the image slowly, and it is easy to end up with
all three describing the same physical drift:

1. The **global offset** rises and falls because the optics housing and the focal plane physically
   warm and cool (M3.3 self-emission on the housing node M9.3, and the FPA node M9.2).
2. The **NUC residual** grows between FFC events because the correction was calibrated at one
   T_FPA and is being applied at another (M9.6, and ADR 0053 makes it the *only* ΔT_FPA-driven
   residual).
3. Whatever this step adds.

## Decision

### The process is Ornstein–Uhlenbeck, advanced by its exact update

    x ← x e^{−dt/τ} + σ √(1 − e^{−2dt/τ}) ξ,     ξ ~ N(0, 1),     τ = `noise.fpn_drift_tau_s`

**Not a random walk.** A random walk is the obvious way to make a pattern move, and its variance
grows without bound. The configured 3-D ratios — σ_V, σ_H and σ_VH against σ_TVH — are the
sensor's identity: they are what makes a frame look like it came from *this* camera. A random walk
destroys them a little per frame, while every individual frame still looks like a plausible thermal
image, so nothing in a visual check or a single-frame test would catch it. The OU form is
stationary by construction: if Var(x) = σ² then

    Var(x′) = σ² e^{−2dt/τ} + σ² (1 − e^{−2dt/τ}) = σ²

exactly, at any step size. The ratios hold for a ten-minute sequence and a ten-hour one alike.

**Exact, not Euler–Maruyama.** The decay and innovation terms above are the closed-form solution of
the OU SDE over the interval. A first-order discretisation would give a *different stationary
variance at coarse dt*, which is how the same camera modelled at 9 Hz and at 60 Hz would end up
with measurably different FPN — a frame-rate-dependent sensor fingerprint, and a genuinely hard
thing to diagnose later.

### All three fixed terms drift: V, H and VH

They share one physical cause — the per-row, per-column and per-pixel spread in how the detector
and the ROIC respond to bias and to temperature. Two reasons not to restrict it to VH:

- Slowly breathing **column stripes** are among the most recognisable artefacts in real uncooled
  imagery. Breathing VH alone leaves the stripes frozen, which is visibly wrong in a way that
  matters for sim-to-real transfer, where low-level structure dominates (see the
  `sensor-noise-chain` skill).
- ME.3 measures fixed-pattern *growth* between FFC events as a whole-pattern figure and does not
  separate the three. A model that drifts only VH would have to be tuned to reproduce that figure
  with one component doing the work of three.

`FpnDrift.components` can restrict the set, for ablations (ME.8) and for isolating which component
an artefact came from.

### The global term does not drift here, and passing it is an error

`components=("t", …)` raises. The DC level is produced *physically* by the housing and FPA nodes;
adding a stochastic global drift on top would count the same effect twice, and the result would
look like a plausible drift rate rather than a bug. This is the same separation ADR 0053 drew
between the detector's raw response and the NUC residual, applied one layer out.

### Drift is the one sequential stream in the noise chain

Every §10.2 per-frame term is drawn from the counter-based hash keyed by `(sensor_seed,
frame_index)`, so any frame can be reproduced in isolation (ADR 0022). The OU state cannot be: it
is the accumulation of every innovation before it. `drift_rng(sensor_seed)` therefore returns a
sequential `np.random.Generator` on the reserved `VH_DRIFT` stream, and a sequence is reproducible
from its start rather than from an arbitrary frame. This is also why M10.7 holds the Warp twin to
*statistical* equivalence with this path rather than to bit-equality.

## Consequences

- A frame in the middle of a drifting sequence cannot be rendered without replaying the sequence
  from its start. For golden fixtures and for M10.7 this means fixing the start, not the frame.
- Because the OU process is stationary, it contributes **no growth** between FFC events. All the
  between-FFC growth ME.3 measures must therefore come from M9.6's ΔT_FPA residual. If the measured
  growth cannot be reproduced by M9.6 alone, the right response is to revisit M9.6 — or this ADR
  deliberately — not to quietly make the drift non-stationary.
- `noise.fpn_drift_tau_s` is currently 120 s for the Boson and is **ESTIMATED**: no published
  figure exists for it. ME.3's between-FFC measurements are what should eventually set it, together
  with the amplitude. Until then, a τ this short mostly affects how quickly a pattern decorrelates
  from a frozen reference, not the per-frame appearance.
- `float('inf')` freezes the pattern bit-identically and consumes no draws. That is the ideal-sensor
  control case and the baseline every drift test is written against.

## Revisit when

ME.3 produces a measured between-FFC growth band and a decorrelation time from public imagery, and
either the OU τ or the amplitude falls outside it; or a temporal PSD from real flat-sky footage
shows the pattern drift has a 1/f^β shape rather than the OU Lorentzian, in which case the process —
not the component set — is what changes.
