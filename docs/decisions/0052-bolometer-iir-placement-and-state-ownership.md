# 0052 — Bolometer IIR: where the thermal lag is applied, and who owns the state

- **Status:** Accepted
- Date: 2026-09-12
- Step: M9.1 (`feat(detector): bolometer thermal time constant — stateful per-pixel IIR`)
- Spec: docs/physics-model.md §9.2; spec issue S8

## Context

§9.2 models the microbolometer membrane as a first-order thermal system,
`C_th dΔT/dt = α_abs Φ(t) − G_th ΔT`, with `τ_th = C_th/G_th` of 8–12 ms for VOx. Sampled once per
frame with the flux held constant across the frame, the exact solution is the exponential IIR

    S_n = S_{n−1} + (S_ideal_n − S_{n−1})(1 − e^{−Δt/τ_th}).

Two things about it are decisions rather than transcriptions of the spec, and both are easy to get
wrong in a way that still produces a plausible-looking image.

## Decision

**1. The filter is applied to the ideal signal, before noise.**

The membrane integrates *incident flux*. It does not integrate the ROIC's read noise, the FPA's
fixed-pattern terms, or the 3-D noise components, all of which are injected downstream of the
thermal integration in the real device. So the IIR sits between the ideal detector transfer (M3.7)
and the detector's own noise addition (M4.8) — the same place the physics puts it.

The alternative — filtering the detector's noisy output — is worse than merely unphysical. It
would correlate σ_TVH across frames with correlation coefficient `1 − α`, cutting the per-frame
temporal variance by roughly `α/(2 − α)`. At the reference 60 Hz / 10 ms that is a factor of 0.68
in variance, so the NETD anchor (M4.6) would silently stop delivering the datasheet NETD, and the
error would be invisible in a single frame and only show up in a temporal statistic. Placing the
filter before noise keeps the anchor meaningful and keeps M4.4's 3-D decomposition valid.

**2. The per-pixel state is float32 and owned by the pipeline, not the detector.**

The state array carries signal, so CLAUDE.md non-negotiable #2 applies: float32 or better, never
float16. It lives in `PipelineState.buffers` (M9.8 wires it) rather than inside the detector
object, for three reasons: detector instances stay immutable and safe to share between cameras;
every cross-frame buffer in the project has exactly one owner and one reset path, which matters
once drift (M9.4), NUC residual (M9.6) and FFC (M9.7) add their own; and a Warp port (M10.6) needs
the state as an explicit device buffer anyway, so keeping it explicit here makes that a transport
change rather than a redesign.

`BolometerLowPass` is therefore a small mutable holder that the pipeline drives, not a detector
subclass.

**3. The first frame adopts its input — the filter starts settled.**

A camera already looking at the scene is in thermal equilibrium with it. Starting the state at
zero would put a frame-long ramp at the head of every sequence, every golden fixture and every
exported dataset. `reset()` restores the cold-start behaviour for anyone who wants to model
power-on explicitly.

## Consequences

- The photon path has no filter at all: cooled MWIR/SWIR detectors are memoryless on these
  timescales, which is the §15 T3 phenomenology check (LWIR smears, cooled MWIR does not).
- `responsivity_rolloff(f, τ)` is the *continuous* response `1/sqrt(1 + (2πfτ)²)` — the datasheet
  quantity — not the sampled IIR's transfer function. They agree well below Nyquist. Temporal MTF
  work should use the former; frame propagation uses the latter.
- Spec issue S8 is resolved in the code's favour: §9.2's "smears over roughly 0.6 frames" is
  `τ_th/Δt`, not a smear extent (one frame of the IIR already reaches 81 %). The quantity worth
  quoting is the trailing exponential decay length of a moving edge, `v τ_th/Δt` pixels, which
  `trailing_decay_length_px` computes and `tests/unit/test_bolometer_lowpass.py` measures from a
  rendered trail. The spec wording should be corrected; this ADR records the interpretation the
  code uses in the meantime.

## Alternatives considered

- **Filter after noise.** Rejected: breaks the NETD anchor as quantified above.
- **State inside the detector object.** Rejected: makes detectors mutable and stateful, scatters
  cross-frame state across two owners, and complicates the Warp port.
- **Convolutional smear from motion vectors.** Rejected for the bolometer: §13.3 suggests motion
  vectors for "bolometer smear", but the smear is an inter-frame temporal effect, not an
  intra-frame spatial one. Motion vectors feed `MTF_motion` for photon detectors (ADR 0059, spec
  issue S20); the bolometer gets its smear from state, for free, and correctly for arbitrary
  motion including objects that stop.
