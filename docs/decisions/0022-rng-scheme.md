# ADR 0022 — RNG scheme and the CPU/GPU reproducibility contract

**Status:** Accepted
**Date:** 2026-09-11

## Context

Every noise term (temporal, striping, fixed pattern, drift, bad pixels, shot) needs random numbers.
Runs must be replayable exactly, frames must be generatable out of order and in parallel, two sensors
in one scene must not share noise, and the Warp/SPG fast path must be comparable to the NumPy oracle.
NumPy's default generators (PCG64, MT19937) cannot be reproduced inside a Warp kernel, and Warp's own
hash does not match NumPy, so a sequential generator forces the GPU comparison to be statistical only.
The renderer's `rtx.frameId` was measured advancing 6–7 per orchestrator step (M2 spike), so it is not
a sensor frame count.

## Options considered

1. **One global sequential generator** — replay needs every earlier frame; parallel rendering changes
   results; sensors interleave.
2. **`SeedSequence(sensor, frame, stream)` → PCG64 per draw site** — replayable per frame, but still
   sequential within a frame (traversal-dependent) and not reproducible on the GPU.
3. **Counter-based per-element hash**: value = f(sensor_seed, frame_index, stream, element_index)
   via a 64-bit mixer, uniform from the top 24 bits, normal by Box–Muller in float32 — order-free,
   any pixel regenerable alone, and the *same integer mixer* is a few lines of Warp.

## Decision

Option 3 for every Gaussian term; option 2 only where no cheap counter-based form exists
(`irsim.noise.seeding`):

- `stream_key(sensor_seed, frame_index, stream)`: splitmix64 finaliser chained over the triple.
  `hash_u64(key, index, lane)` = mix64(key + ((index << 2) | lane) · φ), `hash_uniform` = (top 24
  bits + 1) · 2⁻²⁴ ∈ (0, 1] (exact in float32), `hash_normal` = Box–Muller on lanes 0/1 in float32,
  `field_normal(key, shape)` with element index = flat C-order index.
- **`frame_index` is the sensor's own counter** (`PipelineState.frame_index`), never `rtx.frameId`.
- `NoiseStream` values are part of the contract (TVH 1, TV 2, TH 3, T 4, fixed 10–12, drift 20–22,
  bad-pixel map 30, RTS 31, shot 40, dark 41) — append, never renumber. A test pins hash properties.
- `noise_rng` / `sensor_rng` (PCG64 from `SeedSequence`) remain for **Poisson** shot/dark draws and the
  bad-pixel map; those are statistically equivalent on the GPU, not bit-identical.
- **CPU/GPU contract:** the Warp stage re-implements the integer mixer and the uniform exactly;
  the equivalence test asserts bit equality on the uniforms and a few-ulp tolerance on the normals
  (`log`/`cos` differ in the last ulp between libms). Noise-free stages are compared exactly as before.
- No global state: `np.random.seed` / legacy `np.random.*` are never used by the core.

## Consequences

Memoryless components (T, TV, TH, TVH) are exactly replayable per frame and per pixel. Components with
memory (fixed-pattern drift, bolometer IIR, FFC) depend on history; `PipelineState` carries that state
and tests check sequential replay. Box–Muller costs a `log` and a `cos` per element — negligible
against the Simpson-free pipeline. Photon shot noise stays sequential PCG64 Poisson on the CPU; the GPU
will use its own Poisson (or a Gaussian approximation above ~1000 e⁻), compared statistically.

## Revisit when

A counter-based Poisson (e.g. inversion from the hashed uniform for small means, Gaussian above a
threshold) is needed for bit-exact photon-band comparison, or the stream list needs a new entry.

## Addendum (M10.7a, 2026-09-12): the device noise stage

Stage 5 is the first stage that cannot be held to bit-equality with the CPU oracle, which is what
this ADR anticipated. Warp's generator is not NumPy's, so the two paths draw different numbers from
the same seed by construction. Four decisions made the statistical comparison mean something.

**Counters follow `frame · H · W + pixel`, with a per-stream offset.** `wp.rand_init` is seeded
from `frame_index · 1000003 + sensor_seed`, plus a small distinct constant per component (7 for T,
13 for TV, 29 for TH, none for TVH), and indexed by the pixel, row, column or zero as that
component requires. Without the per-stream offsets the frame term and the row term would be drawn
from the same state and correlate, which is the kind of fault that shows up only in a 3-D
decomposition.

**The device fixed pattern is seeded from the CPU's own realisation, not redrawn.** On first use
the V, H and VH buffers are uploaded from `NoiseStage.unit_fixed`. Had the device drawn its own,
"statistically equivalent" would have compared two *different cameras* — a much weaker statement
than it sounds, and one that would pass even if the device pattern had the wrong spatial structure.
With the same realisation on both sides, what is being measured is the two generators.

**The NUC residual's ξ fields are uploaded too, so that stage stays bit-comparable** (to 2e-6
relative, i.e. float32 rounding). The residual is reset by an *FFC event*, not by a frame, so
redrawing it on device would mean a second copy of the epoch counter to keep in step across a
shutter. Uploading is both cheaper and safer, and it means the one part of the chain with an exact
CPU answer keeps one.

**The stage boundary differs between the paths, and the composition does not.** On the CPU the
per-pixel TVH term belongs to the *detector* (stage 4) and `NoiseStage` adds only the six
correlated terms; the Warp detector stage is the ideal transfer alone, so the device stage 5
supplies all seven. Comparing `NoiseStage.apply` directly against the device kernel therefore
compares a six-term image with a seven-term one — which is a real trap rather than a theoretical
one: it produces a PSD ratio of about ten and looks exactly like a broken kernel. The oracle is the
detector and the stage *together*, which is what each path actually produces for a frame.

Stage 5 is deliberately **not** registered in `EQUIVALENCE_STAGES`. That harness asserts bit-level
agreement, and a stage that cannot achieve it would have to loosen the harness for everyone else.
Its equivalence lives in `tests/integration/test_warp_noise_stage.py`, measured with
`irsim.validation`'s own `decompose_3d`, `spatial_psd` and `compare_psd` — the same statistics the
CPU path is validated with, rather than a second set invented for the comparison.
