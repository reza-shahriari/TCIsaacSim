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
