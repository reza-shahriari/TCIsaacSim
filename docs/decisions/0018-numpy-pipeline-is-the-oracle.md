# ADR 0018 — `irsim.pipeline` (NumPy) is the reference; Warp and SPG are fast paths

**Status:** Accepted
**Date:** 2026-09-11

## Context

Three implementations of the same stage chain will exist: the engine-free NumPy pipeline, the Warp
path inside Isaac Sim (§13.6) and the SPG CUDA kernels (§13.5). CLAUDE.md calls `irsim_isaac/pipeline`
the "Warp-based reference pipeline" while the skills make the NumPy path the oracle. One of them has
to be the thing the others are tested against; nobody can eyeball a thermal image for correctness.

## Options considered

1. **Warp is the reference** — runs only with Isaac Sim and a GPU; the physics loop then needs the
   engine, which is exactly what non-negotiable #1 forbids.
2. **NumPy is the reference; GPU paths are validated against it** on synthetic G-buffers to float32
   rounding (5 mK / 1e-4 relative, ir-sim-testing skill) — the reference runs in the default gate.

## Decision

Option 2. `src/irsim/pipeline/` is an engine-free package with:

- `Stage`: a protocol `(planes, config, state) → planes` on the G-buffer dict contract, float32 in
  and out for anything carrying temperature or radiance;
- `PipelineConfig` (sensor config, band LUT, material table, FPA params, supersample factor) and
  `PipelineState` (frame index, housing temperature, persistent buffers for IIR/drift/FFC);
- stage 1 `band_radiance_stage`: L = ε₀ · L_B(T) per material id, **no π, no aperture, no cos⁴** —
  radiance is a scene property; the optics stage converts it to pixel power.

`irsim_isaac/pipeline` is the Warp *fast path*; its docstring wording and the CLAUDE.md layout
(proposed additions `src/irsim/pipeline/`, `src/irsim/validation/`, `src/irsim_eval/`, commit scope
`pipeline`) are open question 12. Interfaces respect ADR 0014: ids never blend, temperature is
per-facet, `sky_view_factor` is an input plane, supersampling is a G-buffer resolution choice and
anti-aliasing happens by box-filtering radiance downstream.

A minimal `irsim.materials.MaterialTable` (id → ε₀ per band, id 0 = UNMAPPED and refused) lands with
this step; M7.18 extends it with the angular/reflectance/transmittance columns.

## Consequences

Every GPU kernel gets a test of the form "same synthetic G-buffer through NumPy and through the
kernel, agree to 5 mK". The NumPy path is slow (seconds per frame at 4× supersample) and that is
fine: it is the oracle, not the product.

## Revisit when

A stage cannot be expressed in NumPy without the engine (e.g. path-traced reflections at L3) — then
that stage's oracle is a different, still engine-free, reference.
