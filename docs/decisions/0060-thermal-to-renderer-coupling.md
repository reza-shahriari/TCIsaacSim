# ADR 0060 — Thermal-to-renderer coupling: a float32 facet table on a 1 Hz tick, sky from the ray

**Status:** Accepted
**Date:** 2026-09-12

## Context

§13.1 and the `isaac-sim-spg` skill both describe the thermal coupling as *the renderer carries the
surface temperature*: encode `c = (T − 200)/800` into an emissive channel, decode it in the kernel.
ADR 0014 measured that and rejected it — every colour AOV on this build is float16 and
exposure-scaled, giving ≈ 100 mK at 300 K against a 50 mK NETD — and decided that the renderer
transports **instance ids and geometry**, with temperature looked up from a float32 table.

This ADR settles what that table actually is for phase 1 (M10.18, aerial targets): where its values
come from, how often they are recomputed, and what happens to the pixels that are not geometry at
all. The full ThermalField over a facet mesh is phase 2 (M10.3) and deliberately not decided here.

Three questions had to be answered together, because a wrong answer to any one of them produces a
plausible image rather than an error.

## Options considered

**1. What fills the table.**

  a. A per-frame call into the energy balance for every prim — correct and pointlessly expensive;
     surface temperature does not move measurably in 33 ms.
  b. **The M6.6 solvers the M6.17 `Scene` already built, keyed by a per-prim → target-name map**
     (chosen). The solvers are the tested thermal model, they already hold the one shared
     `WeatherSeries` (CLAUDE.md #6), and the map is scene data rather than code.
  c. USD attributes carrying temperature per prim — puts physics in the asset, where nothing
     validates it.

**2. How often it ticks.**

  a. Once per frame — makes the result depend on the frame rate, which is a rendering setting.
  b. **A fixed 1 Hz thermal tick with linear interpolation between the bracketing ticks**
     (chosen). Decouples the physics from the frame rate and costs one solver step per second of
     simulated time regardless of how many frames are rendered.
  c. A tick per weather sample (minutes) — too coarse for an engine spooling up under a prescribed
     schedule, which is exactly the phase-1 target signature (MS.7).

**3. What the non-geometry pixels get.**

  a. An emissive sky dome — reintroduces the fp16 colour path this whole design exists to avoid.
  b. One sky temperature for the frame — wrong by tens of kelvin across a 50° field, and the error
     is largest exactly where small targets are hardest to detect.
  c. **`T_sky(θ)` from MS.2 evaluated at each pixel's own ray elevation** (chosen), with rays
     *below* the horizon taking the environment preset's ground temperature instead.

## Decision

- **`AerialThermalBridge(scene, prim_to_target, band=...)`** owns the coupling. It refuses a target
  name the scene does not define, and refuses a `SkyModel` holding a different `WeatherSeries` than
  the scene — CLAUDE.md #6 enforced at the engine boundary, not only inside `irsim`.
- **The table is float32, indexed by instance id**, built from `idToLabels` (prim path per id) and
  the per-prim solver map. Temperature is therefore exact at facet granularity: 317.25 K authored
  reads back as 317.25 K, and two targets 50 mK apart stay 50 mK apart — the pair that float16
  collapses onto one value.
- **A rendered prim with no thermal node raises.** The non-strict fill is **0 K**, chosen to be
  obviously broken rather than plausible: if it ever reaches a radiance kernel the image is
  visibly wrong instead of quietly being an image of a different scene.
- **Thermal tick 1 Hz, linear interpolation at render time.** The interpolation error is computed
  and reportable (`tick_error_k`), not assumed: for an exponential node it is bounded by
  `|ΔT|·h/(8·τ)`, which is ≈ 3 µK for τ = 900 s at h = 1 s — four orders of magnitude inside the
  10 mK budget. The clock refuses to run backwards, because the solvers are stateful and a rewind
  would silently produce a different history than a forward run of the same scene.
- **Background pixels are split at the horizon.** Elevation ≥ 0 takes `T_sky(θ)` (ADR 0044/MS.2);
  elevation < 0 takes `T_ground` from the environment preset's ground mode — the same quantity
  ADR 0045's reflected term uses. The sky model is only defined on [0°, 90°] and raises outside it;
  extrapolating it downwards would report a 40–60 K cold sky where warm ground actually is, which
  **inverts the contrast** of anything silhouetted against it. For a downward-looking aerial camera
  that is most of the frame, so this is not an edge case.

## Consequences

- Temperature is constant within a prim. A target needing sub-prim structure (a motor hotter than
  its arm) is modelled by splitting the mesh into prims, which is also how the thermal model
  already discretises it. Sub-facet gradients wait for M10.3.
- One ground temperature for everything below the horizon. Under an open sky — phase 1, ADR 0003 —
  that is all the environment preset claims anyway; a ground scene with structure needs M6.12's
  environment solver before this is better than a placeholder.
- The bridge holds no device memory. The float32 table is a small NumPy array that a Warp stage
  uploads; when the whole frame moves onto the device (M10.9a) only the upload changes.
- Replaying a frame needs a fresh `Scene`. That is a real constraint on a dataset generator that
  wants to re-render a timestamp, and it is stated loudly rather than papered over with a reset
  that would not restore solver state correctly.

## Revisit when

- A target needs sub-prim temperature structure, or a scene has enough prims that a per-prim map is
  unmaintainable — both point at M10.3's ThermalField and a temperature atlas.
- Ground scenes arrive (phase 2): the single `T_ground` below the horizon becomes the limiting
  approximation and wants M6.12's environment solver, or a real horizon/terrain hit.
- The thermal tick becomes visible in a phenomenology test — raise `tick_hz` and re-measure
  `tick_error_k` rather than changing the interpolation.
