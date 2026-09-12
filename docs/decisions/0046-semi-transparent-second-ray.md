# ADR 0046 — Semi-transparent materials: τ scalar per band, closure on band-effective values, L_behind = L_env until a second ray exists

**Status:** Accepted
**Date:** 2026-09-12

## Context

§4.4 is blunt about why this matters: glass is opaque in LWIR and transparent in SWIR, so a
windshield shows *the windshield's own temperature* in one band and *the driver* in another. That
is one of the headline capabilities of a multi-band simulator, and §4.4 warns that retrofitting
transmission into a surface-only model is painful. Until now the pipeline carried the two-term
form L = ε L_B + (1 − ε) L_env, where (1 − ε) silently stood for ρ + τ. That renders a
windshield identically in both bands, plausibly, and wrongly.

## Options considered

1. Keep folding τ into (1 − ε). Cheapest, and it makes the band transition unrepresentable —
   the one thing §4.4 says to build in from the start.
2. Spectral τ(λ) with a full second-ray path traced through the material. Correct, and far beyond
   what the renderer supplies in phase 1: there is no second-hit AOV yet.
3. **Scalar τ per band, a single second ray, closure enforced per pixel on band-effective
   values** (chosen).

## Decision

- **L = ε L_B(T_s) + ρ L_env + τ L_behind**, implemented in
  `irsim.materials.surface.surface_radiance` as the scalar oracle and used by stage 1.
- **τ is a scalar per band**, authored in the material file; **ρ is derived** as 1 − ε − τ and
  never read back from the packed table, so the closure cannot drift from the authored values.
  `MaterialTable.properties_for` returns the triple and refuses a derived ρ below zero.
- **Closure is checked per pixel**, not per material, to 1e-6. A closure check on the library
  alone would pass while a hand-built table used in a test or by the Isaac adapter violated it.
- **L_behind = L_env when no second-ray plane is supplied.** The three-term form then collapses
  to exactly ε L_B + (1 − ε) L_env, so opaque scenes, every existing golden and the two-term
  behaviour of M7.13 are bit-identical. This is what makes the change safe to land before the
  renderer can supply the second hit.
- **The plane is `radiance_behind`**, which needs no change to the G-buffer contract: the
  contract already admits any plane whose name begins with `radiance` and treats it as
  precision-critical, so float16 is refused for free.
- **A second-ray plane with no environment model raises.** Without `PipelineConfig.sky` there is
  no L_env, hence no ρ term, and quietly dropping the τ term would render a transparent material
  as opaque — the silent-plausible-wrong failure this project exists to avoid.

## Consequences

- The committed windshield now behaves correctly across bands: dL/dL_behind is 0.0 in LWIR,
  0.02 in MWIR, 0.70 in SWIR and 0.77 in NIR, from one material and one code path.
- Only one ray deep: what is behind the glass is itself rendered without anything behind *it*,
  so two stacked transparent surfaces are wrong. Scenes needing that want option 2.
- τ is band-grey, so a material with structure inside a band (a coating with a cut-on) is not
  represented; and there is no refraction, so the second ray is not bent.
- `L_behind` is currently a fixture key. The Isaac adapter will fill it from a second-hit AOV
  (M10.x); until then only synthetic scenes exercise the τ > 0 path.

## Revisit when

The renderer gains a second-hit AOV, or a scene needs stacked transparent surfaces, refraction,
or spectral τ(λ) within a band.
