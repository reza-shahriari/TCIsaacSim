# ADR 0040 — Material files: one YAML per material, `source` required, data directories; the closure reading for τ > 0

**Status:** Accepted
**Date:** 2026-09-11

## Context

§12.3 shows a `materials:` mapping with one entry; the thermal-materials skill shows the same entry
without the wrapper. CLAUDE.md #4 says "author one of the three and derive the others" — which is
under-determined once τ > 0: a windshield's SWIR triple (ε, ρ, τ) has two degrees of freedom, and
"author one" cannot fix both. The loader has to decide what may be authored, where the data files
live, and how a violation is caught.

## Options considered

1. One `materials.yaml` with a mapping of all materials — merge conflicts for everyone, no per-file
   provenance, and the §12.3 example's `materials:` wrapper on every load.
2. **One YAML per material, `material:` top-level block** (chosen; mirrors `sensor:` and
   `atmosphere:`), `source` (measured | literature | estimated) required, file name = material name.
3. Author all three and check closure — the failure the rule exists to prevent.

## Decision

- **Layout.** `configs/materials/<name>.yaml` with a top-level `material:` block (the `materials:`
  wrapper is refused with a message); property spectra under `data/spectra/materials/`, n/k tables
  under `data/nk/`, both referenced relative to the data root of ADR 0008. `source` is required and
  a `reference` string says where the numbers came from; ESTIMATED values are marked in the files.
- **Closure reading of non-negotiable #4 for τ > 0.** Exactly **one** of {ε, ρ} is authored — as a
  spectral file (`spectral_emissivity` / `spectral_reflectance`, reduced by the ADR 0010 Planck-
  weighted band average) or as a scalar per band (`emissivity_per_band` / `reflectance_per_band`).
  τ may be authored per band (`transmittance_per_band`, default 0). The third quantity is always
  derived: ρ_B = 1 − ε_B − τ_B or ε_B = 1 − ρ_B − τ_B, on band-effective values (ADR 0046). An
  authored pair that leaves no room (ε_B + τ_B > 1) is refused — at the schema for scalars, after the
  band average for spectra, so a file cannot smuggle a violation through. Proposed CLAUDE.md wording
  (open question 12): "Author exactly one of ε or ρ (spectral or per band); τ may be authored with
  it; the remaining quantity is derived, never authored."
- **Band keys are free strings**, not an enum: a material may declare a band no current sensor has
  (`lwir_wide`); consumers request the bands they need, and a spectral material needs the sensor's
  response (or the registry's nominal top-hat) to answer for a band.
- **The library walk exists now** (`test_committed_library_closes_in_every_band`): every material,
  every standard band, closure to 1e-6 — the test CLAUDE.md #4 promised (spec issue T3).

## Consequences

- Six starting materials from §16.2 (paint black/white, bare aluminium, windshield glass, dry
  asphalt, skin); NIR/SWIR values and all roughness/angular parameters are ESTIMATED and say so.
  The black paint carries a piecewise spectral ε(λ) so the spectral path is exercised end to end.
- `Material.content_hash()` keys tables and goldens (files by content, like `config_hash`).
- Fresnel n/k files are resolved and hashed but not yet evaluated (M7.6); the angular model is
  carried through to the table as Level C placeholders (M7.18).

## Revisit when

Measured spectra arrive (M7.5, ADR 0041) or a material needs a spectral τ(λ) (today τ is scalar
per band, ADR 0046).
