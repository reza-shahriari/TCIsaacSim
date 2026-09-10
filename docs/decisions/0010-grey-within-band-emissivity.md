# ADR 0010 — Grey-within-band material properties via Planck-weighted averaging

**Status:** Accepted
**Date:** 2026-09-11

## Context

Materials have spectral ε(λ), ρ(λ), τ(λ); the simulator carries one scalar per band per material
(`docs/physics-model.md` §12.3, Appendix A #3). Something must reduce the spectrum to the scalar, and
how it does so decides (a) whether Kirchhoff closure survives the reduction and (b) how large the
error is when the property slopes across the band.

## Options considered

1. **Unweighted mean over the band edges** — ignores both the response shape and the Planck
   spectrum; wrong by the full slope of the property.
2. **Response-weighted mean** ∫R s dλ / ∫R dλ — respects the camera, ignores that a 300 K scene
   fills the band unevenly (LWIR at 300 K peaks near 9.7 µm).
3. **Planck-and-response-weighted mean at a reference temperature** — ∫R s B(T_ref) dλ / ∫R B(T_ref) dλ,
   energy form for bolometers, photon form for photon FPAs. Exact for the radiance emitted at T_ref;
   linear in s, so closure survives.
4. **Runtime spectral integration per pixel** (§3.2 c) — exact, but it is hyperspectral rendering, not
   an imaging camera, and it multiplies the LUT by the material count.

## Decision

Option 3, `irsim.radiometry.band_average.band_average(response, spectrum, t_ref_k=300, form)`:

- Weighting is R(λ)·B(λ, T_ref) with `form='energy'` (bolometers) or `form='photon'` (photon
  detectors); the same Simpson grid as the band-integration oracle.
- **T_ref = 300 K** by default: the scene temperature the sensor is characterised at (NETD is quoted
  at 300 K, §9.4) and where most pixels sit.
- Spectra arrive as callables; `tabulated(λ, values)` builds one from a file table, holding the end
  values beyond the table (a material file that stops at 14 µm still has an ε at 14.2 µm, where R is
  already ~0).
- Because the operator is linear, pointwise ε + ρ + τ = 1 gives ⟨ε⟩ + ⟨ρ⟩ + ⟨τ⟩ = 1 to 1e-9 — tested, so
  non-negotiable #4 holds after reduction without any renormalisation step.

## Consequences — the error accepted

A property that slopes across the band averages differently at other temperatures. Measured with the
Boson response for ε falling from 0.95 to 0.85 across 7–14 µm: ⟨ε⟩ changes by ~2e-3 between 300 K and
600 K (energy form), and the energy and photon forms differ by ~1e-3 at 300 K. For LWIR scene
materials (ε slopes of ≲ 0.1 across the band) the resulting radiance error is below 0.3 % at any
scene temperature, i.e. below ~0.2 K apparent — under the 2 K Tier 4 per-class target. Materials with
strong spectral structure inside the band (reststrahlen bands of silicates in the 8–10 µm region,
narrow-band paints) are **not** well represented; they are the L3 case (§3.2 c) and must be flagged in
their material file.

## Revisit when

Tier 4 shows a per-class bias above 2 K that tracks scene temperature, or a target material with
in-band spectral structure matters (e.g. quartz-rich soils in phase 2) — then either a second table at
a hotter T_ref or per-material spectral integration (option 4) for that material only.
