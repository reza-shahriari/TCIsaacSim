# 0048. One grey extinction coefficient per band (the L2 atmosphere)

Date: 2026-09-11
Status: accepted

## Context

The physics model (§7.2, Appendix A #2) models the atmosphere as a single band-averaged extinction
coefficient γ_B per band: τ_B(d) = exp(−γ_B d). Real extinction is spectrally structured (water-vapour
continuum and lines across the LWIR, the opaque CO₂ band at 4.2–4.4 µm inside the MWIR), and the
exact band transmittance

    τ_B(d) = ∫ R(λ) B(λ,T) e^{−γ(λ) d} dλ / ∫ R(λ) B(λ,T) dλ

is a Planck-weighted mean of exponentials, not an exponential. By convexity it is always *larger*
than exp(−⟨γ⟩ d): the strong lines saturate first and the remaining, more transparent part of the
band carries the signal ("curve of growth"). The effective γ_eff(d) = −ln τ_B(d)/d therefore
decreases with distance, and any grey γ_B fitted at short range over-attenuates at long range.

M8.7 (`scripts/validate_atmosphere_band_average.py`, `irsim.atmosphere.spectral`) quantifies this
with two synthetic spectra chosen to bracket the real cases, with the grey γ_B fitted by least
squares on ln τ over 0–300 m (the range the spec's presets are meant for):

| Case | γ_B fitted 0–300 m | error at 500 m | error at 1000 m |
|---|---|---|---|
| LWIR 8–12 µm, two-level γ (2e-4 /m below 10 µm, 1.2e-3 /m above) | 6.67e-4 /m | −1.7 % | −8.7 % |
| MWIR 3–5 µm, opaque CO₂ notch 4.2–4.4 µm (5 /m) over 3e-4 /m | 9.78e-4 /m | −18 % | −42 % |

(error = exp(−γ_B d)/τ_B(d) − 1; negative means the grey model transmits too little.) The MWIR case
is pathological on purpose: the notch is fully saturated within metres, so it should be removed from
the band *before* fitting a grey coefficient (i.e. γ_B describes the transparent part of the band and
the response effectively excludes 4.2–4.4 µm). With the notch excluded the MWIR band behaves like the
LWIR case.

## Decision

Keep the single grey γ_B per band for phase 1 (the L2 atmosphere), with these rules:

1. Presets (M8.3) specify γ_B for the *transparent* part of each band; an opaque sub-band (CO₂ in
   MWIR) is treated as excluded from the band's response, not folded into γ_B.
2. The model is documented as valid to ~500 m (README limitation) with the measured error budget:
   ≤ 2 % transmittance error at 500 m and ≤ 9 % at 1 km for LWIR-like spectral structure.
3. The grey model errs on the side of *less* contrast at range. For the sky-target application
   (slant paths of kilometres) this is the conservative direction for detection performance, and a
   curve-of-growth correction (γ_eff(d) tabulated from a line-by-line or MODTRAN-class run) is the
   upgrade path, kept as a per-band table so the radiance kernel does not change.
4. Beer–Lambert stays in radiance space with path radiance (1 − τ) L_B(T_air): the isothermal
   invariance and the contrast identity hold exactly for the grey model.

## Consequences

- `irsim.atmosphere.beer_lambert` is the phase-1 kernel; `irsim.atmosphere.spectral` is the oracle
  that measures what the grey model loses, and the unit test pins the numbers above so a change in
  the fitting convention is noticed.
- Ranges beyond ~1 km, MWIR without the CO₂ exclusion, and airborne slant paths through the full
  column need the curve-of-growth table (phase 2, together with MS.2's sky radiance model).
