# ADR 0049 — Aerosol extinction from visibility (Koschmieder) with per-band ratios; fog regimes

**Status:** Accepted
**Date:** 2026-09-11

## Context

§7.1 splits the band extinction into a molecular and an aerosol part, γ = γ_mol + γ_aer. §7.3
gives γ_mol,B = γ₀,B + β_B w (ADR 0048 fixes the coefficients). The aerosol part must come from the
weather's *visibility* (the only aerosol observable in a weather file, ADR 0032) and it must
reproduce the §7.2 table's decisive facts: LWIR beats everything in fog, SWIR beats LWIR in humid
clear air, and the visible dies first in haze and fog. The table's haze row is also internally
inconsistent (below).

## Options considered

1. **Per-condition τ tables** (τ_B at 200 m per preset, interpolated in visibility) — no physics,
   cannot extrapolate in distance, and every new band needs new authored numbers.
2. **Koschmieder + per-band extinction ratio** (chosen) — γ_aer,vis = 3.912/V is the *definition*
   of meteorological optical range (2 % contrast threshold), and γ_aer,B = r_B γ_aer,vis with one
   ratio per band per aerosol regime. Two numbers per band, physical in V and d.
3. **Mie extinction from a droplet/particle size distribution** — right in principle, but the
   size distribution is not in any weather file and the ratios it produces are what option 2
   authors directly. The upgrade path if measured fog PSDs become available.

## Decision

`irsim.atmosphere.extinction`:

    γ_aer,vis(V) = max(0, KOSCHMIEDER / V − γ_mol,vis)         KOSCHMIEDER = ln(1/0.02) = 3.912023…
    γ_B(w, V)    = γ₀,B + β_B w + r_B · γ_aer,vis(V)

MOR is defined on the *total* visible extinction, so the Rayleigh part of the visible band
(γ₀,vis = 1.2e-5 m⁻¹) is subtracted; V → ∞ recovers clear air and the Rayleigh-limited
visibility (~330 km at sea level) gives zero aerosol. The ratios r_B live in the preset per
`aerosol_regime`:

| regime | LWIR | MWIR | SWIR | NIR | source |
|---|---|---|---|---|---|
| rural (clear presets, haze) | 0.10 | 0.19 | 0.43 | 0.70 | fitted to the §7.2 haze row (below) |
| maritime (tropical) | 0.30 | 0.40 | 0.60 | 0.80 | ESTIMATED: large sea-salt particles flatten the wavelength dependence |
| droplet (fog presets) | 0.16 | 0.24 | 0.40 | 0.60 | fitted to the light-fog row |

NIR has no table row; its ratios are interpolated between the visible (1) and SWIR. The fog
presets carry `aerosol_regime: droplet`; `regime_for_visibility(V)` (WMO: fog is V < 1 km) says
which regime the weather implies so an `Atmosphere` can warn when a preset is used outside the
conditions it was fitted for. The molecular coefficients are the same in every preset (one physics
of water vapour); presets differ in aerosol regime, profile shape and provenance.

**The haze row.** Labelled "5 km vis", its visible column reads τ(200 m) = 0.45–0.65. Koschmieder
at V = 5 km gives τ_vis(200 m) = exp(−200 · 3.912/5000) = 0.855, far outside that range, and the
SWIR column (0.70–0.85) would need an aerosol ratio above 1 — extinction rising with wavelength,
which no small-particle aerosol does. At **V = 1.5 km** all four columns are reproduced with
wavelength-ordered ratios (0.10 / 0.19 / 0.43 / 1): τ = 0.886 / 0.862 / 0.774 / 0.594 against
rows 0.85–0.92 / 0.82–0.90 / 0.70–0.85 / 0.45–0.65. The row's τ values are therefore taken as
authoritative and its visibility label as the error (spec issue T18); the `haze` preset's
provenance is checked at V = 1.5 km.

**Fog.** Droplet ratios fitted on the light-fog row (V = 200 m, 283.15 K, RH 1.0) reproduce the
dense-fog row (V = 50 m) *without* fitting: τ(200 m) = 0.074 / 0.022 / 0.0018 / 1.6e-7 against
0.02–0.10 / 0.01–0.06 / ~0 / ~0. τ_vis(200 m) in 200 m fog is exactly 0.02, the table's lower
edge, because that is the Koschmieder threshold.

## Consequences

- Every preset × band reproduces its §7.2 row at the conditions in `tests/unit/test_atmosphere_presets.py`;
  the fog ordering τ_LWIR > τ_MWIR > τ_SWIR > τ_NIR > τ_vis and the humid/fog LWIR–SWIR crossover
  follow from the weather alone with a single preset.
- Errors: the table rows are ±5–10 % wide and ESTIMATED themselves; expect ±30 % on any γ_B.
  Humidity does not affect the visible/NIR aerosol (hygroscopic growth ignored). The ratios are
  band-grey, so a band edge near a strong aerosol feature is not represented. Nothing here
  extends beyond `valid_range_m` (ADR 0048).
- `KOSCHMIEDER` lives in `irsim.radiometry.constants` as the exact ln 50 (the conventional 3.912 rounded
  would put τ(V) at 0.0200005, failing the 1e-12 identity).

## Revisit when

A MODTRAN/LOWTRAN-derived per-band ratio table or measured fog PSDs are available (option 3), or
when MS.1 needs the aerosol scale height (currently only the water-vapour scale height is in the
profile block).
