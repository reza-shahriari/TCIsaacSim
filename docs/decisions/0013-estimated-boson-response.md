# ADR 0013 — Estimated Boson VOx spectral response and its expected error

**Status:** Accepted
**Date:** 2026-09-11

## Context

The first configured band is LWIR on the FLIR Boson 640 (§16.1). No measured relative spectral
response for the Boson's VOx core plus its germanium optics is publicly available, and no camera is on
hand to measure one (ADR 0003). The LUT, NETD anchoring and every LWIR radiometric statement depend on
R(λ), so the shape used must be stated as an estimate with a bounded expected error, not presented as
data.

## Options considered

1. **Exact top-hat 7.5–13.5 µm** — the closed form; edges are unphysical (real curves roll off over
   ~0.5 µm) but Lb differs from a soft-edged curve by well under 1 %.
2. **Raised-cosine edges of ±0.25 µm centred on 7.5 and 13.5 µm, flat between** — smooth, integrates
   to exactly the top-hat width, keeps the 50 % points at the quoted band edges; still an estimate.
3. **A digitised generic VOx + Ge curve from the literature** — more structure (Ge AR-coat ripple,
   VOx absorber roll-off beyond 12 µm) but of unknown provenance for this core, and it would carry a
   false air of measurement.

## Decision

Option 2, committed as `data/spectra/responses/boson_vox.csv` (M1.3) with `ESTIMATED` in its
provenance header, 0.01 µm grid, zero outside 7.25–13.75 µm. The generated LUT (M1.10, `make luts`,
band hash `f0446a1f…`) gives, at 300 K:

| quantity | value | check |
|---|---|---|
| Lb(300 K) | 55.4777 W m⁻² sr⁻¹ | top-hat closed form 55.4916; −0.03 % (−15.8 mK equivalent) |
| Lb(600 K) | 692.32 W m⁻² sr⁻¹ | top-hat 692.10; +0.03 % (+67.5 mK equivalent) |
| λ_eff(300 K) = hc·Lb_q/Lb | 10.44 µm | inside 9.5–11.5 µm |
| dLb/dT(373)/dLb/dT(300) | 1.737 | [R9] anchor 1.736 (39 → 23 mK NETD) |

## Consequences — the error accepted

Against a *measured* Boson curve, the in-band radiance is expected to differ by **up to ~10 %** (a real
VOx absorber rolls off towards 14 µm and the Ge optics add a few percent of ripple); the apparent-
temperature scale (§3.3) is affected far less because inversion uses the same table, but the
contrast-to-NETD relation and the atmosphere weighting inherit the shape error. The band-averaged
derivative ratio, which anchors NETD scaling, is insensitive to the edges (top-hat and cosine agree to
0.1 %). Every LWIR radiometric number in this repo carries this caveat until the curve is replaced.

## Revisit when

A measured relative response for the Boson (or the Halmstad set's Boson 320, open question 3) becomes
available: replace the CSV, keep the provenance history, rerun `make luts`, and re-run the Tier 2
benches; the stale-LUT check (ADR 0012) and the golden slice (M1.11) will flag every dependent result.
