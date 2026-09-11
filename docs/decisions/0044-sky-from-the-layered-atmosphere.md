# ADR 0044 — Sky radiance is the layered atmosphere's column emission; the spec's cos^q form is derived, not authored

**Status:** Accepted
**Date:** 2026-09-11

## Context

§5.3(a) offers T_sky(θ_zen) = T_air − ΔT_clear (1 − cloud) cos^q θ_zen with ΔT ≈ 55–70 K and
q ≈ 0.5–1.0 as the practical LWIR sky. For sky targets the sky *is* the background, and its shape
versus elevation sets the detectability of a small target near the horizon. MS.1 provides the
physical quantity directly: the column's own emission L_path(∞, θ), calibrated to R13 (−40 °C at 15°
in LWIR, warmer in MWIR).

## Options considered

1. Author (ΔT, q) per environment preset and evaluate the power law — cheap, but the layered
   model shows the form is wrong in shape: the sky warms toward the horizon as a saturating
   column (emission ∝ 1 − e^{−τ_z / sin θ}), not as a power of cos θ_zen. Fitted over 5°–90° to
   the layered model the best power law has q ≈ 0.2–0.35 (outside the spec's 0.5–1.0) and a
   maximum error of several kelvin near 5–15° -- exactly where the sky background of a
   low-elevation target matters.
2. Evaluate the layered model per pixel — correct, ~1 ms per quadrature: too slow for a
   640×512 sky in the adapter.
3. **A 1-D elevation LUT over the layered model** (chosen): 181 points at 0.5°, linear
   interpolation, rebuilt per weather time; within 0.5 K of the direct evaluation for every
   preset over 5°–90°; plus a **1-D tilt LUT** for the cosine-weighted sky a tilted plane sees,
   with the azimuth integral in closed form so the tilt LUT is a single 1-D quadrature per tilt.

## Decision

`irsim.atmosphere.sky.SkyModel(atmosphere, environment, band, lut)`:

- `radiance(θ)` / `apparent_temperature_k(θ)`: the elevation-LUT fast path; the adapter fills
  `temperature_k` under the sky mask with it.
- `effective_radiance(β)` and `effective_radiance_from_sky_view(V_s)` (β from V_s = (1 + cos β)/2,
  the unoccluded relation) for the reflected term (M7.13).
- Cloud: L_sky = (1 − c) L_clear + c L_B(T_base) with T_base = T_air until MS.3 supplies the LCL
  base and the structured field; c = 1 gives L_B(T_air) at every angle and tilt (§5.3: overcast
  → T_sky → T_air, the "flat" cloudy-day image).
- `broadband_downwelling()` delegates to M6.5 (ADR 0035): the in-band sky is never used as the
  broadband sky.
- `fit_cos_q()` derives (ΔT, q) from the layered model and reports max/rms error; the environment
  preset's authored (ΔT, q) are validation ranges and an L1 option, never the model.
- The constructor takes the `LayeredAtmosphere` (which holds the one `WeatherSeries`) and exposes
  `.weather`; a scalar T_air or a path is refused (CLAUDE.md #6).

## Consequences

- The sky background and the reflected-term sky come from one object and one weather; MS.3 adds
  cloud structure on top without changing the interface.
- Cost: ~0.4 s to build both LUTs per band per weather time (181 column quadratures); cached.
- The horizon (θ < 0.5°) is evaluated at 0.05° on a flat earth: the last degree above the
  horizon is the least trustworthy part of the model (no refraction, no earth curvature).

## Revisit when

MS.3 lands (cloud base temperature and structure), when a per-pixel slant-path stage 2 needs the
same LUT for the path radiance of *targets* (MS.8), or when refraction/curvature matter (< 2°).
