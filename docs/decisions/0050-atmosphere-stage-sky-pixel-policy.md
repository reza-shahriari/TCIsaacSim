# ADR 0050 — Stage 2: path radiance at a uniform T_air, sky pixels to the sky model, no scattered sunlight

**Status:** Accepted
**Date:** 2026-09-11

## Context

Stage 2 of the pipeline (§13.4) applies the atmosphere per pixel: L' = τ(d) L + (1 − τ(d)) L_B(T_air).
Three choices had to be made: what temperature the air radiates at along the ray, what happens to
pixels that hit no geometry (the sky), and whether scattered sunlight along the path is modelled.

## Decision

1. **Uniform T_air along the ray.** The path radiance uses the surface air temperature of the
   shared `WeatherSeries` at the frame time (`PipelineState.t_s`), for every pixel. This is exact
   for horizontal paths near the ground (the §7.2 regime, valid to ~500 m) and the L2 form the spec
   asks for. For slant paths to aerial targets the air cools with height and the path radiance is
   over-estimated; MS.1's layered model replaces the scalar γ, L_air pair with per-ray integrals
   without changing the kernel's shape.
2. **Sky pixels pass through untouched.** The G-buffer's `sky_mask` (agreed with the Isaac lane:
   the renderer reports +inf distance and id 0) marks them; their `temperature_k` is the apparent
   sky temperature T_sky(θ) filled by the sky model (MS.2), which by definition already includes
   the whole atmospheric column. Stage 1 treats them as blackbody-equivalent; stage 2 returns them
   bit-identical in both the Beer–Lambert and the constant-τ paths. There is no distance sentinel
   in the contract: `distance_m` is 0 under the mask so that a consumer ignoring the mask sees τ = 1.
3. **No scattered sunlight.** Daytime path radiance in MWIR/SWIR/NIR from sunlight scattered into
   the line of sight is not modelled (Appendix A #2). At 200 m it is negligible against thermal
   emission in LWIR and small in MWIR; at kilometres in haze it can be a few percent of a target's
   MWIR signal and dominates SWIR/NIR path radiance. This is the phase-1 limitation with the
   largest effect on daytime MWIR/SWIR sky-target contrast, and it is documented rather than
   approximated.
4. **Where it runs.** On the k× supersampled grid, after band radiance and before the PSF: the
   atmosphere is a property of each ray, the blur is a property of the optics. Both γ and
   L_B(T_air) are evaluated with the pipeline's own LUT so an isothermal scene stays isothermal to
   round-off (the invariance test) and the constant-τ L1 path keeps the path-radiance term.

## Consequences

- Known answer through the whole chain: 310 K blackbody, T_air 290 K, τ 0.8, 8–12 µm →
  T_app = 306.303 K (within 1 mK, bisection cross-check); isothermal G-buffers read T_air at 10 m
  and 5 km within 1 mK.
- The stage adds no per-frame state; time enters only through `PipelineState.t_s`.
- Error budget: uniform-T_air over-estimates slant-path radiance (order 1 K apparent at 5 km,
  15° elevation, clear mid-latitude air; MS.1 quantifies it); no scattered sunlight (above).

## Revisit when

MS.1 (layered slant-path atmosphere) lands — replace the scalar pair with per-ray τ and L_path —
and when a daytime MWIR/SWIR sky-target scene needs scattered-sunlight path radiance (M11's
solar-path work).
