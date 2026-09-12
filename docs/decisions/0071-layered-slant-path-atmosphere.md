# ADR 0071 — Layered slant-path atmosphere: an exponential-sum band model, sky = column emission

**Status:** Accepted (MS.1 + MS.6)
**Date:** 2026-09-11

## Context

The grey γ_B per band (ADR 0048) is valid to ~500 m on horizontal paths and says nothing about the
sky. The sky-target application needs slant paths of kilometres, the sky radiance versus
elevation as the background (§5.3: a clear dry LWIR sky is −40 °C at 15° elevation while the
MWIR sky is far warmer [R13]), and a curve of growth: M8.7 showed the grey model 9 % low at 1 km
for a two-level LWIR spectrum and 42 % low for a MWIR band with an opaque CO₂ core. MODTRAN is
not available; the model must be calibrated from the §7.2 table and the public R13 anchors.

## Options considered

1. Grey γ_B with a scale height — wrong curve of growth (a single exponential cannot saturate).
2. A correlated-k / band-model radiative transfer code — the right tool (MODTRAN's own approach),
   but it needs line data and a large validation effort; phase 2 if a MODTRAN-class run becomes
   available (its outputs would *replace the class parameters*, not the model's shape).
3. **Exponential sum over spectral classes** (chosen): per band, τ_B = Σ_k w_k e^{−od_k} with one
   term per spectral class (window, water-vapour lines, band-edge lines, an opaque CO₂ core). The
   weights are the Planck-weighted shares of the classes in the sensor's response (300 K for the
   emissive bands, 5800 K for the reflective ones); the surface extinction per class is a
   multiplier on the preset's molecular γ_mol,B(w) ("water" classes, scale height 2 km) or an
   absolute value ("air" classes: CO₂, scale height 8 km); the aerosol part is grey (scale height
   1.2 km). The non-opaque classes are rescaled by one factor so the **horizontal 200 m**
   transmittance equals the grey preset's at the current weather -- M8.3's table stays the
   anchor; opaque classes (CO₂ 4.3 µm, H₂O 2.7 and 1.4 µm) sit on top with fixed extinction
   (ADR 0048 read the table for the transparent part of the band).

## Decision

Option 3 in `irsim.atmosphere.layered`:

- Ray geometry is flat-earth, h = s sin θ. Column integrals of the exponential profiles are
  analytic, so transmittance is closed-form along any ray; path radiance
  L_path = Σ_k w_k ∫ γ_k(h) τ_k(s) L_B(T(h)) ds is integrated **per class in its own optical-depth
  coordinate** (∫ L_B e^{−u} du with s(u) from the analytic od_k(s)), which resolves an opaque
  class emitting from the first metres and a window emitting from kilometres up alike. A uniform
  Simpson grid in s was tried first and gave a MWIR sky *warmer than the air* -- the spike of an
  opaque class was aliased; the optical-depth quadrature reproduces the isothermal invariance to
  1e-7 at 5° and 1e-11 at zenith.
- T(h) = T_air − Γ min(h, h_tropopause) with the preset's lapse rate (USSA 6.5 K/km, tropopause
  11 km); the profile block gains `aerosol_scale_height_m` (1200), `air_scale_height_m` (8000)
  and `tropopause_m` (11000) with defaults (ADR 0017).
- **The sky is the column emission**: L_sky,B(θ) = L_path,B(∞, θ); space contributes nothing.
- At θ = 0 the model is per-term horizontal Beer–Lambert (closed form, any plane shape) and with
  one class it is M8.1 exactly (the visible band has one class, so there the layered model *is* the
  grey model, tested to 1e-12). Stage 2 accepts a `LayeredAtmosphere` and uses this per-term
  horizontal form per pixel; per-pixel slant paths (an elevation plane) arrive with MS.8/M10.5.

**Calibration against R13** (Sensors 21:7067; Tucson, clear, low humidity; FLIR T1020 7.5–14 µm,
TELOPS M1k 2.2–5.5 µm). With the US Standard preset at 288.15 K, RH 0.20, 23 km visibility and
the LWIR window multiplier 0.3:

| elevation | 0.5° | 2° | 5° | 10° | 15° | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|---|---|---|
| LWIR 7.5–14 µm, °C | +11.9 | −4.3 | −20.8 | −32.9 | **−39.7** | −50.4 | −55.9 | −59.2 | −61.6 |
| MWIR 2.2–5.5 µm, °C | +14.4 | +11.9 | +8.6 | +5.0 | **+2.0** | −4.4 | −8.3 | −10.6 | −12.3 |

R13's LWIR anchor (−40 °C at 15°) is reproduced; the sky approaches T_air at the horizon and is
coldest at zenith. R13's MWIR value at 15° (> +10 °C) is a **midday** measurement of a 2.2–5.5 µm
band and includes sunlight scattered into the line of sight in its short-wave part; this model is
thermal-only (ADR 0050 #3) and gives +2 °C -- the right ordering (42 K warmer than LWIR) and the
night-time value; the daytime MWIR sky needs the solar-scatter term of M11. The roadmap's
"T_air 288 K, RH 50 %" for this anchor is not what R13 measured ("low humidity"); RH 20 % is used.

Note also that the roadmap's convergence statement ("a 300 K target at d → 20 km converges to
L_sky") holds only for an opaque column: a target beyond the atmosphere is still seen through the
window, so L'(d) → τ(∞) L_t + L_sky(θ); the test asserts that limit and the monotone decrease of
the excess over the sky.

## Point-target injection below one native pixel (MS.6)

A target of area A_t at range R fills φ = A_t f² / (R² A_pix) of a native pixel. Rasterising it on
the k× supersampled G-buffer gives, for a square target swept through 16×16 sub-pixel phases at
k = 4 (`tests/unit/test_point_target.py`, measured 2026-09-12):

| size (native px) | samples across | mean error | rms error | min | max |
|---|---|---|---|---|---|
| 0.5 | 2.0 | +27 % | 44 % | 0 % | +125 % |
| 1.0 | 4.0 | +13 % | 21 % | 0 % | +56 % |
| 1.2 | 4.8 | −2 % | 13 % | −31 % | +9 % |
| 2.0 | 8.0 | +6 % | 10 % | 0 % | +27 % |
| 4.0 | 16.0 | +3 % | 5 % | 0 % | +13 % |

The phase mean is nearly unbiased from ~1.2 px up, but the *per-phase* flux of a sub-pixel or
one-pixel target is wrong by tens of percent -- a drone's signal would flicker with its sub-pixel
position. Decision: below **φ = 1** (one native pixel) the target is injected analytically in the
fill-fraction form, ΔL = φ Σ_k w_k τ_k(R, θ) [L_t − L_beyond,k(R, θ)], with the excess power
Ω_eff(F) τ_opt A_d ΔL (the aperture factor imported from `irsim.optics.aperture`; F/1 vs F/2 gives
3.4, not 4.0). L_beyond,k = (L_sky,k − L_path,k(R))/τ_k(R) is the column emission beyond the
target per spectral class, so the path between camera and target is counted once and a target at
the effective beyond-radiance has zero excess at every range. The excess is splatted bilinearly at
the sub-pixel position on the k× grid before stage 3, so the PSF and the same box downsample the
resolved side uses spread it; the splat conserves flux to 1e-6 at every phase and the phase-mean
of the rasterised side matches the injected flux within 1 % at the handoff. Above 1 px the
rasteriser is used as is; the table says a k of 4 leaves ±10 % per-phase flicker on 2 px targets,
so scenes that care about 1–3 px targets should render at k = 8 (M10.x).

## Consequences

- Full-band transmittances at 200 m now include the opaque classes: US Standard MWIR 3–5 µm
  τ(200 m) = 0.78 (grey: 0.945 for the transparent part), SWIR 0.83 (opaque 1.4 µm band). This is
  the physically expected band-integrated value for a top-hat response; a real camera's response
  weights the classes differently and the model uses it when given.
- Class edges are spectroscopy (H₂O 0.94/1.1/1.4/2.7/6.3 µm and the rotational band; CO₂ 4.3 µm
  and the 4.45–5.0 µm complex); multipliers are ESTIMATED except the two calibrated windows.
  Expect ±5 K on sky apparent temperatures at mid elevations and larger errors near the horizon
  (flat earth, no refraction, no aerosol vertical structure beyond a scale height).
- `fit_exponential_sum` and `exponential_sum_from_piecewise` are the tools to replace the class
  parameters from a MODTRAN run or a line-by-line calculation without touching the kernel: the
  M8.7 two-level and CO₂-notch spectra are reproduced to round-off at 5 km (the grey model was
  off by up to 42 %).

## Revisit when

A MODTRAN-class transmittance/radiance table is available (replace the class multipliers per
preset), when a daytime MWIR/SWIR sky needs scattered sunlight (M11), or when paths below 2°
elevation or beyond 50 km need spherical geometry and refraction.
