# ADR 0067 — The reflection lobe: GGX, an explicit split, and a capped glint

**Status:** Accepted
**Date:** 2026-09-15

## Context

§4.3 makes a strong qualitative claim and gives no model: an infrared reflection lobe is
"generally **narrower and stronger**" than the same material's visible one, because roughness that
scatters 0.5 µm light diffusely can be optically smooth at 10 µm. So painted bodywork and glass
show near-mirror sky reflections in LWIR — which is why a car roof reads far below ambient on a
clear night — while asphalt and concrete stay near-Lambertian. §5.4 gives solar glint as
`L = R(θ_i) E_B τ / Ω_sun` and nothing about how it spreads.

`roughness_per_band` has been authored in every material YAML since M7.9 and, like
`cold_shield_efficiency` before M11.5, nothing read it.

## Decisions

### 1. GGX, not Beckmann

Both are normalised microfacet distributions and both are defensible. GGX (Trowbridge–Reitz) is
chosen for its power-law tails: a thermal reflection's *tail* is what puts a faint warm smear
around a hot reflection instead of a hard-edged disc, and that smear is the part a detector
trained on synthetic imagery will or will not have seen. Roughness maps to the NDF width as
**α = roughness²** (the Disney/Burley convention), which is what makes the library's authored
numbers behave the way their own comments intend: glass 0.03 → α = 9e-4, a near mirror; asphalt
0.80 → α = 0.64, essentially Lambertian. Using roughness directly as α would leave glass visibly
blurred and compress every interesting material into the top of the range.

### 2. The specular/diffuse split is explicit — because the alternative was tried and measured

The tidy version of this module carries one GGX kernel and lets roughness → 1 produce the diffuse
answer, with no invented weight anywhere. It does not work, and the reason is worth recording:
**a microfacet lobe does not converge to Lambertian.** At α = 1 GGX is uniform in the half-vector,
and the resulting reflection kernel sits a **total-variation distance of 0.30** from the cosine
hemisphere — reading a test sky 4.7 units warmer out of 280, 1.7 % — and it does not improve as α
grows further. A test measures exactly this, so the decision stays visible rather than becoming
folklore.

So:

    K(ω_i) = w_s · K_GGX(ω_i) + (1 − w_s) · cos θ_i / π,   ∫K dω_i = 1,   w_s = (1 − roughness)²
    L_reflected = ρ · ∫ K(ω_i) L_incident(ω_i) dω_i

Energy is conserved by construction — the reflected coefficient is **exactly ρ for every
roughness, to 1e-12** — and both endpoints are exact rather than fitted: roughness 0 reads
L_sky(mirror direction) to 1e-6, roughness 1 reads the M7.13 V_s blend to 1e-6.

The **(1 − r)²** shape is a modelling choice. Its check is §4.3's own prose, reproduced from the
library's authored values without being fitted to them:

| material | LWIR roughness | specular share |
|---|---|---|
| glass windshield | 0.03 | **94 %** |
| painted bodywork | 0.12 | **77 %** |
| asphalt | 0.70 | **9 %** |

That is "painted bodywork and glass show near-mirror sky reflections in LWIR … asphalt and
concrete stay near-Lambertian even in LWIR", as numbers. A linear weight would leave asphalt 30 %
specular and visibly mirror-like, which §4.3 says it is not.

### 3. The specular quadrature is centred on the mirror direction, and that is not an optimisation

The first version put both parts on one grid tied to the surface normal. At α = 1e-4 that grid
cannot see a lobe 1° wide at all, and a glass windshield came out **0.09 units short** of the
exact mirror answer — a bias, not noise, and in the direction that makes shiny surfaces look
less shiny. The specular part is now quadratured in a cone about the mirror direction, sized to
hold 99.99 % of the GGX half-angle distribution; the diffuse part keeps the normal-aligned cosine
quadrature, where it is exact. Below-horizon directions get zero weight and the remainder is
renormalised — the same G = 1 simplification as the kernel itself, applied consistently.

### 4. The sun is not a special case, and the glint is capped

The sun arrives through the same kernel, as a small very bright patch of the incident field:
`L_glint = ρ w_s · min(K(ŝ)·Ω_sun, 1) · E_B τ_sun / (Ω_sun cos θ_s)`, reducing to §5.4's form in
the mirror limit.

The **cap is physics, not a guard**. A microfacet NDF evaluated at its peak claims 1/(πα²) sr⁻¹,
which for glass (α = 9e-4) is 4e5 — enough to render a glint four orders of magnitude brighter
than the sun that caused it. `min(K·Ω_sun, 1)` says a mirror shows you the sun, not something
brighter, and it is what makes the smooth end of the roughness range physical instead of divergent.

### 5. The incident field is sky **and ground**, with `up` as a separate argument

`incident_field` takes the **world** up axis, not the surface normal. A near-vertical windshield
or a ship's flank reflects ground over half its lobe, and a model that sampled only the sky would
render it far too cold. Passing the normal by mistake would give a vertical panel a sky in every
direction; the argument exists to make that mistake explicit, and a test pins it.

## What this measured

MWIR glint, 60° sun, camera in the specular direction, ρ = 0.10, τ_sun = 0.82, against a 300 K
scene (Lb = 2.04 W m⁻² sr⁻¹) — the §5.4 headline as a roughness sweep:

| roughness | glint / Lb(300 K) | apparent temperature |
|---|---|---|
| 0.03 (glass) | **1.4e4** | > 1000 K (off the top of the LUT) |
| 0.12 (paint) | 6.3e2 | 698 K |
| 0.30 | 5.6 | 357 K |
| 0.60 | 0.13 | 252 K — *below* ambient, i.e. invisible |

Glint is a smooth-surface phenomenon and it is gone by roughness 0.3. Off the specular direction it
is below 1e-6 of the peak; at night, or with the sun below the horizon, it is **exactly** zero.

In LWIR, a ρ = 0.9 flat mirror at 30° elevation under a clear US-Standard sky reads more than
**30 K below its own 300 K surface temperature**, and matches `Lb⁻¹(ε Lb(T_s) + ρ Lb(T_sky(30°)))`
to under 0.1 K. That is the car-roof effect, and it is the reason §4.3 exists.

## Consequences

* `irsim/materials/lobe.py` is engine-free and is the **oracle**: ~2·40·64 direction evaluations
  per surface, which is not what a renderer would run. A cubemap probe or an importance-sampled
  estimator is; this is what those get checked against.
* `irsim/pipeline/specular.py` binds it to the `SkyModel` and to M11.3's solar band irradiance.
  Neither is wired into `run_frame` yet: stage 1's reflected term is still the M7.13 V_s blend,
  which is exactly the roughness → 1 limit of this model, so nothing rendered so far changes.
  Wiring it per pixel is a later step and needs the per-pixel **normal**, which the G-buffer
  currently carries only as `normal_dot_view`.
* Fresnel R(θ_i) is *not* multiplied in: ρ from the material's Kirchhoff closure is the magnitude
  and the lobe is only the distribution. Using both would count the reflectance twice.

## Revisit when

* Stage 1 wires the lobe per pixel. That needs a world-space normal in the G-buffer, and it is
  where the oracle's cost stops being acceptable.
* A measured BRDF exists for any material in the library. `(1 − r)²` is the first thing to refit.
* Shadowing/masking matters — i.e. at grazing view angles on rough surfaces, where G = 1 and
  renormalisation start to differ from Smith's G noticeably.
