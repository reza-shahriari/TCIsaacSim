# ADR 0078 — Maritime targets as phase 1b, and the sea as an analytic background

**Status:** Accepted
**Date:** 2026-09-14

## Context

ADR 0003 split the work into phase 1 (sky targets) and phase 2 (ground / automotive). The aerial
lane has now reached the bar it was aiming at: a scene YAML plus one command produces LWIR frames of
a quadrotor mission or an aircraft low pass, with the companion visible frame, the shared weather and
the cloud field. On 2026-09-14 the project owner set the next target — **maritime objects**, built
"with the same idea" as the aerial ones — and placed it *before* ground and automotive.

That ordering is not arbitrary from a physics standpoint. A sea-background target shares almost
everything with a sky-background one: small targets at kilometre ranges, a slant atmospheric path, a
background that is a smooth function of a single angle, and no ground thermal solver. What it adds is
one thing the aerial work stopped short of — the half of the frame *below* the horizon, which today
is the single scalar `T_ground` from the environment preset.

A single scalar is badly wrong there, and wrong in the direction that matters. Water is a Fresnel
reflector, and the viewing geometry is extreme. A camera 20 m above the surface sees the sea at 5 km
at 0.23° depression, which is **89.8° incidence**; at 500 m it is 2.3°, still 87.7°. Water's
band-effective emissivity at 80° is 0.70 and keeps falling. So nearly all visible sea is a mirror of
the sky, not a 290 K blackbody, and its apparent temperature runs from roughly SST looking straight
down to within a couple of kelvin of the sky at the horizon. A vessel warmer than the sky but cooler
than the water therefore appears **dark against near water and bright against far water**, with a
contrast null somewhere in between. Detection performance across range is dominated by where that
null sits, and a constant-temperature sea cannot produce it at all.

## Options considered

1. **Render the sea as displaced water geometry.** A tessellated, wave-displaced plane with a water
   material, letting the renderer produce per-pixel normals. M10.1 established that the `normals` AOV
   is float32, full-resolution and world-space, so unlike the earlier M2.4 reading this is *not*
   blocked by AOV quality. It is blocked by sampling: at 3 km a Boson pixel spans 2.6 m and contains
   thousands of independent wave facets. One normal per pixel is not a coarse version of the right
   answer — it is a different quantity. Reproducing the correct pixel value would need the facet
   distribution resolved *and* supersampled, over a surface extending to a horizon 6–20 km away.
   Cost rises without bound and the answer stays biased.

2. **Analytic sea profile, the way MS.2 does the sky.** Apparent sea temperature as a function of
   depression angle: ε_B(θ) Lb(T_skin) + (1 − ε_B(θ)) ⟨L_sky⟩, with the reflected sky integrated over
   a Cox–Munk slope distribution driven by the shared weather's wind. The sub-pixel facet statistics
   are then handled in closed form, which is exactly the quantity the sensor integrates. Same shape
   as the existing sky path, same reason (ADR 0014/0060: a computed background also avoids the fp16
   colour AOVs), and it costs one evaluation per pixel.

3. **Hybrid: analytic beyond a crossover range, geometry in the near field.** Correct in principle —
   close in, a pixel really is smaller than a wave. Rejected *for now* because nothing in the target
   application looks at water within tens of metres, and a seam between two models is a defect
   generator. The crossover is recorded here so it can be added without re-deciding the far field.

## Decision

**Maritime work is phase 1b**, between ADR 0003's phase 1 and phase 2: sky → sea → ground. It gets
its own milestone (**MM**) in the roadmap, and it promotes M7.3 and M7.5 out of phase 2, because a
sea surface is the one background whose emissivity cannot be a scalar — §4.2 refuses Level C for
water by name. M7.4 (branch-safe complex Fresnel) already landed early for this reason.

**The sea is an analytic background**, option 2. No water geometry in the maritime stage. The
horizon is a computed elevation, and below it the background comes from a `SeaModel` evaluated per
pixel, in the same place the `SkyModel` is evaluated above it.

Deferred deliberately: **wakes, whitecaps and foam** (each is a distinct surface class with its own
emissivity and temperature, and none is needed for the background to be right), and **MWIR solar
glint**, which belongs to M11.7's specular lobe. MM.2's slope distribution is written as its own step
precisely so glint reuses it instead of growing a second wave model.

## Consequences

Easy: a maritime scene costs the same per pixel as an aerial one, needs no new AOV, and runs in the
engine-free core, so the whole sea model is testable with plain `pytest` — including the identity
that matters (T_sky ≡ T_sea ⇒ T_app = T_sea to 1 mK at every angle and wind), which no amount of
looking at a rendered frame would verify.

Hard: anything that requires a *spatial* realisation of the surface. A wake, a glint path, a
reflection of the target in the water beneath it, and any per-wave structure are all outside this
model. The sea will be smooth in appearance, varying only with depression angle plus whatever
statistical texture is added on top of the profile.

**Error introduced.** The reflected sky is integrated over the slope distribution but the surface is
still treated as locally flat for *geometry*: no shadowing or multiple reflection between wave faces.
Published treatments put the shadowing correction below a few percent of reflectance for incidence
angles under about 85° and rising steeply beyond it — which is precisely the regime that dominates a
low camera's view of distant sea. **This error is not currently bounded for the near-horizon band**,
and MM.3's verification does not test it. That is a known gap, stated rather than hidden: the first
comparison against public maritime imagery (MM.8) should be read with it in mind, and a
Smith-type shadowing factor is the cheapest next fidelity step if the horizon band reads too warm.

Neglecting salinity in the optical constants is bounded separately in ADR 0079.

## Revisit when

- A target must be seen at closer than ~50 m of water, where a pixel is smaller than a wave and
  option 3's near field becomes the honest model.
- MM.8 shows a systematic bias in the near-horizon band against public imagery — the shadowing
  term above is the first suspect.
- MWIR arrives (M11) and glint becomes a first-order signal rather than a deferred one.
- Wakes become a detection cue in their own right, which is a known maritime discriminator and
  would reopen option 1 for a bounded patch of surface behind a vessel.
