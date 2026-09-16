# 0084 — The illumination bundle on the Isaac render path

Date: 2026-09-15
**Status:** Accepted
Roadmap: M10.22 (§5.2, §5.4, §5.5)

## Context

M11.2 built the illumination bundle, M11.3 the reflected-solar term and M11.4 the night sky.
`run_frame` reads them out of the plane dict as `l_sun` and `l_night`, `Illumination.for_regime`
gates them by band, and all three are unit-tested.

**Nothing in `irsim_isaac` ever wrote those planes.** Every Isaac render this project has produced
was emission only. For LWIR that is right to within 0.35 %. For the reflective bands it is not an
approximation at all:

| band | sunlit / emission-only, dry asphalt at 300 K, 61° sun |
|---|---|
| NIR | 1.1e15 |
| SWIR | 3.6e7 |
| MWIR | 1.17 |
| LWIR | no bundle — the regime enables no source term |

A NIR frame rendered without this is black. This is the same defect ADR 0077 and ADR 0082 record,
for the third time: a mechanism exists, is tested, and the layer above cannot reach it. It only
became visible when a fourth band was configured and the renders were asked for in it.

## Decision

1. `irsim_isaac.pipeline.illumination_isaac` builds a frame's `l_sun` and `l_night` planes from the
   scene's own site, clock, weather and atmosphere preset, and `IrCamera` adds them to the plane
   dict beside the G-buffer — *outside* the M0.6 contract, as `radiance_behind` already is, because
   they are stage-1 inputs and not geometry.
2. **One sun.** The direction is the same expression `irsim_isaac.stage.add_sky_dome` aims the USD
   `DistantLight` along, so the radiometry and the shadows in the companion visible frame cannot
   disagree about where the light comes from. Computed separately, the two would drift and nobody
   would notice until they were overlaid.
3. `SceneIllumination.for_camera` returns **`None`** for a band whose regime enables no source
   term, and `IrCamera` *refuses* a bundle attached to an emissive band. An LWIR render therefore
   takes exactly the code path it took before and is bit-identical, rather than merely close.
4. The night term is built only when the scene names an environment preset. §5.5's airglow level
   varies by two orders of magnitude between sites and nights, so a default would be a number
   nobody chose.

## Consequences

**Cast shadows are not modelled.** `shadow` is 1 everywhere, so the only shadowing is
*self*-shadowing through max(0, n·s): a surface turned away from the sun gets nothing, a surface
facing it gets all of it, and a surface standing in another object's shadow is lit anyway.

* For the aerial scenes this is **exact** — there is nothing above a drone to shadow it.
* For a vessel it is optimistic on the shaded side of the superstructure.
* For a ground scene with buildings it would be wrong, and the reflective bands are where it would
  show most.

Getting it right needs an occlusion query along the sun direction, and the renderer on this build
delivers no working occlusion AOV (ADR 0014 addendum). The honest interim is to say so in the
module, in the README's limitations and here, rather than to render a plausible shadow nobody
computed. The G-buffer contract already carries an optional `shadow_mask` for when one exists.

**The atmosphere is consulted for the solar slant path**, via `solar_transmittance(preset, band,
zenith)`, so a low sun is dimmer than the cosine alone predicts. That is asserted rather than
assumed: the test compares a 8° sun against the cosine-only prediction and requires it to be less.

## Alternatives considered

* **Derive a shadow mask from the rendered RGB.** The companion visible frame does have ray-traced
  shadows in it. Reading a shadow mask off a tone-mapped, exposure-scaled colour image would make
  the infrared radiometry depend on the visible pass's exposure heuristic, which is exactly the
  coupling ADR 0073 keeps the two frames free of.
* **Put `l_sun` in the G-buffer contract.** It is not geometry, and the contract is what the
  renderer must deliver. Widening it would tell every future adapter to produce a radiance.
