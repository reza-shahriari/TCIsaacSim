# ADR 0076 — Structured cloud in the rendered background, fixed to the sky

**Status:** Accepted
**Date:** 2026-09-14

## Context

For a sky-background sensor the dominant false alarm is not sensor noise. It is a cloud edge: a
warm, high-contrast, target-sized feature with the *same polarity* as the thing being looked for,
since both a drone and a cloud base read warmer than a cold clear zenith. A detector trained on
synthetic imagery with a smooth sky has never seen the thing that will actually confuse it.

MS.3 (ADR 0070) modelled this in M7 — a seeded 1/f^β field thresholded at the weather's own cloud
fraction, with covered pixels reading `ε L_B(T_base) + τ L_clear`. It has been reachable only from
the engine-free scene generator (`irsim.validation.aerial_scene`). Nothing that renders used it.

**A correction to ADR 0073 first, because it is the reason this looked smaller than it is.** That
ADR recorded the infrared background as "the clear-sky profile only". That overstated the gap.
`SkyModel.radiance` has always applied the *uniform* blend `(1 − cε) L_clear + cε L_base`, which is
the **expectation** over the structured field — its own docstring says so. The mean cloud effect
was in every rendered frame all along; what was missing was the structure. That distinction
decides the whole design: this is not adding cloud, it is replacing a mean with a realisation, and
the two must agree on average or every other consumer of the sky model silently disagrees with the
rendered background.

## Options considered

**1. What frame the field lives in.** Only one of three behaves.

  a. **Regenerated per frame.** Flickers: every frame is a different sky. Unusable.
  b. **Fixed to the image plane** — what `SkyModel.cloud_field(shape, seed)` returns, and what the
     engine-free generator uses, where the "camera" never moves. Stable frame to frame, and it
     *travels with the sensor*: a slewing mount carries its clouds along, so the background never
     changes and a tracked target never crosses a cloud edge. That removes exactly the clutter the
     field exists to provide. Fine where the camera is fixed; wrong for the aircraft stage, whose
     pedestal sweeps 50° of azimuth in ten seconds.
  c. **Fixed to the sky** (chosen). A 1/f^β field on an (elevation, azimuth) grid, sampled per ray.
     Stable *and* stationary in the world, so slewing sweeps the camera across it and a target
     crosses in front of it.

**2. Sampling by elevation alone.** Rejected, and worth recording because it is the cheap version:
without azimuth the field can only be indexed by elevation, which bands the sky in horizontal
stripes. That is worse than no cloud — it looks like a deliberate atmospheric layer. A caller with
no azimuth therefore gets the uniform blend, which is merely less detailed rather than wrong.

**3. On by default, or opt-in.** Opt-in, via `cloud_seed`. Turning it on by default would have
moved every committed background by a few kelvin with no commit saying so; without a seed the
background is bit-identical to what it was.

**4. Grid geometry.** Equirectangular in (elevation, azimuth), which stretches structure
azimuthally toward the zenith — a row spans 360° at every elevation. For a sky-target sensor at
low to moderate elevation the distortion is small. Generating on the sphere is the right fix and is
not done here.

## Decision

`irsim.atmosphere.cloud.SkyFixedCloud` + `generate_sky_cloud` build the hemisphere field at half a
degree. `AerialThermalBridge` takes `cloud_seed`, builds the field once from the shared
`WeatherSeries`'s cloud fraction and the environment preset's β, and samples it per ray;
`azimuth_from_rays` supplies the second coordinate and `IrCamera` passes it through.

The field is built **once and never regenerated**: cloud fraction moves on the hour, not on the
frame, and a field that changed with time would flicker. Wind advection — a rotation of the azimuth
axis — is not modelled, because over the seconds a flypast lasts cloud drift is far below a pixel.

## Consequences

**The mean is preserved, and that is the property that makes this an upgrade rather than a
different model.** Averaged over the whole sky the structured field returns the uniform blend it
replaces, measured at ~0.2 % across seeds against a 1 % bound. `SkyModel.radiance` feeds the
elevation LUT, the tilt integral and ADR 0045's reflected term; had the rendered background's mean
drifted from it, the same sky would have been two different skies depending on which code path
asked. The residual is not error but clumping: coverage correlates with elevation, and the clear
radiance varies with it, so which elevations a realisation happens to cover moves the mean slightly.

**Coverage is exact over the sky, not over a frame.** A camera pointed at a gap sees no cloud and
one pointed at a bank sees only cloud — measured, one elevation ring came out at 0.0195 against a
whole-sky 0.0500. That is the behaviour that makes cloud a clutter source rather than a texture,
and it means no single frame can be used to check the cloud fraction.

**What is not modelled.** No wind advection. No multi-layer cloud. No cloud *geometry*, so nothing
is occluded by a cloud and a target always renders in front of one — for a target below the cloud
base that is right, and for one above it is wrong. The base is a single temperature from the LCL of
the shared weather (ADR 0070), so there is no vertical structure within a cloud and no edge
softening; a real cloud edge has a transition region this does not.

**The visible dome should follow.** ADR 0073 left cloud off the dome specifically so the pair would
not show a sky the infrared frame lacked. That reason is now gone, and the two will disagree until
the dome samples the same field. This is the next step, not a defect of this one — but a frame pair
rendered between the two commits shows cloud in infrared and none in visible.

## Revisit when

* The visible dome gains the same field — then the pair agrees again, and the sampling should be
  shared rather than duplicated.
* A scene needs cloud near the zenith, where the equirectangular grid's azimuthal stretch stops
  being small.
* Tier 4 compares cloud spectra against public imagery (ME.4) — the β and the fraction are the
  first two things that comparison should re-fit, and the 1/f^β form itself is the third.
* Anything needs to fly *above* a cloud layer, which needs cloud geometry rather than a background
  field.
