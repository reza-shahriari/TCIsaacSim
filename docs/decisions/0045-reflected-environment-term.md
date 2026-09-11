# ADR 0045 — Reflection fidelity: the sky-view blend; V_s = occlusion·(1 + n·up)/2; L_ground = L_B(T_ground); the tilt LUT

**Status:** Accepted
**Date:** 2026-09-11

## Context

§5.3 makes the case: in LWIR the reflected term is only (1 − ε) ≈ 0.05–0.2 of the signal, but
the clear sky is 40–60 K colder than the air, so low-emissivity surfaces (paint at grazing angles,
glass, bare metal) swing by tens of kelvin between clear and overcast. Three fidelity levels are
offered: (a) the sky-view-factor blend, (b) a low-resolution irradiance cubemap, (c) path tracing
in the radiance domain. The renderer hands the pipeline a per-pixel sky-view factor and nothing
else about the environment.

## Options considered

1. Emission only (the M3 form) — the cold-roof and mirror-like-metal phenomena never appear.
2. **§5.3(a): L_env = V_s L_sky,eff + (1 − V_s) L_ground** (chosen for phase 1).
3. §5.3(b)/(c) — needed for urban scenes with building/vehicle inter-reflections; phase 2.

## Decision

- Stage 1 becomes L = ε L_B(T_s) + (1 − ε) L_env with (1 − ε) = ρ + τ: a transmitting material
  passes the environment behind it (L_behind = L_env until a second ray exists, ADR 0046).
- **V_s source.** The G-buffer's `sky_view_factor` is `occlusion · (1 + n·up)/2`: the exact
  cosine-weighted sky fraction of a plane of tilt β (1 facing up, 1/2 for a wall, 0 facing down)
  times an occlusion factor. Raw ambient occlusion alone is wrong for walls, which see half the
  sky unoccluded. `irsim.pipeline.environment.sky_view_factor` is the helper adapters and fixtures
  call; a seeded cosine-weighted Monte Carlo agrees with it to 1e-3.
- **L_sky,eff** comes from the SkyModel's tilt LUT (ADR 0044) indexed through the unoccluded
  relation β = arccos(2 V_s − 1): occlusion reduces the weight of the sky but not the *shape* of
  the sky the plane sees -- an approximation accepted for phase 1 (an occluded wall sees the
  upper part of the sky, which is colder; the error is a fraction of the (1 − ε) term).
- **L_ground = L_B(T_ground)** with T_ground from the environment preset's ground mode: `air`
  (the shared weather's T_air), `fixed` (authored), `solver` (M6.12's environment solver; refused
  until it exists). One temperature for the whole ground.
- The Scene builds the LayeredAtmosphere and one SkyModel per band on the same WeatherSeries when
  the scene config names an `environment_preset` (scene schema v2); the pipeline config refuses a
  sky model whose weather, band or radiance form differs from its own.

## Consequences

- The isothermal enclosure identity holds to 1 mK for every emissivity and sky-view factor, the
  cold-roof drop under a clear sky is (1 − ε)(L_B(T_air) − L_sky,eff)/(∂L/∂T) to 0.05 K, and
  bare aluminium reads > 10 K colder than paint -- the three phenomena the term exists for.
- Not modelled: inter-reflections between scene objects, a ground with structure, the reflected
  solar term (M11), specular sky reflections (the term is hemispherically averaged; the specular
  lobe of §4.3 arrives with the roughness model).

## Revisit when

Urban scenes need §5.3(b)'s irradiance cubemap, or the renderer can supply a per-pixel horizon
map so the occluded tilt LUT can be evaluated exactly.
