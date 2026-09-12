# ADR 0070 — Cloud clutter: LCL base from the weather, ε_cloud = 1 − τ_cloud, 1/f^β structure (authored, bounded only by display-domain statistics)

**Status:** Accepted
**Date:** 2026-09-12

## Context

The spec has no cloud section; §5.3(a) says only that T_sky → T_air under thick overcast. For
sky targets the cloud is the clutter: its temperature contrast against the clear sky (tens of
kelvin at zenith) and its spatial structure set the false-alarm rate of a detector far more than
any radiometric detail. Nothing in the pipeline could produce a non-uniform sky.

## Options considered

1. Leave the sky uniform (MS.2's blend) — no clutter, unrealistically easy detection.
2. A physical cloud model (microphysics, radiative transfer through droplets) — far beyond what the
   weather file can drive and unvalidatable without a cloud dataset.
3. **A bounded clutter model** (chosen): the physics that the weather *does* constrain (base
   height and temperature, thick-cloud emissivity), one authored optical parameter (τ_cloud) with
   its complement derived, and a seeded scale-free field for structure whose only physical check
   is the display-domain clutter statistics of real imagery (ME.5).

## Decision

`irsim.atmosphere.cloud` + `SkyModel` (MS.3):

- **Base height** z_LCL = 125 m K⁻¹ (T_air − T_dew) (Espy's rule, Lawrence 2005) from the same
  WeatherSeries the atmosphere and the thermal solver read (T_air, RH; the dew point by inverting
  M8.2's Magnus form), clamped to the environment preset's [0, 8 km]; RH = 1 puts the base at the
  surface (the sky reads T_air, the §5.3 overcast limit).
- **Base temperature** T_base = T_air − Γ_env z_LCL with the atmosphere preset's lapse rate; the
  cloud pixel radiates ε_cloud L_B(T_base) + τ_cloud L_clear(θ) per band through the LUT, with
  τ_cloud authored in the environment preset (default 0: thick) and **ε_cloud = 1 − τ_cloud
  derived** (CLAUDE.md #4 applied to the cloud).
- **Structure** from a seeded Gaussian field with PSD ∝ f^{−β} (default β = 1.8), thresholded at the
  (1 − c) quantile so exactly the weather's cloud fraction is covered; the uniform MS.2 blend
  (1 − c ε) L_clear + c ε L_B(T_base) is its expectation and remains what the tilt LUT (the
  reflected term) and the LUT-based `radiance()` use.
- The generator carries a **self-test** (the synthesised PSD slope equals β within 0.1); the
  physical bound is the display-domain clutter slope of real cloud regions (ME.4/ME.5), against
  which β and the coverage statistics will be tuned — that assertion is in the test file, skipped
  with the reason until the reference bands exist.
- Not modelled: sun-lit cloud (MWIR/SWIR daytime, phase 2), cloud sides and multiple layers, cloud
  motion, the temperature gradient across a cloud edge (edges are sharp at the pixel scale).

## The overcast limit: T_base, not T_air (a deliberate deviation from §5.3 a)

§5.3(a) says "under overcast, T_sky → T_air". That is the placeholder this model replaces: a cloud
radiates at the temperature of **its base**, which is above the ground, so a thick overcast sky is
colder than the surface air by Γ_env z_LCL -- 6.5 K for a 1 km base, five times a Boson's NETD and
far too large to round away. The contract is therefore:

    cloud_fraction = 1, τ_cloud = 0  ⇒  L_sky(θ) = L_B(T_air − Γ_env z_LCL)  at every θ and tilt

and the spec's T_air limit is recovered exactly when the air is saturated (RH = 1 ⇒ z_LCL = 0),
which is the physically correct special case: cloud touching the ground is fog at air temperature.
Tests that need a "sky at T_air" reference (the isothermal enclosure of M7.13, its cold-roof
control, MS.2's overcast identity) now ask for cloud = 1 **with RH = 1** rather than assuming the
two coincide. Recorded as spec issue T20.

## Consequences

- Cloud = 1 with a ~1 km base at T_air 288 K reads 281.6 K against a clear zenith sky near 230 K:
  the > 20 K edge contrast a detector must survive.
- A seed makes a scene's cloud field reproducible; the field is not a physical cloud and must not
  be used for anything but clutter statistics.
- The environment schema is v2 (optional `clouds:` block, ADR 0017).

## Revisit when

ME.5 lands (tune β and the threshold statistics to the reference bands; possibly replace the
Gaussian threshold with a log-normal field), or when a scene needs sun-lit or multi-layer cloud.
