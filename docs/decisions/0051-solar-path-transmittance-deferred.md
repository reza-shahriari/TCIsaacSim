# ADR 0051 — Solar-path transmittance: a plane-parallel Bouguer stub now, the real path with M11

**Status:** Accepted
**Date:** 2026-09-12

## Context

§5.4's reflected-solar term needs the top-of-atmosphere irradiance attenuated along the **solar**
path, τ_atm^sun(λ), which is a different path from the camera-to-target one the rest of the
atmosphere model computes: it runs from the sun to the surface through the whole column, at the
solar zenith angle, and it is spectrally weighted by the solar spectrum rather than by Planck at
scene temperature. M11 implements that term. M8.8 needs *something* on the preset schema now, so
that the aerial and environment presets carry a complete band-key set and so the ordering
(clear passes more sun than haze passes more than fog) is pinned by a test before anything
consumes it.

Note also that the weather file already carries **surface** DNI and DHI (ADR 0032/0034), so this
transmittance must never be used to attenuate the file's irradiance a second time. Its job is the
*spectral shape and band weighting* of the reflected term.

## Options considered

1. Leave the field out until M11 — then M11 changes the preset schema for all seven files and
   every scene config hash at once, and nothing checks the preset table's ordering in the meantime.
2. Compute τ_sun from the existing γ_B and a column height — wrong: γ_B is fitted to a
   *horizontal, near-surface* 200 m path (ADR 0048) and the column is neither at surface density
   nor at surface humidity. Extrapolating it vertically would be a fabricated number wearing the
   authority of a fitted one.
3. **An explicit, clearly-labelled ESTIMATED zenith transmittance per band, with the
   plane-parallel Bouguer law for the angle** (chosen).

## Decision

`AtmospherePreset.solar.zenith_transmittance` — one value per band, in (0, 1], authored per preset
and marked ESTIMATED with its condition in the file. The angular dependence is Bouguer's law for a
plane-parallel atmosphere:

    τ_sun(θ_zen) = τ_zenith ** (1 / cos θ_zen)     — airmass = sec θ_zen

`irsim.atmosphere.extinction.solar_transmittance` / `airmass` implement it. Zenith angles beyond
**85°** are refused rather than extrapolated: sec θ diverges at the horizon and the plane-parallel
form is already ~2 % off the refracted airmass by 80°. The atmosphere schema goes to **version 2**
(ADR 0017's policy: an optional-shaped field with authored values, all seven presets updated).

## Consequences

- `τ_sun(60°) = τ_zenith²` exactly, which is the identity the test pins; the ordering
  clear > haze > light fog > dense fog holds in every band, so a daytime scene cannot produce
  MWIR glint through dense fog.
- The values are ESTIMATED whole-column figures, not fitted: expect them to move when M11 arrives.
  Nothing consumes them yet, so moving them costs only the config hashes.
- The fog presets are the least defensible entries — fog is a shallow layer of finite depth, so its
  true solar transmittance depends on layer thickness, which the preset does not carry. Treated as
  "sun through the layer" and flagged in the file.
- Spherical geometry, refraction, and any wavelength dependence inside a band are out of scope.

## Revisit when

M11 implements the §5.4 reflected-solar term: replace the authored zenith values with a computed
column (or a MODTRAN-class table), and take the opportunity to move past the plane-parallel airmass
if low-sun scenes matter.
