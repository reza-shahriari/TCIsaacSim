# ADR 0034 — Solar geometry computed in irsim (NOAA equations); shadow flag and irradiance are inputs

**Status:** Accepted
**Date:** 2026-09-11

## Context

The energy balance (§6.1) needs the sun's direction per facet (cosine ψ) and the solar irradiance;
the reflected solar term (§5.4, M11) needs the same direction. The options are a dependency
(pvlib, astral, ephem) or the standard NOAA Solar Calculator equations (Meeus-derived), which are
~150 lines of NumPy. The renderer already knows what is shadowed (a shadow AOV or a scripted flag),
and the weather file already carries surface DNI/DHI.

## Options considered

1. `pvlib.solarposition` — excellent, but a heavy dependency (pandas) for ~0.01° we do not need.
2. `astral` / `ephem` — lighter, but another install in the Isaac Sim interpreter and different
   conventions to reconcile.
3. **NOAA equations in `irsim.thermal.solar`** (chosen): geometric elevation and azimuth, declination,
   equation of time, solar noon; vectorised over Julian days; better than 0.01° for 1950–2050
   ignoring refraction.

## Decision

Option 3. Conventions: longitude east-positive, azimuth clockwise from north, elevation geometric
(no refraction: < 0.6°, and only at the horizon where DNI ≈ 0), times timezone-aware. The unit sun
vector is in the local ENU frame (east, north, up). `solar_loading` is
Q_sol = S · max(0, n·s) · DNI + V_s · DHI with the shadow flag S ∈ [0, 1] and the sky-view factor
as **inputs**; absorbed power is α_sol · Q_sol (written in that form; the §6.1 draft's complement is
spec issue S2). DNI/DHI are taken from the `WeatherSeries` as surface values and are not attenuated
again; the solar-path transmittance of M11 shapes the reflected term's spectrum only.

Tests use an *independent* implementation (Spencer 1971 Fourier series + spherical trigonometry)
as the oracle at six sites/times within 0.5°, plus analytic checks: solstice declination,
equinox noon elevation 90 − |lat|, Boulder solar noon and sunrise azimuth, the Tromsø midnight sun.

## Consequences

- No new dependency; the thermal tick can evaluate thousands of timestamps in one call.
- Refraction and the ~1′ parallax are ignored; sunrise/sunset times are the geometric ones
  (~3 min late).
- Terrain/self-shadowing is not computed here: the shadow term must come from the renderer
  (or a script), which is where the geometry is.

## Revisit when

A sky-radiance or aureole model needs sub-0.1° accuracy near the horizon, or when the renderer
supplies a per-pixel sun-visibility AOV that should replace the scalar flag.
