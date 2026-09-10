# ADR 0015 — Distortion is applied by the engine; the core uses undistorted pinhole geometry

**Status:** Accepted
**Date:** 2026-09-11

## Context

§8.4 lists distortion models (Brown–Conrady, Kannala–Brandt, f-theta) and §2 puts cos⁴θ natural
vignetting in the master equation. Both depend on where a pixel looks. The renderer already owns the
projection: Isaac Sim's camera can apply a lens model when rendering, and re-warping a rendered
G-buffer in the core would resample ids and temperatures — which must not be interpolated (ADR 0014:
ids never blend; temperature is per-facet).

## Options considered

1. **Core warps the G-buffer** by the configured model — a second projection implementation to keep
   consistent with the engine's, and it interpolates planes that must not be interpolated.
2. **Engine renders with the lens model; core treats the image as the ideal pinhole** for field angles
   and cos⁴ — one projection; the core stores the distortion schema for the engine and for reports.
3. **Ignore distortion** — wrong for the 32° Boson lens at the edges and for any fisheye variant.

## Decision

Option 2. `irsim.optics.vignetting` computes field angle θ = atan(r/f) and cos⁴θ = (f²/(f²+r²))² from
**undistorted** pinhole geometry at sample centres (i + 0.5)/s, optionally supersampled, with an
optional measured multiplicative map in (0, 1]. `optics.distortion` in the sensor config is schema the
engine consumes; the core never applies it. cos⁴ is a rectilinear-lens result: the schema (M3.4) refuses
`vignetting_cos4: true` with `kannala_brandt` or `ftheta`, whose irradiance fall-off must come from a
measured map.

Known answers for the Boson 640 / 14 mm: cos⁴ at the format corner 0.79240 and at the horizontal edge
centre 0.86496; the corner *pixel centre* is half a pixel inside and reads 0.7930.

## Consequences

Field angles are exact for the pinhole and slightly off for the true lens at the field edge
(Brown–Conrady with the estimated zero coefficients in the Boson YAML makes no difference today). The
error in cos⁴ from ignoring a few percent of distortion at the edge is < 1 % of a 21 % effect and is
absorbed by flat-field correction in any case (§11.2). When the engine applies a fisheye model, the
core's cos⁴ must be disabled — enforced by the schema.

## Revisit when

A camera with strong distortion is modelled (fisheye automotive lenses in phase 2), or a measured
vignetting map is available — then the map replaces cos⁴ rather than multiplying it.
