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

## Addendum (M10.9a, 2026-09-12): the forward model as the engine's oracle, and f-theta refused

This ADR's decision stands unchanged — the renderer applies the lens, `irsim.optics.vignetting`
keeps computing field angles from undistorted pinhole geometry, and nothing in the core warps a
G-buffer. What M10.9a adds is the other direction: `irsim.optics.projection`, a **forward** model
of the lens the engine is handed.

The distinction matters and is easy to lose. Applying distortion in the core was rejected because
it resamples ids and temperatures. Asking "where should a ray at this angle land under this
`optics.distortion` block?" resamples nothing — it is a function from a direction to a pixel — and
without it there is no way to tell whether the coefficients written into USD produced the lens that
was configured. Writing coefficients into a schema and judging the result by looking at the picture
is how a lens ends up a few percent wrong and stays that way; a barrel term is a smooth radial
stretch, which is exactly the kind of error the eye accepts.

Two conventions are pinned by the model rather than left implicit, because both are silent when
wrong. USD camera space is +Y up, −Z forward; OpenCV — whose model the engine implements — is
+Y down, +Z forward, so the flip happens in exactly one function. And the principal point is at
the format corner `(W/2, H/2)` with pixel centres at `i + 0.5`, matching
`field_radius_map_mm`; the two modules are tied together by a test rather than by a comment.

**Which schema each config model maps to** was measured on 6.1.0-rc.26, not assumed (the probe
that measures it lands with the camera itself, M10.9a-ii):
`brown_conrady` → `OmniLensDistortionOpenCvPinholeAPI`, whose twelve attributes are in OpenCV's own
`[k1, k2, p1, p2, k3, k4, k5, k6, s1..s4]` order, so a five-term Brown–Conrady block maps
**positionally** with no reordering; `kannala_brandt` → `OmniLensDistortionOpenCvFisheyeAPI`
(`k1..k4` on θ).

**`ftheta` is refused rather than approximated.** Its schema exposes `k0..k4` beside
`nominalWidth`, `nominalHeight` and `opticalCenter`, and nothing this build exposes determines
either whether the polynomial returns a radius in pixels of the nominal image or in normalised
units, or whether `k0` is a constant term — under one reading a one-coefficient block is a lens,
under the other it is a constant radius, which is not. Either guess renders a plausible fisheye
that disagrees with the engine by tens of pixels at the field edge, which is precisely the failure
this module exists to detect, so `project` and the USD writer both raise. No camera in
`configs/sensors/` uses f-theta, so nothing is blocked. The experiment that settles it belongs to
M10.9b's renderer audit: render a grid through an f-theta camera with one coefficient set and fit
r(θ).

## Revisit when (addendum)

The f-theta convention is measured, or a camera config needs a lens family beyond the two verified
here (`RadTanThinPrism` and `KannalaBrandtK3` schemas also exist on this build).
