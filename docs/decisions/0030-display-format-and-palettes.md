# ADR 0030 — Display output is RGBA8; palettes are cited public approximations

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.4 asks for white-hot/black-hot and the Ironbow/Rainbow/Lava/Arctic palettes because perception
models are sensitive to palette; §12.2 lists gray/ironbow/rainbow/lava (spec issue S27); the Isaac
viewport displays RGBA unorm textures (isaac-sim-spg skill). FLIR's palette tables are proprietary.

## Decision

`irsim.isp.palette`:

- Output is **always RGBA8** (H, W, 4), alpha 255 — one layout for the viewport, files and consumers.
- DN8 = round(255 · clip(y, 0, 1)); white-hot indexes the table with DN8, **black-hot with 255 − DN8**
  (so gray black-hot is the exact complement).
- Tables: gray = identity; ironbow, lava and arctic are built by linear interpolation between a
  handful of RGB control points that approximate the palettes of those names as they circulate in
  open-source thermal viewers; rainbow is a full-saturation hue sweep from blue to red. They are
  **not** vendor tables; ironbow and lava have non-decreasing Rec.601 luma, rainbow has 256 distinct
  entries. Both the §11.4 and §12.2 palette sets are accepted (`arctic` included).

## Consequences

Palette identity is part of the config hash, so a train/deploy palette mismatch is detectable
(§15 Tier 5). Colour fidelity to a specific vendor palette is not claimed; anyone comparing against
real palette-rendered video should use gray or fit the control points from that video.

## Revisit when

A dataset's palette must be matched exactly for a Tier 5 experiment — fit the control points from
the real frames' colour histogram and add the fitted table under a new name.
