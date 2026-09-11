# ADR 0031 — Display branch operator order: AGC → gamma → DDE → polarity → palette

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.1 lists the display chain as AGC/DRC → gamma → polarity → palette and omits DDE; the
sensor-noise-chain skill puts DDE after gamma; §13.4 puts it before (spec issue S23). Real cores are
not documented at this level. The order changes the image (a high-pass boost before gamma is not the
same as after), and it must be fixed for train/deploy matching (§15 Tier 5).

## Decision

`irsim.isp.display.run_display_branch(dn16, isp, bit_depth)`:

1. **AGC** on the raw DN16: `linear` (percentile clip, ADR 0027), `plateau_equalization`
   (ADR 0028), or `none` = the exact shift DN16 >> (bits − 8);
2. **gamma** y^(1/γ) in float32;
3. **DDE** unsharp mask (ADR 0029) on the gamma-mapped image, clipped;
4. **polarity** (black-hot = 255 − DN8) and **palette** (ADR 0030) to RGBA8.

Rationale: DDE acts on what is displayed (the tone-mapped image), so its halo amplitude is in display
units and does not depend on the AGC mode; polarity is a table index and is last by construction.
Every operator is a pure per-frame function; there is no temporal memory in the display branch
today (AGC gain smoothing, if modelled later, becomes explicit `PipelineState`). The float32
rounding points R1–R4 are documented in the module so a GPU port can match them exactly. The isp
block's config hash (plus bit depth) is returned with every frame.

## Consequences

The Boson estimate `plateau 0.012, gamma 1.0, dde_gain 0.35, gray, white_hot` defines the reference
look. Because DDE follows gamma, the overshoot on a step is ±gain/6 of the *display* step.

## Revisit when

ME.3's DDE-overshoot and AGC-signature extraction from public video contradicts this order for the
reference camera.
