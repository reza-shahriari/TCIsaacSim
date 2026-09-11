# ADR 0027 — Linear AGC percentiles are histogram-based with in-bin interpolation

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.3's linear AGC clips at the 0.5 % / 99.5 % percentiles of the frame. A percentile can be computed
by sorting (exact, O(N log N), order-dependent tie handling) or from a histogram over the integer DN
bins (what GPU kernels and real cores do). The two differ by up to a bin, and the choice decides
whether the CPU oracle and the SPG kernel can agree exactly.

## Decision

`irsim.isp.agc`: percentiles come from the histogram over 2^bit_depth integer bins, with **linear
interpolation inside the bin** (`percentile_from_histogram`): the value v such that p·N pixels lie
below it when pixels are spread uniformly within their bin. On a ramp with one pixel per bin this
equals p·N to the in-bin offset, so it matches `np.percentile` to a fraction of a DN, and it is what
a histogram kernel computes. Both AGC operators are **global** over the frame (the hot-exhaust
collapse of §11.3 is a feature to reproduce, not a bug to avoid); ROI-weighted and locally-adaptive
variants are future options. A frame with no dynamic range maps to mid-grey (0.5). Gamma is applied
as y^(1/γ). Inputs are uint16 DN or float32 in [0, 2^bits − 1]; float16 is refused.

## Consequences

Percentile positions are quantised to the DN bin (up to 1/range relative), which is invisible at 8
bits. A GPU port reproduces the CPU result exactly by building the same histogram.

## Revisit when

A camera with a documented non-global AGC (ROI weighting, tiles) is modelled.
