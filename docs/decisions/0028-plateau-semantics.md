# ADR 0028 — Plateau equalisation: plateau is a fraction of N_pixels per bin over 2^bit_depth bins

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.3 defines plateau equalisation as histogram equalisation with per-bin counts clipped at a
plateau P before integrating the CDF; §12.2 configures `plateau: 0.012` as a dimensionless number;
the sensor-noise-chain skill says "plateau = 0 behaves like linear", which is degenerate (spec issue
S26). The bin count and the normalisation of the CDF also decide the result.

## Decision

`irsim.isp.agc.agc_plateau(x, plateau, bit_depth)`:

- histogram over **2^bit_depth integer bins** (the container's full range; 14-bit data in a uint16
  container uses `bit_depth: 14`, so 16383 → 1.0);
- **P = plateau · N_pixels** counts per bin, applied before the CDF;
- the CDF is *exclusive* (counts of bins strictly below) and normalised between the darkest and
  brightest **occupied** bins, so the frame's minimum maps to 0 and its maximum to 1 — no dead range;
- **P → 0 limit** (normative): every occupied bin contributes the same clipped count, so the output is
  the rank map of the *distinct occupied bins*. On a dense histogram (every bin between min and max
  occupied) that is the min–max linear stretch to within one bin; on a sparse one it is a
  distinct-value rank map, and that is the documented behaviour ("plateau → 0 behaves like linear"
  holds only for dense histograms). `plateau ≤ 0` is an error, not a mode.
- **P ≥ max bin count** is full histogram equalisation (rank/(N − 1) for tie-free data).

## Consequences

The characteristic thermal look (and the AGC-collapse phenomenology) depends on `plateau`, which is
part of the config hash: train/deploy mismatch is detectable (§15 Tier 5). Output entropy is
non-decreasing in P.

## Revisit when

The reference camera's AGC signature is measured from public video (ME.3) and the fitted plateau
or a tile-based variant is needed to match it.
