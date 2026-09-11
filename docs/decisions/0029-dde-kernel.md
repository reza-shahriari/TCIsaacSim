# ADR 0029 — DDE is an unsharp mask with a 3×3 box, applied after gamma

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.3 models digital detail enhancement as a high-pass boost added back over the compressed image
("why real thermal images look crunchy"); the kernel and the position in the chain are not specified
and the skill/§13.4 disagree on the position (spec issue S23).

## Decision

`irsim.isp.dde.dde(y, gain)`: y + gain · (y − box₃(y)), clipped to [0, 1], with a **3×3 box** mean
and edge replication at the borders (periodic wrap available for spectral tests). Position: **after
AGC and gamma, before polarity and palette** (ADR 0031). The 3×3 box is the cheapest kernel a core
would use; its overshoot on a 0.5 step is ±gain/6 and its transfer is |1 + gain(1 − K̂(f))|² with
K̂ = ((1 + 2cos 2πf_x)/3)((1 + 2cos 2πf_y)/3). `dde_gain` in the config (Boson estimate 0.35) is the
gain; 0 disables the stage bit-identically.

## Consequences

The overshoot halo width is one pixel; a camera with a wider-kernel DDE (Gaussian σ) will show
wider halos than the model. The gain is part of the config hash.

## Revisit when

ME.3's edge-overshoot profiles from public video show a halo wider than one pixel — then a Gaussian
kernel with fitted σ replaces the box.
