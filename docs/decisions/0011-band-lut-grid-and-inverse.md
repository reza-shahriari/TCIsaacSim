# ADR 0011 — Band LUT: grid, range, clamp policy, float32 contract, inverse

**Status:** Accepted
**Date:** 2026-09-11

## Context

Per-pixel band radiance ∫R B(T) dλ is far too expensive to integrate at runtime (§3.2 c); the spec
recommends a precomputed table with linear interpolation (§3.2 b) and sketches the kernel that reads
it (§13.5). The table's grid, range, precision and out-of-range behaviour are shared by the CPU
reference, the Warp path and the SPG kernel, so they are fixed here once. The inverse (§3.3) —
apparent temperature from radiance — is what a radiometric camera actually outputs, and its policy
outside the table matters for cold sky and hot glint pixels (spec issue S14).

## Options considered

- **Grid**: 0.05 K uniform (16001 entries, 64 kB per table) vs 0.1 K (8001) vs non-uniform in Lb.
  Linear interpolation error scales with the step squared; at 0.05 K it is ~0.03 mK, at 0.1 K ~0.1 mK
  — both far under 5 mK. 0.05 K is kept because the cost is four 64 kB tables and it leaves margin
  for the photon table's curvature. Non-uniform grids would break the one-line kernel index.
- **Range**: 200–1000 K covers every scene surface the phase-1 and phase-2 targets contain; cold sky
  below 200 K and glint above 1000 K are handled by clamping *plus flagging*, not by widening (which
  would dilute the grid or double the tables).
- **Precision**: float32 tables built in float64 and cast once (non-negotiable #2: never float16).
- **Inverse**: searchsorted + linear inversion in the bin, vs. storing an inverse table in L (needs a
  non-uniform L grid), vs. Newton on the forward table. Searchsorted is exact for a piecewise-linear
  forward table — the round trip is then limited by float32 only.

## Decision

`irsim.radiometry.lut.BandLUT`:

- Grid `LUT_T_MIN_K = 200`, `LUT_T_MAX_K = 1000`, `LUT_DT_K = 0.05`, `LUT_N = 16001`, defined once in
  `constants.py` beside the temperature encoding; integer indices land on integer-K temperatures.
- Four tables: `lb`, `lb_q`, `dlb_dt`, `dlb_q_dt` (§13.5 needs `lb`; §9.4 NETD needs the derivatives
  in the form matching the detector). Built by the Simpson oracle in float64, cast to float32 once.
- `lookup()` implements the §13.5 kernel arithmetic **in float32**, including the clamp of the index
  to `N − 1.001`, so the CPU reference and the GPU agree to float32 rounding. Out-of-range temperatures
  are clamped; `out_of_range()` returns the mask so callers can flag such pixels.
- Measured against the oracle at off-grid temperatures: 0.03 mK worst case over 200–1000 K. The float32
  path costs 0.086 mK versus float64 interpolation, dominated by the float32 index (`u ≈ 16000` has
  spacing 1e-3 of a step) and the float32 input near 1000 K — not by the tables. Budget is 5 mK.
- Inverse (M1.7): `apparent_temperature(L)` by `searchsorted` on the monotone table and linear
  inversion inside the bin, float32 in/out. Radiance below `Lb(T0)` or above `Lb(T1)` **clamps** to
  the range ends (an image must not contain NaN) and `radiance_out_of_range()` flags it; the sky
  model (MS.1) computes sky radiance by direct quadrature and never goes through the inverse.

## Consequences

Adding a band adds four 64 kB tables. Scenes with surfaces outside 200–1000 K read clamped radiance
and are flagged rather than wrong-and-silent; if such surfaces become routine (exhaust plumes, fires)
the range must be widened together with the encoding span (ADR 0006). The 0.086 mK float32 cost is
accepted and is the reason the temperature encoding, not raw kelvin, feeds the kernel.

## Revisit when

A target class needs T > 1000 K or the sky path needs T < 200 K through the LUT (widen range, revisit
grid), or a Tier 2 NETD bench shows the LUT contributing more than 0.5 mK.
