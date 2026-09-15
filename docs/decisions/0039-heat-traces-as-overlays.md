# ADR 0039 — Heat traces are overlays, not conserved quantities

**Status:** Accepted
**Date:** 2026-09-15

## Context

§6.6's last row is the road under a departed vehicle: −3 … −8 K, decaying over 5–20 minutes. The
spec singles it out — *"That last row is worth implementing. Thermal shadows and residual heat
traces are a signature phenomenon of the band and they routinely confuse detectors trained only on
synthetic data that lacks them."*

There are two ways to produce one. Either the ground solver knows a car was parked there — it
carries a shadow mask, a contact history and a per-cell state — or a trace is painted on afterwards
as a ΔT with its own decay.

## Decision

**Traces are overlays.** A `Footprint` is a mask, an amplitude, a deposit time and a time constant;
`HeatTraceLayer.apply` adds Σ ΔT_i(t) to whatever the thermal field says. The heat comes from
nowhere and goes nowhere.

The cost is real and is stated rather than hidden: a scene with many overlapping traces is adding
energy no balance accounts for. That is acceptable because the alternative is worse — making
traces conserved requires the ground solver to carry exactly the per-cell occupancy history that an
overlay exists to avoid, and it would make a trace's amplitude depend on the ground's own solve
granularity (ADR 0037), which is a cache key, not a physical quantity.

Amplitudes are therefore **per-trace and bounded by §6.6's own ranges**, not derived.

**Superposition is linear and exact, and a cell no footprint touches comes back bit-identical.**
That matters more than it sounds: a layer that blended or clamped would make an untouched road
pixel depend on whether a car drove past somewhere else in the frame — a coupling that is
impossible to notice and impossible to debug.

**Rasterisation is by distance to the segment, not by walking it.** A trajectory sampled at 1 Hz
and the same trajectory at 100 Hz stripe **exactly the same cells**. Walking the line would make a
tyre trace depend on the logging rate of the drive that produced it.

**A tyre trace is two footprints, not one.** A single wide stripe is a skid; two parallel stripes
at a known gauge are a vehicle, and a detector that has only ever seen the first will not learn the
second.

## What this measured

| quantity | value |
|---|---|
| tyre stripe amplitude / τ | +6.0 K / 7 min (§6.6: +3 … +10 K, 5–20 min) |
| body shadow amplitude / τ | −5.5 K / 12 min (§6.6: −3 … −8 K, 5–20 min) |
| stripes at 20 min | **< 0.5 K** — gone |
| shadow at 20 min | **1.0 K** — still there |
| shadow at 60 min | < 0.2 K |
| superposition linearity | 1e-12 |
| untouched cells | bit-identical |

⚠️ The 20-minute pair is the point rather than a discrepancy. §6.6 gives a 5–20 min *range*, and a
trace at the long end of it is still visible when someone looks — which is precisely why it
"routinely confuses detectors". Asserting both below 0.5 K at 20 minutes would have meant choosing
τ for the convenience of the test.

## Revisit when

* A scene needs the trace to interact with what is above it — a puddle that evaporates faster in a
  tyre track, a shadow that changes the ground's own solar absorption. Then the overlay stops being
  separable and the ground solver has to carry the state after all.
* Traces start to dominate a frame's energy budget, i.e. a dense urban scene with many vehicles.
  The non-conservation is bounded per trace but not per scene.
