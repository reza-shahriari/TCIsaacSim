# ADR 0033 — Convection: the §6.2 max(free, forced) form; scalar composition of wind and vehicle speed

**Status:** Accepted
**Date:** 2026-09-11

## Context

The convection coefficient h is the largest single source of uncertainty in surface temperature
prediction (§6.2, DIRSIG [R3]); it must be derived from conditions rather than authored per material,
and for a vehicle it must include the vehicle's own speed (a parked and a moving car have different
hood temperatures — first-order for automotive IR). A weather file gives wind *speed* only; the
vehicle's heading and the wind direction are not both available in phase 1.

## Options considered

1. Authored h per material (MuSES-style) — pushes the largest uncertainty onto the user, and makes
   the vehicle-speed effect impossible.
2. **§6.2 form: h = max(c |ΔT|^{1/3}, a + b v_rel^n)** with a = 5, b = 4, n = 0.8, c = 1.5 (chosen).
   The forced branch is in the family of the Jürges/McAdams flat-plate correlation (5.7 + 3.8 v);
   the free branch is the turbulent natural-convection ΔT^{1/3} law. Two branches, four numbers.
3. Full Nusselt–Reynolds correlations per geometry (plate, cylinder, orientation) — better in
   principle, but the geometry and flow regime of a car panel in traffic are not known well enough
   for the extra terms to be more than decoration.

## Decision

Option 2, in `irsim.thermal.convection`, vectorised. Relative air speed is the **scalar** sum
v_rel = |v_wind| + |v_vehicle|: without headings this is the upper bound (head-wind case) and
is the conservative choice for cooling. Consequences of the numbers: h(0, calm) = 5.000,
h(28 m/s) = 62.51 W m⁻² K⁻¹ (× 12.5 over parked), free convection overtakes calm forced at
|ΔT| = (a/c)³ = 37.0 K and never matters once there is real wind.

## Consequences

- Expect ±30 % on h and therefore a comparable error on the convective share of the balance;
  this is documented as the dominant thermal uncertainty, not hidden.
- Tail-wind driving is over-cooled by the scalar composition (a 10 m/s tail wind at 10 m/s
  vehicle speed is v_rel = 20 here, 0 in reality).
- Surface orientation (a roof vs a door) does not change h.

## Revisit when

Vehicle heading and wind direction are both available (vector composition), or when a scene
needs orientation-dependent free convection (e.g. cold-sky roofs at night).
