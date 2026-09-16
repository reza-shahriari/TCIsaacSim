# 0090 — Clamp summed view factors instead of partitioning nested radiators

Date: 2026-09-16
**Status:** Accepted
Roadmap: PT.3 (§6.1, §6.6)

## Context

ADR 0088 gave the road a computed configuration factor to a rectangular radiator instead of an
authored Gaussian. `CarGeometry.ground_radiators()` supplies three: `underbody` (a broad, low-lying
rectangle standing for the whole flat floor pan), `engine_bay` (a smaller, hotter rectangle for the
bay's own floor), and `exhaust_pipe` (a narrow stripe) — authored that way "because the picture
they make differs: a broad warm footprint from the underbody, a brighter pool under the engine, and
a narrow stripe from the exhaust run."

`build_ground_field` calls `patch_view_factors` once per radiator and sums the resulting
`occluded_longwave_flux` unconditionally. Each call is an exact two-body closed form (ADR 0088) and
individually correct — but `engine_bay`'s 0.72 m × 0.80 m footprint and `exhaust_pipe`'s 0.11 m ×
2.60 m stripe both lie **entirely inside** `underbody`'s 1.64 m × 3.85 m footprint. They are not
three bodies filling disjoint solid angles; they are three different-fidelity descriptions of one
floor pan. Summing their view factors double- (under the exhaust stripe, triple-) counts the solid
angle they share, which a plane element's view factors to any set of surfaces cannot exceed 1 for
(Howell). Measured peak Σ F ≈ 1.40 across 28 cells of the clear-night scene's patch, inflating the
road's clear-night patch by up to ~40% — a number `test_a_clear_sky_puts_far_more_of_the_patch_on_
the_road_than_the_engine_does` was reporting as sky-occlusion physics.

## Options considered

1. **Partition the geometry.** Author `underbody` as the floor pan *minus* the engine-bay and
   exhaust footprints (an L-shaped or multiply-holed region), each rectangle then radiating only
   where its solid angle is actually its own. Physically exact, and the nesting here happens to be
   clean full containment rather than partial overlap, so it is geometrically tractable. But
   `view_factor_to_parallel_rectangle`'s closed form is for a full rectangle; a holed region needs
   either several more rectangle terms (superposing negative cutouts, the same trick already used
   for offset rectangles) or a different derivation, and re-authoring three carefully-dimensioned
   `RadiantRectangle`s to a fourth shape is easy to get subtly wrong. Scoped larger than this step.
2. **Clamp the sum, scaling proportionally.** Rescale each radiator's view factor at every cell by
   the same per-cell ratio wherever the sum exceeds 1, so the total is capped at the physical bound
   and the *relative* weight between radiators — the reason there are three rectangles rather than
   one — is preserved. Simple, and `occluded_longwave_flux` is linear in its `view_factors`
   argument, so the same scale factor caps the occlusion term along with the source term without
   touching that function.
3. **Drop to a single `underbody` rectangle**, absorbing the bay and exhaust into one authored
   emissivity/temperature. Removes the bug by removing the feature ADR 0088 added the three
   rectangles for — the pool and the stripe, which the docstring calls "the feature that says 'this
   car has been running' rather than 'this car is warm'."

## Decision

Option 2. `irsim.thermal.spatial_sources.clamp_view_factor_sum` takes the raw per-radiator view
factor arrays, and wherever their sum at a cell exceeds `1 + 1e-6` rescales all of them at that cell
by `1 / sum`; cells where the sum is already ≤ 1 (almost everywhere — the overlap is local to the
region under the engine bay) are untouched. `build_ground_field` computes all three raw view
factors first, clamps them together, and only then loops to accumulate
`occluded_longwave_flux`. It warns (`warnings.warn`, not a silent branch) whenever clamping changes
anything, naming the peak sum and the affected cell count, because the trigger is a modelling gap
and should stay visible rather than become routine.

## Consequences

**What this buys.** Σ view factors ≤ 1 + 1e-6 for every cell of both car scenes
(`tests/unit/test_spatial_sources.py::test_clamp_view_factor_sum_*`,
`tests/unit/test_car_demo.py::test_ground_radiator_view_factors_never_exceed_one`), closing the gap
`test_a_clear_sky_puts_far_more_of_the_patch_on_the_road_than_the_engine_does` was silently
absorbing into "sky occlusion physics." The visual character ADR 0088 wanted — broad footprint,
brighter pool, narrow stripe — survives, because proportional scaling changes magnitude, not shape.

**What stays hard.** This is an approximation, not a re-derivation of the true partitioned geometry:
proportional scaling has no way to know that, physically, a cell directly under the engine bay
"belongs" more to `engine_bay`'s specific hot floor than to the generic `underbody` estimate at the
same spot — it treats the overlap as shared evenly by ratio of the (uncapped) individual factors,
which is a modelling choice, not a measurement. The error this introduces is bounded by construction
(the sum is always ≤ 1) but its *distribution* between the three radiators at an overlapping cell is
not independently verified against anything.

## Revisit when

A scene needs the engine-bay pool and exhaust stripe to be quantitatively distinguishable from the
underbody footprint at overlapping cells specifically — at which point option 1's partitioned
geometry is the next step, and this ADR's clamp is superseded rather than extended.
