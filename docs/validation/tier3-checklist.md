# Tier 3 thermal: the manual half

`tests/unit/test_tier3_thermal.py` asserts everything that can be asserted about the M6.13 facet
scene. This is the list of things that cannot be, and that a person should look at in
`scripts/validate_thermal_diurnal.py`'s plot before trusting a diurnal run.

Run it with:

    python scripts/validate_thermal_diurnal.py --out outputs/thermal_diurnal.png

## What to look for

1. **Shape, not just extrema.** Every curve should rise smoothly from a pre-dawn minimum, peak
   after solar noon by an amount that grows with the surface's mass, and fall. A curve with a
   kink, a plateau, or a peak *before* solar noon is a solver or forcing bug, not a material.
2. **The dawn convergence should be smooth.** The automated test checks that the spread collapses
   to under 1.5 K; the plot shows whether it converges or whether the curves cross each other in a
   tangle, which would indicate the facets are not being driven by the same sky.
3. **The shaded twin should track the sunlit one at night and separate at first light**, with the
   separation starting at sunrise rather than before it. A separation that begins before the sun
   is up means diffuse radiation is being shadowed, which it should not be.
4. **The air curve should sit inside the envelope by day and above the coldest surfaces at
   night.** A surface never colder than the air on a clear night means the longwave term is not
   reaching the solver.
5. **The moving hood should sit close to the air all day.** Forced convection at 28 m/s pins a
   thin panel to ambient; if it does not, the convection coefficient is not seeing the vehicle
   speed.
6. **Look at the second half of the plot for a discontinuity.** The scene's spin-up wraps the
   weather file (M6.12), so a seam in the *forcing* is possible; the surfaces should not show one,
   because the spin-up ends before t₀ and the rendered day is real data throughout.

## Known, recorded departures from the roadmap's criteria

* **Swing is 47 K, not the quoted 15–40 K.** These facets are single nodes with an adiabatic back,
  so the day's heat has nowhere to go but back out of the surface. M6.8's two-node solver with a
  finite R₂d and T_deep is what pulls this into the quoted band; the single-node figure is an upper
  bound. See `test_the_swing_exceeds_the_roadmaps_band_and_the_reason_is_the_back_boundary`.
* **The scene goes flat at dawn but not at dusk.** §6.3 asks for both. The dawn collapse is
  dramatic — 12.7 K of spread to 0.88 K, a 93 % drop — and there is no dusk equivalent, because
  each pair of surfaces crosses at a different time spread over three hours and the ensemble never
  passes through a common point. A dusk collapse needs a facet set whose members share a solar
  absorptivity as well as differing in mass; this one deliberately does not, because its pairs
  exist to isolate other variables. Engineering the fixture to produce a dusk flat would be
  fitting it to its own acceptance test.
* **The site and the weather file must agree about where noon is** (M6.12). Nothing enforces it.
  If a curve peaks at an implausible local hour, check that first.
