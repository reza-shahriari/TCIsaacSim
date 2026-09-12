# ADR 0072 — Aerial material library (four materials) and prescribed target heat schedules

**Status:** Accepted (MS.7)
**Date:** 2026-09-12

## Context

The first target application is objects in the sky — drones, light aircraft, birds against a sky
background — and until now the material library held only ground-scene surfaces (asphalt, car
paint, glass, skin). Nothing in it is an airframe, and nothing in the repo says how hot a motor
runs. Two things were needed and they have different evidence bases, which is the whole reason
this ADR exists:

- **Optical properties of airframe surfaces.** Well constrained. An airframe is paint, epoxy,
  or rubber, and §16.2 already carries the rows those reduce to; the LWIR emissivity of a
  polyurethane topcoat is not controversial.
- **Internal heat sources.** Badly constrained. No public dataset of instrumented multirotor
  motor, ESC or battery temperatures was found. The magnitudes are estimates.

A ground surface gets its temperature from an energy balance: solar loading, convection, longwave
exchange, storage in a slab. Applying that to an aircraft in flight would be theatre. Forced
convection at flight speed pins an unpowered skin to within a degree or two of the air it is
moving through, there is no ground contact and no diurnal storage, and everything a thermal camera
actually keys on is internal dissipation set by throttle. The solver machinery would run, produce
numbers, and none of them would mean anything.

## Options considered

1. **Full energy balance per airframe panel** (M6.x slab solver with a flight-speed convection
   coefficient). Physically the most complete, but the convective term dominates so heavily that
   the answer is `T_air` plus solver noise, and the parameters that would make it interesting
   (internal conduction paths from motor to arm, boundary-layer heating) are not authored anywhere.
   Expensive, and its extra fidelity is not observable.
2. **A fixed ΔT per part, authored per scenario.** Cheapest. Cannot express a throttle change, so
   it cannot produce the one aerial phenomenon that is unambiguous — a motor heating up on climb.
3. **Prescribed nodes driven by a throttle law** (chosen).

## Decision

### Materials — four, authored as scalar per-band emissivity

`configs/materials/`: `painted_composite`, `carbon_fibre`, `aircraft_aluminium_painted`,
`propeller_rubber`. All `source: literature`, all opaque (τ = 0 in every band), all with the M7.2
scalar `emissivity_per_band` fallback rather than a spectral file — for a paint or an elastomer the
band-effective value is what the literature reports, and inventing a spectral curve to average back
down to it would fabricate structure. ρ is derived, never authored (CLAUDE.md #4).

LWIR/MWIR values come from the §16.2 paint and rubber rows, because the topcoat is what the camera
sees: `aircraft_aluminium_painted` is radiatively a paint (ε_LWIR = 0.90) sitting on a 205 W/m/K
substrate, and pairs with the existing `bare_aluminium` (ε_LWIR = 0.09) as the same metal with the
paint removed. NIR/SWIR values are **ESTIMATED** and tied to each material's solar absorptivity so
the reflective and short-wave paths cannot disagree about how dark a surface is.

### Heat sources — prescribed nodes, ΔT ∝ u²

`irsim.thermal.aerial`. Each node follows

    T_node(t) = T_air(t) + ΔT_max · u(t)^n,      n = 2

with u the throttle fraction and `T_air` from the scene's shared `WeatherSeries` (CLAUDE.md #6 —
injected, never loaded in this module). n = 2 is the ohmic reading: winding and MOSFET loss go as
I²R with current roughly proportional to throttle, so at a fixed convective conductance the steady
rise above ambient goes as u².

| node | ΔT_max | basis |
|---|---|---|
| motor | 45 K | **ESTIMATED** |
| esc | 30 K | **ESTIMATED** |
| battery | 15 K | **ESTIMATED** |
| airframe | 0 K (offset configurable) | forced convection pins the skin to `T_air` |

The **ordering** (windings hottest, MOSFETs below, pack coolest because it is a large mass spread
over many cells) is the defensible part and is what a test asserts. The magnitudes are the first
thing a Tier 4 fit against public aerial IR imagery should replace.

The schedule is not a typed table. `refine_nodes` bisects the node grid until piecewise-linear
interpolation reproduces the analytic law to a stated tolerance (default 1 mK), so a
`PrescribedSolver` built this way is a *discretisation of a model with a bound*, not a list of
numbers someone chose. For the u² law the segment midpoint is the worst point, so the bisection
criterion is exact rather than sampled.

### Target contrast against the sky

`irsim.validation.aerial` closes the loop with MS.1/MS.2:

    C(θ) = τ(R, θ)[ε L_B(T_t) + (1 − ε) L_env] + L_path(R, θ) − L_sky(θ)

`zero_contrast_elevation` finds where it vanishes, returning `None` when there is no crossing —
the ordinary answer for a hot target, and a case callers must handle rather than treat as failure.
L_env reuses `irsim.pipeline.environment.environment_radiance`, so the helper and the pipeline
cannot disagree about the reflected term.

Measured for the specified case (T_air = 300 K, ε = 0.9, V_s = 1, R = 1 km, `us_standard_clear`,
7.5–13.5 µm top-hat):

| ε | 0.80 | 0.90 | 0.95 | 1.00 |
|---|---|---|---|---|
| zero-contrast elevation | 1.97° | 1.26° | 0.78° | none |

| range | 200 m | 1 km | 5 km |
|---|---|---|---|
| zero-contrast elevation (ε = 0.9) | 1.39° | 1.26° | 1.06° |

At zenith the same drone is +64 K of apparent-temperature contrast; at 0.5° it is −2.5 K.

**The ε = 1 column is the load-bearing result.** A blackbody target at air temperature never
inverts. The inversion is therefore not a path-radiance artefact and not a numerical accident: it
is the reflected term (1 − ε)(L_env − L_B(T_air)) going negative because the target's own face is
mirroring a cold sky into a camera that is looking at a warm horizon. A simulator that got the
reflected term wrong would show no crossing, or a crossing at ε = 1, and the test would catch it.

## Consequences

- Four aerial materials pass the CLAUDE.md #4 library walk in all four bands, including after
  float32 packing by M7.18. MS.8 can build the aerial fixture on their names.
- An aerial target is a set of prescribed nodes, so it drops into the existing `Scene` and solver
  plumbing with no new solver type, and Isaac-side work needs no aerial-specific thermal path.
- **The ΔT magnitudes are unvalidated.** Any absolute claim about motor signature strength is only
  as good as the estimates in the table above. Relative claims (a climbing drone is hotter than a
  hovering one; motor hotter than pack) are robust because they follow from the law's shape.
- **L_env is evaluated with the ground-level sky model.** A target at altitude sees slightly less
  atmosphere above it than a surface does, and its belly sees ground at a slant range rather than
  underfoot. Neither is modelled, so the belly case (V_s → 0) carries an unquantified bias — the
  bias is toward too much reflected ground, since the real path attenuates it. V_s = 1 is the
  honest configuration and is what the horizon test uses. The warm-belly/cold-top ordering is
  asserted as an ordering only, never as a magnitude.
- Elevation is the flat-earth geometric ray angle inherited from MS.1; below ~0.5° the layered
  model's own geometry is the larger error, which is why the root bracket starts there.
- No solar term. A sunlit drone in MWIR or SWIR needs M11's solar scatter; `carbon_fibre` carries
  α_sol = 0.90 so it is ready to be the material that shows it when that lands.

## Revisit when

- Public instrumented aerial IR data (or a permitted flight with a borrowed camera) allows the ΔT
  magnitudes to be fitted rather than estimated — the ordering should survive, the numbers should
  not be assumed to.
- M11 lands solar scatter, at which point the reflective-band NIR/SWIR estimates start to matter
  radiometrically instead of only bookkeeping.
- A target-altitude sky model exists, at which point the belly (V_s → 0) case becomes quantifiable
  and the caveat above can be replaced with a bound.
- M7.9 extends the library walk to fifteen materials; these four should need no change.
