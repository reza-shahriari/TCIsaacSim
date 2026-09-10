---
name: thermal-solver
description: Surface temperature prediction for the IR camera simulator — energy balance, lumped-capacitance and slab solvers, convection coefficients, thermal inertia and diurnal crossover, weather ingestion and spin-up, and the multi-regime vehicle thermal model (panels, engine bay, exhaust, brakes, tyres, cabin). Use this skill whenever the task involves surface temperature, heating or cooling, solar loading, wind, humidity, weather files, time of day, thermal transients, Newton's law of cooling, engine or exhaust heat, or why something in the scene is the temperature it is. In LWIR the temperature field IS the image, so consult this before tuning anything radiometric to fix a temperature problem.
---

# Thermal solver

In LWIR your image *is* your temperature field. No amount of radiometric sophistication rescues a bad
`T_s`. When an LWIR image looks wrong, suspect this layer first.

Reference: `docs/physics-model.md` §6.

## The energy balance

For a surface element with area-normalised heat capacity `C = ρ·c_p·δ` (J·m⁻²·K⁻¹):

```
C · dT_s/dt = α_sol·Q_sol                 # absorbed solar
            + ε·Q_LW_down                 # absorbed sky / environment longwave
            − ε·σ·T_s⁴                    # emitted
            − h·(T_s − T_air)             # convection
            − k·∂T/∂z|₀                   # conduction into substrate
            + q_int                       # internal generation
```

Write `α_sol·Q_sol` in that form — not `(1 − α)·Q` — so the sign is unambiguous at a glance.

## Solver architecture: pluggable, not singular

This is the design decision that keeps the system honest. Implement a common `TemperatureSolver`
interface with several implementations, chosen per object:

| Solver | Use for | Notes |
|---|---|---|
| `PrescribedSolver` | Anything measured or scripted: engine bay, exhaust, brakes | Reads a time series or a function of vehicle state |
| `NewtonCoolingSolver` | Objects relaxing with no solar input: a parked car cooling, a body in shade | `dT/dt = −κ(T − T∞)`. Cheap and exactly right in its domain |
| `LumpedTwoNodeSolver` | Environment surfaces: road, buildings, terrain, vehicle panels | Surface + substrate node. **The workhorse.** |
| `SlabSolver` | When the two-node model proves inadequate | 1-D diffusion, N nodes |
| `ImportedSolver` | Results from an external tool | Reads precomputed per-facet temperatures |

Newton's law of cooling is a legitimate solver, not a mistake — it is the linearised, radiation-free,
solar-free special case. It is wrong only when used for something under solar load or exchanging with a
cold sky, because it has no `T⁴` term and no `Q_sol`. Keep it; scope it.

## The two-node model

```
C₁·dT₁/dt = α_sol·Q_sol + ε·Q_LW↓ − ε·σ·T₁⁴ − h·(T₁ − T_air) − (T₁ − T₂)/R₁₂ + q_int
C₂·dT₂/dt = (T₁ − T₂)/R₁₂ − (T₂ − T_deep)/R₂ᵈ
```

with `R₁₂ = δ₁/(2k) + δ₂/(2k)`. Integrate with explicit RK2 at a 1–60 s step. Stability limit:

```
Δt < 2·C₁ / (h + 4·ε·σ·T³ + 1/R₁₂)
```

Assert this at solver construction and fail loudly rather than producing a diverging temperature field —
an unstable thermal solver produces beautiful, completely wrong images.

**Run the thermal tick decoupled from the render loop.** 1 Hz thermal, interpolated to frame rate. The
timescales differ by orders of magnitude and coupling them wastes enormous compute.

## Convection coefficient

`h` is the largest single source of uncertainty in temperature prediction — it depends on surface
roughness, fluid viscosity, vertical stability and humidity. Do not ask users to supply it; derive it:

```
h_free   = c · |T_s − T_air|^(1/3)                  # c ≈ 1.5
h_forced = a + b · v_wind^n                          # a ≈ 5, b ≈ 4, n ≈ 0.8  (SI, v in m/s)
h        = max(h_free, h_forced)
```

**For a moving vehicle, `v_wind` must include vehicle speed.** A car at 100 km/h has a completely
different hood temperature from the same car parked. This is a first-order effect for automotive IR,
it is two lines of code, and it is routinely forgotten.

## Thermal inertia and crossover

`P = sqrt(k·ρ·c_p)` controls diurnal swing amplitude. Low P (dry sand, foliage, thin painted metal):
fast, large swings. High P (water, concrete, wet soil): sluggish, damped.

**Thermal crossover** — contrast collapsing near dawn and dusk as different materials pass through equal
apparent temperature — falls out of this automatically. It is one of the most operationally important
phenomena in thermal imaging.

Make it an acceptance test: *does the scene go flat around 0600 and 1900?* If not, the thermal solver
is not actually running, whatever the code says.

## Weather and spin-up

**One `WeatherSeries` object feeds both the thermal solver and the atmosphere model.** Nothing may allow
summer weather in one and winter in the other. Enforce by construction — both take the same injected
object, neither reads a file itself.

Fields: air temperature, relative humidity, wind speed, cloud fraction, direct and diffuse solar
irradiance, optionally precipitation. Hourly is enough; interpolate.

**Spin-up is mandatory.** Initialise to air temperature 24–48 h before `t = 0` and integrate forward.
Without it, night scenes are wrong — surfaces have no thermal history and everything sits at ambient.
Cheap (a few thousand steps per material class) and it is the difference between plausible and correct.
Cache spin-up results keyed by (material, weather hash, start time).

## The vehicle: five regimes, not one material

A car is not one thermal object. Model it as five:

| Regime | Solver | ΔT over ambient | Notes |
|---|---|---|---|
| Skin panels | `LumpedTwoNode` | environment-driven | Thin, low mass, responds in minutes |
| Engine bay | `Prescribed` | +40 … +90 K | Internal generation + forced airflow. **Measure, don't model.** |
| Driveline (manifold → tips) | `Prescribed`, gradient along path | +80 … +250 K | Hottest at the manifold. Strongest MWIR signature on the vehicle |
| Brakes and tyres | `Prescribed` from vehicle dynamics | brakes +50…+400 K, tyres +10…+35 K | Brakes are the fastest transient in the scene |
| Glazing | `LumpedTwoNode`, coupled to cabin | cabin-driven | Different surface entirely per band |

Plus one more node people forget: **cabin air**. A car in sun becomes a greenhouse, and the cabin drives
roof and glass temperatures from the inside. Model the vehicle as a shell with nothing behind it and
daytime roofs come out too cold. One extra node fixes it.

For the engine bay and exhaust: thermocouples on a warm-up cycle produce a driven schedule more accurate
than any lumped model, and remove the largest uncertainty in the scene. That is a parallel task someone
can run while rendering work proceeds.

## Environmental heat traces

These are signature phenomena of the band and they routinely confuse detectors trained on synthetic data
that lacks them. Implement as decaying overlays on the ground temperature field:

- Warm tyre traces where a vehicle drove (+3…+10 K, decaying over 5–20 min)
- Cool body shadow where a vehicle was parked (−3…−8 K)
- Recently occupied seat, recently touched surface

They are cheap and they matter more for perception realism than another decimal place of radiometry.

## Tests

- **Analytic equilibrium**: constant forcing, no solar → steady state matches the analytic root of the
  balance equation to 0.1 K.
- **Energy conservation**: over one integration step, energy in − energy out equals stored change, 1e-6.
- **Stability guard**: constructing a solver with a too-large `Δt` raises, rather than diverging.
- **Diurnal shape**: a 24 h run of dry asphalt produces max near 1400–1500 local and min near dawn,
  with a swing in a plausible range.
- **Crossover**: two materials of very different thermal inertia cross within an hour of dawn and dusk.
- **Vehicle speed effect**: hood temperature at 0 m/s and 28 m/s differ by a meaningful margin.
- **Spin-up convergence**: 48 h spin-up and 96 h spin-up agree to < 0.5 K at t = 0.
- **Weather coupling**: constructing a scene with mismatched weather sources is impossible by API design
  (test that the constructor requires the shared object).
