# ADR 0038 — Vehicle thermal architecture: a cabin node, and heat sources that are scripted

**Status:** Accepted
**Date:** 2026-09-15

## Context

§6.6 lists nine active heat sources with ΔT ranges and time constants, and says of them: *"These
are scripted, not predicted, and they are where most of the useful signal lives."* It does not say
what "scripted" means in code, nor what is behind a vehicle panel.

## Decisions

**1. Heat sources are schedules driven by a `VehicleState` trace, not a combustion model.**
Predicting an exhaust-tip temperature needs an engine map, a catalyst model and a flow solver, and
the result would still be authored parameters wearing a physics costume. A first-order rise to
§6.6's own ΔT_max with §6.6's own τ is honest about what it is. Every default in
`VEHICLE_HEAT_SOURCES` is a midpoint of a §6.6 range and is ESTIMATED; the module says so once
rather than nine times.

**2. Two exceptions, where a closed form exists and scripting would throw it away.**

* **Brakes are an energy deposit.** A braking event dumps `f·½m(v₁² − v₂²)` into the discs, so
  ΔT = E/(m_disc c_p) is arithmetic and only the *cool-down* is a time constant. Scripting a brake
  temperature directly would make it independent of how hard the car actually braked — the one
  thing a braking cue is supposed to carry. A 1600 kg car stopping from 30 m/s puts **162 K** into
  an 8 kg disc, and a stop from twice the speed puts in exactly four times as much.
* **Tyres follow speed, not time.** §6.6's "+10 … +35 K, rises with speed" is a steady-state
  relation like ADR 0072's aerial sources: the contact patch reaches what its current speed
  implies. Defensible only where the speed varies slowly against the tyre's own 10–30 min
  constant, and the docstring says so.

**3. The exponential step is exact, so a schedule does not depend on the trace's sample rate.**
`SourceHistory.step` uses `1 − e^{−dt/τ}` rather than a forward difference. A trace logged at 1 Hz
and the same trace at 10 Hz agree to **1e-9**; with a forward difference they would not, and the
engine-bay temperature in a scene would depend on how finely someone happened to log the drive.

**4. A vehicle panel's back boundary is a cabin node, not ambient and not adiabatic.**
`CabinNode` is one lumped volume with three terms — solar gain **through** the glazing, conduction
from every bounding panel, and infiltration — solved **together** with its panels rather than
alternately. Stepping them alternately is stable at a 1 s tick and wrong at a 60 s one, in the
direction that under-predicts the greenhouse, which is the effect the node exists to produce.

**5. The cabin's capacity is the trim, not the air.** 3 m³ of air is 3.6 kJ K⁻¹, which a sunbeam
moves in seconds. The seats, dashboard and trim are what actually store the heat (25 kJ K⁻¹,
ESTIMATED), and leaving them out gives a cabin that tracks the sun instantly and cools instantly
too.

**6. ⚠️ The glazing must appear in the panel list, and the constructor enforces it.** Glass is both
the cabin's solar **inlet** and one of its largest conduction **paths** — single glazing is about
5.8 W m⁻² K⁻¹, which for 2.6 m² is 15 W K⁻¹ against 2 W K⁻¹ of infiltration. A cabin given the
inlet without the path reaches **102 °C** at noon instead of **80 °C**: not obviously wrong, just
wrong, which is why it is a constructor error rather than a note.

## What this measured

| quantity | value |
|---|---|
| roof at noon, with cabin vs adiabatic back | **+4.8 K** (§6.6 asks for > 2 K) |
| cabin air at noon, 850 W m⁻² | **80 °C** (a sealed car really does reach 60–80 °C) |
| glazing panel at noon | 38 °C — α_sol 0.10, it transmits rather than absorbs |
| cabin τ, panels pinned | **9.0 min** |
| cabin τ, panels free | **12.0 min** — 33 % slower, because a cooling cabin drags its panels down and they feed heat back |
| cabin and roof on a clear night | **> 2 K below ambient** — why cars frost when the air does not |
| coupled equilibrium vs `fsolve` | < 0.1 K, residual < 1e-9 |
| energy across cabin + panels | closes to **1e-6** |

The pinned-versus-free time constant is worth keeping: the lumped formula
`C_cab/(Σ A/R + ṁ c_p)` is exact for the system it describes, and a scene quoting it is quoting a
number 33 % shorter than the one it will observe.

## Revisit when

* A scene needs the exhaust **plume** rather than the pipe. §6.6 flags it as gas emission, not a
  surface, and band-selective (CO₂ at 4.3 µm) — which makes it a radiance-space injection like
  MS.6's point targets, not a temperature.
* The cabin needs to be more than one node — a dashboard that sees the sun directly is much hotter
  than the air behind the seats, and that difference is visible through a windscreen.
