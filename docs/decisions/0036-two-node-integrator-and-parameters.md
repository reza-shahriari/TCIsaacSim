# ADR 0036 — The two-node solver: integrator, the parameters §6.4 leaves undefined, and its bound

**Status:** Accepted
**Date:** 2026-09-15

## Context

§6.4 gives the practical solver as two lumped nodes,

    C₁ dT₁/dt = net_flux(T₁) − (T₁ − T₂)/R₁₂
    C₂ dT₂/dt = (T₁ − T₂)/R₁₂ − (T₂ − T_deep)/R₂d

with `R₁₂ = δ₁/(2k) + δ₂/(2k)`, "explicit RK2 at a 1–60 s step", and a stability limit
`Δt < 2C₁/(h + 4εσT³ + 1/R₁₂)` which it says "for thin painted metal (C₁ ~ 5 kJ m⁻² K⁻¹) lands
around 60–200 s".

Three things are undefined — **R₂d**, **T_deep**, and what `h` and `T` in the bound should be —
and one is wrong.

## Decisions

**1. RK2 is the midpoint rule, and the reason is the T⁴ term, not the order.** Forward Euler is
also first-order-cheap and would be defensible on accuracy grounds alone. It is not defensible on
*bias*: the radiative term makes dT/dt fall as the surface warms, so Euler consistently overshoots
a warming surface and undershoots a cooling one, and the error does not cancel over a diurnal
cycle — it inflates the swing amplitude, which is precisely the quantity §6.3's thermal-crossover
acceptance test looks at.

**2. R₂d defaults to infinity — an adiabatic back — and a finite value requires T_deep.**
`TwoNodeProperties` raises if one is given without the other. R₂d without T_deep is a conduction
path to an unspecified reservoir; a default of "some plausible ground temperature" would let a
scene lose heat to a number nobody chose, and it would do it slowly enough to look like physics.

**3. T_deep is a scenario input, not a derived quantity.** For ground it is the annual-mean air
temperature at the damping depth; for a panel it is the cabin or the air behind it. The solver
does not guess: the caller that knows what is behind the surface supplies it.

**4. The stability guard is evaluated at the worst case of both arguments and enforced in the
constructor.** `h` is the largest convection coefficient the injected weather can produce, and
T is `GUARD_T_MAX_K = 400 K`, because `4εσT³` grows with T and a guard has to hold for the whole
run. It is checked when the solver is built rather than on each step, because a caller that has
chosen a tick has chosen it once, and discovering on frame 4000 that the tick was unstable is
discovering it after the scene has been rendered.

## The factor of 400 (spec issue S39)

§6.4's formula is right and its evaluation is not. For the §16.2 car-paint row — 1.2 mm, ρ 7800,
c_p 470, k 45 — with both nodes at the authored thickness:

| term | value |
|---|---|
| C₁ | 4 399 J m⁻² K⁻¹ |
| R₁₂ = δ/(2k) + δ/(2k) | 2.667e-5 m² K W⁻¹ |
| **1/R₁₂** | **37 500 W m⁻² K⁻¹** |
| h (worst case) + 4εσ(400 K)³ | 30 + 12.9 = 42.9 |
| ratio | **874×** |
| **bound** | **0.235 s** |

The quoted 60–200 s is what the **single-node** bound `2C₁/(h + 4εσT³)` gives at high wind — the
same formula with the term that dominates it by three orders removed. The bound is not a nuisance
to be worked around: it is telling you that a 1.2 mm layer on a conductive substrate equilibrates
across itself in a quarter of a second, and that resolving that explicitly at a scene tick is not
a sensible thing to want.

**So a thin panel is not this solver's problem.** It is a single node (M6.7) with a resistive back
boundary, whose bound is the comfortable `2C/(h + 4εσT³)` — 264 s for the same paint. The two-node
solver is for surfaces with real depth, where it is comfortable anyway: asphalt's bound is 3 456 s
and concrete's 7 067 s, both far past a 1 Hz tick.

## A second measured correction: which variable orders the diurnal swing

The literature's thermal inertia `P = √(ρ c k)` is the right measure for a **semi-infinite** solid,
where heat has unbounded depth to diffuse into. For finite layers on an adiabatic back it is
exactly wrong: measured over a synthetic 24 h cycle, car paint has the **highest** P of the three
tested (12 800 against concrete's 1 700, because it is backed by steel) and the **largest** swing
(58.6 K against 18.1 K). The ordering variable is the areal heat capacity **C = ρ c δ** — 4.4, 101
and 202 kJ m⁻² K⁻¹ — and the swings order inversely by it, exactly. P becomes the right variable
again once the back boundary is a deep reservoir rather than a wall, which is M6.10's ground case.

## Consequences

* The equilibrium is found on the **surface node alone**: at steady state every flux through the
  stack is equal, so T₂ is determined by T₁ and the two-node root reduces to the one-dimensional
  bisection of M6.7, keeping its uniqueness argument intact.
* Verified against `scipy.linalg.expm` on the linearised 2×2 system: 6 h of stepping agrees to
  **under 10 mK**, so the RK2 truncation error on the non-linear part is small where it matters.
* Energy across the pair closes to **1e-6**: what enters the surface minus what leaves the back
  equals what the two nodes stored.

## Revisit when

* A scene needs more than two nodes — a multi-layer wall, or a ground profile deep enough that the
  damping depth matters. The equations generalise; the bound gets worse as layers get thinner, and
  the same "is this really an explicit problem?" question applies to each new pair.
* The deep boundary becomes dynamic (a cabin node, M6.14). Then T_deep is another state and R₂d is
  the panel-to-cabin resistance, which is a coupling and not a boundary condition.
