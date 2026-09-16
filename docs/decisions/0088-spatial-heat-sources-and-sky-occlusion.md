# 0088 — Spatial heat sources: a computed configuration factor, and the sky the body blocks

Date: 2026-09-15
**Status:** Accepted
Roadmap: MP.2 (§6.1, §6.6)

## Context

ADR 0087 gave a surface a grid of cells. Nothing yet makes their forcing differ, so a
`PlanarThermalField` over a bonnet still solves to a flat answer — the machinery is spatial and the
input is not.

Two spatial sources are needed for a vehicle scene, and they turn out to be the same geometry
problem: an **engine bay** radiating up onto the bonnet skin above it, and a **warm underbody**
radiating down onto asphalt. Both are "a plane surface element exchanging radiation with a parallel
rectangle".

§6.6 supplies the source *temperatures* (`engine_bay`: +65 K, τ_rise 750 s) and says they are
scripted, not predicted. It says nothing about how that heat is distributed across the panel above,
because the spec never contemplated a panel with more than one temperature.

## Options considered

1. **A Gaussian with a fitted width.** One line, and it produces a picture that looks right. The
   width would be a free parameter fitted to nothing, and it would be the parameter controlling the
   single most visible feature of the frame.
2. **A closed-form configuration factor to a parallel rectangle.** Exact, standard (Howell C-11),
   has an independent check (brute-force quadrature of the defining integral), and needs only the
   block's dimensions and its distance below the panel — all of which the scene already knows.
3. **Ray-traced form factors.** More general, needed only when the geometry stops being two
   parallel rectangles, and considerably more machinery.

## Decision

Option 2. `irsim.thermal.spatial_sources.corner_view_factor` is the closed form, extended odd in
each argument so the four-corner superposition gives an arbitrarily offset rectangle;
`patch_view_factors` applies it to a `PlanarPatch`. One kernel serves both the bonnet and the
asphalt. Non-parallel geometry **raises** rather than being evaluated with a formula that does not
apply — the result would be a plausible number, which here is the worst kind of wrong.

The constant is 1/(2π) and was **verified against a brute-force quadrature** of
`c²/(π(x²+y²+c²)²)` rather than taken from memory. A view factor wrong by a factor of two gives a
gradient of exactly the right shape and half the right size, which no image would reveal.

**The second decision is the one that is easy to omit.** `occluded_longwave_flux` returns

    F · ε_s · (ε_r σ T_r⁴ − L_down/V_s)

— the **net** change, not the source term alone. A hot body standing over a surface both radiates
onto it *and* removes the sky it was seeing. Keeping only the first term invents energy.

## Consequences

**What this buys, beyond not being wrong:** the occlusion term is *dominant* on a clear night. A
295 K underbody over asphalt under a 245 K sky adds about half of what the naive source-only term
would claim, and under an overcast 288 K sky only about a tenth of it. That difference is the
reason a parked car leaves a warm car-shaped patch on asphalt **before its engine has ever run** —
a well-known feature of night parking-lot thermal imagery that a source-only model cannot produce
at all, and it is now a consequence of the geometry rather than something anyone authored.

It also settles a scene-design question: a demo meant to show *engine* heat must be shot under
overcast, where the occlusion term nearly cancels. Under a clear sky the car-shaped patch would be
there in the first frame and the engine would be a modest addition to it.

**What stays hard.** ε_s multiplies inside this function because `FacetSolver.net_flux` adds
`q_internal` **unweighted** — it is an internal deposition, not an incident irradiance. That is a
seam a caller can get wrong, and it is documented at the call site rather than defended by a type.

**The error introduced.** Three approximations, none bounded by measurement here: the exchange is
single-bounce (no inter-reflection between the body and the surface, which for ε_s ≈ 0.95 asphalt
and a cavity-like bay is a sub-percent term); both surfaces are isothermal over the rectangle,
which the engine bay is not; and the geometry is a rectangle, so a body with wheels and a
transmission tunnel is one box. All three are small against the fact that the alternative in use
until now was *one temperature for the whole panel*.

## Revisit when

A scene needs a non-parallel pair (a wall beside a radiator, a wheel arch), or Tier 4 shows a
vehicle-vs-ground bias that survives the schedules — at which point option 3 is the next step.
