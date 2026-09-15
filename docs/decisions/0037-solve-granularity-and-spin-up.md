# ADR 0037 — Solve granularity and the spin-up policy

**Status:** Accepted
**Date:** 2026-09-15

## Context

§6.4 says to run the thermal solver "on a fixed thermal tick (say 1 Hz) and interpolate", and that
a scene needs "a day or two" of spin-up. It does not say what a *solve* is a solve of — every
triangle, every material, every distinct facet? — nor how long "a day or two" actually needs to be,
nor what makes two spin-ups the same.

The cost is not academic. A 48 h spin-up at 60 s is 2 880 steps; doing that per triangle of a
street scene is the single most expensive thing this package can be asked to do, and almost all of
it would be recomputing identical answers.

## Decisions

**1. A solve is per (material class × orientation bin × shadow state), not per facet.** Two facets
of the same material, facing within a bin of each other, with the same shadow history, have the
same temperature — the balance has no other inputs. The facet solver runs over `(N,)` arrays so
that N is the number of *distinct* solves, not the number of triangles, and the scene maps
triangles onto them.

**2. Spin-up is cached on the materials, the weather and t₀ — and on nothing else.** The key is a
SHA-256 over the facet properties' bytes, the weather hash, t₀, the span and the step. Not the
scene, the camera, or the frame: two scenes made of the same materials under the same weather at
the same hour have the same surface temperatures, and the cache exists to say so. A **0.01 K**
perturbation of any property changes the key, because the properties are hashed as bytes rather
than rounded into buckets — a bucketed key is a key that occasionally returns someone else's
answer.

**3. Spin-up ends where the scene begins.** `spin_up(..., t0_s, hours)` integrates the `hours`
*before* t₀ and returns the state at t₀ itself, rather than a state at some earlier time the
caller then has to advance. The alternative invites two different callers to advance it by
different amounts.

**4. The default is 48 h, and it is not adequate for everything.** Measured as the difference
between a 48 h and a 96 h spin-up under the same synthetic diurnal weather:

| material | 48 h vs 96 h |
|---|---|
| thin steel (2 mm) | **< 0.5 K** |
| asphalt (5 cm effective) | **< 0.5 K** |
| concrete (10 cm) | **< 1.0 K** — reported, not asserted at 0.5 K |

Concrete at 202 kJ m⁻² K⁻¹ is still remembering the day before yesterday after two days, and
that is a property of concrete rather than a deficiency of the spin-up. The policy is to use 48 h
by default, report the residual for slow materials, and let a scenario that cares raise `hours`.

**5. float64 inside, float32 only at the boundary.** The balance subtracts terms around
400 W m⁻² to leave a residual of a few, and a diurnal run accumulates ~10⁵ steps of that. In
float32 the subtraction alone loses four significant digits before the accumulation starts.
`FacetSolver` refuses float16 outright and exposes `as_float32()` as the one place the state
narrows (CLAUDE.md #2).

## What this measured

* 64 random facets over 6 h agree with 64 separate scalar runs to **under 0.1 mK** — the
  vectorised form is the scalar one in a different loop order, not an approximation of it.
* Shadowing one facet changes that facet by more than 1 K and its neighbours by **exactly zero**,
  which is the check that vectorising has not accidentally coupled anything.
* Starting a scene at the air temperature instead of spinning up is wrong by **more than 5 K** on
  a sunlit asphalt surface at 14:00 — and wrong in a way that decays over hours, i.e. across
  exactly the part of the diurnal cycle a thermal camera is most interesting in.

## Revisit when

* The orientation binning is exercised by a real scene (M6.12). The bin width is a trade between
  cache hits and the error at the bin edges, and it should be chosen by measuring that error, not
  by taste.
* A scenario needs a material slower than concrete — a thick masonry wall, a body of water. Then
  48 h is not the default any more and the spin-up cost stops being negligible.
