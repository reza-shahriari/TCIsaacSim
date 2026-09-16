# 0079 — Sea-water optical constants and the Cox–Munk slope model

Date: 2026-09-16
**Status:** Accepted
Roadmap: M7.5, MM.1, MM.2, RP.4 (§4.2, §5.3, §7.4, §12.3)

## Context

The maritime background (ADR 0078) needs two things the rest of the project does not: optical
constants standing for **sea** water, and a statistical description of the **surface slope** inside
a pixel. Both were decided and shipped with MM.1–MM.3; neither was recorded. Two source modules and
one test file cite this ADR, and ADR 0078 explicitly defers its salinity bound to it — "Neglecting
salinity in the optical constants is bounded separately in ADR 0079" — so until now that bound
existed only as a forward reference. Supplying it is the substantive new content here; the rest is
written from the code.

The physical constraint is scale. At 3 km a Boson pixel spans 2.6 m and contains thousands of
independent wave facets, so one surface normal per pixel is not a coarse version of the right
answer — it is a different quantity. What the detector integrates is the *distribution* of facet
orientations, which Cox & Munk give in closed form from the wind alone. And because water is a
Fresnel reflector viewed at brutal incidence — a camera 20 m up sees the sea at 5 km at 89.8° of
incidence — the band-effective emissivity is doing most of the work in the answer, so which
constants feed it matters more than it would for a ground scene.

## Options considered

**For the optical constants:**

1. **Pure water, Segelstein (1981).** Freely redistributable, covers 0.65–15.6 µm in one table,
   already needed for the non-maritime water material. Wrong substance by the salinity of the sea.
2. **A salinity-corrected sea-water table.** Right substance. No redistributable dataset covering
   the LWIR window was found, which is the same wall ADR 0041 hit for soda-lime glass and paint.
3. **Fit an effective k to published sea emissivity measurements.** Reproduces the number we are
   trying to predict, so it cannot then be used to check the model.

**For the slope model:**

1. **Cox & Munk (1954) component regressions** — separate upwind and crosswind slope variances.
2. **Cox & Munk's independently-fitted isotropic total**, σ² = 3.0e-3 + 5.12e-3 U.
3. **A wave spectrum integrated to slope variance** (Pierson–Moskowitz or similar). More general,
   needs fetch and duration, and reduces to Cox–Munk for a fully developed sea anyway.

## Decision

**Optical constants: option 1, pure water, labelled as such.** `data/nk/water.csv` carries
Segelstein (1981) over 0.65–15.6 µm, retrieved from the Oregon Medical Laser Center mirror and
otherwise unmodified, with the `# source:` block the loader *requires* rather than merely prefers
(ADR 0041). Sea water is rendered with pure-water constants, and the cost is bounded below.

Three details of how they are used, each with a tempting wrong alternative:

* **n and k are interpolated separately and linearly**, never a derived quantity. Interpolating ε or
  R instead would be interpolating a nonlinear function of the data, making the result depend on the
  table's sampling rather than on the material.
* **Extrapolation raises.** A table that stops at 15.6 µm has nothing to say about 20 µm, and
  holding the last value there would put a fabricated number into a radiometric result. The single
  exception is inside a band average whose response support has already been checked to lie within
  the table, where end-holding can only touch a padding sample at which R is zero.
* **The band average goes through the one sanctioned route** (`band_average`, ADR 0010), not Fresnel
  at a "representative" 10 µm. Over a Boson's window n runs 1.28 → 1.16 and k runs 0.033 → 0.35 — an
  order of magnitude in k — and the error from a single-wavelength shortcut grows with angle, which
  is exactly where a sea surface lives.

**Slope model: option 1, the component regressions**, `upwind σ² = 3.16e-3 U` and
`crosswind σ² = 3.0e-3 + 1.92e-3 U` for a clean sea, with U the wind speed at 12.5 m.

`slope_variance` returns **both components rather than one isotropic number**, and deliberately does
not return Cox & Munk's published isotropic total. The two regressions were run separately on the
same slicks data and do not agree: the components sum to 3.0e-3 + 5.08e-3 U against a published
total of 3.0e-3 + 5.12e-3 U, a 0.8 % disagreement in variance and 0.4 % in RMS slope — far inside
their own quoted scatter, but a real inconsistency in the source. Returning the components makes a
caller that wants the published total say so explicitly rather than reach it by summing.

Today's consumer takes the isotropic **mean** of the two. The anisotropy is deferred because using
it needs the view-to-wind azimuth, which no scene currently carries. The tilt integral uses 15
Gauss-Hermite nodes, reaching past 4σ on a smooth integrand — convergence, not a compromise.

## Consequences

**The salinity bound ADR 0078 promised.** No sea-water table is checked in, so the substitution
cannot be bounded by direct comparison. What *can* be measured, and is, is the sensitivity of the
shipped result to the constants — the transfer function a future sea-water table would be plugged
into. Perturbing the whole table by **+1 %** and recomputing ε_B(θ) through the project's own
`band_directional_emissivity` over the Boson response gives:

| incidence | Δε_B for n ×1.01 | Δε_B for k ×1.01 |
|---|---|---|
| 0° | −7.5e-4 | −1.0e-4 |
| 45° | −9.4e-4 | −1.4e-4 |
| 70° | −2.4e-3 | −4.9e-4 |
| **85°** | **−3.0e-3** | −5.6e-4 |
| 89° | −9.4e-4 | −1.7e-4 |

Three things follow. **ε_B is about five times more sensitive to n than to k**, so a future
sea-water table's value of n is what matters. **The sensitivity peaks near 85°**, which is precisely
the near-horizon band a low camera spends most of its pixels on — the error is worst where the
scene is. And the magnitude is small: against a 20 K sea-to-sky apparent contrast, a 1 % error in n
costs **60 mK at 85° and 15 mK at nadir**, against a 50 mK NETD. So the bound is conditional but
usable: *if* salinity moves the LWIR n of water by less than about 1 %, the substitution costs less
than roughly one NETD at the worst angle and is invisible elsewhere. Establishing whether it does is
what a checked-in sea-water table would settle; this ADR does not claim it.

**What is not bounded** and is stated rather than hidden: the clean-sea regression is used for all
wind speeds, with no slick or whitecap correction, and ADR 0078 already records that wave shadowing
is unbounded in the same near-horizon band where the constants matter most. Those two unbounded
errors live in the same pixels and could partly cancel or could compound; nothing here says which.

**What it makes easy.** The whole maritime path is driven by wind speed off the shared
`WeatherSeries`, so the picture and the radiometry cannot disagree about the weather (CLAUDE.md #6).
Swapping in a sea-water table later is a data change: one file, same loader, same provenance rule,
and the table above says immediately what it would move.

**What it makes hard.** Any sun-glint or direction-aware work needs the anisotropy, which needs a
view-to-wind azimuth threaded through the scene — not a change local to this module.

## Revisit when

- A redistributable sea-water n/k table covering the LWIR window becomes available, or a Tier 4
  comparison against public maritime imagery shows a systematic bias in the near-horizon band.
- Sun glint is implemented, or any result becomes sensitive to the up/crosswind ratio rather than
  to the isotropic mean.
- A scene needs wind speeds high enough for whitecaps to cover an appreciable fraction of the
  surface, where the clean-sea regression stops describing the sea.
