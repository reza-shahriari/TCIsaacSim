# ADR 0043 — Which ε the §16.2 scalars mean, and which one the solver gets

**Status:** Accepted
**Date:** 2026-09-15

## Context

§16.2 tabulates an emissivity per material and does not say which emissivity. There are three
different numbers hiding behind that one column, and they differ by more than the table's own
precision:

* **ε(0)** — normal incidence. What a camera reads looking straight down at a surface.
* **ε_B(θ)** — directional, band-effective. What a camera reads at an angle, and what stage 1
  needs per pixel (M7.7).
* **ε_hemi** — hemispherical, `2∫₀¹ ε(µ) µ dµ`. What an energy balance needs, because a surface
  radiates into the whole sky and not along one ray (§6.1).

For water in the Boson band those are **0.990**, a curve collapsing to 0.70 by 80°, and **0.951**.
Handing a solver the first of them overstates its radiative cooling by about 4 %.

## Decision

**1. The §16.2 scalars are read as ε(0), normal incidence at the band's reference temperature.**
That is what a datasheet and a handbook table almost always mean, and it is the one of the three
that a material file can carry without also carrying an angular model.

**2. The thermal solver gets ε_hemi and cannot reach the directional value.**
`total_hemispherical_emissivity(material, T)` is the entry point, and M6.7's `ThermalProperties`
populates its emissivity from it — never from `optical.emissivity_per_band` directly, and never
from anything authored under `thermal:` at all.

**3. The integral is written in µ = cos θ.** The substitution turns `cos θ sin θ dθ` into `µ dµ`,
so the integrand has no trigonometry in it and Gauss–Legendre converges on something
polynomial-like rather than oscillatory. Measured: 8, 16 and 64 nodes agree to better than **1e-6**
on every material tested, so the default 32 is far past convergence rather than a guess.

**4. The sign is not assumed.** ε_hemi is **below** ε(0) for a dielectric and **above** it for a
metal — aluminium's is 1.29× its normal value, because its ε rises with angle. A model that
clamped ε_hemi ≤ ε(0) would be right for every dielectric in the library and wrong for every
metal, and would stay wrong quietly.

**5. The total form reports how much of itself is an assumption.** At 300 K, **61 %** of Planck's
weight lies outside every configured band — nearly all of it beyond 13.5 µm — and is filled by
extending the nearest band's value outwards. That is the standard assumption for a thermal solver
and is probably close to right for a dielectric, but `TotalHemispherical.extrapolated_fraction`
returns it, because a solver that cannot see how much of its ε rests on an assumption cannot
report its own uncertainty.

**6. A band carrying negligible Planck weight is skipped, not queried.** At 300 K, NIR and SWIR
hold about 1e-9 of the exitance between them. Asking a material for its angular model there asks a
question whose answer cannot move the balance — and it can *fail*, because §4.2's Level C bound is
a statement about thermal-band roughness and a material may legally violate it in a band the
solver will never integrate. The threshold is 1e-4 of the weight.

## What this measured

| quantity | value |
|---|---|
| water (R1 n/k at 10 µm): ε(0) | 0.9898 |
| water: **ε_hemi** | **0.9511** |
| aluminium (n 25, k 68): ε_hemi / ε(0) | **1.29** |
| GL 8 vs 64 nodes | agree to < 1e-6 |
| Planck weight outside all bands at 300 K | **61 %** |

**The roadmap's "> 0.3 K equilibrium difference" is unreachable, and that is a property of the
balance.** ΔT between ε = 0.990 and ε = 0.951 vanishes at both convective limits — with h → 0 the
radiative equilibrium is `T = ε_sky^{1/4} T_air`, independent of ε, and with h → ∞ the surface is
pinned to the air — so it has a maximum in between. Measured: **0.230 K at h ≈ 4 W m⁻² K⁻¹**. The
test finds that maximum over h rather than asserting a single point, which is a stronger statement
and one that survives a change of h. 0.23 K is small; it is also in the same direction for every
water pixel of every maritime scene at every hour, which is the argument for rule 2.

## Consequences

* M6.7's `ThermalProperties` takes its ε from `total_hemispherical_emissivity`, which makes the
  material's angular model a *hard dependency* of the thermal solver — a material with no valid
  angular model cannot be given to it, which is the intended friction.
* Enforcing §4.2's Level C bound (M7.7) means `bare_aluminium` and `asphalt_dry` currently raise.
  Aluminium needs a Level A n/k table (M7.5); asphalt needs Level B with a ≈ 0, which is what §4.2
  itself prescribes for rough dielectrics.

## Revisit when

* A material library entry carries a measured total hemispherical emissivity. Then the extension
  beyond 13.5 µm can be checked rather than assumed, and `extrapolated_fraction` becomes a
  reported error bar instead of a warning.
* The far-infrared matters for its own sake — a cryogenic scene, where Planck's peak moves out
  past 20 µm and the extrapolated fraction approaches 1.
