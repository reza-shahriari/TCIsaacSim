# ADR 0005 — Stefan–Boltzmann identity tolerance and the fractional-exitance series

**Status:** Accepted
**Date:** 2026-09-10

## Context

`docs/physics-model.md` §15 asks for the Tier 1 identity ∫L dλ = σT⁴/π; the ir-sim-testing skill
lists analytic identities at 1e-12 relative; the scaffold's test met it only at 2e-5 with a trapezoid on
0.1–1000 µm, limited by tail truncation and trapezoid error, with no ADR for the loosened number.
Separately, `fractional_exitance` summed the exponential series only, which converges as 1/N³ in the
Rayleigh–Jeans tail (x = C2/λT ≪ 1), so it could not reach double precision there — and it accepted
metre-valued wavelengths silently.

## Options considered

1. **Keep 2e-5 with a trapezoid** — no code change; hides any error below 2e-5 and treats a numerical
   artefact as a physics tolerance.
2. **Simpson quadrature on 0.1–200 µm plus the closed-form tail** (σT⁴/π)(1 − F(200 µm)) from
   `fractional_exitance` — two independent routes (quadrature vs series) covering the whole spectrum;
   reaches 1e-6 easily. Requires the series to be accurate at small x.
3. **Adaptive quadrature to 1e-12** (scipy `quad` over the transformed variable) — reaches the skill's
   number but the oracle then depends on scipy's tolerances rather than on two independent derivations.

## Decision

Option 2, with the tolerance set at **1e-6 relative** (the spec's number) rather than the skill's 1e-12.
Measured closure is 3e-11 at 250, 300 and 800 K, so the margin is 4–5 orders; 1e-6 is chosen because
it is the spec figure and because the remaining error is Simpson's, not physics — any physics error
this test exists for (wrong C1L, dropped π, wrong σ) is a factor of 3 or more.

`fractional_exitance` now uses two expansions of the dimensionless Planck integral ∫ₓ^∞ t³/(eᵗ−1) dt:
the exponential series for x ≥ 0.5 (with a geometric tail bound as the stopping rule) and π⁴/15 minus
the Bernoulli/Taylor series of the complementary integral for x < 0.5. The branches agree at the
crossover to 1e-13 in a test. Both `fractional_exitance` and `band_radiance_tophat` validate units
(micrometres and kelvin) like the spectral functions.

**Correction to the roadmap's verification cell for M0.4.** "F(5000 µm) = 1 to 1e-12" is not a valid
test: F never reaches 1 at finite wavelength. At 300 K, 1 − F(1000 µm) = 5.6e-6 and 1 − F(5000 µm) =
4.5e-8, physically. The convergence test instead checks F(1000 µm, 6000 K) against the Rayleigh–Jeans
limit (15/π⁴)(x³/3 − x⁴/8), residual 7e-10, to 1e-6 relative.

## Consequences

The Stefan–Boltzmann test is a genuine two-route identity and runs in 30 ms. `fractional_exitance` is
double-precision accurate over the whole 0.1–1000 µm × 1–6000 K domain, so it can serve as the oracle for
the band LUT (M1). No physical error is introduced.

## Revisit when

A band integration oracle needs better than 1e-12 (unlikely), or the validated wavelength range grows
beyond 1000 µm (the small-x branch already covers it; only `_validate` bounds would change).
