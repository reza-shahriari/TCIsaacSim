# ADR 0016 — Optics self-emission model and what T_housing means

**Status:** Accepted
**Date:** 2026-09-11

## Context

Warm optics add in-band power to every pixel (§8.2). In LWIR at room temperature this is about
8 % of the scene signal for a τ = 0.92 lens, and its drift with housing temperature is the physical
origin of shutterless drift and the need for flat-field correction (§11.2). The spec offers both the
general N-element sum and a single-lens simplification, and it uses "T_housing" without saying which
body it is (the lens, the barrel, the FPA package, or the FPA itself).

## Options considered

1. **Single-lens form** Φ_self = A_d Ω_eff (1 − τ_opt) L_B(T_housing) — one parameter beyond the
   config; treats the whole train as one grey body at one temperature; ρ_lens = 0.
2. **N-element stack** with per-element τ, ρ, T and downstream attenuation — exact for a known train;
   needs data no datasheet provides (per-element temperatures, AR-coat reflectances).
3. **Empirical drift** (a random walk on the offset) — reproduces the symptom, not the physics, and
   cannot respond to a modelled housing temperature.

## Decision

Both 1 and 2 are implemented in `irsim.optics.self_emission`; the **single-lens form is what the
pipeline uses** (M3.9) with:

- **ρ_lens = 0**: AR-coated germanium reflects ~1–2 % per surface in band; the reflected term would
  be the detector seeing its own cold reflection (narcissus), which is a separate optional effect,
  not part of the emission budget. So ε_lens = 1 − τ_opt exactly.
- **T_housing is a lumped body node** for lens + barrel, distinct from the FPA temperature
  (T_FPA drives detector responsivity and dark current, not self-emission). Its evolution is set by
  `optics.housing_temp_mode` (§12.2): `fixed` (a config value), `ambient` (follows T_air), `coupled`
  (first-order lag on ambient plus self-heating, M9.3). This module takes L_B(T_housing) as an input
  and does not know the mode.
- Elements enforce Kirchhoff closure (ε = 1 − τ − ρ derived; τ + ρ > 1 raises) and the stack applies
  downstream-only attenuation; the stack is the reference the single-lens form is tested against.
- Power is added in radiance/power space, never kelvin (non-negotiable #3).

Known answer recorded: for τ = 0.92 at F/1, a +1 K housing step shifts apparent temperature by
(1 − τ)/τ = 87 mK when the transfer assumes the calibration housing temperature. This is the
per-kelvin drift the NUC/FFC model (M9) must remove.

## Consequences

The 8 % self-emission fraction is exact only if the train really is one grey body at T_housing; a
warm rear element and a cool window average to something else. The error is bounded by the
temperature spread across the train times (1 − τ) — a few mK per kelvin of spread — and is invisible
after FFC. Narcissus and cold-shield terms are separate optional stages (O10, O11).

## Revisit when

A cooled MWIR camera is modelled (cold shield replaces most of the housing term, M3.4/ADR 0066), or a
measured shutterless-drift figure for the Boson disagrees with 87 mK/K by more than 20 %.

## Addendum (M9.3, 2026-09-12): the housing node as built

`irsim.optics.HousingTemperature` implements the three modes this ADR named. Two choices are worth
recording because neither is forced by the physics:

- **The coupled node delegates to M6.6's `NewtonCoolingSolver`** rather than integrating itself.
  That solver already provides the exact exponential update, which is unconditionally stable and
  overshoot-free at any step size, and the housing is precisely the linear, radiation-free,
  solar-free case it was scoped for — it sits inside the camera body, not exchanging with the sky.
  A second lumped-node implementation would be two things to keep in agreement for no physical
  gain. The FPA node (ADR 0053) stays on its own RK2 integrator because it is independently
  parameterised and is not the same node; it is the deliberate exception, not a precedent.

  The one visible consequence is inherited: `NewtonCoolingSolver` evaluates the ambient at the step
  *midpoint*, so a housing whose τ is far below the tick relaxes onto the midpoint air temperature
  rather than the endpoint one — a half-tick lag, bounded by how far the air moves in half a step
  (at 600 s ticks under a 10 K diurnal swing, ~0.1 K). For any realistic housing τ this is far
  below the lag being modelled. It is pinned as a derived bound in
  `tests/unit/test_housing_temperature.py`, not hidden.

- **`housing_temp_mode: coupled` now *requires* `housing_tau_s`** (sensor schema v6). A lumped node
  with no time constant cannot be integrated, and defaulting the lag would put a number nobody
  authored into the drift the whole M9 chain is built on. This mirrors the rule ADR 0053 set for
  `fpa_temp_mode: coupled`. The consequence is that `configs/sensors/flir_boson_640_lwir.yaml`,
  which declared `coupled` and authored neither parameter, now carries `housing_tau_s: 900 s` and
  `housing_self_heating_k: 4 K` — both **ESTIMATED**, with no published figure for either, and
  both candidates for the ME.3 between-FFC drift band to constrain.

The node registers with the `Scene` through the `.weather` property, so non-negotiable #6 covers it
with no new code in `irsim.scene`: a housing built on a different `WeatherSeries` than the
atmosphere and the thermal solvers is refused at construction. That matters more here than it looks,
because a wrong housing temperature produces a smooth radiometric pedestal rather than a
recognisable artefact — nobody would catch it by looking at the image.
