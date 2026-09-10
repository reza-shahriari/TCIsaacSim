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
