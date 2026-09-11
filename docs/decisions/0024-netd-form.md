# ADR 0024 — NETD form: photon-count / +1 only; the 4F² datasheet form is a labelled conversion

**Status:** Accepted
**Date:** 2026-09-11

## Context

§9.4 gives two NETD forms: the datasheet/thermal-detector form NETD = 4F² NEP/(A_d L') and the
photon-count form NETD = σ/(∂S/∂T) with the +1 aperture factor. §8.1 says "NETD scales as F²". At
F/1.0 the paraxial form is 4F²/(4F² + 1) = 0.8 of the +1 form (spec issue S7). A simulator that mixes
them is 20–25 % wrong in noise magnitude at the f-number every uncooled LWIR camera uses.

## Options considered

1. **Use the datasheet 4F² form** because published numbers are quoted that way — internally
   inconsistent with the +1 irradiance (non-negotiable #5) and 20 % optimistic at F/1.
2. **Photon-count / +1 form everywhere; keep the 4F² form only as a documented conversion** for
   comparing against published figures.

## Decision

Option 2 (`irsim.detector.netd`, `irsim.detector.figures_of_merit`):

- NETD(T) = σ_total(T) / (∂S/∂T)(T) in the detector's physical signal unit: pixel power (W) for
  bolometers, photoelectrons for photon FPAs. ∂S/∂T comes from the same transfer as the signal
  (A_d · π τ/(4F² + 1) · ∂L_B/∂T, times η t_int for photon FPAs), so the aperture factor enters only
  through `irsim.optics`. σ² = Poisson terms (photon) + one lumped Gaussian.
- `netd_datasheet_form_k` (4F² NEP/(A_d M')) and `datasheet_over_photon_count_ratio` (4F²/(4F² + 1) =
  0.8 at F/1.0: the paraxial form understates NETD) exist for comparison with published numbers and
  are never used by the noise chain. The §9.4 symbol
  L' ("change in power per unit area radiated by the object") is the in-band *exitance* derivative
  M' = π ∂L_B/∂T, not the radiance derivative; with that reading the two forms differ by exactly
  4F²/(4F² + 1). The 4F² term is written as π/Ω_eff − 1 so the aperture guard still holds.
- Bolometer noise bandwidth: ENBW = 1/(4τ_th) for the first-order membrane; the sampled per-frame
  IIR's ENBW (1/2Δt · a/(2 − a)) is also provided and is ~18 % lower at 60 Hz with τ_th = 10 ms — both
  reported. Johnson and temperature-fluctuation floors are computed from §10.1 and **reported, not
  enforced** (the anchor, ADR 0025, sets the magnitude; Appendix A #7).
- The §10.1 1/f term is represented by the drift model (M9.4), not by σ_total (ADR 0054).

## Consequences

Predicted NETD falls with scene temperature exactly as the in-band derivative ratio for
scene-independent noise (0.576 from 300 to 373 K in 7.5–13.5 µm; the [R9] 39 → 23 mK anchor), and the
read-limited NETD(F/2)/NETD(F/1) is 3.4, not 4.0. Anyone comparing to a datasheet quoted in the 4F²
convention must multiply it by 1.25 at F/1 to get the +1-form value (the conversion function gives
the factor).

## Revisit when

A camera's NETD is published together with its measurement convention and disagrees with the +1
form by more than the 25 %; or a cooled system needs the cold-shield-limited background term in
σ_total (ADR 0066).
