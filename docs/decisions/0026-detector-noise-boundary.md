# ADR 0026 — Detector/noise boundary: TVH in the detector, correlated terms in the noise stage, quantise last

**Status:** Accepted
**Date:** 2026-09-11

## Context

Noise has physical sources in the detector (Poisson shot, Johnson/read) and structural sources in
the readout (row/column striping, fixed pattern, frame bounce; §10.2). §13.4 SPG#4 emits `DN_ideal`
and SPG#5 adds noise *after* quantisation, which contradicts §9.1 (electrons before the floor) and
non-negotiable #3 (spec issue S10). Something must say which module generates what and in which unit.

## Options considered

1. **All noise in one stage after the detector, in DN** — simple, but shot noise is a property of
   electron counts and its variance depends on the signal; adding it in DN after quantisation loses
   that and double-quantises.
2. **Per-pixel temporal terms in the detector (its physical unit), correlated terms in the noise
   stage (signal units), quantise once at the end** — each term where its physics lives, one ADC.
3. **Everything in the detector** — the readout structure (striping, drift, FFC) has no place in a
   pixel physics model.

## Decision

Option 2:

- `DetectorFrame(signal_dn, dn, sigma_dn)` is the uniform detector output: the un-quantised signal
  in DN units *with the per-pixel temporal noise included*, its quantised DN, and the per-pixel
  σ_TVH in DN units.
- `PhotonDetector.response`: N_e = η t_int Φ_q + N_dark + N_bg; Poisson(N_e) (PCG64 `SHOT` stream)
  + Gaussian read (hashed `TVH` stream) in **electron space**, clipped at 0, then DN units, then the
  shared quantiser. Dark current is Arrhenius (§9.1) with band gaps in `constants.py`
  (InSb 0.23 eV, InGaAs 0.75 eV); a T_FPA below 30 K is refused as a Celsius slip.
- `MicrobolometerDetector.response`: static transfer + the anchored Gaussian σ (pixel-power units,
  ADR 0025) in signal space via the DN-per-W gain (ADR 0019), then the quantiser. The τ_th IIR and
  T_FPA gain/offset coupling are M9.
- The **noise stage** (M4.9) takes `sigma_dn` and adds the six correlated components (V, H, VH, TV,
  TH, T) as σ_i = r_i · σ_TVH on the un-quantised signal, then quantises once. The detector's
  `dn` is therefore the "TVH-only" image; the pipeline's `dn16` is the noise stage's.
- **Quantise last, always.** Nothing adds noise to a uint16.

## Consequences

Two-blackbody NETD on the detector alone reproduces the anchor at 300 K and the derivative ratio
0.576 at 373 K for the bolometer, and the shot-noise curve for the photon detector; the DN noise
std is scene-independent for the bolometer while NETD in kelvin falls — the pair of facts that proves
the noise is in signal space. The photon path's Poisson draw is sequential (ADR 0022) and the GPU
will compare it statistically.

## Revisit when

A counter-based Poisson is adopted (bit-exact GPU comparison of photon cameras), or bolometer noise
gains a signal-dependent term (bias-heating, M9) that must move into the detector.
