# ADR 0023 — 3-D noise estimator, its tolerances, and the PSD diagnostics

**Status:** Accepted
**Date:** 2026-09-11

## Context

The NVESD 3-D decomposition (§10.2, [R28]) is both the way the synthesiser is verified (M4.3/M4.4),
the Tier 2 bench (§15), the statistic the CPU-vs-GPU noise comparison uses (ADR 0022), and the
statistic real anti-UAV video is reduced to (ME.5). The naive estimator — the standard deviation of a
directional mean — overstates small components: the mean over N_t·N_h pixels of pure i.i.d. noise
still has σ_TVH/√(N_t N_h) of spread, plus contributions of every higher-order component, so a
0.02 σ_TVH frame bounce cannot be read from a naive column of numbers. Real data adds a codec floor
(that part is ME.2b).

## Options considered

1. **Naive directional-mean standard deviations** — biased upward by leakage; the small components
   (T, TV, TH, V) come out wrong by their own size.
2. **Random-effects ANOVA mean squares with leakage subtraction** — the unbiased estimator: each
   effect's expected mean square is its variance times its averaging count plus the higher-order
   mean squares; solve from TVH downward. Negative estimates from sampling noise are clipped to 0 but
   kept in `raw_variances`.
3. **Spectral fitting of the striping lines** — needs assumptions about spectra; kept as a
   *diagnostic* (PSD lines), not the estimator.

## Decision

Option 2 in `irsim.validation.noise.decompose_3d` — the sole home of the estimator (NOISE-04/12):

- σ²_TVH = MS_TVH; σ²_TV = (MS_TV − MS_TVH)/N_h; σ²_TH = (MS_TH − MS_TVH)/N_v; σ²_VH = (MS_VH − MS_TVH)/N_t;
  σ²_T = (MS_T − MS_TV − MS_TH + MS_TVH)/(N_v N_h), and cyclically for V and H.
- Input any float32-or-better or integer cube (uint16 promoted); float16 refused.
- **Tolerances are derived from the estimator's own sampling floor**, `estimate_floors(shape, σ)`:
  std(σ̂²_Y) ≈ √(2/df_Y) · Σ_{X ⊇ Y} σ_X² / Π_{a ∈ X∖Y} N_a, and every assertion is 3× that. Two
  consequences the roadmap's flat "5 %" missed: a fixed row or column pattern of a 64-wide cube is a
  64-sample quantity, so σ_H and σ_V are known only to ~9 % (1σ) from one such cube no matter how many
  frames; and a small term inherits the sampling noise of the larger mean squares subtracted from it
  (a pure TV cube leaks ~0.05 σ_TV into σ̂_T at 100 frames). On 200 frames of 64×64 with the Boson
  ratios: σ_TVH to 0.6 %, σ_VH to 3 %, σ_H/σ_V to ~27 % (3σ), T/TV/TH to ~0.01–0.03 σ_TVH absolute.
  Wider cubes tighten the fixed terms as 1/√N_axis; more frames tighten the temporal ones.
- The sum of the seven sums of squares equals the total sum of squares (orthogonal decomposition);
  tested to 1e-6 as the algebraic check of the operator set.
- `spatial_psd`: frame-averaged, mean-removed 2-D periodogram normalised to Σ = variance, with the
  radial profile (10 bins to Nyquist) and the k_v = 0 / k_h = 0 lines where column/row noise sits;
  `temporal_psd`: one-sided, pixel-averaged, Σ = temporal variance; `compare_psd`: max ratio of radial
  profiles, the Tier 4(c) "within a factor of 2" statistic.
- **Convention note** (spec issue S25): σ_TVH ("in NETD units") is the anchored temporal σ; σ_total is
  1.06 × NETD for the Boson ratios. The estimator reports both.

## Consequences

Every 3-D claim in this repo carries the cube size it was measured on and its floor. Recovering a
0.02 σ_TVH frame bounce needs ≳ 200 frames; a 5 % σ_H needs ≳ 2000 columns or many independent
cubes; on 10-s public clips at 30 Hz the temporal terms are feasible, on single frames they are not
(ME.5 reports N and CI per statistic from these floors). Codec quantisation biases (8-bit video) are ME.2b's floor
and are reported beside the estimate, never subtracted silently.

## Revisit when

Real-data decompositions show non-Gaussian striping (heavy tails) where a variance-based estimator
misleads, or the GPU comparison needs per-component confidence intervals — then bootstrap CIs join
the estimator.
