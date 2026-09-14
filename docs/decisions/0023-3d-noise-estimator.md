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

## Addendum (ME.2b, 2026-09-14): the codec floor, the flat-region gate, and the temporal shape

The decision above stands unchanged. What ME.2b adds is the half this ADR deferred: *where* a
statistic may be measured on somebody else's clip, and *how much of the answer is the storage
path's*. The numbers below are measured, not quoted, all on `irsim.noise`'s own synthesiser so the
truth is known exactly.

**The floor is `step/sqrt(12)` = 0.289 codes at 8 bits, and the lattice step is measured.** A
recorder that mapped a Y16 stream onto a narrow 8-bit range leaves the codes on a coarser lattice,
so `quantiser_step` reads it off the data (the largest step nearly every sample shares a residue
with) rather than assuming one. Sheppard's correction — subtract `step^2/12` from the variance —
recovers a known sigma to 0.5 % at 0.5 codes and to 3.5 % at 0.3 codes, and collapses below that
(at 0.2 codes it under-reads by 50 %: the information is gone, not merely biased).

**A component within `margin = 2` floors is reported as codec-limited, not measured**, whether or
not the correction happens to land. The consequence is blunt and is the point: on a clip at
sigma_TVH = 1.5 codes — a plausible stretch for a Y16 stream in 8 bits — *every* other Boson
component sits inside two floors, so the striping ratios this project cares about cannot be read
off it at all. Four times noisier and sigma_H, sigma_VH and sigma_TVH come back. `flag_codec_limited`
attaches the flag and the corrected value to each of the seven components, keeping this ADR's rule
that a bias is reported beside the estimate and never silently subtracted.

**Lossy coding removes noise; it does not usually make it blocky.** Through x264 (full-range flags
pinned so a CRF 0 round trip is bit-exact, otherwise the harness itself costs a code), a 320x256x60
cube of Boson-ratio noise at sigma_TVH = 1.5 codes loses 95 % of its temporal noise at **CRF 18** —
a high-quality setting — and becomes a constant image at CRF 23; a 2000 kbit/s CBR stream keeps
75 %, 800 kbit/s keeps 57 %. Meanwhile the 8-pixel blocking z-score reaches only 3.4 at CRF 18 and
nothing at all at 800 kbit/s. So blockiness is reported as an indicator and is **not** the test: the
defensible statement about a lossy set is that its noise statistic is a *lower bound*, and ME.5/ME.6
must carry it as one. (Anti-UAV410 and CST are already excluded from `noise_3d` in the index for
this reason; this quantifies what that exclusion is worth.)

**The blocking null is built from gap positions, not pixels.** A fixed column pattern makes some
columns noisier than others in every frame alike, so a pixel-count error bar is far too tight and
reads blocking where there is none (uncoded Boson-ratio cubes land at |z| ~ 3 that way, and at
|z| <= 2 on the per-gap null).

**Flatness is judged against the window's own noise, never in DN.** `robust_noise_scale` is the
ruler: the MAD of the first differences *about their own median*, which cancels a linear gradient
exactly, taking the smaller of the two axis estimates so that striping (which enters one axis only)
does not inflate it. A window is then scored on the peak-to-peak of a fitted plane and on the
standard deviation of the de-ramped residual after block averaging, with the white contribution
removed in quadrature. Two consequences are deliberate: **striped windows stay flat** (a finder that
rejected them would make column noise unmeasurable by construction), and **a single-pixel target is
caught only by the outlier rule** — averaged into a 4x4 block it is a quarter of the block noise, so
a variance-based finder waves it through. On a clip the judgement runs on the per-pixel temporal
median, which deletes a moving target outright.

**Temporal shape.** A flat sky's temporal spectrum is flat unless something filtered it, so
`temporal_shape` reports the low-over-high band ratio against a null built from the bins' own
scatter, and fits the sampled one-pole response whose time constant in frames is exactly `tau/dt`.
It recovers the 10 ms membrane at 60 Hz (0.6 frames) to 1 % from a 256-frame cube and separates
white from a 0.3-frame filter at 28 sigma. Two details are load-bearing: **DC and Nyquist are
dropped**, because `temporal_psd`'s one-sided convention leaves the Nyquist bin at half the weight
of its neighbours and keeping it makes white noise read 2 % low-pass — five times the null; and the
fit starts above `f_fit_min` so a drift does not become a time constant. That defence is partial and
its cost is stated rather than hidden: a linear ramp of one noise sigma across the clip costs 1.4 %
on tau and one of five sigma costs 28 %, while `drift_fraction` moves 0.14 → 0.23 → 0.75, so the
report says "this clip drifted" instead of claiming a longer membrane.
