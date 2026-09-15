# 0085 — The Tier 4 acceptance checks, and what a gap score can and cannot say

Date: 2026-09-15
Status: accepted
Roadmap: ME.6 (§15 T4/T5)
Extends: ADR 0068

## Context

ADR 0068 replaced §15's radiometric Tier 4 targets with display-domain ones, because the public
sky-target data is 8-bit, post-AGC and codec-compressed and there is nothing radiometric to compare
against. It proposed the targets; ME.6 implements the comparison and had to measure them.

## Decision

1. `irsim.validation.compare` holds the four frame statistics — histogram EMD in codes, contrast
   ratio, PSD shape ratio, ESF width — plus `Tier4Report`, in which **`untestable` is a verdict**.
   A check whose inputs are missing is printed with the reason, never omitted: on this project's
   data most of Tier 4 is untestable, and a report that hid that would be the most misleading
   thing it could produce. `passed` ignores untestable checks; the report lists every one.
2. §15's "apparent temperature within 2 K" is reported as `untestable` in **every** run, with the
   reason, until a radiometric capture exists.
3. `irsim_eval.discriminator` is a **linear probe on interpretable statistics**, not a CNN, fitted
   in NumPy with no torch and no scikit-learn. Its AUC is a *lower bound* on the gap and it names
   the feature that separated the two sets. An AUC near 0.5 means these statistics do not separate
   them; it does not mean a detector cannot, which is ME.7's question.
4. The probe is **cross-validated**. Ten free parameters fitted and scored on the same eighty
   patches will report a gap that is not there.

## Two targets moved, by measurement

**The PSD-shape target: 2.0 → 1.5.** ADR 0068 proposed "PSD shape within ×2". Measured over three
seeds on a smooth scene with per-pixel noise:

| noise mismatch | max normalised radial-PSD ratio |
|---|---|
| ×1 (matched) | 1.05 – 1.08 |
| ×2 | 1.30 – 1.36 |
| ×3 | 1.77 – 1.93 |
| ×5 | 3.15 – 3.35 |

A ×2 threshold passes a simulator whose noise is **three times wrong**. 1.5 separates matched from
×3 with a 1.6× margin on both sides, and openly misses a ×2 error — which is the honest statement
of this statistic's sensitivity rather than a threshold tuned to a wish.

**A fraction-of-peak spectral floor is the wrong shape of filter.** ADR 0068's floor was meant to
exclude bins the codec owns. Keyed off the reference spectrum's peak, it excludes the bins that
carry *noise* instead: a scene with low-frequency structure has a steeply falling spectrum, so
"carries little power" and "is high-frequency" are the same bins. Measured: a 1 %-of-peak floor
took a ×3 noise mismatch from **1.90 to 1.06** — it removed the entire signal the statistic exists
to detect. The floor now defaults to zero, and a real codec floor should be supplied as a fraction
computed from an absolute power level.

**The control's target: "AUC 0.5 ± 0.03" → within two null standard errors.** The AUC has its own
sampling spread, √((n₁+n₂+1)/(12 n₁ n₂)), which at 96 patches per class is already **0.042**. A
synthetic-versus-itself control that lands at 0.54 is one sigma from chance, and a report calling
that a failure would be measuring its own sample size. `DiscriminatorResult.indistinguishable`
applies `|auc − 0.5| < 2σ_null`, which scales with n instead of assuming one.

## Consequences

The acceptance report (`scripts/validation_report.py`) exits non-zero on any failed check and has a
`--self-test` mode that runs synthetic-versus-itself. **That control is the gate on the gate:** if
it does not pass, the thresholds are wrong and nothing the report says about a real comparison is
worth reading. It is what the unit test drives, so the thresholds cannot drift without a failure.

On the reference set two of the five checks are untestable for reasons ME.5 recorded — the
annotation boxes are MATLAB MCOS objects with no Python reader, which costs the contrast-ratio and
ESF checks — so a real Tier 4 run there exercises the histogram, the spectrum and the gap score.
