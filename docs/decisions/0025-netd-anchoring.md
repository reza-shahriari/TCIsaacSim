# ADR 0025 — NETD anchoring: σ_TVH = NETD at the reference f-number; only Gaussian terms scale

**Status:** Accepted
**Date:** 2026-09-11

## Context

§9.4 and Appendix A #7: NETD is anchored to the datasheet, not predicted. Something must decide
*which* noise terms absorb the anchor, at which f-number the datasheet figure applies, and what to do
when the physics cannot reach the target. §10.2 quotes the 3-D ratios "in NETD units", i.e. relative to
σ_TVH, and the datasheet NETD is measured at one lens while the simulator may run another.

## Options considered

1. **Scale every noise term** by a common factor — rescales Poisson shot noise, which is physics
   (variance = mean), not a free parameter; wrong statistics for photon cameras.
2. **Scale only the scene-independent Gaussian term** (read/Johnson/ROIC/1/f lumped) to make
   σ_total(300 K)/(∂S/∂T) equal the target; shot terms untouched; raise if shot alone exceeds the
   target.
3. **Fit a per-temperature σ(T) to a NETD curve** — no camera provides one.

## Decision

Option 2, `irsim.detector.anchor.anchor_noise(sensor, lut, target, T_ref=300 K)`:

- σ_gaussian² = (NETD_target · ∂S/∂T(T_ref, F_ref))² − σ_shot²(T_ref); Poisson shot (signal + dark +
  background) is never rescaled; an unattainable target raises with the shot-limited NETD in the
  message.
- **σ_TVH = NETD** convention: the anchored σ is the per-pixel temporal σ; the 3-D components take
  their σ_i = r_i · σ_TVH from the configured ratios (spec issue S25: σ_total is then 1.06 × NETD for
  the Boson ratios, which is the convention chosen — the datasheet NETD is a temporal figure).
- **F_ref = `noise.netd_ref_f_number`** (the lens the datasheet figure was measured with) or the
  configured f-number. The anchor is solved at F_ref; the camera runs at its configured F, which enters
  only through the transfer, so NETD(F/1.4)/NETD(F/1.0) = 1.768 (ADR 0024), never 1.96, and
  σ_signal does not change when the lens changes.
- Bolometer signal unit is pixel power (W); the DN gain of ADR 0019 converts σ to DN in the noise
  stage, so the anchor is gain-invariant. Photon signal unit is electrons.
- Bolometer first-principles floors (Johnson, temperature fluctuation at ENBW = 1/(4τ_th)) are computed
  and reported in the budget for information; with the typical VOx constants of ADR 0017 they are of
  the same order as the datasheet NETD, which is why the anchor — not the floors — sets the magnitude.

## Consequences

The noise *magnitude* matches the datasheet at 300 K and its *structure* (shot vs Gaussian, spectral
and spatial) is physical. NETD at other temperatures follows the physics: for a bolometer, exactly the
derivative ratio; for a photon camera, the shot-noise curve. Grade variants (40/50/60 mK) differ only
in σ_gaussian, analytically.

## Revisit when

A measured NETD-vs-scene-temperature curve or a 3-D noise measurement for the reference camera
exists (then the ratios and the temporal/total convention can be fitted rather than assumed), or a
cooled camera needs a background-limited term (ADR 0066).
