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

---

## Addendum (M11.6, 2026-09-15) — for a photon FPA the electron datasheet is primary

The rule above makes NETD the calibration handle and solves for the Gaussian σ. That is right for
a bolometer, whose datasheet quotes NETD and nothing else usable. M11.1's SWIR camera is where it
breaks in the open.

NETD is defined against a **300 K blackbody** (§9.4). A 300 K blackbody puts **1.24
photoelectrons per pixel per 16 ms frame** into 0.9–1.7 µm, against 120 e⁻ of read noise, so
evaluating §9.4 honestly on that camera gives **976 K**. It is correct physics and a useless
anchor: solving `σ_gaussian² = (NETD · ∂S/∂T)² − σ_shot²` against it scales the Gaussian term by
about 2e4 and produces a camera whose noise is invented rather than described.

What an InGaAs or InSb datasheet *does* quote is what the noise is made of — quantum efficiency,
well capacity, integration time, read noise **in electrons**, dark current. So:

* **Photon FPA with `read_noise_e` authored → `irsim.noise.electron.electron_budget`.**
  σ_gaussian **is** the configured read noise, used and not solved for; the Arrhenius block
  becomes a Poisson dark term; the cold shield (ADR 0066) a Poisson background term. NETD becomes
  the **cross-check**, not the anchor.
* **Bolometer, or a photon FPA with no `read_noise_e` → `anchor_noise`, unchanged.**

Three rules survive intact, because none of them was ever about which handle is primary: Poisson
terms are never rescaled; noise is added in electron space and never in kelvin (non-negotiable #3);
and a configuration that cannot reach its own claim **raises**. That last check is skipped, with
its reason named in the code, for a **reflective** band — where §9.4's 300 K definition is vacuous
and the configured field is a placeholder by construction. This closes the loop M11.1 opened when
it wrote 976 K into the SWIR file and said nothing may anchor to it.

Measured (synthetic bench: σ of a uniform scene over ∂S/∂T, in electrons, 200 000 samples):

| camera | at | bench NETD | predicted | error | shot-limited floor |
|---|---|---|---|---|---|
| InGaAs SWIR | 700 K | 7.86 mK | 7.87 mK | **0.16 %** | 7.87 mK |
| InSb MWIR | 300 K | 18.33 mK | 18.36 mK | **0.16 %** | 17.93 mK |

The InSb camera sits **2.4 % above shot-limited**, which is what a cooled MWIR core is sold on and
a useful check that the read-noise and dark terms have not been overstated. The SWIR camera is
measured at 700 K rather than 300 K because a NETD bench at 300 K in that band measures the read
noise and nothing else — the scene contributes one photoelectron a frame.

Dark current, as a number, is the reason InSb is cryocooled: **200 e⁻ per frame** for the uncooled
InGaAs array against **0.12 e⁻** for the InSb at 77 K, and warming that InSb array to room
temperature multiplies its dark count by more than 10⁶.
