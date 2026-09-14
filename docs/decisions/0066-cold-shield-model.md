# ADR 0066 — The cold shield: a solid-angle model for §9.1's formula-free knob

**Status:** Accepted
**Date:** 2026-09-15

## Context

§9.1 asks for a parameter and gives no equation:

> A cooled system's cold stop limits the detector's view of warm surroundings. Effective f-number
> for background flux is set by the cold shield, not the lens: if they are mismatched ("cold shield
> inefficiency"), background flux rises and NETD degrades. Expose a
> `cold_shield_efficiency ∈ [0, 1]` parameter; it is a real trade-study knob.

`optics.cold_shield_efficiency` has been in the schema since M0.9 and nothing has ever read it —
which is the failure mode ADR 0077 and ADR 0082 already document twice in this repository, so it
is worth closing while a cooled camera is being added rather than after. Spec issue S31 also notes
that the field lives under `optics:` while the physics is the detector's.

## Decision

**A solid-angle model, with the efficiency as the ratio of the cones.**

```
Ω_lens  = π / (4F² + 1)                    the scene cone (non-negotiable #5)
Ω_admit = min(π, Ω_lens / η_cs)            what the shield lets through
Φ_bg    = A_d (Ω_admit − Ω_lens) L_B(T_surround)
```

η_cs is therefore exactly what §9.1 describes: the fraction of the admitted cone the lens actually
fills. A perfect shield admits the lens cone and cold metal everywhere else — a shield at the
detector's own temperature radiates nothing in band, which is the entire reason a dewar has one.

Three properties are asserted rather than argued:

* **η_cs = 1 gives exactly 0.0**, not 1e-17. A residue would put a spurious Poisson term into the
  noise budget of every well-shielded camera and of every uncooled one.
* **Ω_admit is capped at π**, the *projected* hemisphere ∫cos θ dΩ, not 2π. A pixel on a plane
  cannot receive from behind itself however badly matched the shield is.
* **At η_cs = 0 an isothermal enclosure closes**: scene through the lens plus background around it
  equals A_d·π·L_B(T) with τ = 1. A pixel inside a cavity at one temperature must read that
  cavity whatever the optics do, and this identity holds only because the cap is π.

**τ_opt does not multiply it.** The background reaches the pixel *without going through the lens*
— that is what "outside the lens cone" means — so `background_power` has no transmittance argument
at all, and the test asserts the signature rather than a value. This is also what distinguishes it
from the optics' self-emission (§8.2, `irsim.optics.self_emission`), which is
A_d Ω_lens (1 − τ) L_B(T_housing): the lens glowing *inside* the scene cone. The two add; neither
contains the other, and folding one into the other would look right at η_cs = 1 and be wrong
everywhere else.

**The surround radiates at the housing temperature.** `PipelineConfig.from_sensor` looks up
L_q,B at `t_housing_cal_k`, the same node §8.2 uses. It is the warm structure between the cold stop
and the lens, so the housing node is the closest thing the model has; a separate dewar-wall node
would be a third thermal node for a second-order difference.

**The cost is shot noise, not the offset.** NUC removes offsets, so what a mismatched shield
actually costs is the Poisson noise its DC level carries:

    NETD(η_cs) / NETD(1) = √((N_signal + N_bg) / N_signal)

`netd_degradation_factor` returns exactly that, and the test holds `predict_netd_k` to it to 1e-9
with σ_gaussian = 0 — isolating the shot term, which is the part the shield controls — then checks
that NETD still rises monotonically with the camera's real 350 e⁻ read noise.

**Where it lives.** `irsim/detector/cold_shield.py`, because it is detector physics (S31). The
*schema* field stays under `optics:` where §12.2 puts it: moving it would break every existing
config for a filing-cabinet improvement, and the module docstring names the split.

## What this measured

The MWIR InSb example at F/2, 2 ms, 300 K scene, 8.5 Me⁻ well, N_signal = 2.49e6 e⁻ (29 % of well):

| η_cs | N_bg (e⁻) | % of well | shot-limited NETD |
|---|---|---|---|
| 1.00 | 0 | 0 % | ×1.000 |
| 0.95 | 1.54e5 | 1.8 % | ×1.030 |
| **0.90** | **3.26e5** | **3.8 %** | **×1.063** (×1.060 with the real read noise) |
| 0.70 | 1.26e6 | 14.8 % | ×1.226 |
| 0.50 | 2.93e6 | 34.5 % | ×1.475 |
| 0.20 | 1.17e7 | **138 %** | ×2.389 |

The last row is the trade study as a fact rather than a curve: at η_cs = 0.2 **this camera
saturates on its own dewar before it sees the scene**. That is why cold-shield matching is a
manufacturing specification and not a tuning parameter.

## Consequences

* `configs/sensors/example_mwir_insb_640.yaml` is the first configured camera with η_cs < 1
  (0.90, ESTIMATED). Every uncooled camera in the repository has η_cs = 1.0, so the term is
  identically zero for them and **no golden moves**.
* `NoiseBudget.background_electrons` and `PhotonDetector` already carried the term; only the
  formula was missing. `anchor_noise` now receives it, so the anchored σ_gaussian is solved
  *after* the background's shot noise is counted — which is the right order, and means a
  badly-matched shield can make a NETD target unattainable and say so.
* A third band is configured with no kernel change, which M11.1's guard re-checks automatically
  because it parametrises over the whole `configs/sensors/` directory.

## Revisit when

* A dewar-wall temperature is worth separating from the housing node — i.e. when a cooled camera
  is simulated through a warm-up transient rather than at a steady state.
* The shield's own emissivity matters. The model treats the shield as a perfect cold absorber;
  a real one at 80 K with ε < 1 reflects some of the warm structure back, which raises the
  background slightly and is well inside the ESTIMATED η_cs itself.
