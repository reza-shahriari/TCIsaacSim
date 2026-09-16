# 0053 — Splitting the T_FPA effects between the detector and the NUC residual

- **Status:** Accepted
- Date: 2026-09-12
- Step: M9.2 (`feat(detector): FPA temperature node and gain/offset(T_FPA) polynomials`)
- Spec: docs/physics-model.md §9.2 ("FPA temperature coupling"), §6.4, §11.2

## Context

A TEC-less core's focal plane runs above ambient by its own dissipation and lags ambient by its own
time constant. Bolometer responsivity and offset are functions of that temperature, so T_FPA is the
physical origin of shutterless drift.

The hazard is double-counting. Two separate places in the model want to depend on T_FPA:

1. The detector's **raw** response — what the FPA would produce with no correction applied.
2. The **NUC residual** (M9.6) — the non-uniformity that survives a two-point correction which was
   calibrated at one T_FPA and is being applied at another.

The config already carries `nuc.residual_gain_ppm_per_k` and `nuc.residual_offset_mk_per_k`. If the
detector's own gain/offset(T_FPA) were also applied to a corrected image, the same physical effect
would be counted twice, and the error would look like a plausible drift rate rather than a bug.

## Decision

**The two are separate quantities with separate parameters, and they live at separate points in the
chain.**

*Detector (this step, M9.2).* `gain_of_t` and `offset_of_t` are the raw, uncorrected response.
They are large — parts in 10³ per kelvin for gain, DN-scale per kelvin for offset — and represent
an image that would be unusable without correction. They are parameterised by the new
`fpa_gain_coeffs_per_k` and `fpa_offset_coeffs_dn_per_k`, applied in signal space before noise, at
the same place the membrane IIR sits (ADR 0052).

*NUC residual (M9.6).* `nuc.residual_*` is what is left after the two-point correction, in ppm and
mK per kelvin of ΔT_FPA. It is small by construction, because it models a failure of correction
rather than the uncorrected response.

**Normalisation is structural, not asserted.** Both polynomials take coefficients in ascending
powers of `(T_FPA − T_cal)` **starting at order 1**, so `gain(T_cal) == 1` and `offset(T_cal) == 0`
exactly and cannot be authored otherwise. A schema that accepted a constant term would let someone
write a gain of 0.98 at the calibration temperature, which is not a temperature effect at all —
it is a units error wearing a polynomial's clothes.

**One ambient source (CLAUDE.md #6).** `FpaThermalModel` accepts the scene's shared `WeatherSeries`
*or* an explicit ambient provider, never both and never neither. A camera whose focal plane warms
on a different day from the one the atmosphere and the thermal solver see is precisely what that
non-negotiable exists to prevent, and the constructor raises rather than picking a winner.

**τ and ΔT_self are authored, not C, h and P.** The node is `C dT/dt = P − h(T − T_amb)`, but only
`τ = C/h` and `ΔT_self = P/h` are observable from outside the camera. Authoring three numbers to
determine two invites them to disagree, and there is no measurement that would settle which of the
three was wrong. The energy-balance form is still what the test checks against, by choosing a `C`
and deriving `h` and `P` from it.

## Consequences

- Sensor schema goes to **version 5**: `fpa_temp_mode`, `fpa_t_cal_k`, `fpa_gain_coeffs_per_k`,
  `fpa_offset_coeffs_dn_per_k`. `fpa_temp_mode` is `None` by default, so existing configs are
  unchanged and the node is opt-in until M9.8 wires it into the pipeline.
- `fpa_temp_mode` reuses the housing node's three modes (ADR 0016) deliberately: `fixed` is a
  TEC-pinned core and shows no ambient drift at all, which is the control case in the test;
  `ambient` is the zero-τ degenerate case; `coupled` is the lumped node.
- Integration is RK2 (Heun) on the thermal tick, as the roadmap specifies, even though this
  particular ODE has the same exact exponential solution `NewtonCoolingSolver` uses. The reason to
  keep RK2 is that M9.3's housing source and later coupled nodes will not stay first-order and
  linear, and a single integrator across the node family is worth more than exactness on the one
  case that happens to be analytic today. The step is exact in the energy sense regardless — the
  Heun update satisfies `C ΔT = (P − h(T_stage − T_amb)) Δt` identically, which is what the test
  asserts to 1e-6.

## Alternatives considered

- **One set of T_FPA coefficients, with NUC subtracting most of it.** Rejected: it makes the
  residual a difference of two large numbers, so the small quantity that actually matters for
  sim-to-real would carry the numerical error of the large ones.
- **Fold T_FPA drift entirely into the NUC residual and give the detector no temperature
  dependence.** Rejected: it would make the uncorrected image wrong, so `nuc.mode: none` — a real
  configuration, and the honest way to model a non-radiometric core with no correction — would show
  no drift at all, which is the opposite of the truth.
- **Author C, h and P.** Rejected as unidentifiable, above.
