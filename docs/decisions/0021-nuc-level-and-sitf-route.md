# ADR 0021 — NUC reference level, calibrated SITF route, and radiometric vs non-radiometric cameras

**Status:** Accepted
**Date:** 2026-09-11

## Context

§11.1 forks the signal chain into a radiometric branch (calibrated apparent temperature) and the
AGC branch. The calibration must say what "DN 0" and the gain mean, and which housing temperature the
camera assumes when it inverts — that assumption is the physical origin of shutterless drift (ADR 0016).
The spec also has no field declaring a camera non-radiometric (spec issue T14); `outputs.apparent_
temperature` is the only proxy.

## Options considered

1. **Invert with the true instantaneous housing temperature** — the simulator would hide the very
   drift it exists to reproduce.
2. **Invert with the housing at its calibration temperature** (the two-blackbody SITF of the ideal
   chain) — the camera's belief, not the truth; drift appears in T_app until an FFC re-references.
3. **Empirical DN→T polynomial** as real cameras ship — needs a fitted calibration nobody has.

## Decision

Option 2, `irsim.isp.radiometric.RadiometricCalibration`:

- The calibrated transfer is built by `from_scene_range(sensor, lut, T_min, T_max, T_housing_cal)`:
  blackbodies at T_min and T_max through the ideal optics (housing at T_cal) set the DN gain so that
  range spans the ADC (bolometer gain per ADR 0019). `lb_housing_cal` is the NUC reference level: the
  self-emission the camera believes it sees. Pipeline default range 233.15–473.15 K (−40 to +200 °C,
  the Boson high-gain-like span); T_cal = the housing temperature at the pipeline's first frame.
- **Two routes, kept distinct.** `apparent_t` in the pipeline outputs comes from the **float32 signal**
  (un-quantised), so the sub-mK round-trip claims are meaningful. `apparent_temperature_from_dn` applies
  the same calibration to the uint16 image (reconstructed at bin centre, +0.5 LSB) as a *validation*
  output whose error is bounded by half an LSB — at 300 K on the Boson span that is ~2 mK, at 250 K
  ~4 mK.
- T_app is always L_B⁻¹(L) through the band LUT; kinetic temperature never enters (§3.3).
- **Non-radiometric cameras** are declared by `outputs.apparent_temperature: false` (and
  `radiance_linear: false`); the pipeline then emits no such key — never zeros. A dedicated flag is
  left to the spec owner (T14).

## Consequences

A housing drift of +1 K reads as +87 mK of scene temperature until FFC (ADR 0016), by construction.
The DN gain is a scene-range choice, not a hardware constant (ADR 0019); changing the range changes
every DN value but no T_app. Photon cameras use the same structure with `lb_q` and the photon transfer
(M11.1).

## Revisit when

A radiometric Boson calibration (DN→T table) becomes available, or NUC (M9) introduces gain/offset
maps — then the calibration gains per-pixel gain and offset and the reference level moves with FFC.
