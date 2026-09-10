# ADR 0019 — Bolometer DN gain representation and the shared quantiser

**Status:** Accepted
**Date:** 2026-09-11

## Context

§9.2 gives the bolometer physics up to a responsivity R₀ = α β I_bias R / G_th in volts per watt.
Turning volts into DN needs the ROIC's integration/amplifier chain and the ADC's volts-per-LSB, none
of which a datasheet camera publishes. Meanwhile NETD is anchored to the datasheet value (Appendix A
#7), so the absolute gain does not determine noise; what matters is that the DN transfer is linear in
absorbed power and that the ADC spans a known scene range.

## Options considered

1. **Model the full chain** (R₀ × amplifier × ADC) with guessed constants — many unknowns, no
   validation route, and the guesses would masquerade as physics.
2. **One effective gain in DN per watt plus an offset in watts**, sized so a chosen scene-temperature
   range fills the ADC; R₀ computed and reported for the record only. Linear, honest, one number.
3. **DN linear in temperature** — wrong physics (the spec's SITF is linear in *radiance*, not T).

## Decision

Option 2, `irsim.detector.bolometer.BolometerTransfer(gain_dn_per_w, offset_w, bit_depth)` with
`from_power_range(Φ_min, Φ_max, bits)`; `signal_dn(Φ)` is the un-quantised float32 signal to which
noise is added; `dn(Φ)` quantises; `power_from_signal_w` is the inverse for the radiometric branch
(M3.10). The scene range that sizes the gain is chosen by the pipeline from a temperature span through
the LUT and the optics (M3.9/M3.10; the NUC/level convention is ADR 0021).

Shared quantiser `irsim.detector.quantise.quantise(signal_dn, bits)`: **floor** then clip to
[0, 2^bits − 1], uint16, never wraps, refuses float16 and NaN. Floor, not round, because an ADC
compares against thresholds; the quantisation error is then uniform on [0, 1) LSB, 0.29 LSB rms,
which the SITF bench tolerance (0.5 LSB rms) is set against. Noise is added before quantisation
(spec issue S10; quantise last).

Guards: the bolometer takes energy-form radiance; a pixel "power" above 1 mW is a photon rate by
mistake and raises. Membrane ΔT = αΦ/G_th and R₀ are reported and tested (R₀ = −8.4e5 V/W for the
typical VOx constants) but do not feed the DN.

## Consequences

Absolute DN values are not comparable to any real camera's raw counts — only their linearity and
their relation to the configured range are. Two cameras of the same design with different gain
settings differ by the range choice alone. Bolometer dynamics (the τ_th IIR) are M9.1 and multiply
this static transfer.

## Revisit when

A camera's raw 16-bit output and its radiometric calibration become available (Y16 data from the
Halmstad set may give a partial handle) — then the gain and offset can be fitted and this ADR gains a
"calibrated" mode.
