# ADR 0006 — Temperature encoding for the render G-buffer

**Status:** Accepted
**Date:** 2026-09-10

## Context

The renderer must carry surface temperature from the scene to the radiometry kernel through an AOV.
Emissive channels are commonly fp16 or unorm. At 300 K, fp16 spacing is 0.25 K, five times a 50 mK
NETD; an 8-bit unorm over 200–1000 K is 3.1 K per code. Either destroys the sensor silently while the
image still looks thermal (`docs/physics-model.md` §3.2 precision trap, §13.3; CLAUDE.md #2). The
CPU reference path, the Warp/SPG kernels and the Isaac adapter all need the *same* encoding, or the
encoder and decoder drift apart.

## Options considered

1. **Raw kelvin in a float32 AOV** — simplest; works only if every stage really is float32, and a
   silent fp16 downgrade anywhere costs 0.25 K with no error raised.
2. **`c = (T − 200)/800` in a float32 AOV** (the spec's recommendation) — c ∈ [0, 1] so any clamp or
   tone-mapping stage that assumes unit range passes it through; the constants live in one place;
   float32 round trip is at the representation floor.
3. **Coarse/fine fp16 pair** — the spec's fallback for engines with no float32 target; two channels,
   more shader work, and the pair must be split and recombined in exactly one way.

## Decision

Option 2 as the contract, with option 3 implemented as the documented fallback.

- `T_ENCODE_REF_K = 200`, `T_ENCODE_SPAN_K = 800` and the LUT grid (`LUT_T_MIN_K/MAX_K/DT_K/N` =
  200/1000/0.05/16001) are defined once in `irsim.radiometry.constants` and imported everywhere.
- `irsim.radiometry.encoding.encode_temperature` / `decode_temperature` compute in float64 and cast
  once to float32. Both **refuse float16 and integer (unorm) inputs** with a message that cites the
  non-negotiable; float64 is accepted.
- `encode_temperature_fp16_pair` splits `c` into `coarse = floor(256c)/256` (exact in fp16) and
  `fine = 256c − floor(256c)` (fp16, 2^-11 relative → 1.5 mK), clamped to [200, 1000] K.
- Verified: float32 round trip 0.054 mK over 200–1000 K at 0.01 K steps (floor set by float32
  representation of T near 1000 K); fp16-pair round trip < 10 mK. **Negative controls:** raw kelvin
  through fp16 → 125 mK at 300 K; encoded `c` through fp16 → ≥ 40 mK at 300 K and ≥ 150 mK over
  800–1000 K. These controls are what make the 10 mK test meaningful: if either ever passes the bound,
  the test has lost its sensitivity.

## Consequences

Every engine adapter has to produce `c`, not kelvin, and the AOV must be float32 (the in-engine
round trip of §13.3 is roadmap M2.2, `tests/integration/test_temperature_encoding.py`). The encoding
introduces no physical error: 0.05 mK is three orders below the tightest NETD modelled. Temperatures
outside 200–1000 K have `c` outside [0, 1]; the float32 path passes them through, the fp16-pair path
clamps them — the LUT (ADR 0011) decides the clamp policy for radiance.

## Revisit when

A scene needs temperatures outside 200–1000 K (exhaust plumes, fires) — widen the span and the LUT
together; or Isaac Sim 6.0 turns out to offer only fp16 custom AOVs (M2.1), in which case the pair
codec becomes the primary path and its 1.5 mK floor is the new budget.
