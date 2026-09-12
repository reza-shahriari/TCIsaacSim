# 0056 — The millikelvin → DN conversion reference for the NUC residual

- Status: accepted
- Date: 2026-09-12
- Step: M9.6 (`feat(noise): NUC residual — gain/offset drift with ΔT_FPA and growth between FFC`)
- Spec: docs/physics-model.md §11.2, §2 (g_ij, o_ij), §12.2 `nuc`

## Context

`nuc.residual_offset_mk_per_k` is authored in **millikelvin per kelvin** — 45 mK/K for the Boson.
That unit is right for the config: it is how datasheets and the literature quote residual
non-uniformity, because millikelvin is what a user sees.

It is the wrong unit to *apply* in. Non-negotiable #3 exists because the blackbody thermal
derivative ∂L/∂T rises steeply with temperature, so a fixed kelvin-space error is only correct at
the one temperature it was tuned at. Measured on the committed Boson chain, ∂DN/∂T is

| T (K) | ∂DN/∂T |
|---|---|
| 250 | 97.2 |
| 300 | 178.0 |
| 373 | 309.0 |
| 450 | 439.6 |

so the same physical residual is **0.576×** as large in apparent-temperature terms at 373 K as at
300 K. (That 0.576 is the same ratio ADR 0026 arrived at independently for the two-blackbody NETD
bench — a useful cross-check that the two derivations are describing one chain.) A residual added
in kelvin would be flat across that range, and every image would still look completely plausible.

So the question is not *whether* to convert but **where the conversion reference sits**, and how
many times it is applied.

## Decision

**Convert once, at construction, at 300 K.**

`NucResidual` takes `dn_per_k` = ∂DN/∂T evaluated at `RESIDUAL_REFERENCE_K = 300.0`, and from that
moment the residual is a fixed number of DN. It is never reconverted, never re-evaluated at the
scene temperature, and never applied in kelvin.

- **300 K** because that is where §9.4 anchors NETD and where datasheet residual figures are
  quoted. The config's "45 mK/K" therefore means the same thing here as on the datasheet it was
  copied from, which is the only way an ESTIMATED value can later be replaced by a measured one
  without silently changing meaning.
- **Once**, because the point of the conversion is to stop being in kelvin. Re-deriving the DN
  amount per frame from the current scene temperature would reintroduce exactly the flat-in-kelvin
  behaviour the conversion exists to prevent, while looking like extra rigour.
- **Supplied by the caller**, not computed inside `irsim.noise`. `∂DN/∂T` comes from
  `irsim.isp.dn_per_kelvin`, which needs the radiometric calibration; taking it as a constructor
  argument keeps `irsim.noise` independent of `irsim.isp` and puts the "converted once" rule at the
  call site where a reader can see it rather than buried two layers down.

`dn_per_kelvin` is a central difference through the real forward chain (LUT → optics stage →
detector transfer) rather than an analytic derivative, so it is the same object the Tier 2 SITF
bench measures.

### The consequence this buys

Because the residual is a fixed DN, its apparent-temperature equivalent falls with scene
temperature exactly as the derivative ratio says it should. `test_residual_not_kelvin_flat` asserts
that the ratio of the apparent-temperature error at 373 K to that at 300 K equals
(∂DN/∂T at 300)/(∂DN/∂T at 373) within 5 %, and that at the reference temperature the error is the
configured 90 mK at ΔT_FPA = 2 K. A kelvin-space implementation passes the second and fails the
first — which is why the second alone would not be enough.

## Consequences

- The reference is a module constant, not a config field. Making it configurable would let two
  cameras interpret the same millikelvin figure differently, which defeats the purpose of using the
  datasheet's unit.
- A camera whose residual figure is quoted at some other temperature must have it restated at
  300 K before it goes in the YAML. This is a documentation burden, not a modelling one, and it is
  preferable to carrying a per-camera reference temperature that nobody would keep consistent.
- Both `residual_gain_ppm_per_k` and `residual_offset_mk_per_k` for the Boson are **ESTIMATED**.
  ME.3's between-FFC growth measurement is what should set them.
- The gain residual needs no conversion at all: parts per million of the signal is already
  dimensionless and already multiplicative, so it is correct at every temperature by construction.
  Only the offset has this problem, which is exactly why it is the one §11.2 states in temperature
  units.

## Revisit when

A camera is modelled whose published residual is specified at a materially different temperature
and restating it at 300 K proves lossy; or the chain gains a non-linearity large enough that a
single-point derivative misrepresents the transfer over the working range, in which case the
conversion becomes a curve rather than a constant and this ADR is replaced rather than amended.
