# ADR 0007 — Sensor configuration schema

**Status:** Accepted
**Date:** 2026-09-10

## Context

`docs/physics-model.md` §12 makes "a band is a data file" the project's scalability claim: adding SWIR
must touch YAML and a response CSV, never the radiance kernel. That only holds if the file has a typed,
validated contract that every stage reads from, so nobody re-parses a dictionary or invents a default.
The scaffold shipped `configs/sensors/flir_boson_640_lwir.yaml` with no schema behind it.

## Options considered

1. **Plain dictionaries with ad-hoc checks** in each consumer — zero dependencies; typos default
   silently, units are folklore, constraints get re-implemented inconsistently.
2. **Dataclasses + hand-written validation** — typed, no new dependency; every constraint and the
   discriminated FPA union are manual and easy to leave incomplete.
3. **pydantic v2 models** — already a dependency; `extra='forbid'`, `frozen=True`, `Field` bounds,
   `Literal` enums and discriminated unions cover §12.2 with little code; JSON-schema export for free.

## Decision

Option 3, in `irsim.config.sensor`:

- Every model is `extra='forbid'` and frozen: `f_stop` fails, and a loaded config cannot be mutated
  behind a LUT or a golden that was keyed on it.
- Units are in field names (`pitch_um`, `netd_mk_at_300k`); every `a | b | c` comment of §12.2 is a
  `Literal`. Palettes accept both the §11.4 and §12.2 sets (spec issue S27).
- `fpa.type` is a discriminated union: `BolometerFpa` requires `thermal_time_constant_ms`/`tcr_per_k`
  and accepts photon fields only as `null` (the §12.2 example writes them that way); `PhotonFpa` the
  reverse, and its integration time must fit the frame period.
- Physical cross-field checks: band edges within the radiometry bounds in micrometres; regime vs
  wavelength (`emissive` needs λ_max ≥ 2.5 µm, `reflective` needs λ_min ≤ 3 µm, per §12.1);
  `ratios_3d.tvh == 1.0` exactly; ordered clip percentiles; bounded transmittance, cold-shield
  efficiency, fill factor, bit depth 8–16, bad-pixel fraction ≤ 1 %.
- Derived read-only properties: `pixel_area_m2`, `nyquist_cyc_per_mm`, `hfov_deg`, `frame_period_s`,
  `dn_max`. **No aperture-factor property** — π/(4F² + 1) is defined once in `irsim.optics`
  (non-negotiable #5); a test scans the config package's code for any such expression.
- `schema_version` (top level, default 1) is checked, so a future breaking change fails loudly.
- `types-PyYAML` joins the dev extras so `mypy --strict` accepts the YAML loader (M0.8).

## Consequences

The Boson file loads exactly (every §16.1 value asserted). The derived HFOV for 14 mm / 640 × 12 µm is
30.7°, not the datasheet's 32°; the test records the discrepancy (open question 7). No physical error is
introduced; the schema only refuses inputs the physics could not use.

## Revisit when

A camera needs a field §12.2 lacks (non-radiometric flag, ADR 0021; cold-shield formula, ADR 0066) —
bump `schema_version` when a change is not backward compatible.
