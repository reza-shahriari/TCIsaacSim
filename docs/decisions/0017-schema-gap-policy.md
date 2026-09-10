# ADR 0017 — Schema-gap policy: fields the spec omits

**Status:** Accepted
**Date:** 2026-09-11

## Context

`docs/physics-model.md` §12.2 is the configuration contract, but the physics in §8–§11 needs
parameters the block does not list: the housing temperature and its dynamics (§8.2, ADR 0016), the
supersample factor (§8.3), the aberration Gaussian and the diffraction reference wavelength (§8.3), a
measured vignetting map (§2), and later detector constants (§9). Each could be hard-coded, put in a
side file, or added to the schema.

## Options considered

1. **Hard-code defaults in code** — invisible in the YAML that keys every LUT, golden and report; two
   cameras cannot differ.
2. **Add required fields** — every existing YAML breaks; the spec block and the files diverge.
3. **Add optional fields with physically sensible defaults, validated like everything else, and bump
   `schema_version`** — files without the fields keep loading; the config hash captures the defaults
   because they are part of the validated model dump; the spec block is patched to list them.

## Decision

Option 3. Rules:

- A field the spec omits is added to the pydantic schema as **optional with a default**, with units in
  its name, bounds, and cross-field checks. It is listed in this ADR's table and proposed for §12.2.
- `schema_version` is bumped for every additive change (now **2**); the loader accepts only the
  current version so a mismatch is loud. Files without `schema_version` take the current default.
- Defaults must be *conservative physics*, not convenient zeros, unless zero is the physical
  "feature absent" (aberration σ = 0, self-heating 0).

| field | default | meaning / rule |
|---|---|---|
| `optics.housing_temp_k` | none | required when `housing_temp_mode: fixed` |
| `optics.housing_tau_s` | none | first-order lag for `coupled` (M9.3) |
| `optics.housing_self_heating_k` | 0 | steady offset above ambient for `coupled` |
| `optics.vignetting_map` | none | path to a measured multiplicative map; required source of fall-off for fisheye models |
| `optics.supersample_factor` | 4 | render/box-filter factor, 1–8 (§8.3) |
| `optics.mtf.aberration_sigma_um` | 0 | Gaussian fitted from a slant edge (§8.3) |
| `optics.mtf.reference_wavelength_um` | band centre | λ for ξ_c = 1/(λF) |
| `optics.mtf.apply_motion_mtf` | false | motion term for photon FPAs (ADR 0059) |

Derived read-only quantities added: `detector_active_area_m2` (= `pixel_area_m2`), `active_width_um`
= √fill · pitch (the MTF_det box width, ADR 0020), `reference_wavelength_um`, `cutoff_cyc_per_mm`.
Rule: `vignetting_cos4: true` with a non-rectilinear distortion model is refused (ADR 0015).

## Consequences

Every config hash changes with a schema bump (new leaves in the dump); goldens keyed on config
hashes must be regenerated at that point — none exist yet. §12.2 in the spec should be patched to
list these fields (spec issue to be added to `docs/spec-issues.md` by the next docs step).

## Revisit when

The spec owner adopts or rejects the proposed fields, or a field's default proves to matter for
Tier 4 results (then it becomes required).
