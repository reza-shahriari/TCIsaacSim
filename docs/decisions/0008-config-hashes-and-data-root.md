# ADR 0008 — Config hash, band hash and the data root

**Status:** Accepted
**Date:** 2026-09-10

## Context

LUTs, goldens and pipeline outputs are all derived from a sensor config plus the data files it names.
Reusing one with a config that has changed is the silent-staleness failure the ir-radiometry skill
warns about ("generating a LUT from one config and using it with another"). The spectral-response path
was stated four ways across the spec, the skill and the YAML (`responses/`, `spectra/`,
`data/spectra/`), and the referenced file did not exist (spec issue T4).

## Options considered

1. **Hash the YAML text** — trivially sensitive to comments and key order, so every cosmetic edit
   invalidates every LUT; and it ignores the data files the config points at.
2. **Hash the validated model's canonical JSON, with file paths replaced by file-content hashes** —
   invariant to text cosmetics and to `60` vs `60.0`; moves for every physical leaf; moves when a
   referenced file's bytes change; independent of where the data root lives.
3. **Hash only the fields a consumer says it depends on** — precise, but every consumer must
   enumerate its dependencies correctly, and a forgotten field is exactly the bug this exists to stop.

## Decision

Option 2, with two hashes (`irsim.config.loader`):

- `config_hash` — SHA-256 of `json.dumps(model_dump(), sort_keys=True)` after replacing every field in
  `DATA_PATH_FIELDS` by `{"sha256": <file bytes hash>}`. Keys goldens and pipeline outputs.
- `band_hash` — the same over `sensor.band` only. Keys LUTs, so the consumer/professional/industrial
  NETD grades of one detector (§16.1) share one table; a 1e-3 change in one response sample changes it.
- The test suite perturbs **every numeric leaf** of the Boson file by 1 % (downward at an upper bound)
  and flips every enumerated leaf; the hash must move each time. Only `ratios_3d.tvh`, pinned to 1.0
  by the schema, is exempt.
- **Data root** — `data_dir` argument, else `$IRSIM_DATA_DIR`, else `<repo>/data`. Data-file fields
  are relative to it; the loader stores the resolved absolute path on the model and raises
  `FileNotFoundError` naming the field, the raw value and the root when a file is missing.
- **Canonical layout** — `data/spectra/responses/<name>.csv` for R(λ); later `data/materials/`,
  `data/solar/`, `data/nk/`. The Boson YAML now points at `spectra/responses/boson_vox.csv`; the spec
  block is corrected in M0.10; M1.3 authors the file. Until then the loader fails by name on the repo
  file — a test records that state and flips to "loads" when the file lands.
- Round trip: `dump_sensor_config` → `load_sensor_config` gives an equal model and equal hashes.

## Consequences

Any new file-referencing field must be added to `DATA_PATH_FIELDS` — one line — for both hashes to
see it. Hashing reads the referenced files each time (64 kB-scale, negligible). No physical error.

## Revisit when

Material or atmosphere configs reference files (M7.2, M8) — extend `DATA_PATH_FIELDS` and decide
whether a material hash joins `band_hash` in the LUT key.
