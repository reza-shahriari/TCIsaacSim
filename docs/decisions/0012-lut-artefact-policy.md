# ADR 0012 — LUT artefacts: generated, not committed; `.npy` + `.f32` + sidecar

**Status:** Accepted (provisional on roadmap open question 6)
**Date:** 2026-09-11

## Context

Each band needs four 64 kB float32 tables (ADR 0011) consumed by NumPy, by Lua/SPG kernels and
eventually by Unreal. The ir-radiometry skill said "committed"; `.gitignore` excluded `data/lut/`;
open question 6 asked the team. A stale table — one built from a response file or band block that has
since changed — is the failure mode that matters, and it is silent unless the artefact knows its inputs.

## Options considered

1. **Commit the binaries.** Clone-and-run works; but every regeneration is an unreviewable binary diff,
   tables built on different libm/NumPy versions differ in the last ulp (spurious diffs), and the
   build takes 0.6 s per band anyway.
2. **Generate on demand (`make luts`), gitignored; sidecar with input hashes; loader refuses stale.**
   Nothing binary in git; determinism is tested (two builds are bitwise equal on one machine);
   staleness is detected by content hash, not by trust; the committed regression is a small golden
   slice (M1.11) at 1 mK tolerance, which tolerates libm ulps.
3. **Git LFS** — infrastructure for 256 kB per band; not justified.

## Decision

Option 2, `irsim.radiometry.lut_files`:

- Bundle = `<key>_{lb,lb_q,dlb_dt,dlb_q_dt}.npy` (float32, asserted on save and on load; a
  hand-edited float16/float64 file is rejected), the same tables as raw little-endian `.f32` for
  Lua/SPG (`cuda.array`) and Unreal (`PF_R32_FLOAT`), and `<key>_lut.json`. `<key>` is the first
  16 hex digits of the band hash (ADR 0008).
- Sidecar: schema version, sensor name, band block, full band hash, config hash, spectral-file
  SHA-256, grid (T0, T1, N, dt), dtype, file names, raw layout, NumPy and irsim versions.
- `load_band_lut_for_config` recomputes the band hash and the spectral SHA-256 from the *current*
  config and data. A bundle for the current hash loads; a bundle for the same sensor with a different
  hash raises `StaleLUTError` naming whether the **spectral file** or the **band block** changed and
  the remedy (`make luts`); no bundle raises `FileNotFoundError` with the same remedy. Sensor configs
  that differ outside the band block (NETD grades) share one bundle.
- `data/lut/` stays gitignored. Tests build LUTs in memory (session fixtures); the golden slice is the
  committed regression. CI does not need `make luts`.
- Regeneration must be deterministic: tested bitwise for two builds in one environment. Across
  environments the last-ulp differences are absorbed by the 1 mK golden tolerance, not by exact
  comparison.

## Consequences

A fresh clone must run `make luts` (0.6 s per band) before a pipeline run; the error message says so.
Open question 6 is answered provisionally here; if the team prefers committed binaries, only
`.gitignore` and this ADR change — the sidecar and stale detection stay.

## Revisit when

A band's build cost grows past seconds (finer grid, spectrally varying QE), or a consumer without a
Python build step (Unreal) needs the tables shipped — then commit the `.f32` files only, via LFS.
