# ADR 0009 — Spectral response files: peak-1 shape, QE applied once, edge precedence

**Status:** Accepted
**Date:** 2026-09-11

## Context

Band radiance is ∫R(λ) B(λ, T) dλ (§3.2 b). Everything downstream — the LUT, NETD anchoring, the
photon-count chain — scales with how R(λ) is normalised, and four conventions are in common use:
peak-normalised shape, area-normalised shape, absolute QE(λ), and per-nanometre tables. A file in the
wrong convention loads without error and produces a wrong radiance that no image will reveal. The
repo also had no R(λ) file at all: the Boson YAML referenced one that did not exist.

## Options considered

1. **Absolute QE(λ) in the file** — physically complete, but bolometers have no QE, most vendors
   publish only relative curves, and the QE would then live in two places once a config also states it.
2. **Area-normalised shape** — convenient for "effective wavelength" maths; the peak carries no
   meaning and a top-hat's value depends on its width, so cross-checks against closed forms get
   confusing and a renormalisation step hides file errors.
3. **Peak-normalised relative shape, QE separate** — matches vendor data; the file is dimensionless
   and comparable across cameras; a top-hat file reproduces `band_radiance_tophat` exactly.

## Decision

Option 3 (`irsim.radiometry.spectral_response`, `irsim.radiometry.band`):

- **R(λ) is a peak-1 shape**: max(R) = 1 ± 1e-6 asserted, never renormalised. Area-normalised and
  QE-scaled files are refused with a message that says where QE belongs.
- **QE is `fpa.quantum_efficiency`**, a scalar in the sensor config, applied exactly once in the
  photon-detector model (M3/M4). Bolometers have none; their responsivity is anchored by NETD (§9.4).
- **Units**: wavelength strictly increasing in micrometres within 0.1–100 µm (7500–13500 is refused
  as nanometres); response in [0, 1]; no NaN; `#` lines are provenance and are kept on the object.
- **Zero outside support**: `resampled(grid)` interpolates linearly and returns 0 beyond the file's
  first/last sample. Files should therefore extend to where R is genuinely ~0.
- **Config edges vs file**: the CSV *is* the response. `band.lambda_min_um/max_um` are the nominal
  edges used for `band.id`, top-hat cross-checks and reporting. `Band.from_spec` requires the file's
  half-power points to sit within 0.5 µm of the configured edges, else the wrong file is attached.
- **Boson estimate**: `data/spectra/responses/boson_vox.csv` is flat over 7.75–13.25 µm with
  raised-cosine edges of ±0.25 µm centred on 7.5 and 13.5 µm, 0.01 µm grid, marked ESTIMATED in its
  header. Its integral equals the 6 µm top-hat width exactly (symmetric edges), so `Lb(300 K)` will
  sit within a few percent of the top-hat closed form; the error against a measured curve is expected
  to be up to ~10 % and is recorded again when the LUT is generated (ADR 0013).

## Consequences

Adding a camera means adding a CSV in this contract plus a YAML. The estimate's error is unbounded
until a measured VOx curve is obtained; every radiometric statement about the Boson carries that
caveat. No renormalisation means a vendor curve published in percent must be divided by its peak
before committing — deliberately, by a person, with the provenance line saying so.

## Revisit when

A measured Boson response becomes available (ADR 0013), or a photon-band camera needs a spectrally
varying QE — then QE(λ) would become a second file, still separate from the shape.
