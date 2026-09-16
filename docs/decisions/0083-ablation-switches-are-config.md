# 0083 — Ablation switches are config, and `schema_version` is not in the config hash

Date: 2026-09-15
**Status:** Accepted
Roadmap: ME.8 (§15 Tier 5 caution 1)

## Context

§15's first caution about a Tier 5 sim-to-real study is that a gap has to be *attributed*: run the
same scenario with one mechanism off, and see whether the gap moves. That requires ablation
variants, and it requires being able to tell two variants apart a year later.

Every ablation this project needs already existed. Four of them existed as **keyword arguments**:

| mechanism | where it lived |
|---|---|
| detector + 3-D noise | `PipelineConfig.from_sensor(noise_enabled=...)` |
| optical PSF | `PipelineConfig.from_sensor(psf_enabled=...)` |
| dead/hot pixels | `attach_sensor_chain(defects_enabled=...)` |
| NUC residual | `attach_sensor_chain(residual_enabled=...)` |

A keyword argument is invisible to `config_hash`. Two ablation variants therefore produced
*identical* hashes, so a stored result could not say which one produced it — and a golden or a
reference-statistics table keyed on the hash would silently describe the wrong camera. This is the
same shape of defect as ADR 0077 and ADR 0082: a mechanism exists, is tested, and the configuration
layer cannot reach it.

The other ablations §15 names were *already* explicit, hashed config:

* **AGC** — `isp.agc: linear` (or `none`).
* **FFC** — `nuc.mode: ideal`, a core that never shutters.
* **Clouds** — cloud *fraction* is weather (`WeatherSeries.cloud_fraction`), never authored, so a
  cloud-free run is a weather file; `environment.clouds.tau` sets what a cloud is.

## Decision

1. A `fidelity:` block in the sensor config carries exactly the four switches that had no config
   home: `noise`, `optical_psf`, `bad_pixels`, `nuc_residual`. All default `true`.
2. The four keyword arguments default to `None`, meaning *read the config*. An explicit bool still
   overrides, as a bench affordance; a run whose provenance matters sets the config, because only
   the config reaches the hash.
3. The AGC, the FFC and the clouds are **not** duplicated into `fidelity:`. A second way to set
   them would raise the question of which copy wins, and the answer would have to be arbitrary.
4. A `fidelity:` block equal to the all-`true` default is **dropped from the canonical dump**, so
   full fidelity written out longhand hashes the same as full fidelity left unwritten. The hash
   tracks the ablation, not the notation.
5. `schema_version` is **excluded from the config hash**. It describes the document format, not the
   sensor: a v8 file and a v9 file that describe the same camera are the same input and must
   produce the same reference.
6. The sensor schema gains a readable **range** (`MIN_SCHEMA_VERSION = 8`, `SCHEMA_VERSION = 9`),
   as the scene schema already has. v9 added `fidelity:` as an optional block with a full-fidelity
   default, so every v8 document is a valid v9 document describing exactly the camera it described
   before; refusing one would be refusing it for a change that cannot affect it.

## Consequences

**The risk in decision 5** is a future version that changes what an existing field *means* without
changing any field. Two such documents would hash the same while describing different cameras. The
guard is decision 6's floor: when a version genuinely stops being readable, `MIN_SCHEMA_VERSION`
rises and the old document is **refused outright** rather than quietly hashing like a new one. That
makes the floor load-bearing, not bookkeeping, and it must be raised whenever meaning changes.

**Evidence that this moved no physics.** The version bump changed every sensor's config hash, so
every golden went `STALE`. Regenerating them rewrote twelve `.json` sidecars and left **every
`.npy` array byte-identical** — which is the check that the change is provenance metadata and
nothing else. (`make golden-update` is for deliberate changes; this is what "deliberate" looks
like.)

**What M12.3 gets.** Sixteen combinations of four switches produce sixteen distinct config hashes,
so an ablation table can be stored keyed on the hash of the configuration that produced each row.

## Alternatives considered

* **Keep `schema_version` in the hash.** Honest in a narrow sense — it is part of the document —
  but it makes every additive, backwards-compatible schema change invalidate every golden, which
  trains people to run `make golden-update` without reading the diff. That is a worse failure mode
  than the one decision 5 risks, and decision 6 bounds that risk.
* **Duplicate the AGC and FFC into `fidelity:`.** Rejected under decision 3.
* **Drop the keyword arguments entirely.** Cleanest in principle, but the benches legitimately want
  to isolate one mechanism without authoring a file, and `defects_enabled`'s docstring already said
  so. Decision 2 keeps that and names it as an override.
