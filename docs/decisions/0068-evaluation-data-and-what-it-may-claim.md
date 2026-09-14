# ADR 0068 — Evaluation data: what the public sets are, and what may be claimed from them

**Status:** Accepted
**Date:** 2026-09-14

## Context

There is no IR camera on this project and there will not be one soon (ADR 0003), so every
comparison with reality is made against imagery somebody else published. That is workable, but only
if the *provenance* of each frame is treated as part of the measurement rather than as background
information. The same flat patch of sky supports a three-dimensional noise decomposition if it came
off a Y16 stream and supports nothing at all if it came out of an undocumented ISP through a lossy
codec; a box size converts to a range only if the pitch and focal length are known; a histogram
shape measures the recorder, not the camera, when the recorder did the 16→8-bit conversion.

The failure this ADR exists to prevent is quiet and one-directional: a number measured on a public
clip gets written into a table, the simulator is then tuned until it reproduces that number, and the
simulator has been fitted to a codec. Every decision below is a gate designed to make that harder
than doing it properly.

This ADR was opened at ME.1a (the index) and is extended by each ME step. ME.2b's codec floor is
recorded in ADR 0023's addendum because it is an extension of that estimator; the reading rules and
the Tier 4 acceptance targets live here. **The DN8 acceptance targets that replace §15's radiometric
ones are added to this ADR at ME.6**, when the comparison code that uses them exists.

## Options considered

1. **Index the sets informally in a README.** Cheapest, and what most projects do. Rejected: prose
   cannot gate an analyser, and the decision about whether a measurement is legitimate then gets
   made silently, differently, every time somebody writes a script.
2. **Machine-readable index with required provenance fields, and analysers that refuse.** A set that
   cannot state its licence, its signal path and the analysers it supports cannot be added; each
   analyser checks the property it depends on before producing a number.
3. **Take the data at face value and validate against it directly.** Rejected outright: three of the
   six indexed sets have an undocumented sensor, so there is nothing to convert a measured contrast
   or box size into.

## Decision

Option 2, in three layers.

**1. The set must say what it is (ME.1a).** `data/validation/datasets.yaml` + `irsim_eval.manifest`
make `licence`, `signal_path` and `analysers` required fields, with `excluded_analysers` beating
`analysers`. `licence: unstated` is a recorded fact, not a default — checked against each
publisher's own page on 2026-09-13, only the Halmstad set (CC0-1.0, Boson 320×256, Y16 → 8-bit →
mp4) states one, so it is the single `primary` and the other five are `unstated`.
`scripts/fetch_validation_data.py` refuses an `unstated` set unless explicitly asked, never
auto-downloads a `manual` one, and always hashes what is on disk.

**2. One canonical form, and 8 bits enforced (ME.1b).** Every converter writes
`irsim_eval.data.Sequence`; no analyser reads a publisher's layout. The reader refuses anything but
8-bit, because every indexed set *is* 8-bit and silently widening to float would invite a claim the
data cannot support.

**3. The analyser refuses what its data cannot support.** Three gates so far:

- **Clip length.** An FFC *interval* may only be reported from a clip of at least
  `MIN_INTERVAL_CLIP_S = 180 s` — the Boson's own schedule. Halmstad's clips are 10 s, so they can
  show a freeze *length* and can never show an interval; `freeze_intervals_s` raises rather than
  returning the spacing of two events that happened to be close. A real core also fires on ΔT_FPA,
  so the interval is a **distribution whose upper edge** is the configured schedule: a mean below
  180 s is the expected observation, not evidence against the schedule.
- **Frame-to-frame change.** Freeze detection compares each gap with the clip's own median gap.
  A codec that has removed the temporal noise leaves no median gap to compare against, and
  `find_freezes` raises instead of reporting the whole clip as one freeze. Measured through x264 on
  a 64×64 clip with a 42-frame freeze every 180 frames: at 4 codes of temporal noise the events
  survive CRF 18 **exactly** (start frame and length); at 1 code, CRF 23 flattens the clip and the
  detector refuses, and CRF 18 is worse than refusing — it invents an event at a frame the camera
  never froze on. This is the ME.2b codec floor in a second statistic.
- **Signal path.** Histogram-shape and edge-overshoot statistics measure an ISP, so they run only
  on sets whose frames *are* the camera's display output. (The gate itself lands with those
  extractors in ME.3b; the Halmstad entry already excludes `agc_signature` and `dde_overshoot` for
  this reason.)

**A freeze is a run of still frames; an FFC is a freeze followed by a new fixed pattern.** Counting
repeated frames alone cannot tell a shutter from a dropped chunk of recording, and the two would
contribute identically to an interval distribution. `find_freezes` therefore reports
`pattern_change` beside every run: the step in the time-averaged frame across the run, in units of
what the temporal noise alone would produce. A shutter lands far above 1 (8–20 on the synthetic
clips), a stall lands at 1, and neither is discarded — a stalled recording is a reason to distrust
every temporal statistic from that clip.

**What grows between shutter events grows from zero, and the factor of two is in the variance.**
After a shutter the correction is recalibrated, so the part of the pattern the shutter owns is zero
and relaxes back toward its stationary level (§11.2, §10.3). For an OU pattern of correlation time
τ started at zero the variance recovers as `1 − e^{−2t/τ}`, so `fit_pattern_growth` fits
`σ²(t) = floor + A(1 − e^{−2t/τ})` with `floor` absorbing the white noise and any pattern the
shutter does not recalibrate. Fitting the variance with the amplitude's law returns 2τ and looks
entirely reasonable, which is why the test asserts the true τ *and* rejects 2τ. The projection is
the column and row means: §10.3's stripe noise lives there, they average white noise down by the
width of the array, and a NUC residual moves them. Scene structure lives there too, so these run on
the flat windows ME.2b's finder returns, not on a whole frame with a horizon in it.

## Consequences

Measurements from this data come with a gate that can refuse, which means some questions have no
answer from the public sets and the reports will say so rather than fill the cell. Concretely:
Halmstad can never give an FFC interval; Anti-UAV410 and CST can never give a 3-D noise ratio (the
index already excludes it); no set gives a radiometric bias, so §15's "< 2 K per class" target is
declared untestable and reported as open in every Tier 4 report rather than quietly dropped.

The gates cost real coverage and that is the point: the alternative is a table of numbers whose
provenance nobody can reconstruct six months later. Everything measured here is a **lower bound or a
shape**, never an absolute, and ME.5/ME.6 have to carry it as one.

## Revisit when

A radiometric (16-bit, documented-sensor) public set appears, or a camera arrives — either makes the
absolute targets testable and turns most of these gates off. Also revisit if a licence status
changes: the fetch script's `unstated` refusal is the only thing standing between an experiment and
an unlicensed redistribution.
