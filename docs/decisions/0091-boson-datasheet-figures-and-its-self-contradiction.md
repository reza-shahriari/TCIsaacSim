# 0091 — Which Boson datasheet numbers the configs take, and which of its two answers

Date: 2026-09-17
**Status:** Accepted
Roadmap: SC.3 (§9.4, §10.2, §16.1)

## Context

`configs/sensors/flir_boson_640_lwir.yaml` and `halmstad_boson_320.yaml` carried three values that
disagree with FLIR's published figures for the part they claim to model:

* `ffc_interval_s: 180`, which matches **no** published default;
* `thermal_time_constant_ms: 10.0`, marked ESTIMATED against "typical VOx 8–12 ms", while the
  datasheet states the number outright;
* `ratios_3d` with `th = tv = 0.05` and no provenance marker at all, in a file whose header says
  "Values marked ESTIMATED are literature/typical figures, not measurements".

The source is [R24], *FLIR Boson® Thermal Imaging Core Product Datasheet*, Doc. # 102-2013-40
Release 340, March 2021 — free, public, and marked EAR99. No camera is needed to read it, which is
why this step exists in a project with no camera.

## The datasheet contradicts itself about the FFC defaults

This is the part worth writing down, because the next person to check will find both numbers.

**Section 5**, the normative parameter description, says:

> FFC Period is specified in seconds (e.g., the factory-default value of 300 represents a 300
> second (5 minute) maximum time between successive FFC events).

> FFC Temp Delta is specified in tenths of a Celsius degree (e.g., the factory-default value of 10
> represents a 1 deg temp change between successive FFC events).

**Table 8**, "Partial List of Modes, Parameters, and Operations Controllable through the CCI", in
the same document, gives `FFC Period` as **1200 (20 minutes)** and `FFC Delta Temp` as **30 (3.0
Celsius degrees)**.

Three things decide it for us:

1. Table 8 carries its own staleness warning immediately above it — "Note: Table 7 to be updated
   for Release 3.0 in a later revision" — and is described as a *partial* list, i.e. a reference
   convenience rather than the specification.
2. Table 8's pair (1200 s, 3.0 °C) matches the **2018** *Boson FFC/NUC Control Application Doc*
   (rev 120, 21 Sep 2018), which states "FFC Temp Delta is at its factory-default value, 30
   (3.0C)". Table 8 preserves an older release's defaults; Section 5 is the 2021 one.
3. Section 5 is internally consistent across paragraphs in a way Table 8 is not: the FFC Start-up
   Period paragraph says an FFC occurs "every 1/3rd degree" when Temp Delta "is set to its
   factory-default value, which results in an FFC event every 1 degree when at steady-state" —
   which is true of 1.0 °C and false of 3.0 °C.

**Decision: the configs take Section 5.** `ffc_interval_s: 300`.

## The three temporal-noise components, and what they cannot tell us

Table 13, *Temporal NEDT in high-gain state*, is the closest thing FLIR publishes to a noise
specification, and it is not a scalar NETD:

| Camera grade | Random temporal noise (tvh) | Column noise (th) | Row noise (tv) |
|---|---|---|---|
| Industrial | < 40 mK | < 14 mK | < 14 mK |
| Professional | < 50 mK | < 18 mK | < 18 mK |
| Consumer | < 60 mK | < 21 mK | < 21 mK |

with the conditions stated beside it: "the lensless configuration with an f/1.0 aperture
installed", "operating in the high-gain state at 20 °C, with the averager disabled, in free-running
mode, imaging a 30 °C background", and "NEDT values with averager enabled are approximately 20%
lower". With a lens the limits scale by (f/#)²/τ.

Two consequences.

**The `netd_mk_at_300k` field is σ_TVH, not a total.** That is already the convention ADR 0025
chose ("the anchored σ is the per-pixel temporal σ; the 3-D components take their ratios from it"),
and Table 13 confirms it was the right reading rather than a convenient one. The configs now say
so where the number is authored.

**The ratios are not derivable from the limits, and are left as ESTIMATED.** Every grade gives
th/tvh = tv/tvh = **0.35** (14/40, 18/50, 21/60), against the **0.05** both configs carry — so the
committed cameras are modelled seven times more spatially uniform than FLIR guarantees. The
temptation is to write 0.35 in and call it sourced. It would be wrong:

* Table 13 publishes **acceptance-test limits**, i.e. upper bounds. A camera that ships at 35 mK
  tvh and 5 mK th passes every line of it. The ratio of two upper bounds is not the ratio of two
  typical values, and nothing in the document says the two components sit at their limits together.
* The configured 0.05 is therefore **compliant** — at the professional grade's 50 mK it puts
  σ_th at 2.5 mK against an 18 mK limit — and compliance is all the datasheet can adjudicate.
* The project already has a better source for this specific question: ME.5's *measured* ratios off
  the Halmstad Boson 320 set (vh 2.64, v 0.58, h 0.16, t 0.38 against 0.30/0.08/0.15/0.02).
  Substituting those is `SC.2`, and open question 6 fixes which file gets which: datasheet limits
  asserted on the 640, field-measured ratios on the 320, never mixed in one `ratios_3d` block.

So `SC.3` marks both blocks ESTIMATED with the limits and the conditions recorded beside them, and
`tests/unit/test_boson_datasheet.py` asserts **compliance with Table 13** rather than equality with
a ratio the table does not contain. The bench reproduces the stated conditions, and it is a real
check: the same bench at the industrial grade's 14 mK limit fails if the ratio is raised past 0.35.

## Consequences

* Every Boson `config_hash` moves, which is correct and is the mechanism working. Five golden
  fixtures were regenerated deliberately (`make golden-update`), not to silence a failure.
* The membrane is 20 % faster. τ/frame period falls from 0.60 to 0.48 at 60 Hz, so the first
  frame of a step now shows α = 0.8755 of it instead of 0.8111 — a *smaller* lag, i.e. this
  project had been modelling a slower detector than FLIR ships.
* The shutter fires every 300 s instead of every 180 s, so a 60 Hz sequence closes on frame 18000
  rather than 10800, and the NUC residual has 1.67× as long to grow between events.
* `MIN_INTERVAL_CLIP_S` moves 180 s → 300 s with it. Its docstring derives it from the camera's
  schedule — "a clip below this can only ever catch one event and an interval measured on it is
  an artefact" — so it is not an independent constant, and leaving it behind would have let a
  250 s clip report an FFC interval it cannot possibly have measured.
* The 1.0 °C temperature trigger and the 90 s start-up period at one-third delta are **not
  modelled**. `NucSpec` has a time-based trigger only. This is a real gap — a camera warming up
  shutters far more often than every 300 s — and it is left open rather than approximated, because
  the FPA-temperature track that would drive it is the `housing_tau_s` model and nothing currently
  connects the two. Recorded as roadmap open question 12; `SC.11` is a
  different step and is not this.
