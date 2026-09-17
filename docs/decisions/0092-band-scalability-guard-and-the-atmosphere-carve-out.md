# ADR 0092 — The band-scalability guard denies by default, and the atmosphere keeps one carve-out

**Status:** Accepted
**Date:** 2026-09-17

## Context

CLAUDE.md names "bands are data, not code" the main scalability requirement of the project: adding a
band must be a YAML change, not a kernel change. `tests/unit/test_band_scalability.py` asserts it —
an AST walk that flags any band name (in a string or an identifier) and any nominal band edge used
as a wavelength.

The guard scanned a **positive list** of six packages:

```python
CORE_PACKAGES = ("radiometry", "optics", "detector", "noise", "isp", "pipeline")
```

That shape has three defects, all measured before this change:

1. **The exemption was an omission, not a decision.** `atmosphere`, `materials`, `thermal`, `io`,
   `validation`, `config` and `scene.py` were unguarded because nobody had listed them. Four of
   them — `materials`, `io`, `thermal`, `validation` — and `scene.py` carried **zero** offences and
   had for their whole lives, so the carve-out that named `irsim.materials` rescued nothing. This is
   the same shape as the float16 carve-out `IG.8` removed.
2. **The guard could switch itself off in silence.** Each test re-globbed its own package
   directory, so a package renamed or moved in the tuple walked a path that does not exist, which
   yields no files, no offences, and a green test. The one coverage assertion counted the six
   packages' union against a floor of 30 files; the six hold 72, so dropping even the largest of
   them still passed.
3. **The guard was blind to the fifth band the codebase already contains.** Both halves tested
   membership in `BAND_IDS` = `("nir", "swir", "mwir", "lwir")`. But `visible` has a spectral-class
   table, a Planck weighting temperature, a mandatory key in all seven atmosphere presets, a
   wavelength range of its own (`VISIBLE_RANGE_UM`) and an `if band == "visible"` branch — and not
   one of those tripped a guard whose forbidden set stopped at four names. `AT.4`'s exit bar ("a
   fifth band added to a sensor YAML renders with no edit under `src/`") was therefore already
   falsified by a worked counter-example in the tree.

## Decision

**The guard scans every module under `src/irsim` and carries the carve-out as a tested constant.**

`BAND_AWARE` maps a path to `(ceiling, why)`. The ceiling is a ratchet: it may fall freely, and it
rises only by someone editing the table. Three tests keep it honest — every entry must name a file
that exists, must still have at least one offence (a carve-out that rescues nothing is deleted), and
must not have grown past its ceiling. A module is guarded the day it is created, not the day someone
remembers to list it.

The forbidden **name** set widens to `BAND_KEYS` = the four camera bands plus the anchor. The
forbidden **edge** set deliberately does *not* widen to the anchor's `(0.4, 0.7)`: those are
photopic definition points, and adding them buys one true positive against one false one —
`irsim.radiometry.solar.TRUSTED_MIN_UM = 0.7` is the wavelength below which the 5778 K solar model
departs from the real spectrum, a fact about the sun rather than a band limit, and exempting a clean
radiometry module to catch it costs more coverage than it wins.

**The registry gains the anchor band.** `ANCHOR_BAND`, `ANCHOR_RANGE_UM`, `ANCHOR_REGIME`,
`BAND_KEYS`, `regime_for()` and `nominal_range_for()` live in `irsim.config.bands`. The anchor is a
*reference* key, not a camera band — no `BandId` names it, `band_id_for` cannot produce it (0.4–0.7 µm
overlaps no nominal range by 50 %) and no sensor YAML may declare it — but the atmosphere needs it
spelled, and spelled by hand it was five independent copies of one convention.

**Two of the three couplings the roadmap row named are removed:**

* `WEIGHT_T_REF_K`, a five-row dict keyed by band name, is replaced by
  `weight_reference_temperature(band)`, which asks the registry for the regime. All five shipped
  rows are reproduced exactly.
* `ATMOSPHERE_BAND_KEYS` now derives: `frozenset(BAND_KEYS)`. `VISIBLE_RANGE_UM` and the
  `if band == "visible"` branch (with its `cast(BandId, band)`, which lied about the type) are gone.

**`BAND_CLASSES` keeps its carve-out**, recorded here with its reason. The roadmap row explicitly
permits this ("if the carve-out is kept, it is recorded with its reason").

## Why `BAND_CLASSES` was not fixed in this step

The table is spectroscopy — H₂O at 0.94/1.1/1.4/2.7/6.3 µm, CO₂ ν₃ at 4.3 µm, the rotational band
past 13 µm — filed under camera names. Unifying it into one wavelength-ordered ladder is the right
end state, and two things measured during this step say so:

* **The per-band multipliers are a gauge, not a physical disagreement.** `exponential_sum` solves one
  scale factor per band so the horizontal 200 m transmittance matches the grey preset's. Multiplying
  every non-opaque multiplier in a band by 3 moves τ at 200 m / 1 / 5 / 20 km by at most **3.2e-14**
  in all four bands. So "LWIR window 0.3 vs SWIR window 0.5" carries no information; only ratios
  within a band do. The objection that a flat table cannot reproduce the fit does not survive
  measurement.
* **The band keying has already produced a physics defect.** SWIR's table and NIR's table disagree
  about the same air. NIR resolves the 0.94 µm water band (`h2o_0p94`, 0.90–0.98 µm, ×10); SWIR's
  window swallows the same interval at ×0.5. Resolving it in favour of the feature moves **16.9 %**
  of the shipped InGaAs band's Planck-weighted response out of "clear window", and τ by −1.5 % at
  1 km, **−6.4 % at 5 km** and +37.8 % at 20 km. The 200 m anchor is exact by construction, which is
  precisely why no existing test caught it. A second, smaller conflict sits at 0.70–0.80 µm
  (`visible` window 1.0 vs NIR window 0.5), and the union of the shipped intervals has holes at
  1.80–2.00 µm and 6.00–7.00 µm that no band claims.

That is a physics change with a golden update, an API change to the public `class_weights`, and
migration of three test files. Folding it into a guard commit would have shipped a −6.4 % τ change
under a refactor's message. It is `AT.10`, with the measurements above already taken.

## Consequences

* A new module under `src/irsim` is band-guarded from creation. 140 of 146 modules are guarded today;
  six are exempt, each with a written reason and a ceiling.
* Five of the six carve-outs are a fact about the world (`config/environment.py`'s LWIR-only sky
  depression, `config/sensor.py`'s solar/thermal crossover) or a pydantic field name already written
  into YAML on disk (`aerosol_ratio_to_visible`, in seven presets). One — `atmosphere/layered.py` — is
  neither, and a test asserts that so it cannot quietly become permanent.
* A fifth band still does **not** render without an edit under `src/`: it needs a `BAND_CLASSES`
  entry. `AT.4`'s exit bar is met on its second clause (the carve-out is recorded and tested), not
  its first. `AT.10` owns the first.
* `WEIGHT_T_REF_K`'s removal came within one line of a silent calibration change. The table weighted
  MWIR at 300 K while `DEFAULT_REGIME["mwir"]` is `"mixed"`, so the derivation a reader reaches for
  first — "emissive keeps 300 K, everything else is solar" — reproduces four rows and moves MWIR to
  5800 K, rescaling every MWIR class share, its anchor solve, and τ_MWIR at every range but 200 m.
  The rule that reproduces all five turns on *reflective*. A test asserts the two rules still differ,
  so it cannot stop guarding the drift it was written for.
* `SOLAR_WEIGHT_T_K = 5800.0` sits beside `irsim.radiometry.solar`'s 5778 K for the same sun. The
  22 K disagreement moves the shipped SWIR and NIR class shares by 1.2e-3 relative (measured) —
  below this fit's own uncertainty, but it is two constants for one fact, and it is recorded rather
  than reconciled here.

## Revisit when

`AT.10` unifies `BAND_CLASSES` into a wavelength ladder — at which point `atmosphere/layered.py`'s
ceiling should fall to its two Koschmieder identifiers, and the "only carve-out that is not physics
or a schema key" test should be deleted rather than updated.
