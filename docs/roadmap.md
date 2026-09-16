# irsim roadmap

**Revision 4, 2026-09-15.** Replaces revision 3 (2026-09-10) in full. The plan for taking `irsim` from
where it is today to an L2-fidelity multi-band IR camera simulator in Isaac Sim 6.1, with a clean upgrade
path to L3. Derived from `docs/physics-model.md` (cited as §N.M), the seven project skills, five subsystem
audits run against the code and `git log`, and a survey of production IR simulators, published camera
datasheets, measurement standards, public dataset primary sources and the 2026 sim-to-real literature.

---

## How to read this

**Provenance of every number.** Every figure below marked *measured* was measured on 2026-09-15 on the
working tree at commit `ca5a663` **plus** the uncommitted work in it at that moment: modified
`CHANGELOG.md`, `README.md`, `docs/roadmap.md`, `src/irsim_isaac/maritime_demo.py`,
`tests/unit/test_maritime_stage.py`, and untracked `configs/scenes/vessel_departure_clear_day.yaml` and
`scripts/render_vessel_departure.py`. That is not a clean tree and the distinction matters: several
findings cite `render_vessel_departure.py` by line. Test collection moved 2746 → 2756 during the audits
because parallel sessions were adding files, so a test count is a timestamp, not a property.

**Picking a step.** Take the first step in phase order whose `deps` are all ticked. Ship it with the
`ship-step` skill. Tick it in the **Shipped ledger** in the same commit, with the commit hash.

**Why this document no longer contains 182 fat rows.** Revision 3 held 224,508 of its 298,272 characters
inside markdown table cells, averaging 1,234 characters per row with a 4,958-character maximum. Those rows
were a second changelog: unrenderable, unreviewable as a diff, and unmergeable between two sessions
without a manual rewrite. Shipped work now lives in a compact ledger; the narrative of what shipped lives
where it belongs — in `git log`, in the ADRs and in `CHANGELOG.md`. **Step rows are capped at 600
characters and ledger rows at 200**, enforced by a parser in `make check` (RP.9). A row that needs more
space needs an ADR instead.

**Row format.** `id · what · verification · deps · size · phase`. The **verification** cell states what
would fail if the physics were wrong, and why the tolerance is that number — not that a function was
called. Sizes: S (hours), M (a day or two), L (several days). A step that turns out bigger gets split,
per CLAUDE.md.

**Lane prefixes.**

| lane | subject |
|---|---|
| `RP` | Repair: damage already done to README, CHANGELOG, the roadmap, the ADR corpus and the spec-issues log |
| `PT` | Point-wise surface temperature — the owner's headline requirement |
| `WM` | Warp mesh surface parameterisation — point-wise on curved geometry |
| `AT` | Atmosphere, sky and materials physics |
| `SE` | Sea and maritime |
| `SC` | Sensor chain and published-reference anchoring |
| `EV` | Evaluation methodology and sim-to-real |
| `XD` | External data anchors: public datasets and public measurements |
| `IG` | Isaac glue integrity |
| `GT` | Gates, tests and tooling |
| `DC` | Decisions, deferrals and probes |

Legacy ids (`M0`–`M12`, `ME`, `MS`, `MM`, `MP`, `IU`) are preserved in the ledger and in ADRs. They are
not reused for new work, and eleven of revision 3's `deps` cells named ids that existed only as
letter-suffixed rows (`ME.1`, `ME.2`, `ME.3`, `M10.7`, `M10.9a`), which broke the picking rule for eleven
rows including `M10.9b`. New ids are two letters and a number, and every `deps` cell below names a row
that exists in this document.

**ADR numbers are allocated at write time, never forward.** The highest existing ADR is 0089 and four
numbers below it are cited but were never written (0042, 0062, 0069, 0079). Revision 3 also instructed a
future author to "record as ADR 0073 when scoped" for the exhaust plume, and ADR 0073 is the
visible-companion environment dome, Accepted 2026-09-14 — following that instruction would overwrite a
live decision record. In a three-session shared tree a forward-assigned number is a scheduled collision.
**This document names ADR subjects, not ADR numbers.** RP.4 writes the two that shipped code cites; DC.4
records the rule.

**Commit scopes.** `pipeline`, `validation`, `eval` and `io` remain proposed additions to CLAUDE.md's
list (open question 9). Until approved, use `build` for infrastructure and the physics scope a step tests.

---

## Current state, measured 2026-09-15

Revision 3's "Current state (2026-09-10)" section is deleted. Every claim in it was false — it said
radiometry had "no LUT, no inverse, no R(λ) loader" (all shipped 2026-09-11), that every other
`src/irsim` package was a one-line stub, that `src/irsim_isaac` had no AOV code, and that configs and
data were empty. It was the second section of the document, so a session reading top-down formed a wrong
model of the project before reaching any table.

| thing | measured |
|---|---|
| Physics core | 145 modules under `src/irsim`, engine-free, enforced by an AST scan |
| Isaac glue | 30 modules under `src/irsim_isaac`; `src/irsim_isaac/spg/` holds one file, `README.md` |
| Tests | 2,756 collected across 163 unit files and 6 golden files; `2748 passed, 8 skipped` at the audit run |
| Of which structural | 540 of 2,756 come from two per-file AST scanners (`test_layering.py` 359, `test_aperture_guard.py` 181). They scale with file count, not with verified physics. Quote coverage, not the raw count |
| Coverage | `src/irsim` 93.9 %, `src/irsim_eval` 85.6 %, `src/irsim_isaac` 43.5 %. Six glue files at 0 %, totalling 1,101 statements; `warp_stages.py` 38 % |
| Gate time | ~130 s for `tests/unit tests/golden`, against CLAUDE.md's stated 30 s. Slowest single test 12.04 s, exercising a solver no scene can reach |
| Integration tests | 15 files, 4,480 lines, all auto-marked `isaac`, run in no automated job anywhere |
| Configs | 5 sensors, 19 materials plus a mapping, 7 atmospheres, 8 scenes |
| Demo lanes | Six render drivers produce frames from one command: aerial demo, quad flight, aircraft pass, maritime demo, vessel departure, car ignition |
| Point-wise temperature | Real, correct, end-to-end — and reaching **two prims in one of eight scenes**, on the lane the owner ranked third |
| ADRs | 85 files, 0001–0089 with gaps at 0042, 0062, 0069, 0079. None is ever marked Superseded although the template offers it |
| Docs | `docs/roadmap.md` 888 lines / 304 KB; `CHANGELOG.md` 3,025 lines / 270 KB in one `[Unreleased]` section with 20+ repeated headings; `docs/maps/` 729 KB across 12 files with one commit ever |

---

## The owner's priorities, restated as acceptance criteria

These are requirements, not preferences. Each has a lane that owns it and a criterion that can be read
off a rendered frame.

1. **Per-point surface temperature, not one value per object.** This is the defect that drove the owner
   off their previous simulator. A lane satisfies it when a rendered frame of that lane's target shows a
   temperature gradient *across one prim*, authored from a scene config. Today: the car's bonnet, only.
   Owned by `PT` and `WM`.
2. **Aerial first, then maritime, then ground.** A lane is done when **a scene config plus one command
   produces frames** — and, from this revision on, frames that carry float32 planes and a config-hash
   sidecar (IG.13), because three of the six drivers currently emit 8-bit PNGs only and two of those three
   are the aerial and maritime flight renders.
3. **No IR camera.** All validation against reality uses freely available public data. **No step in this
   document proposes a hardware purchase.** Note additionally that acquiring a camera would not rescue
   §15's radiometric target (see *Not adopted*), so the constraint costs less than it appears to.
4. **Isaac Sim is the only engine.** Engine portability stays a *property* — CLAUDE.md non-negotiable #1,
   enforced by `test_layering.py`, and cheaply extended by keeping the Warp kernels free of
   `omni`/`pxr`/`isaacsim` imports. No step is spent on an Unreal deliverable (open question 2).
5. **Grayscale, white-hot.** All five sensor configs carry `palette: gray` and `to_display8` defaults to
   it. SC.11 adds black-hot as a *config switch and an evaluation axis*, not as a default — real maritime
   sets carry both polarities and a detector trained on one fails on the other.
6. **Evaluation is a first-class deliverable.** The `EV` lane is not a reporting afterthought; it ends in
   EV.9, a measurement that can return a negative about the project's own headline feature.

---

## Do next

**Ask the script, not the table.**

```bash
python scripts/next_step.py            # the one step to start now
python scripts/next_step.py --queue 20 # the next twenty, in order
python scripts/next_step.py --lane PT  # restrict to one lane
```

The picking rule stated in *How to read this* — "the first step in phase order whose deps are all
ticked" — **does not pick a step**. Measured on this revision the day it landed: **51 of 109** open
steps satisfied it at once, thirteen of them in phase 0 alone, and **67 of 109** open steps have no
dependents at all, so the dependency graph cannot order the majority of the plan. What was actually
choosing the next step was its row's position in a table. That is not a rule, it changes whenever
anyone reorders a table, and it cannot be argued with.

`scripts/next_step.py` is the rule. It is a **topological sort**, so a step never comes before
something it depends on, with ties broken by a total order:

| | tiebreak | why |
|---|---|---|
| 1 | **Promoted** | The documented exceptions, and the only place judgement enters. The mechanical key counts dependents; it cannot see that a step prevents a recurring loss. Kept to three at most — a long list means the rule itself is wrong |
| 2 | **Phase**, 0 → A → B → C → X | Where the owner's ordering lives. A dependency may still pull an X step forward; the sort does that on its own |
| 3 | **Dependents, descending** | Transitive, not immediate. A step unblocking ten outranks one unblocking two |
| 4 | **Size, ascending** | Between equal leverage, the small one first |
| 5 | **Step id** | A total order, so two sessions asking "what next" get the same answer |

**The queue is deliberately not written into this document.** A pasted list goes stale the moment a
step is ticked, and a stale queue is worse than none because it still answers. Ticking a step in the
table is all that is needed; the next call recomputes.

`tests/unit/test_roadmap_queue.py` is the guard, and it is what makes the answer safe to act on
without reading this document: the order is **total** (one head, two runs agree), **sound** (no step
before its dependencies), **complete** (every open step appears exactly once, so nothing is silently
dropped), and every pick is **the best available one at that moment** — so a disagreement with the
queue is a disagreement with the key above, which you can change.

---

## Phase plan

Ordered by the owner's application order, with one phase before all of them for damage that is live in
the working tree right now.

**CPU correctness before any acceleration.** The owner's rule, stated 2026-09-15: no GPU or device
work while the CPU reference still has known defects. The engine-free NumPy pipeline is the oracle
every fast path is tested against (ADR 0018), so a wrong oracle makes an accelerated path worth less
than nothing. Consequences, applied throughout this document:

* The **SPG/CUDA shader lane is not scheduled at all.** Legacy `M10.12` and `M10.13a`-`M10.13e` are
  dropped from the delivery phases; `DC.1` keeps only the *probe* that would tell us whether they are
  even possible on SPG 0.4.0, and `R2` stays open in the risk register. Nothing depends on them.
* **No step moves an existing stage onto a device.** Warp stage twins already exist and are already
  equivalence-tested; making more of the frame device-resident is a speed change and waits.
* Where a step needs a GPU-capable *library for a capability rather than for speed* — `WM` uses Warp's
  mesh closest-point query because it is the only exact per-pixel surface parameterisation available
  without a renderer change — it **runs on the Warp CPU backend by default**, keeps a brute-force NumPy
  closest point as its oracle (ADR 0018/0061), and is justified by what it makes possible, not by a
  timing figure. The RTX 5090 numbers quoted in `WM` are a headroom note, not a reason.
* `WM` is therefore scheduled **behind** the phase-A CPU defects rather than beside them: `AT.1`,
  `SC.1`, `PT.1` and `PT.2` are the four critical CPU-path defects and none of them needs a GPU.

| phase | contents | exit |
|---|---|---|
| **0 — Repair** | `RP.1`–`RP.9`, `PT.3`, `PT.4`, `IG.1`, `IG.5`, `IG.8` | The three shared documents are true and mergeable; no shipped physics result rests on a measured error |
| **A — Aerial to the bar** | `AT.1`–`AT.4`, `PT.1`, `PT.2`, `PT.5`, `PT.9`, `IG.2`, `IG.6`, `IG.13`, `SC.1`–`SC.4`, `GT.1`, `GT.2` — **CPU only** | An aerial scene config plus one command produces float32 frames whose target carries a gradient across one prim, with a per-pixel slant path behind it |
| **B — Maritime to the same bar** | `SE.1`–`SE.3`, `PT.10`, `WM.1`–`WM.5`, `IG.12`, `XD.3` | A maritime scene config plus one command produces the same, with the sea model's angular envelope recorded |
| **C — Ground and automotive** | `PT.6`–`PT.8`, `PT.11`–`PT.16`, `WM.6`, `AT.6`–`AT.9`, `XD.10` | Deferred breadth stays deferred (see *Deferred deliberately*); what lands is depth on surfaces already modelled |
| **X — Cross-cutting, continuous** | `EV.*`, `XD.*`, `DC.*`, `GT.3`–`GT.6`, `SC.5`–`SC.13`, `IG.3`, `IG.4`, `IG.7`, `IG.9`–`IG.11`, `IG.14`, `IG.15` | Runs alongside; `EV` gates nothing but is gated by `PT.9`/`PT.10` for its headline measurement |

**Dependency shape.** Phase 0 blocks nothing technically but blocks *knowing what is true*, and three
sessions share this tree. `AT.1` and `SC.1` are the two critical-priority physics defects and are
independent of each other. `WM.1` is a probe and gates `WM.2`–`WM.4`, and the whole `WM` lane is gated on phase A closing — it is a capability step, not an acceleration step, but it waits regardless. `EV.9` — the per-point ablation —
depends on `PT.9` or `PT.10`, because a negative from a feature that is not switched on in the measured
targets is not a negative about the feature.

---

## Shipped ledger

Revision 3 carried 182 step rows averaging 1,234 characters, which is what made it unmergeable and what
let four subsystems be over-reported. Shipped work is now one line per milestone. **The narrative of what
shipped lives in `git log`, in the ADRs and in `CHANGELOG.md`** — those are per-commit and per-file, so a
whole-file overwrite cannot silently revert them the way it reverted the MP rows.

Rows are capped at 200 characters. `RP.6` backfills the commit hash on every row; until then a row reads
`pending`. Legacy step ids inside a milestone remain valid citations in ADRs and docstrings.

| milestone | state | hash | note |
|---|---|---|---|
| M0 build, layering, encoding, goldens | done | pending | `test_layering.py` 359 cases, `test_temperature_encoding.py`, `GoldenStaleError` machinery |
| M1 radiometry: Planck, band integration, LUTs | done | pending | Four-quantity float32 LUTs, 200–1000 K at 0.05 K, kernel-identical lookup, band-hash sidecar |
| M2 Isaac gate spike | done | pending | ADR 0014 and four addenda; M2.4 corrected the position frame against an independent oracle |
| M3 optics | done | pending | Aperture factor defined once, AST guard, 181 cases. `SC.4` adds the missing aberration term |
| M4 detector | done | pending | NETD anchoring, NETD(373)/NETD(300) = 0.576. `SC.1` wires the photon-FPA electron budget |
| M5 ISP | done | pending | NUC, AGC, DDE, grayscale default. `SC.10` adds the temporal behaviour `display.py:286` requires |
| M6 thermal solvers and weather | done | pending | Two-node and cabin shipped but unreachable from a scene: `PT.15`. Heat traces likewise: `PT.16` |
| M7 materials | done | pending | 19 materials, Kirchhoff walk, Fresnel, n/k. **M7.9's stated Beer-Lambert derivation never ran**: `RP.7` |
| M8 atmosphere, grey and layered | done | pending | Layered exponential-sum slant path, R13-anchored. Per-pixel slant path was never in scope: `AT.1` |
| M9 sensor chain | done | pending | 3-D noise, FPN, bad pixels, NUC residual, budget test. M9.8's IIR wiring is recorded by ADR 0082 |
| M10 Isaac pipeline | partial | pending | AOVs, IrCamera, Warp stage twins, six demo stages. M10.12 and M10.13a/b/e blocked: see `DC.1` |
| M11 multi-band and aerial extras | done | pending | `72e8142` shipped the NIR config and response. Specular lobe not wired per pixel (ADR 0067) |
| M12 Tier 4 acceptance | partial | pending | The run fails and says so. `EV.1`-`EV.4` redo it before its attribution is used |
| ME evaluation data lane | partial | pending | ME.5 measured 365 clips on a hashed archive. ME.7 blocked on labels, not compute: `EV.11`, `XD.11` |
| MS sky targets (phase 1) | done | pending | Sky model, cloud clutter, point targets, aerial scene. Per-pixel slant path outstanding: `AT.1` |
| MM maritime (phase 1b) | done | pending | Analytic sea, Cox-Munk facets, sea skin `c6f98aa`, Tier 3 `ca5a663`, MM.7 vessel-departure film |
| MP point-wise temperature | partial | `9c3fd83` `ab7d445` `231d3c6` | MP.1-MP.4b shipped; the working-tree roadmap reverted them to open. MP.5 is now `PT.7` |
| IU engine interface | open | — | `IU-29` is an Unreal deliverable and is open question 2, not a step |

**Corrections this ledger makes to the working-tree roadmap**, all verified against `git log`: MP.3,
MP.4a and MP.4b are shipped and were reverted to un-started prose by a whole-file write; MP.5's row was
deleted entirely; the NIR Si-CMOS deferral at line 536 is stale because `72e8142` shipped it; M9.8's
"wire the IIR into `run_frame`" claim is corrected by ADR 0082; M6.14's cabin node, M6.16's heat traces
and M6.8's two-node solver are shipped and unreachable; and M11.6's electron-budget claim is true of the
module and false of the render path.

---

## RP — Repair (phase 0)

Three sessions share this working tree, and the clobbering this document diagnoses is **live right now**:
`git diff --numstat` shows `CHANGELOG.md` at 22 insertions and 135 deletions, `README.md` at 5 deletions
and `docs/roadmap.md` at 3 insertions and 5 deletions, and the deleted lines are the MP.3, MP.4a, MP.4b
and ADR 0087/0088/0089 entries. Capping row length does not stop a whole-file write. `RP.3` does.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| RP.1 | ✅ **done.** README's four broken status cells restored — `materials` had 1,473 chars of changelog prose where the state token belongs, `detector` 978, `noise` 976, `isp` 669. Prose moved to Notes. MP.5's limitation paragraph was clobbered a **third** time; `RP.3` is the structural fix, not this row. | **Measured.** `test_readme_status_table.py` was red on 4 of 15 rows, now green, with a negative control that reintroduces the defect and fails. Pins shape not content: Notes may say anything, State holds one of four tokens, Tier a tier. Zero CHANGELOG deletions. | — | S | 0 |
| RP.2 | Restructure `CHANGELOG.md`. One `[Unreleased]` section of 3,025 lines and 270 KB with 20+ repeated Added/Changed/Fixed headings is a single merge hotspot for three sessions. Split into dated sub-sections with one heading set each. | Parser asserts no heading repeats inside a section. Two commits already exist whose subject is restoring lost CHANGELOG entries (`68acd1c`, `303b56b`); the parser makes a third detectable before it is committed. | — | M | 0 |
| RP.3 | Make `scripts/stage_own_hunk.sh` the default path, not an available one: `.pre-commit-config.yaml`, a `make stage` target, and the instruction in the `ship-step` skill. It was written and tested (8 cases) for exactly this failure and nothing reaches it. | Hook refuses a commit staging a hunk in CHANGELOG.md, README.md or docs/roadmap.md the session did not author, driven by a synthetic two-author tree. Red today: no hook exists. | — | S | 0 |
| RP.4 | Write the two ADRs shipped code cites and does not have: the Level B angular-emissivity model (cited by `directional.py:21`, `angular.py:24` and four tests) and the sea-water n/k plus Cox–Munk slope model (cited by `sea.py:36`, `nk.py:23`, ADR 0078). Numbers allocated at write time. | Parser walks every `ADR NNNN` citation in `src/`, `tests/`, `docs/` and asserts the file exists. Red on four (0042, 0062, 0069, 0079); 0062 and 0069 are roadmap-only forward allocations and become subjects. | — | M | 0 |
| RP.5 | ADR status hygiene. No ADR is ever marked Superseded although the template offers it. ADR 0059 says bolometer smear is "never a second blur" while `optics/stage.py:77` applies one at duty 1.0 (ADR 0077). ADR 0006's use case was removed by 0014; 0048's range clause by 0071. | Parser asserts every ADR carries a `Status:` line in the template spelling (three spellings in use: 70 / 8 / 7 files) and flags the 0059/0077 pair, seeded as the self-test, until one records the supersession. | — | M | 0 |
| RP.6 | Backfill commit hashes into the shipped ledger from `git log`. Revision 3 prescribed `done YYYY-MM-DD <hash>` and **zero of its 162 done rows carried one**, M0.1 included — the row it cited as the example. | Parser fails any ledger row marked done with no 7-hex hash; red on every row today. A forward-only rule leaves state unrecoverable after the next clobber, which is the failure this revision exists to fix. | RP.1 | M | 0 |
| RP.7 | `docs/spec-issues.md`: one of sixty rows is Resolved and line 20 still says "everything else is open" while fourteen-plus issues have shipped ADRs. **Reopen S13:** glass `mwir: 0.02` was authored in `412f019`, six commits before `data/nk/glass.csv` existed (`1a0f13c`). | A test recomputes τ_mwir by Beer–Lambert over 5 mm of the checked-in k(λ) and asserts the YAML matches within 10 % or carries `derivation: authored` naming an open issue. Red today: the table gives 0.134, a factor of ~7. | — | M | 0 |
| RP.8 | CLAUDE.md factual corrections. `docs/maps/` is advertised as navigation aids; it is 729 KB, one commit ever, and its own README calls it frozen 2026-09-10 snapshots that are "not maintained". `spg/` is advertised as holding .cu/.cu.lua/.usda and holds one README. | Parser asserts each path CLAUDE.md's layout block names exists and its file-type claims hold. Red on two paths. Move the maps to `docs/history/` with a dated banner or drop the claim. Line 13 is open question 2 and is not touched. | — | S | 0 |
| RP.9 | The row-length and ledger parsers in `make check`. Step rows capped at 600 characters, ledger rows at 200. | Parser fails this document on an over-length row, self-tested on a synthetic one. Scope is stated in the parser: step tables and the ledger, not prose. This is a lint, not a physics verification, and is not counted as one. | — | S | 0 |

---

## PT — Point-wise surface temperature

The owner's headline requirement. What shipped (ADR 0087 `PlanarThermalField`, ADR 0088 spatial sources,
MP.3 `point_bridge`, ADR 0089 `VehicleSourceSolver`, MP.4b the car demo) is real, correct and end-to-end —
and reaches **two prims in one of eight scene configs**, on the lane ranked third, at night, authored from
hand-written Python. Six things stand between that and the requirement, and revision 3 carried none of
them: no per-cell solar or shadow, no patch in the scene schema, a world-frame-only bridge so nothing that
moves can carry a field, a nested-rectangle view factor summing to 1.400, an axis guard that is 4.9×
wrong on a rotated patch, and no binding on either priority lane.

This is also the single most externally supported item in the plan. The current state-of-the-art LWIR
drone-detection study ranks target-crop Sobel gradient variance as the **second** largest measured
sim-to-real gap driver (Cohen's d = 1.224), writing that simulated targets "act as uniform silhouettes
rather than noisy, physical heat sources"; DIRSIG's own manual names the bonnet-over-engine case verbatim
as its documented limitation. `EV.9` turns that from a conviction into a cited number.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| PT.1 | **Per-cell solar and shadow.** ADR 0087's own headline case — a wall half in sun spanning 10–20 K — is unimplemented: nothing varies `q_solar_w_m2` or shadow across a surface, which is why both point-wise scenes shipped are pre-dawn. The per-facet direct/diffuse split exists; nothing joins it to a patch. | A vertical concrete wall at 14:00 half occluded by a slab: every lit cell holds its own §6.1 `steady_state_temperature` root to 1 mK and the lit/shaded step is ≥ 8 K at the measured insolation. A per-surface `shaded` bool gives one temperature and fails by that step. | — | M | A |
| PT.2 | **A patch declarable in a scene config.** `SurfaceSpec` carries name, material, tilt, azimuth, shaded, vehicle_speed and nothing spatial; `surface_fields=` is passed at one call site, `render_car_ignition.py:197`. Schema v7 adds a `patch:` block: origin, axes, extents, cell size, thickness, prim binding. | The car demo's two hand-built patches reproduce from YAML with bit-identical cell centres and temperature after 1500 s. Non-orthonormal axes or zero thickness raise. Red today: "a scene config plus one command" is unmeetable for any new point-wise scene. | — | M | A |
| PT.3 | **Fix the nested-rectangle double count.** `build_ground_field` sums `occluded_longwave_flux` over underbody, engine-bay floor and exhaust, which are geometrically nested. Partition them, or clamp with a loud warning. | Σ view factors ≤ 1 + 1e-6 for every cell of both car scenes. Measured today: peak 1.400 with 28 cells above 1.0, against a hard bound of 1 for a plane element — inflating the clear-night road patch by up to ~40 %, which `test_a_clear_sky_puts_far_more_of_the_patch_on_the_road` currently reports as a physics result. | — | S | 0 |
| PT.4 | **Guard the in-plane axes in `patch_view_factors`.** It checks the normals are parallel, then compares an offset measured in the receiver's axes against extents given in the radiator's — so an in-plane rotation is evaluated as if the rectangle had rotated too. | A radiator rotated 90° in-plane against an unrotated patch raises. Measured: 0.01573 against 0.07662, a 4.9× error with no exception — the "plausible number, the worst kind of wrong" the module docstring refuses. Latent only because `car_demo` uses world EX/EZ; the first ship deck hits it. | — | S | 0 |
| PT.5 | **Point-wise on a moving prim.** `point_bridge.py:98-103` rejects any patch frame but `world`, while `surface_field.py:78-80` documents a prim path as supported — the docstring promises what the code refuses. Compose world-from-local from the prim's USD transform at the frame's timestamp. | A patch on a prim translated 10 m and yawed 90° between frames samples the same cell for the same material point to under one cell width; the world-frame path is bit-identical. Red today: every aerial and maritime target moves. | PT.2 | M | A |
| PT.6 | **Thermal properties from the material library.** `car_demo` authors bonnet C = 8000, asphalt C = 60 000, ε 0.92/0.95, α 0.88, bypassing `ThermalProperties.from_material` and violating ADR 0043's single-source rule. | `build_car_demo` with no overrides reproduces `from_material('asphalt_dry')` exactly. Measured disagreement today: C 60 000 against the library's 101 200 J/m²/K — 1.7× in the *time constant* — so the headline 6.21 K gradient is a result on a panel nobody can find in `configs/materials/`. An override raises unless the scene declares it. | PT.2 | S | C |
| PT.7 | **Spin the field up with the scene present** (the MP.5 limit). Fields start uniform: the bonnet at air temperature, the road from one broadcast scalar, so frame 0 is a road no car has ever stood on — the dominant feature of real night parking-lot imagery. | Frame 0 of the clear-night scene carries a road patch within 10 % of its own equilibrium; the overcast scene stays under 0.3 K; a field spun up with no radiators is bit-identical to today. `test_the_ground_patch_needs_time_because_the_field_starts_uniform` names this fix explicitly and must be inverted. | PT.3 | M | C |
| PT.8 | **Bound the tick history.** `ThermalField._ticks` is appended to and never pruned; only the two ticks bracketing a query are read. A two-tick ring plus an optional history hook keeps `state_hash` working. | A 24 h run of the 10 400-cell road patch holds resident memory under 16 MB. Measured today: 2 880 ticks × 10 400 cells × 8 B ≈ 240 MB for that patch alone, gigabytes at 10⁵ cells — which blocks the full-diurnal time-lapse ADR 0074 wants to film. `state_hash` and interpolation unchanged over 500 queries. | — | S | C |
| PT.9 | **Point-wise on the aerial lane.** An airframe skin field bound from the scene config, with per-cell solar and the ram-heating source already in ADR 0075. A fuselage is the most obviously curved thing in the project, so this is the first consumer of `WM`. | The aircraft-pass frame shows a leading-edge-to-shaded-underside gradient across one prim, against 0.000 K today, with each cell holding its own equilibrium to 1 mK in the engine-free oracle. The per-prim path is bit-identical when the binding is absent. | PT.1, PT.2, PT.5, WM.3 | M | A |
| PT.10 | **Point-wise on the maritime lane.** Deck and superstructure fields on the vessel scenes. The sea is already per-ray (ADR 0078/0080) by a different mechanism; the hull is not. | The vessel frame shows a sunlit-deck vs shadowed-superstructure step of ≥ 5 K across the hull prim, against 0.000 K today. Runs the same conservation test as WM.2: the area-weighted mean matches the per-prim value it replaces to within the forcing difference, so the change is provably a redistribution. | PT.9 | M | B |
| PT.11 | **Lateral conduction between cells.** Cells on one steel bonnet are independent columns. Derived diffusion lengths over §6.6's 750 s engine-bay rise: aluminium 270 mm, steel 99 mm, asphalt 16 mm, against 71–150 mm cells — metal panels render sharper than reality. Implicit or ADI: `tick_s: 60` is 6.6× past the explicit limit (≈ 9 s at 5 cm in aluminium). | Against the analytic Gaussian spreading of a step in a semi-infinite sheet to 1 %; the k → 0 limit is bit-identical; forward Euler at the shipped tick diverges. Per-material switch. | PT.3, PT.7 | L | C |
| PT.12 | **N-layer through-thickness stack per cell**, generalising `two_node.py`. MuSES evaluates properties per thermal node through the element thickness; Fraunhofer stores 2+N temperatures per triangle and used 10 layers. | N = 1 reproduces the existing two-node result to 1 mK. A 1 mm steel skin and 0.3 m of asphalt in one scene each show their own time constant, ordered by areal capacity; a single-layer asphalt gets the night curve wrong by > 2 K. | PT.11 | M | C |
| PT.13 | **Temperature-map and parameter-map ingest.** DIRSIG's Map Temperature Solver is a single-band raster in °C applied by UV or drape projection; MappedTherm does the same for parameters. `PlanarPatch` is already a raster with a projection. The escape hatch for prescribed aerial skins, externally solved hulls and draping public thermal frames onto geometry. | A float32 raster round-trips through a patch to 1 mK. A °C raster mis-declared as K raises. A parameter map varying α_sol gives the per-cell equilibrium the scalar solver predicts. | PT.2 | M | C |
| PT.14 | **ADR: temperature granularity tiers.** Production tools select a tier per surface — DIRSIG offers per-material, per-solid, per-facet and per-pixel, and imports MuSES for a real 3-D field. irsim has three tiers in code and only ADR 0087's prose describing the boundary; CLAUDE.md requires an ADR for a chosen fidelity level. | A record, not a test. It states where irsim sits, what each tier costs, and the rule a scene author uses to pick one. It supersedes ADR 0087's "a real limit, not a temporary one". | WM.5 | S | C |
| PT.15 | **Make `LumpedTwoNodeSolver` and `CabinNode` reachable.** Neither is importable from `irsim.thermal`'s `__init__`, neither is a `solver:` kind, no demo constructs either — so every solved surface in every scene has an adiabatic back. | ADR 0036/0038's measured results appear in a rendered frame: a roof +4.8 K with a cabin against adiabatic, both > 2 K below ambient on a clear night. Red today: no scene can construct either, so those are results no rendered frame can show. Note the 12.04 s gate leader exercises this unreachable solver. | PT.12 | M | C |
| PT.16 | **Make `HeatTraceLayer` reachable** (ADR 0039, M6.16). §6.6 calls heat traces a signature phenomenon of the band, and the sim-to-real literature says detectors trained on synthetic data lacking them are confused by them — so this is an evaluation deliverable, not a nicety. | A rendered frame shows the trace ghost at the authored offset and amplitude; the overlay is absent bit-identically when unbound. Red today: `irsim.thermal.traces` is imported only by its own unit test. | PT.15 | M | C |

---

## WM — Warp mesh surface parameterisation

**This lane is the one genuinely new unlock in revision 5, and it deserves its own error budget, its own
oracle and a superseding ADR.** ADR 0087 records "there is no UV AOV, no per-triangle id" and concludes
that curved geometry is "a real limit, not a temporary one". That conclusion is true only if the surface
parameterisation has to arrive through an AOV. It does not.

Warp 1.16.0 — already installed in this build, verified at
`warp/_src/types.py:7579,7613` — returns `MeshQueryPoint{result, sign, face, u, v}` from
`wp.mesh_query_point_no_sign` and `MeshQueryRay{..., face, t, u, v, normal}` from `wp.mesh_query_ray`.
Seeded by the float32 `Camera3dPositionSD` that `point_bridge.world_positions` already decodes, a
closest-point query against a bound prim's own mesh yields exact `(face, u, v)` per pixel, with no
renderer capability, no asset change and no new dependency. Measured on this machine: 327,680 queries
against a 516,960-triangle mesh in **0.63 ms** on an RTX 5090, ray queries 0.51 ms, LBVH build ~0.9 ms —
three orders of magnitude under the radiance kernels.

Conditions this lane must carry, stated up front: the **3.4 mm measured position residual** is the error
budget for the closest-point tolerance; `mesh_query_point*` takes no `root` argument in 1.16, so it is
**one `wp.Mesh` per bound prim**, which also removes the thin-panel closest-point ambiguity structurally
instead of by a tuned `thickness_m`; a brute-force NumPy closest point is the oracle the Warp path is
tested against, per ADR 0018/0061; and if PT.11's lateral conduction follows onto a mesh it needs an
intrinsic-Delaunay-safe Laplacian (WM.6), because a plain cotan Laplacian on an imported obtuse triangle
gives negative weights and a cell can leave the physical range.

Leaving ADR 0087 standing as written will cost another session a week, so WM.5 is not optional.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| WM.1 | **Probe: exact per-pixel (face, u, v) from Warp on this build.** Build a `wp.Mesh` from a prim's own points and indices, seed `mesh_query_point_no_sign` with the position AOV, recover face and barycentrics. A probe script and numbers, not a pipeline. | Recovered face and barycentrics reproduce the M10.1 probe scene's known geometry; the closest-point residual is inside the 3.4 mm position budget; `mesh_eval_position(face, u, v)` returns the queried point. Red today only in the sense that the measurement does not exist — this is the step that settles ADR 0087's premise. | — | M | B |
| WM.2 | **`TriangleMeshField` in `irsim.thermal`**, engine-free, composing `ThermalField` as `PlanarThermalField` does so the fixed tick and never-mutate-on-query rule carry over. Cells per face with a per-face resolution, Ptex style. `PlanarPatch` stays as the special case. | A sphere under a directional sun holds each face to its own cos θ equilibrium to 1 mK — a single facet fails by the pole-to-terminator span. Conservation: the area-weighted mean matches the per-prim value it replaces to within the forcing difference, so the change is provably a redistribution. | WM.1 | M | B |
| WM.3 | **`MeshPointBridge`**: per-pixel instance id selects the prim's `wp.Mesh`, the query gives `(face, u, v)`, the field gives the temperature. Additive like `point_bridge`: an unbound prim keeps the per-instance path bit-identically. | Brute-force NumPy closest point is the oracle; Warp and oracle agree on face and on the sampled temperature for 100 % of pixels on a curved fixture. A hit beyond the tolerance raises rather than snapping. A wheel, tyre and exhaust pipe stop being one temperature — the case ADR 0087 lists as Hard. | WM.2 | M | B |
| WM.4 | **Per-cell geometry from the mesh**: true per-face normals for the solar incidence term, ray-traced sky view factor and self-shadowing via `mesh_query_ray` (0.51 ms per 327 k rays measured). This is ADR 0088's own "revisit when". | The analytic parallel-rectangle form stays the oracle and the Monte Carlo estimator converges to it inside its stated standard error. A hull at 45° shows per-cell view factors varying across the surface where one shared patch normal gives one value. | WM.3 | M | B |
| WM.5 | **ADR: the surface temperature field lives on the mesh**, superseding ADR 0087's curved-geometry limitation. Records the closest-point parameterisation, why it needs nothing from the renderer, the error budget, and the options rejected with reasons. | A record. The rejected options must be named or they will be rediscovered: closest-point-method narrow bands need a grid finer than a panel's thickness; a UV atlas as the *solver* domain carries metric distortion, seam severing and conservative-rasterisation taxes; transient surfels cannot hold a 48 h spin-up memory. | WM.3 | S | B |
| WM.6 | **Intrinsic-Delaunay-safe Laplacian** if PT.11's lateral conduction moves onto a mesh. A cotan Laplacian gives negative edge weights whenever two opposite angles sum past π, breaking the discrete maximum principle. | On a deliberately obtuse imported mesh, no cell leaves the range spanned by its neighbours and the forcing; the plain cotan operator fails this and produces a bright speck that looks like a bad pixel. Backward Euler is prefactored once per asset, so the 1 s fixed tick survives. | PT.11, WM.3 | M | C |

---

## AT — Atmosphere, sky and materials

`AT.1` is the audit's one critical-priority radiometry gap and it sits on the lane ranked first.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| AT.1 | **Per-pixel slant path.** `pipeline/atmosphere.py:72` passes elevation `0.0` unconditionally and `warp_stages.py:936` mirrors it, so every *resolved* pixel gets surface-density extinction over its whole slant range — while the unresolved point-target path uses `target.elevation_rad`, as does the sky behind it. | Measured on `us_standard_clear` at 5 km: τ 0.5217 horizontal against 0.6517 at 45°, **25 %**; L_path 26.94 against 18.22 W/m²/sr, **48 %**. Contrast must not jump across the resolved/unresolved handoff. The elevation plane is float32 and its value is asserted. | — | L | A |
| AT.2 | **Thread the camera's R(λ) into `LayeredAtmosphere`.** `Scene.from_config` never passes the fourth `responses` parameter, so every band's spectral-class weighting uses a nominal top-hat and the model silently describes a different camera. | Measured on the shipped InSb MWIR response: the `h2o_wing` class weight goes 0.0238 → 0.1303, a **5.5×** change, and MWIR sky apparent temperature at 15° is off by **0.65 K — thirteen NETDs**. Test asserts the weights match the response and that no path builds the model without one. | — | S | A |
| AT.3 | **NIR spectral-class coverage.** `BAND_CLASSES['nir']` spans 0.70–1.05 µm; `data/spectra/responses/nir_si.csv` reaches 1.10 µm, so supplying the real response raises. SWIR and MWIR have the same edge shortfalls and pass only because the response is under the 1e-6 threshold there. | `class_weights('nir', nir_si)` returns rather than raising, and a test walks `configs/sensors/*.yaml` asserting every response is inside its band's class support. Latent only because of AT.2, so the two must land together. | AT.2 | S | A |
| AT.4 | **Band-scalability guard covers the atmosphere and materials.** The AST guard exempts `irsim.atmosphere` and `irsim.materials`; `BAND_CLASSES`, `WEIGHT_T_REF_K` and `ATMOSPHERE_BAND_KEYS` hard-code band names and micron edges, so a fifth band is a code change in the atmosphere. | A fifth band added to a sensor YAML classifies, tabulates and renders with no edit under `src/`. If the carve-out is kept, it is recorded with its reason and the exemption list becomes a tested constant rather than an omission. | — | M | A |
| AT.5 | **Guard the grey `Atmosphere`.** `Scene.from_config` builds both models unconditionally; `scene.atmosphere` is the primary attribute while every render script passes `scene.layered`, and `pipeline/atmosphere.py:87` branches between them. A caller taking the primary attribute gets a materially different τ and L_path from the sky model standing beside it. | A `Scene` carrying a sky model refuses to hand out the grey one, or marks it L1-only in a way a test asserts. Not a feature today; a divergence waiting to be stepped on. | — | S | A |
| AT.6 | **Spec issue S40: the Level B (a, p) table.** Twelve of sixteen Level B materials carry ESTIMATED (a, p); the four fitted use `paint_proxy.csv`, whose header says "PROXY: PMMA, not paint", giving a = 0.75 against §4.2's 0.15–0.35. Oblique surfaces are most of a maritime or urban frame. | Either a measured pigmented-paint n/k table in the fit, or the four painted materials' `a` moved inside §4.2's range against a public angular measurement. S40 closes, or is restated with what remains unexplained. | RP.7 | M | C |
| AT.7 | **Surface the extrapolated fraction at scene level.** `total_hemispherical_emissivity` fills everything outside the four nominal band ranges by extending the nearest band. Measured `extrapolated_fraction = 0.607` for every material at 300 K — 61 % of the weight that sets every surface temperature is an assumption, reported on the object and invisible to a scene author. | A scene build reports the worst fraction across its materials, and any Tier 4 report quoting an absolute apparent temperature carries it. | — | M | C |
| AT.8 | **Angle-dependent τ and the second-hit ray.** `directional_properties_for` moves ε(θ) and re-derives ρ but holds τ at its normal-incidence value, and `surface_radiance` defaults `L_behind = L_env`. The limb of every windscreen and shop window keeps full normal-incidence transmittance in NIR/SWIR. | The limb of a glass panel at 75° shows the transmittance falling toward total reflection, against a flat value today, with Fresnel as the oracle. Needs a second-hit AOV from the Isaac side before the behind-radiance half can improve; the angular half does not. | IG.4 | M | C |
| AT.9 | **Urban aerosol preset, and an honest bound on the seven that exist.** `AerosolRegime` already admits `urban` and nothing uses it; all seven presets declare `status: ESTIMATED` from §7.2 midpoints calibrated to one dry Tucson anchor, with `valid_range_m: 500`. | An urban preset exists and a test asserts its per-band extinction ratio differs from rural by more than the presets' own stated uncertainty. Any range claim beyond 500 m in a report names the calibration anchor. | — | M | C |

---

## SE — Sea and maritime

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| SE.1 | **Record the sea model's angular validity envelope.** Published in-situ radiometry validates Masuda only to ~50° from nadir; beyond that multiple reflections need Wu–Smith, and emissivity falls 2–3 % by 55°. A shore-based maritime camera views most of the sea **past** 50°, outside the validated envelope. | The model flags an angle outside the envelope and the maritime Tier 3 report prints the frame fraction beyond it. The isothermal identity is preserved but is not the test — it holds for a wrong angular emissivity too. | — | M | B |
| SE.2 | **Maritime and illumination in-sim tests.** The maritime stage feeds water prims into `background_prim_paths` so their pixels join the sky mask and take the analytic sea profile; `illumination_isaac` fixed "every Isaac render was emission only". Both are verified engine-free only. | In-sim: the intended water prims are masked and no others; the horizon lands where the Earth-curved mesh puts it to a stated pixel count; sea apparent temperature varies monotonically with depression angle. Red today: no such file in `tests/integration/`. | — | M | B |
| SE.3 | **Sea-surface temperature against ECOSTRESS SST.** The skin model shipped at `c6f98aa` has no external check. ECOSTRESS L2 carries an SST layer valid over all water. | Bias and RMSE against ECOSTRESS SST for a matched place, time and weather record, reported beside ECOSTRESS's own validation accuracy (bias −1.6 K, RMSE 3.1 K against SURFRAD) so the bar is the instrument's, not an aspiration. | XD.9 | M | B |

---

## SC — Sensor chain and published-reference anchoring

The NETD anchor is the strongest part of the repo and the rest of the chain is shipped and tested. What
is missing is **wiring and anchoring**: one path is not reached at all, one term is identically zero in
every config, and three configured values disagree with the project's own measurement or with FLIR's
published acceptance limits. None of this needs a camera — a Boson Engineering Datasheet is free.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| SC.1 | **Wire the electron budget for photon FPAs.** `pipeline/core.py:176` calls `anchor_noise` unconditionally for both FPA kinds with no `dark_electrons=`, and `electron_budget`/`dark_electrons_for` have no caller in `src/`. M11.6's row claims the opposite. | Measured: the MWIR InSb renders at σ 533.3 e⁻ against a 350 e⁻ datasheet read noise, **1.52×**, and every photon camera renders with dark = 0, so the SWIR InGaAs is missing its own 200 e⁻. After: `from_sensor` selects the electron budget for a photon FPA and NETD becomes the cross-check. | — | M | A |
| SC.2 | **Mutation test on the electron path**, and substitution of the measured 3-D ratios. Perturbing `read_noise_e` must move the rendered σ. It cannot today, for a reason that is *wiring*, not noise-space correctness — which is why this sits behind SC.1 rather than beside it. | Perturbing `read_noise_e` by 10 % moves rendered σ by the predicted amount for every photon config. `halmstad_boson_320.yaml` takes ME.5's measured ratios (vh 2.64, v 0.58, h 0.16, t 0.38) against the configured 0.30/0.08/0.15/0.02 — **7–19× out**. | SC.1 | S | A |
| SC.3 | **Boson datasheet corrections.** FLIR publishes no scalar NETD; it publishes 3-D components with conditions. Fix `ffc_interval_s` 180 → 300 (the factory FFC Period, with its 1.0 °C trigger and 90 s start-up at one-third delta), `thermal_time_constant_ms` 10.0 → 8 ms published, and add the missing ESTIMATED marker to both Boson `ratios_3d` blocks. | A bench reproduces FLIR's acceptance conditions (f/1.0 lensless, 20 °C camera, 30 °C scene, averager off) and asserts tvh/th/tv against the grade limits, which give th/tvh = tv/tvh = **0.35** against the configured 0.05. | — | M | A |
| SC.4 | **The optical PSF's second factor.** `aberration_sigma_um` defaults to 0.0 and no sensor YAML carries an `mtf:` block, so every rendered camera is diffraction × detector-box only, while the schema docstring calls the value "the Gaussian fitted from a measured slant edge". | FLIR publishes 42 % nominal on-axis MTF at Nyquist for the configured 14 mm f/1.0 lens. Cascaded with the ideal 12 µm detector box (sinc at Nyquist = 0.637) that predicts **0.27 ± 0.03**; the slant-edge estimator must land there. Today one factor is identically zero. | — | M | A |
| SC.5 | **State the de-trending convention on every noise statistic.** Published measurement moves an uncooled imager's temporal noise by up to **3×** with the filter alone (1.00 unfiltered, 0.68 poly2, 0.34 Gaussian σ=8), and the effect differs between imagers. Larger than the codec floor already guarded. | `decompose_3d` and `temporal_shape` take an explicit `detrend`; every reported σ carries it; the Tier 4 report prints two conventions side by side. Red today: no rendered-versus-real noise bound is reproducible by a third party. | — | M | X |
| SC.6 | **ADR: which NETD irsim means.** NETD is N_im/SiTF and there is no agreed N_im — five incompatible definitions are in current use, and NVESD's own recommendation changed in 1992, 2005 and 2023. | A record plus a docstring change wherever the project writes "NETD". A measured NETD quoted without its definition and filter is not comparable to anything, so this is a prerequisite for SC.3 and SC.9 meaning what they say. | SC.5 | S | X |
| SC.7 | **Adopt the standard bench conditions** so irsim's Tier 2 numbers are comparable rather than project-local: SITF as a differential sweep −10…+20 °C in 5 °C steps with a linear fit over −5…+15; 3-D noise from 128 frames at 25 °C; MTF by ISO 12233 at 0.5 cycles/pixel from 128 averaged frames. Report N_temp and N_spat. | The bench reproduces its own previous numbers under the new conditions within the estimator's stated sampling floor, and the two summary quantities every external source quotes are printed. Red today: the frame counts and sweeps are project-chosen. | SC.6 | M | X |
| SC.8 | **Retire "for when a camera arrives".** All four Tier 2 measured comparisons skip forever against directories never created, keyed to `flir_boson_640_lwir` while the only public measurement belongs to `halmstad_boson_320`. Datasheet limits go on one camera YAML, field-measured ratios on the other, with a schema guard that the two never mix in one `ratios_3d` block. | The four benches run instead of skipping; the framing in `validation/measured.py` and the Tier 2 protocol doc, which contradicts ADR 0003, is deleted. | SC.3 | M | X |
| SC.9 | **NV-IPM Measured System Component export**: four 3-D components in Kelvin, pre-sample MTF arrays, normalised response, pitch, FOV, frame rate. | A third party range-checks irsim's Tier 2 benches independently in a free model with no camera. The export round-trips through NV-IPM's own reader. This is the cheapest external credibility the project can buy under the no-camera constraint. | SC.7 | M | X |
| SC.10 | **AGC temporal behaviour.** Every AGC operator is a pure per-frame function with no memory, and `display.py:286` says any temporal behaviour must become an explicit `PipelineState` field. Real cores damp frame to frame, so a target entering frame gives a smooth transient in reality and a one-frame step in the renders. | `lag1_autocorrelation` of the rendered display stream lands inside the real set's measured band, where it does not today. The ablation ranks this switch first (AUC 1.000, EMD 61.7 codes). | — | M | X |
| SC.11 | **Thermal polarity as a config switch and an evaluation axis.** MaCVi 2026 had to normalise inverted thermal scaling across real maritime clips. White-hot stays the default (the owner's preference, and all five configs carry `palette: gray`). | A polarity-flipped condition appears in the detector evaluation and in the ablation switch set. Red today: a detector trained only on white-hot fails on half the real world and nothing measures that. | — | S | X |
| SC.12 | **Record the size-of-source effect** (~0.8–1.0 K for uncooled microbolometers per VDI/VDE 5585, against ~0.1–0.2 K for cooled MCT) as a known omission in `docs/spec-issues.md`. | A spec-issue row with the number. It is larger than several effects the chain does model, so leaving it unrecorded misstates the chain's own error budget. | RP.7 | S | X |
| SC.13 | **Close the ISP temporal-filter question.** `sensor_chain.py:241` says the filter "stays an identity until ME.5's temporal PSD on flat sky shows whether real cores low-pass their output at all". ME.5 landed and measured a one-pole time constant of 2563 frames at a drift fraction of 0.9276, above the 0.5 the report itself calls untrustworthy. | The answer — "not measurable on this set" — is recorded in the ADR and the dangling conditional is removed, so the next session does not re-open a question that was answered. | — | S | X |
| SC.14 | **Carry M9.9's two open criteria rather than dropping them.** The Tier 3 sensor-chain bench landed amber with an aerial edge-asymmetry band (legacy ME.4) and the real-vs-synthetic comparison (legacy ME.6) open; both need statistics measured from public real imagery. This row keeps them tracked and is what `test_tier3_sensor_chain.py` asserts against. | The test reads this row and fails if it stops naming both, or if it is marked done while `EV.6` and `XD` have not supplied the measured bands. An untracked criterion is one nobody will close. | EV.6, XD.1 | S | X |

---

## EV — Evaluation methodology and sim-to-real

The Tier 4 acceptance run at `docs/validation/tier4-2026-09-15` fails and says so, which is right. But
**two of its three failing statistics and its entire attribution table rest on an artefact**, so its
conclusion — that the dominant term is the signal path — is currently not established by the experiment
that reports it. `EV.1`–`EV.4` redo the run before anything is built on top of it.

`EV.9` is the lane's terminus and it is named here, not gestured at: it measures target-crop Sobel
gradient variance and GLCM entropy on irsim renders with the point-wise field **on** versus a per-object
constant, against real target crops. That is the experiment that converts the owner's headline
requirement into an external, cited number, and it can return a negative.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| EV.1 | **Stop pooling the mosaic across clips.** `validation_report.py:100-112` flattens every clip's frames into one list and `_mosaic` medians the whole stack, so the histogram EMD (24.69) and PSD shape ratio (5.70) are computed on a composite image neither set contains. | Measured: per-clip synthetic mosaic means are 63.2 / 58.3 / 67.1 / 93.3 / 113.1 / 102.0 against a pooled 77.5 — the per-clip mean spans **55 codes** against an 8-code EMD target. After: each check carries a per-clip distribution and a stated aggregation. | — | M | X |
| EV.2 | **Gate the real side.** `sorted(...)[:limit]` takes the first six clips alphabetically with none of the three gates the project built — `classify_clip`, flat-region, codec floor. | Measured on the same archive: 81 of 365 clips are moving (every per-pixel temporal statistic invalid on them), 306 of 365 have a robust noise scale at or below one code, and only 28 support a noise table at all. That is exactly the condition under which `noise_scale` separates the sets for reasons unrelated to the simulator. | — | M | X |
| EV.3 | **Make the synthetic side admissible.** `find_flat_regions` returns `[]` for every rendered clip, so the synthetic side cannot be measured by the route the real side was measured by, in either direction. | Measured on `matched_000`: the best window has structure_ratio 1.50 against a 0.25 limit; a raw 64×64 sky window reads σ_TVH **0.329 codes** against the real set's 7.272 — about **20× too quiet**, the opposite sign to the report's headline. The gap is then stated as a number rather than as a refusal. | EV.2 | M | X |
| EV.4 | **Permutation null for the discriminator.** The analytic `null_sigma` assumes independent patches; patches are drawn four per frame from 60 frames of 6 clips. | The report says the sets are separated "38 null standard errors from chance". At the clip level the null σ is ~0.17 and the run sits ~2.4 σ out — a **~16× overstatement**, with the caveat living in the code and not in the report. A permutation null (repeated refits on shuffled labels) absorbs the dependence and needs no new dependency. | — | M | X |
| EV.5 | **Content-matched patch sampling.** Patches are drawn uniformly at random anywhere in the frame; the real clips are tripod shots containing ground, trees, buildings and horizon, the synthetic ones are sky-only renders. | `gradient_median` and `skew` are the 2nd and 3rd heaviest discriminator features — exactly the statistics content mismatch moves. After: patches are drawn from matched content classes and the report separates the physics question from the scene-composition question. | EV.1 | M | X |
| EV.6 | **Add the statistics a published LWIR study found actually separate the sets**: Histogram Total Variation (d = 1.242, the AGC quantisation comb), target-crop Sobel gradient variance (d = 1.224), target-crop GLCM entropy (d = 0.974), dynamic-range-used (d = 0.899). Plus rank-order correlation of object brightness. | Three of the four strongest published discriminators are absent from irsim's feature set and all four are pure NumPy. Rank-order correlation is DIRSIG's oldest comparison, AGC-invariant and codec-robust — the one Tier 4 metric that works on 8-bit sets. | — | M | X |
| EV.7 | **Score target crops separately, and put a real-vs-real control beside every gap number.** Published controls: FID 41.40 synthetic against 38.74 between two *real* sets; KDD 332 against 262; histogram EMD 29.9 against 13.6–18.8. | Full-image and crop-level gaps told opposite stories in the one study that measured both, and the crop level is where the per-point requirement lives. Without the control, ADR 0068/0085's thresholds stay guessed; with it they are measured. `datasets.yaml` already indexes enough sets to compute it. | EV.5 | M | X |
| EV.8 | **Make the Tier 4 artefact reproducible from itself.** `tier4-2026-09-15.json` records only label, checks, passed and attribution — no clip list, archive SHA-256, `config_hash`, CRF, seed, patch size or count, all of which exist elsewhere and all of which change the numbers. | Re-running from the artefact alone reproduces every number bit-for-bit. ME.5's own report already meets this standard (it records the archive hash and states every number was measured on those bytes); the headline Tier 4 artefact does not. | EV.1 | S | X |
| EV.9 | **The per-point ablation.** Target-crop Sobel gradient variance and GLCM entropy on irsim renders with the point-wise field on versus a per-object constant, against real target crops. Pre-committed: a negative redirects the effort to the ISP. | The published gap driver is d = 1.224, with the note that simulated targets "act as uniform silhouettes rather than noisy, physical heat sources". It must run where the feature is switched on, hence the dependency — a negative from a feature not active in the measured targets is not a negative about the feature. | PT.9, EV.7 | M | X |
| EV.10 | **Viewpoint-distribution alignment.** Sample camera elevation, slant range and target aspect from distributions matched to the intended validation set. | The largest single published ablation in this literature: drone mAP@0.5 **0.464** with fixed pitch, **0.981** with random pitch, **0.995** with pitch drawn from the real set's metadata. It is a config change, not physics, and it is currently unmeasured in irsim's aerial and maritime scene configs — the cheapest leverage available on the sim-to-real deliverable. | — | S | X |
| EV.11 | **Sweep the mixed_to_real ratio** instead of testing one mix. Published: synthetic pre-training plus **100 real images** beat real-only in every configuration measured, and the KAIST optimum is only 10–20 % synthetic. | This reframes the ME.7 blocker: the project needs ~100–200 Python-readable boxes, not the full Halmstad MATLAB MCOS export. The sweep reports the curve, not a point, and the exit criterion is beating the published synthetic-only band of 0.41–0.54 mAP@50:95. | XD.11 | M | X |
| EV.12 | **Training-free dataset-quality proxy.** Evaluate SDQM (public code, Pearson r = 0.87 with YOLO11 mAP50) as a stand-in for the torch-blocked half of ME.7. | A defensible sim-to-real number inside the existing no-GPU gate, making the torch install a confirmation rather than a prerequisite. A clear negative result is an acceptable outcome and must be recorded as one. | — | M | X |
| EV.13 | **Publish the paired RAW-16 / AGC-8 artefact** (open question 7 gates the publication, not the format). No public thermal set offers the pairing and the literature names the 16→8 mapping as the dominant sim-to-real factor. | The float planes invert to apparent temperature within the project's existing 10 mK encode/decode budget while the 8-bit stream fails the same bound — the fp16 negative control that already discriminates. Note honestly that this is a self-consistency check on irsim's own encode/decode, not an external radiometric check; XD.5 is the external one. | IG.13 | M | X |

---

## XD — External data anchors

Every set in `data/validation/datasets.yaml` today is 8-bit, post-recorder and mostly lossy-coded, which
is why ADR 0068 declares most of Tier 4 untestable. That is an **indexing** problem, not a fact about the
world: radiometric public data exists, and two of the strongest anchors are not imagery at all.

Nothing here proposes buying a camera, and `XD.12` is the one action with the highest value per unit of
effort in the whole plan.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| XD.1 | **Correct four `datasets.yaml` fields.** Anti-UAV410 is 640×512 at 25 Hz (indexed `null`); LRDDv3's paper states CC BY 4.0 and names an Autel EVO II Dual 640T V3 (indexed `unstated` / `undocumented`, export-control access note kept); Anti-UAV600, 723 k frames and the largest IR anti-UAV set, is missing entirely. | These are gates in `fetch_validation_data.py` and analyser preconditions, so a wrong field is a silently wrong refusal or permission. A 25 Hz clip fitted at 30 Hz returns a time constant wrong by a factor and looks plausible. | — | S | X |
| XD.2 | **A `radiometric` signal path and a `bit_depth` field.** `irsim_eval.data.Sequence` refuses anything but 8-bit — right when every indexed set was 8-bit, and now the thing standing between the project and its own fix. Each analyser declares which path it needs. | A display-output set still cannot reach a radiometric-only analyser, and a radiometric set reaches the noise analysers that currently skip. This is the enabling refactor for XD.3–XD.5 and XD.10 and lands before them. | XD.1 | M | X |
| XD.3 | **MassMIND** — 16-bit LWIR maritime, FLIR ADK, published NETD < 50 mK, 640×512, CC BY-NC-SA 4.0, 2,916 Boston Harbor images with 7-class sky/water/obstacle masks. | Brings the maritime lane to the bar the aerial lane reached, and the masks give **labelled flat windows**, replacing the heuristic finder that EV.3 shows returns `[]` on every rendered clip. Index honestly: the 16-bit values are ADK counts, linear in radiance, not calibrated temperature. | XD.2 | M | B |
| XD.4 | **LTIR v1.0** — the only 16-bit public source found that is made of *sequences* (20, 8-/16-bit variant), so the only one that can carry temporal PSD, FFC and fixed-pattern-growth work without a codec floor. | The temporal analysers run on real 16-bit sequences instead of skipping. Terms of use are not stated on the page, so it indexes as `licence: unstated` and the per-sequence sensor table is read by hand before any absolute claim. | XD.2 | M | X |
| XD.5 | **FLAME 3** — per-pixel Celsius from a calibrated radiometric response, the only absolutely calibrated public imagery found, open access on IEEE DataPort. | The external oracle for apparent-temperature inversion and the encode/decode round trip, in Kelvin rather than DN8 — independent of any AGC. Frame-level labels only and saturation near 500 °C, so it is a radiometry check and not a detector set. | XD.2 | M | X |
| XD.6 | **ARM Infrared Cloud Imager** — radiometrically calibrated full-sky downwelling LWIR, 7.3–14 µm, 320×240 uncooled microbolometer, in W/(m²·sr) to better than 0.5, from a 9-month ARM SGP deployment with netCDF in a free archive. | **The only external check that exists** for ADR 0070 (cloud clutter), ADR 0086 (scattered-sunlight sky) and ADR 0071 (layered slant path) — the physics of the owner's first lane, validated against nothing at all today. Calibrated sky imagery, in the units the sky model predicts, with clouds. | AT.1 | L | X |
| XD.7 | **SURFRAD / BSRN downwelling longwave** — 1-minute pyrgeometer flux with air temperature, RH, wind and solar measured at the same station and minute, US public domain. | A physical cross-check on the weather-to-atmosphere coupling and on the sky total in W/m². Its limit is stated up front: a pyrgeometer constrains the hemispheric broadband total, not radiance versus elevation, and good clear-sky parameterisations sit at ~23 W/m² RMSE against BSRN — so it cannot detect a second `WeatherSeries` holding identical values and does not replace the identity assertion. | AT.1 | M | X |
| XD.8 | **ECOSTRESS / ASTER spectral library** — over 3,000 measured spectra covering asphalt, concrete, soils, vegetation, water, snow and metals, converted to emissivity by Kirchhoff. All 19 irsim materials are `source: literature` and none is anchored to a measurement. | A per-material band-integrated check with a **stated tolerance** — not 1e-6, since the existing closure is exact by construction and this is a different quantity with its own uncertainty — a stated integration convention, and a record of which spectrum each YAML was checked against. | AT.7 | L | X |
| XD.9 | **ECOSTRESS LST / SST** as an absolute surface-temperature check in Kelvin over a known place and time, matchable to the weather that drives the solver. | Sets a defensible bar rather than an aspiration: a 70 m spaceborne product with a full atmospheric correction validates at bias −1.6 K and RMSE 3.1 K against SURFRAD. irsim's solver is assessed on that scale, not on a 0.1 mK convergence tolerance that measures arithmetic. | XD.7 | M | X |
| XD.10 | **FLIR ADAS v2 pre-AGC frames** — 16-bit-in-14 TIFFs in `analyticsData`, Tau 2 640×512, 13 mm f/1.0, T-linear at 0.04 K per count, so the quantisation floor is **11.5 mK**, under a 50 mK NETD. Fires ADR 0068's own "revisit when". | 3-D noise on a flat region in Kelvin, comparable to FLIR's published limits with no SITF inference and no codec floor. Recorded constraints: it is the **ground** lane; it has no in-scene truth so it cannot support a per-material bias claim; Terms of Use are form-gated. | XD.2 | L | C |
| XD.11 | **Python-readable label sources** for the detector half: BIRDSAI (real **and** AirSim-synthetic aerial TIR, a ready-made sim-to-real testbed with a published baseline to beat), HIT-UAV and MONET (CC BY 4.0, readable boxes), RGBT-Tiny (115 sequences spanning sky and sea, >81 % of targets under 16×16 px). | The recorded ME.7 blocker is that Halmstad's boxes are MATLAB MCOS objects. That is a fact about Halmstad, not about the world, and EV.11 needs only ~100–200 boxes. Each set gets ADR 0068's provenance fields before download. | XD.1 | M | X |
| XD.12 | **Ask the Halmstad authors for the Y16 originals.** Their Data-in-Brief paper states the Boson was run in raw Y16 16-bit mode and that "the raw format is used in the database", then that "all videos are in mp4 format" — the 16 bits existed at capture and the encode destroyed them. | If they survive, the **primary** reference set becomes radiometric, on the **aerial** lane, with the **exact** Boson core the sensor configs model, under a CC licence. Time-boxed (open question 1); the plan does not depend on the answer. | — | S | X |

---

## IG — Isaac glue integrity

The layer that produces the owner's images is the least verified layer: `src/irsim_isaac` is at 43.5 %
coverage with six files at 0 % totalling 1,101 statements, and `tests/integration` runs in no automated
job. Several of these rows are not new features but *documented invariants that are not true*.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| IG.1 | **Split `strict_materials` into three flags** — material resolution, thermal-node coverage, patch coverage — and turn the patch-coverage guard back on by default. All six render scripts pass `strict_materials=False`. | `point_bridge`'s docstring says a pixel on a bound prim outside every patch must raise, because the alternative is "a seam that looks like physics". That seam is the shipped default in every frame the project has produced. After: a deliberate gap raises rather than falling back. | — | S | 0 |
| IG.2 | **In-sim test for the point bridge and the car demo.** `test_point_bridge.py:154-173` builds `camera_space` by inverting the formula it then checks, so the only test is a tautology; the production arithmetic is duplicated in `geometry_probe.py:255` and only the duplicate runs in-sim. | The same transpose already produced a silently wrong frame once — ADR 0014's M10.19 addendum, horizon 164 rows out. After: an in-sim test reads a float32 plane out of a real render and inverts it against an independent oracle. | IG.1 | M | A |
| IG.3 | **Enumerate the build's annotator registry** instead of hand-written candidate lists. `SURVEY_CANDIDATES` names 3 motion and 3 occlusion strings; the build registers `Motion2dXYZ`, `Motion2dXYZDilated`, `MotionWorld`, `MotionVectors`, `HighResMotion2d`, `OcclusionSD`, `SemanticOcclusionSD`, `Roughness`, `SpecularAlbedo`. | Every ADR 0014 negative currently means "none of the names we thought of", and three design decisions rest on those negatives. The registry is enumerable in Python. | — | S | X |
| IG.4 | **Probe for UV, texcoord and primitive-id channels.** README and ADR 0087 state their absence as *measured*; no probe has ever requested one. Two untested leads: Isaac 6.0's RTX-sensor `objId` upper index carries a per-primitive index for procedural geometry, and GeomSubsets register distinct `StableIdMap` entries. | Either the claim becomes evidence or the gate opens. GeomSubsets would give sub-prim thermal zones — a wheel, its arch, its tyre — with no new machinery. `WM` makes this non-blocking. | — | S | X |
| IG.5 | **The bogus `motion_px` plane.** `gbuffer_isaac.py` documents `motion_px` as omitted; it is not. `motion_vectors` sits at a ~6e-5 floor and `_reject_reason` applies its all-zero check only to *required* channels, so the plane reaches the G-buffer and the optics stage runs the smear path on it. | A no-op numerically; the hazard is the convention. `geometry_planes` defaults to `motion_convention="pixels"` and Replicator's docs give both signs opposite to the contract. After: the channel is rejected or its sign asserted against a known displacement. | — | S | 0 |
| IG.6 | **Wire `MotionTracker` into `IrCamera`.** M10.1b built it and verified it in-sim to 0.1 px, and its only caller is the test — so `capture()` never sets `aovs.motion` and ADR 0077's smear has never run on a rendered frame, with M9.8 and M10.1b both ticked. | A crossing target at the aircraft stage's 34 °/s — eleven Boson pixels per frame period — renders smeared, and a cooled MWIR config in the same scene does not. Red today: every frame is sharp regardless of scene velocity. | IG.5 | S | A |
| IG.7 | **Re-probe the three ADR 0014 negatives that look like capture-protocol artefacts.** (a) `motion_vectors` with `rt_subframes=0` and the timeline running — IsaacSim #722 says pausing zeroes the motion g-buffer, exactly the 6e-5 signature. (b) `/rtx/rendermode` with NVIDIA's casing `RayTracedLighting`. (c) The Replicator `occlusion` annotator. | One line each in an existing probe. If (b) flips, several negative AOV results were taken under whichever mode that token selected. If (c) delivers, it replaces the unoccluded geometric `V_s`. | IG.3 | S | X |
| IG.8 | **Remove the normals fp16 carve-out.** `gbuffer_isaac.py:156-158` allows float16 through a physics plane on the claim that "the renderer delivers the normals AOV as float16 whether we like it or not (measured; ADR 0014 addendum)". | ADR 0014's own addendum table records `normals` as **float32 ×4 at full resolution**, and the build's registry agrees (`AnnotatorParams("NormalSD", np.float32, 4, ...)`). A wrong claim is the stated justification for letting fp16 through a plane non-negotiable #2 protects. After: the dtype guard admits no fp16 anywhere. | — | S | 0 |
| IG.9 | **Check the companion RGB's resolution.** `rgb` is omitted from `required` so it is exempt from the shape guard, and `_native_rgb` box-filters by the supersample factor without comparing against the render product. | The pair's whole claim is that it is "registered by construction rather than by calibration", and that pair is the sim-to-real training artefact. Several colour AOVs on this build return at half resolution regardless of AA. After: a mismatched shape raises rather than misaddressing every pixel. | — | S | X |
| IG.10 | **Mark UNMAPPED pixels in the float outputs** and write `id_coverage` to a sidecar. They are ORed into `sky_mask`, so they skip the atmosphere *as well as* taking ε = 1 against their own temperature; the magenta overlay is display-only; the per-frame coverage number computed in `material_ids.py` is never recorded. | With `strict_materials=False` everywhere this is the live path, not a debug one, and a detector would train on pixels nobody computed. After: `radiance` and `apparent_t` carry a mask plane and every frame's coverage is in the sidecar. | IG.1 | S | X |
| IG.11 | **`heading_deg` and `refresh_pose()`.** `IrCamera.heading_deg` is stored and never read — the live copy is on `SceneIllumination` — so a caller passing it to the camera alone gets the sun in the wrong compass direction silently. `refresh_pose()` is manual and one production caller remembers it. | Two silent-pose footguns in the object that already produced one silently wrong frame. After: the dead parameter goes, and `planes()` compares the cached pose against the prim and raises rather than rendering geometry from the new aim and sky from the old. | — | S | X |
| IG.12 | **A cheap per-frame invariant on every annotator plane** — finite, right shape, not constant — generalising `_reject_reason` beyond required channels. Replicator annotators are documented to return empty arrays on random frames under multi-GPU (IsaacSim #507), and the known-issues page lists AOV texture-dimension mismatches that cause missing exports. | The rule `AovReader` already has at probe time applies on every frame. A silently empty frame in a 10 k-frame dataset export is unrecoverable after the fact, and the microseconds are free next to the radiance kernels. | IG.9 | S | X |
| IG.13 | **Float32 planes and a config-hash sidecar from every render driver.** `write_frame` appears only in `render_aerial_demo`, `render_maritime_demo` and `render_car_ignition`; `quad_flight`, `aircraft_pass` and `vessel_departure` are PNG-only — the aerial and maritime flight renders. `render_multiband.SCENES` covers 3 of 8 scenes. | ADR 0068 says 8-bit lossy frames cannot support a radiometric claim, so the owner's exit bar is amended to require float32 planes and a sidecar. After: all six drivers write them, all eight scenes covered. | — | M | A |
| IG.14 | **One lane driver behind the six render scripts.** Measured: 281 identical lines between `render_quad_flight` and `render_aircraft_pass`, 73 % line similarity. They have already drifted — only three call `write_frame`, only one exposes `flat_field_enabled`. | Adding a lane stops meaning copying 500 lines, and a fix like IG.13 stops meaning fixing it three times. Each existing driver's output is bit-identical before and after the refactor, which is the test. | IG.13 | M | X |
| IG.15 | **Live IR in the Isaac viewport.** The owner's standing requirement. `display_render_var` is RGBA-unorm-only, so this means publishing an 8-bit grayscale AOV (or deliberately overwriting `LdrColor`, which is the documented way to reach existing consumers). | The white-hot display stream appears in the viewport during a render, matching the written PNG to within the AGC's own quantisation. Note the hazard the SPG docs state: AOV name collisions are **silent** and the built-in shadows yours, so the `Ir*` prefix is load-bearing. | DC.1 | M | X |

---

## GT — Gates, tests and tooling

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| GT.1 | **Implement the `slow` marker R11 promised and the Makefile never grew**, and restore the fast gate. Measured: 2,756 tests in ~130 s against CLAUDE.md's stated 30 s, a 4.3× overrun; slowest single test 12.04 s, exercising `LumpedTwoNodeSolver`, which no scene can reach. | `make test` runs under 30 s with `-m 'not slow'`; `make test-slow` runs the rest; the top-15 durations are printed either way. This is a prerequisite for open question 8, which currently asks the team to choose a tier that does not exist. | — | M | A |
| GT.2 | **Goldens beyond LWIR.** All eight golden arrays come from one Boson LWIR config. Add one MWIR, one SWIR, one NIR, one aerial scene, one maritime scene, one layered slant-path atmosphere, one thermal field and one point-wise frame. | A single MWIR golden would have caught SC.1's 1.52× σ and the zero dark current. The two subsystems where drift is hardest to see by eye — the reflective-band chain and the sky/sea background — have no reference array at all. The `GoldenStaleError` machinery already separates stale from failing, so this is data. | SC.1 | M | A |
| GT.3 | **Run `tests/integration` in an automated job.** 15 files, 4,480 lines, all auto-marked `isaac`, running nowhere; `warp_stages.py` at 38 % and six glue files at 0 % in the only gate that executes. | Split into a Warp-only subset needing a CUDA device but not Kit (ADR 0014's 2026-09-12 addendum measured `env.ensure_warp_on_path` from a bare `python.sh`) plus an Isaac subset run manually with its result recorded. Coverage rises above a stated floor and a regression fails a job. | GT.1 | L | X |
| GT.4 | **`scripts/` into `make typecheck`, and the aperture guard's scope extended to it.** 6,495 lines, linted but never type-checked, containing every lane entry point; 21 of 29 scripts have no test, including all six render drivers, `fidelity_ablation.py`, `eval_detector.py` and `train_detector.py`. | mypy runs clean over `scripts/`. The AST aperture guard walks `scripts/` too — today it covers `src/irsim` and `src/irsim_isaac` only, which is the one real hole in non-negotiable #5's coverage. | — | M | X |
| GT.5 | **Test `irsim_eval.decode`.** It is the entry point to the whole Tier 4 public-data lane and has no test at all; its own docstring names two decode facts (luma-plane-only, an unflagged colour range worth a 255/219 gain plus a 16-code offset) that bound every downstream number. Coverage 43 %, all incidental. | A synthetic clip encoded at a known range round-trips, and a range-flag regression fails. Needs ffmpeg and the `validation` extra, neither of which CI installs, so it follows the existing ffmpeg-gated pattern in `test_codec_floor.py`. | — | S | X |
| GT.6 | **Record Tier 3 manual passes with their commit hash** in `docs/validation/tier3-checklist.md`, and commit a small contact sheet per pass. `outputs/` is gitignored, so the owner's stated way of reading these renders is invisible to everyone but the author, and no manual pass has ever been recorded although the roadmap requires each with a hash. | The checklist carries a dated, hashed row per pass and the five Tier 3 rows that point at open steps (M10.11, M10.19, MM.8) are marked open rather than reading as a plan of record. | RP.6 | S | X |
| GT.7 | **A cost-budget test that pins the cell and prim budget.** Fraunhofer ran 1,313,410 triangles with a 10-layer stack through five day–night cycles in **252 s** in MATLAB on one i7-8700, so 10⁵–10⁶ cells is affordable and nobody should coarsen a patch for speed. | A 10⁵-cell field over a 48 h spin-up completes inside a stated budget, in the **slow** tier — which is why GT.1 lands first. Also pins the render cost: `PointwiseTemperature.apply` measured 79 ms per frame at 640×512 for one bound prim, ~30 % of it a duplicated `local_coords` pass. | GT.1, PT.9 | M | C |

---

## DC — Decisions, deferrals and probes

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| DC.1 | **The SPG probe, split honestly into two.** (a) Write the host-upload experiment into `spg_probe.ALL_EXPERIMENTS` — it lists eleven experiments and the deciding one is not among them, so it does not exist as code. (b) Run it, time-boxed. | Also run the two cheapest temporal-state tests on **this** build: the `:-N` previous-frame `sourceName` suffix, a pure RenderVar rename that fails *silently* by reading the live AOV, so the test must be moving-versus-still; and a `cuda.stateful` output, since the 0.4.0 core binary already contains `allocatePersistentRpResource`. | — | M | X |
| DC.2 | **One ADR for the deferrals that currently have none.** §8.2 narcissus has a deferral row with a reason and a revisit trigger but no ADR, which CLAUDE.md requires for a deliberately skipped fidelity level; same for §13.8 performance, the §6.6 exhaust plume, the composed all-device Warp frame, the ground breadth, and the grey `Atmosphere` as L1-only. | The "Deferred to L3" table is headed "each with its ADR" and has no ADR column. After: it has one, and the plume's "record as ADR 0073" instruction — which would overwrite a live Accepted record — is gone. | RP.4 | M | X |
| DC.3 | **ADR: background photorealism and scene-library breadth are not sim-to-real goals**, labelled as *external* evidence. A published study measured a synthetic set's full-image Vendi diversity at 134.6 against 39.6 for the real set — 3.4× inflated, because real thermal backgrounds wash toward equilibrium — and that inflated diversity drove the gap. | The record says this is one external study on one non-irsim generator: a prior, not an irsim measurement. It also disambiguates "demo scenes must look real", which is about the human reading the render. | EV.7 | S | X |
| DC.4 | **ADR: allocate ADR numbers at write time, never forward.** The corpus has four numbers cited but never written, and revision 3 instructed a future author to write over ADR 0073, which is Accepted. | The parser from RP.4 enforces it: a roadmap or source citation of `ADR NNNN` must resolve to a file. The roadmap names subjects until a record exists. | RP.4 | S | X |
| DC.5 | **Verify the undocumented `add-thermal-emission` material modifier** in the local Isaac 6.1 build, and record the result either way. A forum post surfaced it in the docs with no reported adoption; NVIDIA's own position (Jan 2025) is that Omniverse has no thermal-IR solution and it is a tracked feature request. | A grep of the local MDL and schema tree settles it in under an hour, and a negative is worth recording so nobody chases it again. This is CLAUDE.md's "flag uncertainty rather than guessing" rule applied to the exact case it was written for. | — | S | X |
| DC.6 | **Post the ADR 0014 fp16 measurement back to IsaacSim discussion #298.** NVIDIA's answer to this project's own feature request recommends encoding temperature as OmniPBR emission and reading `PtSelfIllumination` — the route ADR 0014 measured at ~100 mK against a 10 mK bound. | A concrete, reproducible correction to public guidance, on the project's own thread. **No step is spent on the route itself**, because `WM` needs nothing from the renderer. | — | S | X |

---

## Non-negotiable enforcement map

CLAUDE.md's six rules, with what enforces each **today**, what this plan adds, and — where one exists —
the live exception that is not enforced. A step whose only contribution is *not violating* an existing
guard is not listed as an enforcer.

| # | rule | enforced today | this plan adds | live exception |
|---|---|---|---|---|
| 1 | Engine-free core | `test_layering.py`: 359 AST cases over `src/irsim`, `src/irsim_isaac`, `tests/unit`, `tests/golden`, both directions plus the ML/imaging ban, with a scanner self-test. The best-enforced rule in the repo. | `WM.2`/`WM.3` keep the Warp path split so the field stays in `irsim.thermal` and only the query lives in `irsim_isaac` — the cheap way to keep engine portability as a property (see priority 4) | none found |
| 2 | float32 or better wherever T or L flows | `test_temperature_encoding.py` round trip under 10 mK with an fp16 negative control; `GBuffer` dtype guard and `PRECISION_CRITICAL_KEYS`; LUT float32 in memory and on disk with fp16 files rejected; goldens refuse fp16 and float64; in-Isaac AOV dtype; EXR writers refuse fp16 | `IG.8` removes the fp16 carve-out; `IG.13` puts float32 planes on the three drivers that emit PNG only; `AT.1`'s elevation plane is float32 **and its value asserted**; `IG.2` is the in-engine end-to-end enforcer that replaces M10.13c | `gbuffer_isaac.py:156-158` lets float16 through the normals plane, justified by a claim ADR 0014's own addendum and the build's registry both contradict. **`IG.8`** |
| 3 | Noise in radiance or electron space, never Kelvin | NETD(373)/NETD(300) = 0.576 within 5 % at the detector boundary; `test_residual_not_kelvin_flat`; the NUC residual's mK-to-DN conversion happens once at construction (ADR 0056) | `SC.1` wires the electron budget so the rule reaches photon FPAs at all; `SC.2`'s mutation test then discriminates noise-space correctness rather than wiring; `EV.3` stops the rendered-versus-real comparison being stated in DN8 alone | **True for bolometers only.** `core.py:176` calls `anchor_noise` unconditionally; MWIR renders at σ 533.3 e⁻ against 350 e⁻ datasheet, dark = 0 everywhere. **`SC.1`** |
| 4 | Kirchhoff closure to 1e-6, every material, every band | Structurally unbreakable: the schema refuses more than one authored member of {ε, ρ, τ}, ρ is always derived, and `directional_properties_for` re-derives it at **every angle**. Library walk over 19 materials × 4 bands | The closure is not the exposure; the **inputs** are. `RP.7` reopens S13 (glass τ authored before its k table existed, 0.02 against 0.134 recomputed); `XD.8` tests the values against measured spectra with a stated tolerance; `SE.1` records the sea's angular envelope | All 19 materials are `source: literature`; every NIR and SWIR value is ESTIMATED; the one spectral file is a 7-row sketch its own header calls ESTIMATED |
| 5 | Aperture factor π/(4F²+1) defined once | AST guard, 181 parametrised cases with obfuscated-variant self-tests, forbidding any expression squaring an f-number outside `irsim/optics/aperture.py` | `GT.4` extends the guard's walk to `scripts/` — 6,495 lines containing every lane entry point, currently unscanned. That is the one real hole in this rule's coverage | `scripts/` is not walked |
| 6 | One `WeatherSeries` for thermal and atmosphere | Not code review: an `is` identity assertion in three test files across five consumers (`test_scene.py:42-47`, `test_reflected_environment.py:223`, `test_thermal_scene.py:74`), plus constructors with no path parameter | `XD.7` adds a *physical* consistency check, with its limit stated: SURFRAD measures air T, RH, wind and downwelling longwave at one station and minute, but a pyrgeometer constrains the hemispheric broadband total, not radiance versus elevation, and clear-sky parameterisations sit at ~23 W/m² RMSE — so it **cannot** detect a second object holding identical values. The identity assertion stays the enforcer | none found |

Two enforcers named by revision 3 are removed from this map. **M10.13c** (the 300.000 versus 300.050 K
ΔDN test) was formally deferred by ADR 0061 because SPG 0.4.0 has no persistent device buffer, so the
enforcer cited for rule #2 can never ship; `IG.2` replaces it. **M10.7** is not a row — it was split into
M10.7a and M10.7b and the id dangled; rule #3 lost nothing. **M10.9b** is open and is therefore not
listed as enforcing anything.

---

## Risk register

Carried forward with the rows that need treatment. R1, R4–R10, R12, R14–R25 are unchanged from revision 3
and are not restated here; the ones below either changed state or were being reported wrongly.

| # | risk | state | retired by |
|---|---|---|---|
| R2 | `omni.rtx.spg` cannot hold cross-frame state or read a LUT file from Lua | **Not retired.** Revision 3 said "retired by M2.3, M10.12" and M10.12 is open — and is removed from this plan. Two untried tests exist on the shipped 0.4.0 (the `:-N` suffix; a `cuda.stateful` output) | `DC.1`, or the `WM` lane making the capability unnecessary |
| R3 | The AO AOV is not sky visibility | Bounded, not retired: there is no AO AOV on this build at all, and the fallback `V_s = (1 + n·up)/2` is unoccluded | `IG.7`(c) if the Replicator `occlusion` annotator delivers; otherwise `WM.4`'s ray-traced view factor |
| R11 | The unit suite drifts past 30 s | **Materialised.** Measured 130 s, a 4.3× overrun. The `slow` marker the mitigation names was never implemented in the Makefile or pyproject | `GT.1` |
| R13 | Estimated data is mistaken for measurement | **Violated where it matters most.** The three example configs mark `ratios_3d` ESTIMATED; the two Boson configs that actually get rendered and compared against reality do not — and those are the values `SC.2` shows are 7–19× out | `SC.3` |
| R26 | The per-point ablation returns a negative | Live. The pre-committed answer is to redirect to the ISP, which the same evidence ranks first — but the two effect sizes are a near-tie (HTV d = 1.242 against target Sobel variance d = 1.224) measured on a different generator, so "ranks first" is not a ranking that survives. See open question 10 | `EV.9` |
| R27 | **New.** Point-wise temperature reaches one lane, and it is the lane ranked third | Live. Two prims in one of eight scene configs, authored only from Python, world-frame only, night only | `PT.1`, `PT.2`, `PT.5`, `PT.9`, `PT.10` |
| R28 | **New.** A whole-file write silently reverts another session's work | **Materialised three times**: `68acd1c`, `303b56b`, and the 135-line CHANGELOG deletion live in the working tree now. The tested mitigation `scripts/stage_own_hunk.sh` exists and nothing reaches it | `RP.3` — not the row-length cap, which only makes the diffs reviewable |
| R29 | **New.** A "measured" figure in a project document cannot be reproduced | Live. `tier4-2026-09-15.json` records no clip list, archive hash, `config_hash`, CRF, seed or patch count; no noise statistic anywhere states its de-trending convention, which alone moves the answer by up to 3× | `EV.8`, `SC.5` |

---

## Deferred deliberately

Each of these is a fidelity level skipped on purpose, and CLAUDE.md requires an ADR for that. `DC.2`
writes the one that covers them; until it lands, this table is the record and it carries the reason and
the revisit trigger, which is more than the old "Deferred to L3" table did — it was headed "each with its
ADR" and had no ADR column, two rows describing work that had since shipped, and one instruction to write
over ADR 0073.

| item | why deferred | revisit when |
|---|---|---|
| §8.2 narcissus | §8.2 gives only a phenomenological form and no amplitude data | A Tier 4 flat-field PSD shows a radial low-frequency term |
| §13.8 performance | No implementation and no citation outside one open step's spec column | Dataset throughput binds — see the composed Warp frame below |
| §6.6 exhaust plume | Needs a participating-medium term absent from §2; a surface-radiometry pipeline cannot represent it. **Deferred without a number**; the old instruction to "record as ADR 0073" would overwrite a live Accepted record | MWIR Tier 3 |
| The composed all-device Warp frame | ~2,200 lines of op-for-op stage twins exist and are held to the CPU oracle, `EQUIVALENCE_STAGES` registers four of them, and nothing composes them into a frame. `ir_camera.py:44-51` still blames M10.7b, which landed 2026-09-13, so the real blockers are recorded nowhere: composing the stages, the AGC/FFC schedule question, and where the host/device seam sits — M10.7a already showed two correct stages can disagree tenfold when the seam moves | Dataset throughput actually binds. State the measurement rather than the belief: `PointwiseTemperature.apply` is 79 ms per frame at 640×512 for one bound prim on the CPU path and the plan binds more prims, so `GT.7` is what decides this |
| Ground and automotive breadth | Building-envelope and roofing materials, the street-canyon reflected environment, and the solver's missing occlusion input are real gaps — low-e glazing alone is a ~40 K apparent-temperature error on every modern window — but they serve the lane ranked third | Phase C, after the aerial and maritime lanes meet the exit bar |
| The grey `Atmosphere` as a live path | Two models coexist and `Scene.from_config` builds both; every render script uses `scene.layered` while `scene.atmosphere` is the primary attribute | `AT.5` marks it L1-only or guards it; a divergence, not a feature |
| §13.7 options 2, 3 and 4; `THM-16`, `THM-17` | Unchanged from revision 3, and each already carries a reason and a trigger there | Unchanged |
| Turbulence, polarisation, spectral fine structure, scattered-sunlight *path* radiance | App. A #2, #5, #6; §7.4. ADR 0086 shipped the scattered-sunlight **sky**; only the path radiance remains | Ranges beyond 500 m or airborne use |
| `E_star`, `E_artificial`, headlights | No illumination data; airglow and moon cover the night SWIR case | A night urban SWIR scene is needed |

---

## Not adopted, and why

Things a reader might reasonably expect to find here. None of these is in the repository today; this
section exists so nobody adds them.

**MRTD and MDTD (ASTM E1213, ASTM E1311, STANAG 4349).** Never present in this repository — a grep over
every `.md`, `.py` and `.yaml` outside `outputs/` returns zero hits — and they should stay absent. Both
are **observer-in-the-loop by definition**: a human looking through the actual imager at a four-bar
target through a calibrated collimator. The simulator can *predict* them with a TTP-style model; it can
never validate them, with or without a camera. Adding them to a validation ladder would imply a reachable
state that does not exist.

**§15's flat "apparent temperature within 2 K per class", as a pending target.** It should be restated in
`docs/physics-model.md`, not silently retired here — a roadmap can raise a spec issue and propose a
restatement, it cannot amend the document CLAUDE.md names as the source of truth. The reason to restate
it is stronger than "no radiometric capture exists": **no uncooled core is specified well enough to
adjudicate 2 K.** Boson is ±3 °C at 25 °C ambient and ±5 °C at 50 °C, and only under laboratory
conditions (steady state, FPA within 0.2 °C of the last FFC, blackbody emissivity > 98 %); Lepton is the
greater of ±5 °C or 5 %. For scale, DIRSIG — validated against its own instrumented 24-hour collection
with in-scene thermistors — reports roughly 1.8 °C RMS in actual temperature and 5–6 °C in apparent
temperature. So acquiring a camera would not rescue this target, and reporting it as pending implies
otherwise. Raise it as a spec issue against §15 and replace it with a tiered target: solver accuracy
against contact thermometry, and apparent-temperature accuracy through the full chain, each against a
cited precedent. *(Owner-visible: this touches the spec, so it is proposed, not done.)*

**"For when a camera arrives" as a framing anywhere in the repo.** It contradicts ADR 0003 and the
standing public-data-only constraint, and it keeps four measured comparisons skipping forever against
directories that were never created and a camera key (`flir_boson_640_lwir`) that does not match the only
camera anything was measured on (`halmstad_boson_320`). `SC.8` replaces it. **No step in this document
proposes a hardware purchase.**

**Multi Matte as a per-pixel attribute route.** Evaluated and rejected: the id is per mesh or per
material, so it is no finer than `instance_segmentation`, which the project already uses at exact uint32
full resolution. Recorded so nobody rediscovers it — along with the **"Primvar AOV"**, which is a
Houdini/Karma feature and not an Omniverse one. Omniverse's primvar path is MDL-side only and terminates
in fp16 colour or uint8 albedo, so it cannot carry a precise per-pixel parameter out of the renderer.
That is a plausible-looking dead end that costs a day.

**fp16 colour encoding of temperature, UVs or patch coordinates.** ADR 0014 measured ~100 mK at 300 K
against a 10 mK bound, and NVIDIA's own answer to this project's feature request (IsaacSim discussion
#298) recommends exactly that route. Keep the measurement, keep the refusal, post the correction back
(`DC.6`) — but spend **no step** on it. Note the honest form of the argument: fp16's 11 significant bits
are fatal for `(T-200)/800` and would be ample for a 16×8 cell index, so the obstacle for a *UV* is not
precision but the undocumented per-camera exposure scale (~2.8e-4) that puts the ramp in the subnormal
range. The drop stands for a different reason than the one revision 4 gave: `WM` needs nothing from the
renderer, so there is no reason to try.

**GAN image translation and Cosmos Transfer as a fidelity route.** Cosmos Transfer is an RGB appearance
model with no thermal prior; claiming it makes an IR frame look real would violate ADR 0068's rules on
what evaluation data may claim. What transfers is only its posture — structure-preserving augmentation
judged by downstream detector performance — and irsim's equivalent is ADR 0083's hashed
physical-parameter ablation switches. Be precise about the limit of that equivalence: **those switches
have only ever been run against image statistics, never against detector AP**, so the equivalence is
untested, and running it (`EV.9`, and the wider ablation) is itself the publishable result the literature
names as its own unfilled future work.

**The raw test count as a coverage figure.** 540 of 2,756 collected tests come from two per-file
structural scanners that scale with file count. Quote the coverage figures instead: `src/irsim` 93.9 %,
`src/irsim_eval` 85.6 %, `src/irsim_isaac` 43.5 %, with six glue files at 0 % totalling 1,101 statements.

**Revision 3's "Current state (2026-09-10)" section.** Deleted, with the reasons given above. It was the
second section of the document, so a session reading top-down formed a wrong model before reaching a
table.

---

## Recommended, but for the owner to settle — not edited unilaterally

**`IU-29` "engine-interface contract + Unreal port doc", the §14 Unreal mapping, and
`docs/maps/isaac-and-unreal.json`.** The standing guidance is that engine portability stays a *property* —
already CLAUDE.md non-negotiable #1, enforced by `test_layering.py`, and cheap to keep by holding the Warp
kernels free of `omni`/`pxr`/`isaacsim` imports — while no roadmap step, ADR or document is spent on
Unreal-specific deliverables. **CLAUDE.md line 13 ("A port to Unreal Engine follows later") is part of the
same question.** A parallel session may hold the other half of that guidance, so this needs the owner's
word. `IU-29` is marked open in the ledger and left in place until then, and `RP.8` deliberately does not
touch line 13.

---

## Open questions

Decisions only a person can make. Each names the step it blocks, a default answer, and what happens if no
answer arrives — so the plan cannot stall on silence.

| # | question | blocks | default if unanswered |
|---|---|---|---|
| 1 | **Email the Halmstad authors for the Y16 originals.** Action zero, and not engineering work. A positive answer would supersede `XD.10`'s role entirely and switch on every analyser ADR 0068 gates off, on the *aerial* lane, with the *exact* Boson core the configs model | `XD.12`; scopes `XD.10` | Send the email now; time-box the wait at two weeks; on expiry proceed with `XD.3`/`XD.4` and treat any later reply as a bonus. The plan must not depend on the answer |
| 2 | **Unreal.** Does CLAUDE.md line 13 stand, and does `IU-29` leave the plan? | Nothing technical; blocks knowing whether the §14 mapping is dead weight | Leave `IU-29` and line 13 untouched. Spend no step either way |
| 3 | **Detector framework, licence and GPU.** torch plus which small detector — a YOLO-family model (AGPL) or an Apache-licensed alternative? Where does it live, which machine trains it? | `EV.11` | Run `EV.12`'s training-free proxy first, which needs neither, and treat the detector as a confirmation |
| 4 | **FLIR ADAS v2's Terms of Use.** A form-gated click-through with no open terms stated | `XD.10` | Derive statistics on one machine; redistribute nothing; settle before the ingest, not after |
| 5 | **ECOSTRESS / ASTER redistribution terms.** Cite-and-extract is clearly fine; checking CSVs into the repo needs the terms read | `XD.8` | Cite and extract; keep the CSVs out of git, as `docs/maps/materials-surface.json` already flags |
| 6 | **Which camera is "the" reference** — the Boson 640 / 14 mm the spec assumes, or the Halmstad set's Boson 320 the public data was recorded with? | `SC.3`, `SC.8` | Both YAMLs: datasheet limits asserted on the former, field-measured ratios on the latter, never mixed in one `ratios_3d` block, with a schema guard that enforces it |
| 7 | **Publish the paired RAW-16 / AGC-8 dataset?** No public thermal set offers the pairing and the literature names the 16→8 mapping as the dominant sim-to-real factor, so it is a genuine contribution rather than another synthetic drone set. Under what licence? | The *publication* half of `EV.13` only | Build the format and the assertion regardless; hold publication. `EV.13` is split so the unblocked work does not wait on the blocked question |
| 8 | ~~Which tier does the cost-budget test belong to?~~ **Withdrawn as posed.** There is no slow tier: R11's marker is unimplemented in both Makefile and pyproject and the fast gate is already 4.3× over budget. `GT.1` implements the tier; `GT.7` then lands in it | — | `GT.1` before `GT.7`; no decision needed |
| 9 | **Commit scopes.** `pipeline`, `validation`, `eval` and `io` are still proposed additions to CLAUDE.md's list | Nothing; a convention | Use `build` for infrastructure and the physics scope a step tests |
| 10 | **What happens if `EV.9` returns a negative** (R26)? | `EV.9`'s interpretation | Redirect to the ISP — but record the caveat *before* the measurement: the two published effect sizes are a near-tie (1.242 against 1.224) measured on a different generator, so the redirect is a decision, not a reading of the evidence. Confirm the intent so the result is not relitigated afterwards |

---

## ADR number allocation

The highest ADR is 0089. Four numbers below it are cited and were never written: **0042** (the Level B
angular model, cited by `directional.py:21`, `angular.py:24` and four test files), **0079** (sea-water
optical constants and the Cox–Munk slope model, cited by `sea.py:36`, `nk.py:23` and ADR 0078's own
Consequences), and **0062** and **0069**, which are cited only by the roadmap itself as forward
allocations. `RP.4` writes the two that shipped code cites and replaces the two roadmap-only citations
with subjects.

**No number in this document is allocated forward.** In a three-session shared tree a forward-assigned
number is a scheduled collision, and revision 3 demonstrated both failure modes at once: it left four
numbers dangling and it instructed a future author to write over ADR 0073, which is the visible-companion
environment dome, Accepted 2026-09-14. `DC.4` records the rule and the RP.4 parser enforces it.

---

## Review notes

**Revision 5, 2026-09-15.** Replaces revision 3 (2026-09-10) in full, and supersedes an unpublished
revision 4 that was rejected in review for three reasons this revision fixes.

*What revision 4 got wrong, recorded so it is not reintroduced.* It asserted repository states that do
not hold: that MRTD and MDTD were being dropped from a validation ladder they were never in (zero hits
repo-wide); that §8.2 narcissus was "neither implemented nor properly deferred" when it has a
Deferred-to-L3 row with a reason and a revisit trigger, lacking only an ADR; that §13.8 has "zero
citations anywhere" when it is cited in M10.9b's spec column; and that §13.7 options 3–4 are
"unreferenced" when they hold their own deferral row. It cited ADR 0069 as an existing record and
forward-allocated 0095 and 0102 in a tree whose maximum is 0089. It claimed every number was measured "on
this tree at commit `ca5a663`" when the tree was dirty in seven paths, and two of its own headline figures
(2,748 tests, 298,272 characters) no longer reproduced. And it shipped **no step table at all**, so no
coverage claim about any step could be checked.

*What this revision does differently.* (1) The plan is published: 109 steps across eleven lanes, each with
a verification cell that states what would fail and why the tolerance is that number, plus deps, size and
phase. (2) It is organised by the owner's application order, with a per-lane exit bar taken from the
owner's own words and amended only to require float32 output. (3) The three defects the audits ranked
critical are steps, not omissions: the per-pixel slant path (`AT.1`, 25 % τ and 48 % L_path error at
5 km/45°, on the first lane, where the point-target path already does it correctly so a target disagrees
with its own sky), the photon-FPA electron budget no render path reaches (`SC.1`, MWIR σ 1.52× wrong,
dark current zero everywhere), and the three Tier 4 methodology defects that make the acceptance run's own
attribution table unsound (`EV.1`–`EV.4`). (4) The headline requirement gets six steps that carry it out
of the lane ranked third: per-cell solar, a patch in the scene schema, a local-frame bridge, and bindings
on the aerial and maritime lanes. (5) `WM` is a lane with an error budget, an oracle and a superseding
ADR, because ADR 0087's "a real limit, not a temporary one" is false on this build and leaving it standing
will cost another session a week. (6) The repair lane lands first and fixes the mechanism, not the
symptom: `RP.3` makes `stage_own_hunk.sh` the default, which is what stops a whole-file write; the row cap
only makes the diffs reviewable.

*Sources.* Five subsystem audits run against the code and `git log` on 2026-09-15; a survey of production
IR simulators (DIRSIG, MuSES/RadTherm-IR, OKTAL SE-WORKBENCH, CAMEO-SIM, Fraunhofer IOSB, VIRSuite,
ThRend, AirSim); published camera datasheets (FLIR Boson Rev 340, Lepton Rev 400, Tau 2 Rev 141);
measurement standards (NVESD 3-D noise, NV-IPM, ISO 12233, VDI/VDE 5585, ASTM E1543); public dataset
primary sources; NVIDIA's own ovrtx and Isaac Sim 6.1 documentation and the installed Warp 1.16.0 and
`omni.rtx.spg` 0.4.0 in this build; and the 2026 sim-to-real literature on LWIR drone detection.

*A standing caution about this document's own numbers.* Everything marked *measured* carries the commit
and the tree state above. Anything quoted from outside the repository is labelled as external evidence,
not as an irsim measurement — that distinction is what R13 exists to protect, and revision 4 broke it by
attributing another group's correlation to an irsim ADR that did not exist.
