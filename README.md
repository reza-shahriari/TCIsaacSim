# irsim — physically-based multi-band IR camera simulator

Simulates what a real LWIR / MWIR / SWIR / NIR camera would see, with radiometry that closes in physical
units. Targets NVIDIA Isaac Sim 6.x (every engine fact is measured on the 6.1.0-rc.26 source build,
ADR 0014); the physics core is engine-free so an Unreal Engine port is a rewrite of the glue only.

**Physics specification:** [`docs/physics-model.md`](docs/physics-model.md) — the source of truth for
every equation here. Code cites it by section.

**Spec issues:** [`docs/spec-issues.md`](docs/spec-issues.md) — contradictions found in the spec, with the
resolution the code assumes and what has been applied; raise new ones there, never diverge silently.

**Plan:** [`docs/roadmap.md`](docs/roadmap.md) — milestones, one-commit steps, risks, ADR backlog. Phase 1
targets objects in the sky against a sky background, validated against public anti-UAV thermal video;
phase 2 is ground scenes and automotive (ADR 0003).

---

## Status

Validation tiers: **T1** unit/analytic · **T2** radiometric bench · **T3** phenomenology ·
**T4** vs. real data · **T5** task-level. See `docs/physics-model.md` §15.

What promotes a row (the table claims nothing above its evidence):

- **T1** — analytic identities, independent implementations or known answers, in physical units,
  in `tests/unit` (ir-sim-testing skill). Every row starts here.
- **T2** — a standard lab characterisation reproduced *inside the simulator* on the CPU reference
  (SITF, two-blackbody NETD, 3-D noise decomposition, MTF from a slant edge) with a stated tolerance.
  Until a camera is available these are self-consistency checks (ADR 0003).
- **T3** — a phenomenology item of §15 that emerges from the model and is asserted as a scalar (the
  hot-exhaust AGC collapse, thermal crossover, fog vs band, FFC freeze …); a manual look does not count.
- **T4** — a statistic measured on public imagery (`docs/validation/`) and matched within the
  ADR 0068 targets, with N and a confidence interval.
- **T5** — a detector-transfer experiment (real↔synthetic); external to this table and reported
  in `docs/validation/`, never as a row state.

A 🟢 row means the L2 scope of `docs/physics-model.md` §1 is implemented for that module and its
tier evidence is in the suite; 🟡 means partial with the notes saying what is missing.

| Component | State | Tier | Notes |
|---|---|---|---|
| `radiometry` | 🟢 done | T1–T2 | Planck (both forms, derivatives, exitances; σ, σ_q to 1e-6), encoding (0.05 mK), R(λ) contract, Simpson oracle, band averaging, float32 LUT (0.03 mK) + inverse (< 1 mK), bundles + `make luts`; golden LUT slice at 1 mK; ADRs 0005–0013 |
| `materials` | 🟡 partial | T1 | Material schema (one YAML per material, `source` required, author exactly one of ε/ρ, τ optional; ADR 0040) + library deriving the third quantity per band via the ADR 0010 band average; closure library walk to 1e-6 (CLAUDE.md #4); six §16.2 materials; scalar-per-band `MaterialTable` with the UNMAPPED sentinel (npz + sidecar, stale guard); engine-free USD mapping resolver + `scripts/audit_materials.py` coverage gate (ADR 0047); four-material aerial library -- painted composite, carbon fibre, painted aircraft aluminium, propeller rubber (MS.7, ADR 0072); Fresnel/roughness/reflection pending; semi-transparent second ray L = ε L_B + ρ L_env + τ L_behind with ρ derived and closure enforced per pixel, so the committed windshield shows itself in LWIR and what is behind it in SWIR (M7.15, ADR 0046) |
| `thermal` | 🟡 partial | T1 | `WeatherSeries` (one injected object, ADR 0032; unit guards, interpolation identities, hash), project CSV loader/writer (bit-exact round trip), synthetic clear day + committed 48 h sample; convection h = max(free, forced) with vehicle speed (ADR 0033); NOAA sun position + facet solar loading (ADR 0034, checked against an independent Spencer oracle); broadband longwave down from Brunt/Idso clear-sky emissivity (ADR 0035: the LWIR-window T_sky is not a broadband proxy); `TemperatureSolver` protocol with Prescribed and Newton (exact exponential) solvers; aerial target nodes -- motor/ESC/battery ΔT = ΔT_max u² above the shared weather's T_air and an airframe node, on a grid refined to 1 mK (MS.7, ADR 0072; magnitudes ESTIMATED); the two-node environment solver and vehicle regimes are phase 2 |
| `atmosphere` | 🟡 partial | T1 | Beer–Lambert kernel (isothermal invariance to 1e-12), Magnus humidity, grey-band error study (ADR 0048), seven presets reproducing the §7.2 table rows with Koschmieder aerosol (ADR 0049: fog ordering and the humid/fog LWIR–SWIR crossover from weather alone); `Atmosphere(preset, weather, luts)` bound to the one shared `WeatherSeries` (object only; T_air identity with the solver's ambient; humid/fog crossovers from weather alone); stage 2 in `run_frame` with the sky-pixel pass-through and the constant-τ L1 fallback (ADR 0050; known answer 306.303 K within 1 mK); layered slant-path model (exponential sum over spectral classes, sky = column emission; R13's −40 °C LWIR sky at 15° reproduced; ADR 0071); `SkyModel` elevation/tilt LUTs, cloud blend, broadband delegation, the spec's cos^q form derived and its error recorded (ADR 0044); cloud clutter with the LCL base from the weather, ε = 1 − τ and seeded 1/f^β structure (MS.3, ADR 0070; display-domain bound deferred to ME.5); Tier 3 phenomenology over every preset (band orderings, the humid and fog crossovers from weather alone, short-slant self-consistency) and a Bouguer solar-path stub, `τ_sun(θ) = τ_zenith^sec θ` (M8.8, ADR 0051) |
| `optics` | 🟢 done (L2) | T1–T2 | Aperture factor π/(4F²+1) defined once (AST guard), FPA irradiance, pixel power; pinhole field angles and cos⁴ vignetting (ADR 0015); self-emission single-lens form + Kirchhoff-closed element stack, 87 mK/K shutterless drift (ADR 0016); box downsample + composed optics stage with inverse (ADR 0020); MTF cascade and optical PSF at the supersampled pitch, slant-edge MTF bench 0.31 at Nyquist (ADR 0059); `HousingTemperature` source for the self-emission term -- `fixed` / `ambient` / `coupled`, the coupled node a first-order lag on the shared weather's T_air plus ΔT_self via the M6.6 exact-exponential solver, checked against the closed-form step response at τ/2τ/5τ to 0.01 K and against the first-order Bode amplitude and phase under a diurnal drive (M9.3, ADR 0016 addendum; `coupled` now requires `housing_tau_s`, sensor schema v6) |
| `detector` | 🟡 partial | T1–T2 | `FpaParams`; ideal bolometer/photon transfers (ADR 0019), shared quantiser; NETD predictor (ADR 0024), anchoring (ADR 0025); `PhotonDetector` / `MicrobolometerDetector` responses with seeded per-pixel noise in physical units, two-blackbody NETD benches reproduce the anchor and the 0.576 derivative ratio (ADR 0026); membrane thermal time constant as a pre-noise per-pixel float32 IIR, settled start, moving-edge tail length v τ_th/Δt measured from a rendered trail (M9.1, ADR 0052; wired into the pipeline by M9.8); FPA temperature node (tau and ΔT_self authored, RK2, fixed/ambient/coupled) with the raw gain/offset(T_FPA) polynomials normalised at T_cal by construction — the NUC residual keeps its own parameters so drift is not double-counted (M9.2, ADR 0053) |
| `noise` | 🟢 done | T2 | Counter-based per-pixel hash RNG (ADR 0022); NVESD synthesiser; `NoiseStage` (correlated 3-D terms on the detector's σ_TVH) wired into `run_frame`; DN-domain NETD within 10 % of the anchor, ratios recovered within floors; mean-reverting (OU) drift of the V, H and VH fixed terms by the exact stationary update, so the configured ratios survive a 2000-frame run where a random walk would destroy them invisibly -- lag-τ autocorrelation e⁻¹ ± 0.05 (a random walk control reads > 0.9), σ stationary within 5 %, zero mean, τ = ∞ frozen bit-identical, and the global offset refused because it drifts physically via M3.3/M9.3 (M9.4, ADR 0054); bad-pixel map by a Neyman--Scott cluster process (parents uniform, 1 + Poisson(λ) offspring in a 2 px disc) with the four §10.4 classes and a two-state RTS chain for the flickering and blinking ones -- count within 10 % of the configured fraction, mean nearest-neighbour distance 0.26× the uniform-Poisson expectation (a uniform control reads 1.0), stuck pixels bit-identical over 100 frames, RTS occupancy within 5 % and geometric dwell by a Monte-Carlo-calibrated KS (a memoryless control is rejected) (M9.5a, ADR 0055; schema v7); NUC residual `g = 1 + ppm·1e-6·ΔT_FPA·ξ`, `o = mK/K·ΔT_FPA·(∂DN/∂T)·ξ` with the millikelvin→DN conversion done **once** at 300 K (ADR 0056) -- exactly unity/zero at ΔT = 0, 90 mK at ΔT = 2 K and exactly 2× the 1 K value, and `test_residual_not_kelvin_flat` pins the apparent-T error at 373 K to 0.576× the 300 K one, the measured ∂DN/∂T ratio (the same 0.576 ADR 0026 found for NETD); `ffc_reset` redraws, giving |r| < 0.05 across the event -- the pattern is replaced, not faded (M9.6, ADR 0053/0056); `DriftingPattern`, the FFC-resettable holder for the breathing pattern |
| `isp` | 🟢 done (L2) | T2–T3 | Radiometric branch (ADR 0021); linear AGC (ADR 0027), plateau equalisation with the hot-exhaust collapse as a scalar test (ADR 0028), DDE (ADR 0029), polarity + palettes → RGBA8 (ADR 0030); display branch in the fixed order with documented rounding points and the isp config hash (ADR 0031); ideal two-point NUC operator; wired as stage 6 — the first thermal-looking image (`outputs/first_image_*.png` from the golden test); bad-pixel replacement by the valid 4-neighbour mean, iterated so a 3×3 cluster fills in two synchronous passes and a partial fill raises rather than leaking a stuck value — exact on a ramp for an isolated defect, σ²/4 on white noise where copy-one-neighbour gives σ² and an 8-neighbour mean σ²/8, and a local Laplacian under half the untouched one, which is §10.4's detectable smoothed footprint (M9.5b, ADR 0055 addendum); `FfcController` — the §11.2 schedule, the freeze that **holds the last good frame** rather than blanking (60 Hz/180 s/700 ms fires at frame 10800 for exactly 42 bit-identical frames; 9 Hz gives 6, round not ceil), a `Resettable` hook for the residual and the drift, and ownership of the ΔT_FPA the residual sees — raw for `shuttered`, a scene-based-correction first-order lag for `shutterless` (540 s at 1.15× the 180 s value where an uncorrected core is at 3×), zero for `ideal` (M9.7, ADR 0057; schema v8) |
| `config` | 🟡 partial | T1 | `GBuffer` contract; pydantic `SensorConfig` (ADR 0007); YAML loader with data-root resolution, `config_hash`/`band_hash` (ADR 0008); band registry with derived/checked `band.id` and regime-driven illumination terms; optics/detector/noise extensions (schema v4, ADR 0017); 3-D ratio vector and variance closure `total_over_tvh`; scene config + `Scene` builder: one `WeatherSeries` loaded once and injected into the atmosphere and every target solver (M6.17, ADR 0032); environment/illumination presets (`configs/environments/`, ranges checked against §5.3/§5.5; M7.11) |
| `pipeline` | 🟡 partial | T2–T3 | Engine-free NumPy oracle (ADR 0018): `run_frame` emits radiance, apparent_t, dn16 and RGBA8 display8 (stage 2 atmosphere identity); end-to-end golden frames; hot-exhaust AGC collapse reproduced; stage 1 reflected environment term ε L_B + (1−ε)(V_s L_sky,eff + (1−V_s) L_ground) from the SkyModel (M7.13, ADR 0045); analytic point-target injection below one native pixel with the per-class sky-beyond term (MS.6, ADR 0071); the **M9 sensor chain wired into `run_frame`** (M9.8, ADR 0058) — housing and FPA nodes, breathing fixed pattern, defects then replacement, NUC residual, an identity temporal filter and the FFC freeze, in §11.1's order. Attached explicitly by `attach_sensor_chain`; `chain=None` stays the default and is the ideal camera the radiometric goldens describe. The spatial-noise budget is **two mechanisms in quadrature, not four** (3-D spatial ⊕ NUC residual, within 10 %), measured against a subtracted flat-field reference and over averaged frames, at two drift rates so the quadrature is actually under test; the assembled chain has its own golden |
| `validation` | 🟡 partial | T2 | Tier 2 SITF, two-blackbody NETD and slant-edge MTF benches; leakage-corrected NVESD 3-D decomposition, spatial/temporal PSD, `compare_psd` (ADR 0023); aerial target-vs-sky contrast and the zero-contrast elevation (MS.7, ADR 0072: 1.26° for ε = 0.9 at 1 km, none at ε = 1 -- the inversion is the reflected term); synthetic aerial scene generator (sky elevation gradient, horizon, 1/f^β cloud, resolved and sub-pixel targets) behind the `gbuffer_aerial` fixture, with the Tier 3 sky phenomenology suite (MS.8) |
| `irsim_isaac` | 🟡 partial | T1 | `env.py` probes; **M2 gate spike done (ADR 0014):** no float32 colour AOV carries temperature (all fp16, exposure-scaled) → the renderer transports **instance ids + float32 geometry** and temperature comes from a Warp table; `omni.rtx.spg` 0.4.0 present, float32 pass-through bit-exact, **no cross-frame state** (stateful stages stay in Warp), LUT baked into the `.cu`. `probe.py`/`spg_probe.py` + `scripts/probe_isaac_*.py` reproduce it; `tests/integration` (22 tests, one Kit per session) pin it. **Geometry AOVs assembled into the M0.6 `GBuffer` (M10.1):** `gbuffer_isaac.py` builds `distance_m`, `normal_dot_view` (against the per-pixel ray, not the optical axis), `normal_dot_up`, V_s = occlusion·(1+n·up)/2 and `sky_mask`; the ADR 0014 addendum records the survey -- `normals` is float32, full-resolution and **world space**, while `PtWorldNormal` is fp16, half-resolution and all-zero, **no AO AOV delivers** (V_s is the unoccluded form) and **no motion AOV transports motion** (`motion_px` omitted; the analytic form is M10.1b). **Material-ID transport (M10.2):** `material_ids.py` maps instance id → prim path → the M7.17 resolver's material id by exact integer lookup (no interpolation -- a blended id is a different substance), with the UNMAPPED miss painted magenta in the display branch only; `materials_usd.py` walks a stage for bindings, semantics and the `thermal:material` override and `scripts/audit_materials.py` gains `--stage`. The id channel is **`instance_id_segmentation`**: `instance_segmentation` is per-prim only for *labelled* prims (ADR 0014 addendum). Temperature still comes from the facet table; **Warp stage 1** (`warp_stages.py`) twin of the CPU oracle with the CPU-vs-GPU equivalence harness — ≤ 2 ulp on CUDA, bit-identical on Warp CPU, LUT uploaded once (ADR 0061); **aerial thermal bridge** (`aerial_bridge.py`, M10.18/ADR 0060) filling the float32 temperature-by-instance-id table from the M6.6 solvers on the scene's one `WeatherSeries`, 1 Hz thermal tick interpolated per frame, background pixels taking MS.2's `T_sky(θ)` from their own ray elevation and `T_ground` below the horizon where the sky model is undefined and extrapolating it would invert silhouette contrast. **Warp needs no Kit** (ADR 0014 addendum): `env.ensure_warp_on_path` puts the `omni.warp.core` extension on `sys.path`, so the equivalence harness runs on `cpu` and `cuda:0` from a bare `python.sh` in ~1.4 s rather than behind a 35 s Kit boot, and `gpu`-marked tests are selected on Warp rather than on Isaac Sim. **Warp stages 2-3 (M10.5):** one atmosphere kernel covers the grey, layered-exponential-sum and constant-τ branches (Σ w_k = 1 collapses the per-term path radiance), and the optics kernels do PSF-then-box in that order before Ω_eff τ_opt cos⁴θ A_d + Φ_self, with every radiometric scalar evaluated by `irsim.optics` on the host -- measured ≤ 1.9e-7 (stage 2) and 8.5e-7 with a 53×53 PSF (stage 3) against the oracle on both devices, and a +1 K housing step reading +87.43 mK on both. **Warp stage 4 (M10.6):** the membrane IIR runs in place on a persistent device buffer owned by `WarpPipelineState` (kept in `PipelineState.buffers`, ADR 0052's single-owner rule), the photon path is memoryless and allocates none -- a 20-frame flux step tracks the oracle to 1.74e-7, the frame-1 fraction is α = 0.8111 exactly, and photon DN matches the CPU `floor()` code for code across a saturation sweep. **Warp stage 5 (M10.7a):** the seven §10.2 components, the OU drift of the device-resident fixed fields and the NUC residual as kernels — the first stage held to *statistical* rather than bit equivalence (ADR 0022 addendum). The device pattern is seeded from the CPU's own realisation, so the comparison measures the two generators and not two different cameras; the residual's ξ fields are uploaded and stay bit-comparable to 2e-6. Measured on an RTX A6000 and Warp `cpu` over 200 frames: NETD within 5 % of the CPU path and 10 % of the config, every 3-D component within 15 % or its own estimator floor, NETD(373)/NETD(300) = 0.576, `compare_psd` ≤ 1.5, and τ = ∞ frozen bit-identical. Note the **stage boundary differs**: the CPU puts per-pixel TVH in stage 4 and the device in stage 5, so the oracle is detector+stage together. **Warp stage 6 (M10.8):** the display branch on device — atomic histogram, the plateau clip and `array_scan` CDF, the AGC as a 2^bit_depth lookup table, an edge-clamped 3×3 DDE and the palette → RGBA8. Both §11.3 modes are monotone functions of DN alone, so each *is* a table, which is what lets the port be held to ±1 display code: ≥ 99.9 % of pixels agree with the CPU branch on ramp / exhaust / constant scenes in all three AGC modes, DN16 is untouched under any of them, and the ADR 0028 exhaust collapse is reproduced on device |

Tier 2 today is self-consistency (no camera, ADR 0003): SITF strictly increasing, linear in L_B(T) to
0.29 LSB rms, blackbody T_app < 10 mK; two-blackbody NETD within 10 % of the anchor; 3-D ratios within
the estimator floors. Tier 3 items landed: the hot-exhaust AGC collapse (linear AGC halves a
pedestrian's 8-bit contrast; plateau 0.012 keeps > 50 % of the background std).

**First image.** `make check` writes `outputs/first_image_boson_hot_patch.png` (the golden test): a
295 K background with a 305 K block and a 600 K patch through the Boson configuration, noise on,
plateau equalisation. The dark corners are the cos⁴ vignetting that plateau equalisation stretches
on an un-NUC'd camera; the NUC stage (M9) removes it. The atmosphere (stage 2) is still identity.

Bands configured: **LWIR** (`flir_boson_640_lwir`, estimated VOx response — ADR 0013; `make luts` builds the table) · Cameras modelled: _none yet_ (radiometry only; the pipeline starts at M3)

---

## Quick start

```bash
# The project interpreter is Isaac Sim's bundled Python (docs/decisions/0002); any CPython >= 3.10
# works for the engine-free core. Every make target honours PYTHON=.
export PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
make install
make check        # lint + typecheck + unit tests — must be green before any commit
```

Isaac Sim is **not** required for anything in `src/irsim/` or `tests/unit/`. Requires NumPy ≥ 2.0.

Interpreter matrix (ADR 0002): the Isaac Sim interpreter (CPython 3.12) locally for everything
including integration tests; plain CPython 3.10 and 3.12 in CI (`.github/workflows/check.yml`, no GPU,
no Isaac). `make ci` reproduces the CI job locally in a `.venv-ci` built from `python3.10`.

## Commands

| Command | Does |
|---|---|
| `make install` | Editable install + dev dependencies |
| `make test` | Unit + golden tests (fast, no GPU) |
| `make test-all` | Adds integration tests: the `gpu` ones need only Warp and a CUDA device, the rest need Isaac Sim (ADR 0014 addendum) |
| `make lint` / `make fmt` | ruff check / ruff format |
| `make typecheck` | mypy on `src/irsim` and `src/irsim_isaac` |
| `make check` | lint + typecheck + test |
| `make ci` | The CI job locally: plain CPython 3.10 venv + `make check` (no GPU, no Isaac) |
| `make luts` | Regenerate band LUTs from configs and spectral data into `data/lut/` (gitignored; loader detects stale bundles, ADR 0012) |
| `make golden-update` | Regenerate golden reference arrays deliberately (ADR 0004) |

## Layout

```
src/irsim/          engine-free physics core (pure Python + NumPy); irsim.pipeline is the reference oracle (ADR 0018)
src/irsim_isaac/    Isaac Sim glue — the only place engine imports are allowed
tests/unit/         fast, no GPU, no Isaac Sim (default gate, with tests/golden)
tests/conftest.py   synthetic G-buffer fixtures: ramp, uniform, two-material, grazing sphere,
                    4x-supersampled 5.5° step edge, moving edge — the engine-free kernel test bed
tests/integration/  needs the engine; auto-marked isaac, skipped cleanly when absent. The gpu-marked
                    Warp equivalence tests need only Warp + CUDA and run with no Kit (ADR 0014 addendum)
tests/golden/       regression fixtures: .npy + JSON sidecar with config hash (ADR 0004)
configs/            sensor / material / atmosphere YAML
data/               data root (`$IRSIM_DATA_DIR` overrides): spectra/responses/, n/k tables, LUTs, weather
docs/               physics-model.md and ADRs
```

## Current limitations

Stated deliberately — see `docs/physics-model.md` Appendix A for the full list and reasoning.

- No 3-D conduction. Engine bay and exhaust are prescribed, not solved.
- Band-averaged atmosphere (Beer-Lambert). Valid under ~500 m; not for airborne work. Measured
  (ADR 0048): a grey γ_B fitted over 0–300 m over-attenuates a two-level LWIR band by 1.7 % at 500 m
  and 8.7 % at 1 km, and a MWIR band with an opaque CO₂ notch by 18 % at 500 m and 42 % at 1 km.
  The layered exponential-sum model (ADR 0071) removes that error and supplies the sky, but its class
  multipliers are ESTIMATED (two calibrated to R13) and it carries no scattered sunlight.
- Emissivity is grey within a band.
- Reflections are approximate — sky-view-factor blending, not full path tracing.
- No polarisation, no atmospheric turbulence.
- NETD is anchored to datasheet values, not predicted from first principles.
- Weather is prescribed; there is no coupling back from the scene to the atmosphere.
- No IR camera is available: validation uses public datasets (8-bit, lossy-coded, some through an
  unknown ISP), so absolute radiometry (SITF, NETD) is checked for self-consistency only, and the
  target-vs-range behaviour is validated by data only to ~200 m — 0.5–5 km is modelled, not measured
  (ADR 0003).
- Clouds, slant-path atmosphere beyond 500 m and point-target radiometry are additions the physics
  specification does not cover; each carries its own ADR and error statement.
- In-sim the renderer supplies no ambient occlusion and no working motion vectors (ADR 0014
  addendum, measured on 6.1.0-rc.26). The sky-view factor is therefore the unoccluded geometric
  form -- exact under an open sky, optimistic in a street -- and there is no image-plane velocity
  in the Isaac G-buffer yet, so motion MTF and bolometer smear are engine-free only until M10.1b
  synthesises motion from the per-prim transforms.

## Contributing

Read `CLAUDE.md` first — it defines the non-negotiables (engine-free core, checked in both directions by
`tests/unit/test_layering.py`, float32 everywhere
temperature flows, noise in radiance space, Kirchhoff closure) and the per-step workflow.

Every step: `make check` green → README status updated → CHANGELOG entry → ADR if a decision was made →
one commit.

**Several people work in one tree at once.** Commit named paths, never `git add -A`: a blanket add
stages whoever else had `README.md`, `CHANGELOG.md` or `docs/roadmap.md` open, and their prose lands in
your commit under your message. For shared files, `scripts/stage_own_hunk.sh` stages your edit alone —
it three-way merges your change onto HEAD, so a change someone else *committed* meanwhile is a no-op
rather than a conflict:

```bash
export STAGE_OWN_HUNK_ID=my-session          # snapshots are keyed by this; make it unique
scripts/stage_own_hunk.sh snapshot README.md CHANGELOG.md   # immediately BEFORE you edit
# ... edit ...
scripts/stage_own_hunk.sh stage README.md CHANGELOG.md      # then promptly after
git commit -m ...
```

Snapshot late and stage promptly: the snapshot is the only record of what the file looked like before
you touched it, so an *uncommitted* edit someone makes inside that window is attributed to you. `stage`
prints the hunks it staged for exactly that reason. Also remember `make check` lints and type-checks the
whole tree, so an uncommitted error in your files turns everyone's gate red.
