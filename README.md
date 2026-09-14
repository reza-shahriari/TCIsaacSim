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
| `materials` | 🟡 partial | T1 | Material schema (one YAML per material, `source` required, author exactly one of ε/ρ, τ optional; ADR 0040) + library deriving the third quantity per band via the ADR 0010 band average; closure library walk to 1e-6 (CLAUDE.md #4); six §16.2 materials; scalar-per-band `MaterialTable` with the UNMAPPED sentinel (npz + sidecar, stale guard); engine-free USD mapping resolver + `scripts/audit_materials.py` coverage gate (ADR 0047); four-material aerial library -- painted composite, carbon fibre, painted aircraft aluminium, propeller rubber (MS.7, ADR 0072); **branch-safe complex Fresnel (Level A, M7.4)** -- `fresnel_reflectance(n, k, cos θ)` with the §4.2 A/B reparameterisation, so the passive square-root branch is selected by construction (the rejected one returns R > 1) and an absorbing medium grows no total-internal-reflection knee; water at 10 µm reproduces [R1] (R(0) = 0.01018, ε(60°) = 0.961, ε(80°) = 0.697), which is the angular collapse a sea surface is made of; **real water optical constants** (`data/nk/water.csv`, Segelstein 1981, 2.0–15.6 µm) behind a loader that refuses a table with no `# source:` header and never extrapolates, plus `band_directional_emissivity` giving the band-effective ε_B(θ) — water over the Boson band reads 0.988 at nadir and 0.673 at 80°, where Fresnel at a single 10 µm would say 0.711 (M7.5 water, MM.1); glass/Al/paint n/k, roughness and the reflection lobe pending; semi-transparent second ray L = ε L_B + ρ L_env + τ L_behind with ρ derived and closure enforced per pixel, so the committed windshield shows itself in LWIR and what is behind it in SWIR (M7.15, ADR 0046) |
| `thermal` | 🟡 partial | T1 | `WeatherSeries` (one injected object, ADR 0032; unit guards, interpolation identities, hash), project CSV loader/writer (bit-exact round trip), synthetic clear day + committed 48 h sample; convection h = max(free, forced) with vehicle speed (ADR 0033); NOAA sun position + facet solar loading (ADR 0034, checked against an independent Spencer oracle); broadband longwave down from Brunt/Idso clear-sky emissivity (ADR 0035: the LWIR-window T_sky is not a broadband proxy); `TemperatureSolver` protocol with Prescribed and Newton (exact exponential) solvers; aerial target nodes -- motor/ESC/battery ΔT = ΔT_max u² above the shared weather's T_air and an airframe node, on a grid refined to 1 mK (MS.7, ADR 0072; magnitudes ESTIMATED); the two-node environment solver and vehicle regimes are phase 2 |
| `atmosphere` | 🟡 partial | T1 | Beer–Lambert kernel (isothermal invariance to 1e-12), Magnus humidity, grey-band error study (ADR 0048), seven presets reproducing the §7.2 table rows with Koschmieder aerosol (ADR 0049: fog ordering and the humid/fog LWIR–SWIR crossover from weather alone); `Atmosphere(preset, weather, luts)` bound to the one shared `WeatherSeries` (object only; T_air identity with the solver's ambient; humid/fog crossovers from weather alone); stage 2 in `run_frame` with the sky-pixel pass-through and the constant-τ L1 fallback (ADR 0050; known answer 306.303 K within 1 mK); layered slant-path model (exponential sum over spectral classes, sky = column emission; R13's −40 °C LWIR sky at 15° reproduced; ADR 0071); `SkyModel` elevation/tilt LUTs, cloud blend, broadband delegation, the spec's cos^q form derived and its error recorded (ADR 0044); cloud clutter with the LCL base from the weather, ε = 1 − τ and seeded 1/f^β structure (MS.3, ADR 0070; display-domain bound deferred to ME.5); Tier 3 phenomenology over every preset (band orderings, the humid and fog crossovers from weather alone, short-slant self-consistency) and a Bouguer solar-path stub, `τ_sun(θ) = τ_zenith^sec θ` (M8.8, ADR 0051). **Sea surface as a background (MM.2/MM.3, ADR 0078):** `SeaModel` gives apparent sea temperature vs depression angle — Cox–Munk slope statistics from the shared weather's wind, a facet-tilt quadrature moving emissivity and reflected-sky elevation together, spherical horizon geometry. The isothermal identity is exact to 0 mK. The profile is **not** a monotone ramp: there is a cold band 2–15° below the horizon (283.7 K against 289.5 K at nadir for a 290 K sea), and the atmospheric path pulls the far field back toward T_air, so range and not angle alone sets maritime background contrast. |
| `optics` | 🟢 done (L2) | T1–T2 | Aperture factor π/(4F²+1) defined once (AST guard), FPA irradiance, pixel power; pinhole field angles and cos⁴ vignetting (ADR 0015); self-emission single-lens form + Kirchhoff-closed element stack, 87 mK/K shutterless drift (ADR 0016); box downsample + composed optics stage with inverse (ADR 0020); MTF cascade and optical PSF at the supersampled pitch, slant-edge MTF bench 0.31 at Nyquist (ADR 0059); `HousingTemperature` source for the self-emission term -- `fixed` / `ambient` / `coupled`, the coupled node a first-order lag on the shared weather's T_air plus ΔT_self via the M6.6 exact-exponential solver, checked against the closed-form step response at τ/2τ/5τ to 0.01 K and against the first-order Bode amplitude and phase under a diurnal drive (M9.3, ADR 0016 addendum; `coupled` now requires `housing_tau_s`, sensor schema v6); lens projection as the **oracle for the lens the engine is handed** -- OpenCV rational-polynomial and Kannala-Brandt forward models, the USD/OpenCV frame flip and the principal-point convention tied to the vignetting geometry by test, distort/undistort round trip < 1e-6 px against the 0.2 px in-sim budget, and `ftheta` **refused** because its polynomial convention is undetermined on this build (M10.9a, ADR 0015 addendum); **within-frame motion smear** -- the spatially varying spatial twin of the cascade's `mtf_motion`, which had been described since M5 and never applied, held to within 0.015 of `|sinc(s·f)|` and split by integration duty so a shutterless bolometer smears over the whole frame and a cooled photon detector does not (ADR 0077); **rotor discs as a time-averaged veil** -- a running mean of blade passage over the swept angle, composited in radiance, mean-preserving under every shutter, with tilt entering only as the projected area of a pitched plate (ADR 0081) |
| `detector` | 🟡 partial | T1–T2 | `FpaParams`; ideal bolometer/photon transfers (ADR 0019), shared quantiser; NETD predictor (ADR 0024), anchoring (ADR 0025); `PhotonDetector` / `MicrobolometerDetector` responses with seeded per-pixel noise in physical units, two-blackbody NETD benches reproduce the anchor and the 0.576 derivative ratio (ADR 0026); membrane thermal time constant as a pre-noise per-pixel float32 IIR, settled start, moving-edge tail length v τ_th/Δt measured from a rendered trail (M9.1, ADR 0052; wired into the pipeline by M9.8); FPA temperature node (tau and ΔT_self authored, RK2, fixed/ambient/coupled) with the raw gain/offset(T_FPA) polynomials normalised at T_cal by construction — the NUC residual keeps its own parameters so drift is not double-counted (M9.2, ADR 0053) |
| `noise` | 🟢 done | T2 | Counter-based per-pixel hash RNG (ADR 0022); NVESD synthesiser; `NoiseStage` (correlated 3-D terms on the detector's σ_TVH) wired into `run_frame`; DN-domain NETD within 10 % of the anchor, ratios recovered within floors; mean-reverting (OU) drift of the V, H and VH fixed terms by the exact stationary update, so the configured ratios survive a 2000-frame run where a random walk would destroy them invisibly -- lag-τ autocorrelation e⁻¹ ± 0.05 (a random walk control reads > 0.9), σ stationary within 5 %, zero mean, τ = ∞ frozen bit-identical, and the global offset refused because it drifts physically via M3.3/M9.3 (M9.4, ADR 0054); bad-pixel map by a Neyman--Scott cluster process (parents uniform, 1 + Poisson(λ) offspring in a 2 px disc) with the four §10.4 classes and a two-state RTS chain for the flickering and blinking ones -- count within 10 % of the configured fraction, mean nearest-neighbour distance 0.26× the uniform-Poisson expectation (a uniform control reads 1.0), stuck pixels bit-identical over 100 frames, RTS occupancy within 5 % and geometric dwell by a Monte-Carlo-calibrated KS (a memoryless control is rejected) (M9.5a, ADR 0055; schema v7); NUC residual `g = 1 + ppm·1e-6·ΔT_FPA·ξ`, `o = mK/K·ΔT_FPA·(∂DN/∂T)·ξ` with the millikelvin→DN conversion done **once** at 300 K (ADR 0056) -- exactly unity/zero at ΔT = 0, 90 mK at ΔT = 2 K and exactly 2× the 1 K value, and `test_residual_not_kelvin_flat` pins the apparent-T error at 373 K to 0.576× the 300 K one, the measured ∂DN/∂T ratio (the same 0.576 ADR 0026 found for NETD); `ffc_reset` redraws, giving |r| < 0.05 across the event -- the pattern is replaced, not faded (M9.6, ADR 0053/0056); `DriftingPattern`, the FFC-resettable holder for the breathing pattern |
| `isp` | 🟢 done (L2) | T2–T3 | Radiometric branch (ADR 0021); **flat-field correction wired into the display branch (M9.12)** -- `TwoPointNuc` existed from M5 and nothing applied it, so the 8-bit picture carried cos⁴ vignetting (21 % at the Boson's corner) that plateau equalisation stretched into black corners, which no real camera shows. Calibrated from two synthetic blackbody frames through the real forward chain, not from the analytic cos⁴ field, so it removes whatever fixed structure is actually there. **Display branch only**: `invert_optics` already divides cos⁴ out per pixel, so `dn16` and the radiometric outputs stay bit-identical either way and correcting both would remove the same term twice; linear AGC (ADR 0027), plateau equalisation with the hot-exhaust collapse as a scalar test (ADR 0028), DDE (ADR 0029), polarity + palettes → RGBA8 (ADR 0030); display branch in the fixed order with documented rounding points and the isp config hash (ADR 0031); ideal two-point NUC operator; wired as stage 6 — the first thermal-looking image (`outputs/first_image_*.png` from the golden test); bad-pixel replacement by the valid 4-neighbour mean, iterated so a 3×3 cluster fills in two synchronous passes and a partial fill raises rather than leaking a stuck value — exact on a ramp for an isolated defect, σ²/4 on white noise where copy-one-neighbour gives σ² and an 8-neighbour mean σ²/8, and a local Laplacian under half the untouched one, which is §10.4's detectable smoothed footprint (M9.5b, ADR 0055 addendum); `FfcController` — the §11.2 schedule, the freeze that **holds the last good frame** rather than blanking (60 Hz/180 s/700 ms fires at frame 10800 for exactly 42 bit-identical frames; 9 Hz gives 6, round not ceil), a `Resettable` hook for the residual and the drift, and ownership of the ΔT_FPA the residual sees — raw for `shuttered`, a scene-based-correction first-order lag for `shutterless` (540 s at 1.15× the 180 s value where an uncorrected core is at 3×), zero for `ideal` (M9.7, ADR 0057; schema v8); **ROI-weighted and locally-adaptive AGC** (M9.10) -- every histogram takes per-pixel weights, and `agc: plateau_local` tiles the frame, equalises each tile's own histogram and blends the four nearest mappings bilinearly. Both are held to *identities* rather than tolerances: uniform weights and a single tile each reproduce the global operator **bit for bit**, so the new path cannot become a second, slightly different AGC. The global modes stay the default, because a real core is global and §15 Tier 5 needs the simulator to be too |
| `io` | 🟢 done (L2) | T1 | Float32-preserving frame writers (M10.10a): radiance/apparent-T to float32 `.npy` or float32 EXR, DN16 to a uint16 PNG, display to RGBA8 PNG, plus a JSON sidecar with the config/band/ISP hashes, the frame index, the scene time and its UTC, and each plane's dtype, shape and **unit**. A plane in physical units never reaches an integer or half-float container -- a 16-bit PNG over 233-473 K quantises to 3.66 mK and half-float at 300 K to 250 mK, exactly 5x a 50 mK NETD, both of which open and look right; float16 and `half=True` are refused. The EXR writer is stdlib `struct`/`zlib` (single-part, scanline, uncompressed, FLOAT only), so no imaging dependency enters the core, and its header is checked against the format spec rather than only round-tripped |
| `eval` | 🟡 partial | T4 infra | Validation-data index (ME.1a): `data/validation/datasets.yaml` + `irsim_eval.manifest` record licence, access, sensor, **signal path**, bit depth, codec and the analysers each public set may and may not support -- required fields, because a set that cannot say what its frames are cannot be measured. Checked against each publisher: only the Halmstad set (CC0-1.0, Boson 320x256, Y16 -> 8-bit -> mp4) states a licence, so it is the one `primary`; the other five are `unstated`, which records that the terms are unknown rather than assuming they are permissive. `noise_3d` is usable on Halmstad alone; `agc_signature` is *excluded* there because those clips never met an AGC. `scripts/fetch_validation_data.py` plans, hashes and mostly refuses; `data/validation/README.md` is generated and tested for staleness. **`Sequence`/`Frame`/`Box` + canonical layout (ME.1b)** so converters target one form and analysers never read a publisher's -- 8-bit enforced, boxes in the project's pixel-edge convention. **Static-clip gate** in front of every per-pixel temporal statistic, discriminating on cumulative displacement from frame 0 (jitter is bounded and static; a drift accumulates and is moving), by phase correlation with Foroosh's ratio rather than a parabolic fit, which under-reads a Dirichlet peak ~30 % and is biased toward calling a drifting clip static. **`transcode.h264_round_trip` (ME.2b)** calibrates the codec floor by sending a cube of known noise through libx264: full-range flags pinned so CRF 0 is bit-exact, and at CRF 18 -- a *high quality* setting -- 95 % of a Boson-ratio cube's temporal noise is gone while the blocking score barely moves, which is why a lossy set gives a lower bound and not a measurement |
| `config` | 🟡 partial | T1 | `GBuffer` contract; pydantic `SensorConfig` (ADR 0007); YAML loader with data-root resolution, `config_hash`/`band_hash` (ADR 0008); band registry with derived/checked `band.id` and regime-driven illumination terms; optics/detector/noise extensions (schema v4, ADR 0017); 3-D ratio vector and variance closure `total_over_tvh`; scene config + `Scene` builder: one `WeatherSeries` loaded once and injected into the atmosphere and every target solver (M6.17, ADR 0032); environment/illumination presets (`configs/environments/`, ranges checked against §5.3/§5.5; M7.11) |
| `pipeline` | 🟡 partial | T2–T3 | Engine-free NumPy oracle (ADR 0018): `run_frame` emits radiance, apparent_t, dn16 and RGBA8 display8 (stage 2 atmosphere identity); end-to-end golden frames; hot-exhaust AGC collapse reproduced; stage 1 reflected environment term ε L_B + (1−ε)(V_s L_sky,eff + (1−V_s) L_ground) from the SkyModel (M7.13, ADR 0045); analytic point-target injection below one native pixel with the per-class sky-beyond term (MS.6, ADR 0071); the **M9 sensor chain wired into `run_frame`** (M9.8, ADR 0058) — housing and FPA nodes, breathing fixed pattern, defects then replacement, NUC residual, an identity temporal filter and the FFC freeze, in §11.1's order. Attached explicitly by `attach_sensor_chain`; `chain=None` stays the default and is the ideal camera the radiometric goldens describe. The spatial-noise budget is **two mechanisms in quadrature, not four** (3-D spatial ⊕ NUC residual, within 10 %), measured against a subtracted flat-field reference and over averaged frames, at two drift rates so the quadrature is actually under test; the assembled chain has its own golden |
| `validation` | 🟡 partial | T2, T4 infra | Tier 2 SITF, two-blackbody NETD and slant-edge MTF benches; leakage-corrected NVESD 3-D decomposition, spatial/temporal PSD, `compare_psd` (ADR 0023); aerial target-vs-sky contrast and the zero-contrast elevation (MS.7, ADR 0072: 1.26° for ε = 0.9 at 1 km, none at ε = 1 -- the inversion is the reflected term); synthetic aerial scene generator (sky elevation gradient, horizon, 1/f^β cloud, resolved and sub-pixel targets) behind the `gbuffer_aerial` fixture, with the Tier 3 sky phenomenology suite (MS.8); **the Tier 4 reading layer (ME.2b, ADR 0023 addendum)** -- `flat.py` picks the windows a noise statistic may be measured in (flatness scored against the window's own noise, so the thresholds mean the same thing on any DN scale; striped windows stay flat because column noise is what is being measured; a single-pixel target is caught by the outlier rule alone, and on a clip the judgement runs on the temporal median, which deletes a moving one), and `codec.py` measures the floor under them -- lattice step read off the data, floor `step/sqrt(12)` = 0.289 codes at 8 bits, Sheppard's correction applied *and* flagged within two floors of it, per-component codec-limited marks. At sigma_TVH = 1.5 codes only the temporal white term of a Boson-ratio clip survives that test. `temporal_shape` fits the sampled one-pole (10 ms membrane at 60 Hz recovered to 1 %) and reports drift separately, since on flat sky a low-pass is the fingerprint of an in-camera temporal filter; **the shutter signature (ME.3a, ADR 0068)** -- `find_freezes` reads the FFC freezes out of the video (42-frame freezes every 10 800 recovered exactly, and the real `FfcController` loop agrees frame for frame) and reports `pattern_change` beside each one, because a run of repeated frames is equally a dropped chunk of recording and both would feed an interval distribution identically; the interval itself is refused below 180 s, the Boson's own schedule, so Halmstad's 10 s clips give freeze length and never an interval; `fit_pattern_growth` recovers a 30 s OU correlation time from a 5-minute sequence within 15 %, fitting the **variance** (`1 - e^(-2t/tau)`) because the amplitude's law returns 2τ and looks reasonable; **the display-output extractors (ME.3b)** -- the recorder conversion is a required argument of `agc_signature` and `edge_overshoot`, so a Y16-derived set is refused in the function and not only in the index; plateau equalisation separates from a linear stretch by 20× on every sky-like frame (and a naturally uniform scene is reported `indeterminate`, not guessed); DDE overshoot is the analytic `gain/3` of a 3×3 unsharp mask; `replaced_pixel_map` finds the §10.4 footprint by its vanishing Laplacian -- 69 of 69 on float, 93 % on 8-bit codes, no false positives, and **nothing at all through a codec**, which is reported as not measurable rather than as zero defects |
| `irsim_isaac` | 🟡 partial | T1 | `env.py` probes; **M2 gate spike done (ADR 0014):** no float32 colour AOV carries temperature (all fp16, exposure-scaled) → the renderer transports **instance ids + float32 geometry** and temperature comes from a Warp table; `omni.rtx.spg` 0.4.0 present, float32 pass-through bit-exact, **no cross-frame state** (stateful stages stay in Warp), LUT baked into the `.cu`. `probe.py`/`spg_probe.py` + `scripts/probe_isaac_*.py` reproduce it; `tests/integration` (22 tests, one Kit per session) pin it. **Geometry AOVs assembled into the M0.6 `GBuffer` (M10.1):** `gbuffer_isaac.py` builds `distance_m`, `normal_dot_view` (against the per-pixel ray, not the optical axis), `normal_dot_up`, V_s = occlusion·(1+n·up)/2 and `sky_mask`; the ADR 0014 addendum records the survey -- `normals` is float32, full-resolution and **world space**, while `PtWorldNormal` is fp16, half-resolution and all-zero, **no AO AOV delivers** (V_s is the unoccluded form) and **no motion AOV transports motion** (`motion_px` omitted; the analytic form is M10.1b). **`IrCamera` + the aerial demo stage (M10.9a-ii, M10.19):** a stage renders end to end and the frames land on disk. The stage that made it worth doing also found ADR 0014's one real error: `Camera3dPositionSD` is **camera** space, not world -- invisible until a camera is rotated, and read as world it moves the horizon 164 rows, paints the upper sky with the ground temperature and tilts every view cosine, plausibly. **Material-ID transport (M10.2):** `material_ids.py` maps instance id → prim path → the M7.17 resolver's material id by exact integer lookup (no interpolation -- a blended id is a different substance), with the UNMAPPED miss painted magenta in the display branch only; `materials_usd.py` walks a stage for bindings, semantics and the `thermal:material` override and `scripts/audit_materials.py` gains `--stage`. The id channel is **`instance_id_segmentation`**: `instance_segmentation` is per-prim only for *labelled* prims (ADR 0014 addendum). Temperature still comes from the facet table; **Warp stage 1** (`warp_stages.py`) twin of the CPU oracle with the CPU-vs-GPU equivalence harness — ≤ 2 ulp on CUDA, bit-identical on Warp CPU, LUT uploaded once (ADR 0061); **aerial thermal bridge** (`aerial_bridge.py`, M10.18/ADR 0060) filling the float32 temperature-by-instance-id table from the M6.6 solvers on the scene's one `WeatherSeries`, 1 Hz thermal tick interpolated per frame, background pixels taking MS.2's `T_sky(θ)` from their own ray elevation and `T_ground` below the horizon where the sky model is undefined and extrapolating it would invert silhouette contrast. **Warp needs no Kit** (ADR 0014 addendum): `env.ensure_warp_on_path` puts the `omni.warp.core` extension on `sys.path`, so the equivalence harness runs on `cpu` and `cuda:0` from a bare `python.sh` in ~1.4 s rather than behind a 35 s Kit boot, and `gpu`-marked tests are selected on Warp rather than on Isaac Sim. **Warp stages 2-3 (M10.5):** one atmosphere kernel covers the grey, layered-exponential-sum and constant-τ branches (Σ w_k = 1 collapses the per-term path radiance), and the optics kernels do PSF-then-box in that order before Ω_eff τ_opt cos⁴θ A_d + Φ_self, with every radiometric scalar evaluated by `irsim.optics` on the host -- measured ≤ 1.9e-7 (stage 2) and 8.5e-7 with a 53×53 PSF (stage 3) against the oracle on both devices, and a +1 K housing step reading +87.43 mK on both. **Warp stage 4 (M10.6):** the membrane IIR runs in place on a persistent device buffer owned by `WarpPipelineState` (kept in `PipelineState.buffers`, ADR 0052's single-owner rule), the photon path is memoryless and allocates none -- a 20-frame flux step tracks the oracle to 1.74e-7, the frame-1 fraction is α = 0.8111 exactly, and photon DN matches the CPU `floor()` code for code across a saturation sweep. **Warp stage 5 (M10.7a):** the seven §10.2 components, the OU drift of the device-resident fixed fields and the NUC residual as kernels — the first stage held to *statistical* rather than bit equivalence (ADR 0022 addendum). The device pattern is seeded from the CPU's own realisation, so the comparison measures the two generators and not two different cameras; the residual's ξ fields are uploaded and stay bit-comparable to 2e-6. Measured on an RTX A6000 and Warp `cpu` over 200 frames: NETD within 5 % of the CPU path and 10 % of the config, every 3-D component within 15 % or its own estimator floor, NETD(373)/NETD(300) = 0.576, `compare_psd` ≤ 1.5, and τ = ∞ frozen bit-identical. Note the **stage boundary differs**: the CPU puts per-pixel TVH in stage 4 and the device in stage 5, so the oracle is detector+stage together. **Warp stage 6 (M10.8):** the display branch on device — atomic histogram, the plateau clip and `array_scan` CDF, the AGC as a 2^bit_depth lookup table, an edge-clamped 3×3 DDE and the palette → RGBA8. Both §11.3 modes are monotone functions of DN alone, so each *is* a table, which is what lets the port be held to ±1 display code: ≥ 99.9 % of pixels agree with the CPU branch on ramp / exhaust / constant scenes in all three AGC modes, DN16 is untouched under any of them, and the ADR 0028 exhaust collapse is reproduced on device. **Stage 5's remainder (M10.7b):** the defect map (uploaded, not redrawn -- one focal plane has one map), its RTS chain, the iterated 4-neighbour replacement in float64 and the FFC hold, all on device. Mostly **exact** rather than statistical: replacement is bit-for-bit against `replace_bad_pixels` on 1x1/2x2/3x3 clusters and on the real map, injection bit-for-bit against `apply_defects`, a held frame bit-identical across a freeze, and the **composition** bit-for-bit against `SensorChain.finish_frame` -- that last because M10.7a showed two correct stages can disagree tenfold if the seam moves. Only the RTS chain is statistical (occupancy 25 %, mean dwell 35 %, dwell > 2 frames where a memoryless redraw gives ~1). **`IrCamera` (M10.9a-ii):** the object that turns a stage into a frame -- authors the camera prim from the sensor YAML (aperture = width x pitch, so f_px = f/pitch), creates the render product at `supersample x native`, attaches the M10.1 annotators, assembles the G-buffer from geometry + the M10.2 ids + the M10.18 temperature table, and runs `run_frame`. Takes the `Scene` rather than a separate atmosphere preset so one weather object reaches everything (#6). **Distortion writes every attribute of the USD schema**, because the schemas default to a 2048x1024 lens with fx = 900 and a non-zero fisheye k1. In-sim: a rendered grid reprojects through the config to **< 0.2 px** undistorted and under barrel, the barrel term moves the outer quads 2.1 px with the axis fixed, and the zero-coefficient model fails on that render by > 1.5 px; a whole frame comes out float32/float32/uint16/RGBA8 with the two eps = 0.09 prims collapsing 27 K of authored difference to under a quarter. Runs the **CPU reference**, not the Warp stages: the device path has no post-ADC chain until M10.7b, and a frame from a different camera must not be labelled the same. **A generated environment dome for the companion visible frame** (`visible_sky.py`, ADR 0073): a Preetham daylight sky, Lambertian terrain hazed into the horizon by Koschmieder's law, and a 0.53° distant light for the solar disc, all driven by the scene's own NOAA sun position, the shared weather's visibility (turbidity as a *column* optical-depth ratio -- the ground-level one gives T = 14 on a clear 23 km day) and its DNI/DHI. The map is authored in **direction space**, because the renderer's lat-long convention is not the documented one and had to be measured: on this build the RTX dome light's pole is the stage's **+Z**, so an elevation/azimuth layout put the whole camera field inside one texture pole and filled the frame with ground. The infrared radiance and apparent-temperature planes are **bit-identical** with the dome and without it, and the solar disc is re-measured against the renderer each run of the integration suite. **The quadrotor flight stage (M10.20, ADR 0074):** `quadrotor.py` lays out a heavy-lift multirotor from primitives -- engine-free arithmetic, unit-tested, USD authoring separate -- so four motor bells, four speed controllers, a pack and a carbon frame are separate prims with their own materials and thermal nodes; `quad_flight.py` is the tracked stage, with attitude driven by the same throttle the thermal model reads. `IrCamera` gains a `frame_period_s` override that makes it a time-lapse camera -- every stage told the truth about the interval, not a fast-forward. The dome and the visible looks moved to a shared `stage.py` so a second stage cannot drift from the first. **The aircraft pass stage (M10.21, ADR 0075):** `aircraft.py` lays out a light jet with rear-mounted engines so both nozzles sit on the centreline; `aircraft_pass.py` carries the track, an azimuth/elevation look-at mount (not the minimal rotation, which rolls the horizon as it slews) and the stage. `Part`/`author_parts` moved to a shared `airframe.py` so the material override and the prim→node map have one implementation. `IrCamera.refresh_pose()` lets a camera move between frames -- the pose behind every ray is cached at `open()`, so a re-aimed camera without it renders geometry from the new aim and sky from the old one.  **Maritime demo (MM.5–MM.7, ADR 0078):** `maritime_demo.py` + `scripts/render_maritime_demo.py` put vessels on open water — a wave-displaced, Earth-curved water surface for the **visible** frame that joins the *background* mask, so its infrared temperature is the analytic sea profile at each ray's depression angle and `--no-water` leaves the IR frame unchanged. Rings are spaced uniformly in depression angle (a pixel row covers ground as r²) and the wave train is scaled by the shared weather's wind as U², so the picture and the radiometry cannot disagree about the weather. |

Tier 2 today is self-consistency (no camera, ADR 0003): SITF strictly increasing, linear in L_B(T) to
0.29 LSB rms, blackbody T_app < 10 mK; two-blackbody NETD within 10 % of the anchor; 3-D ratios within
the estimator floors. Tier 3 items landed: the hot-exhaust AGC collapse (linear AGC halves a
pedestrian's 8-bit contrast; plateau 0.012 keeps > 50 % of the background std).

**The global AGC is why a target reads as one flat white shape, and there is now an alternative
(M9.10).** One hot object sets the stretch for every pixel in the frame — that is not a defect to
fix, it is what a real core does, so the global operators stay the default. But §11.3 offers two
ways out and both are here. Measured on a broad exhaust plume: a global *linear* stretch keeps
**8 %** of the background's contrast and a global plateau stretch **61 %**, while `agc:
plateau_local` keeps **93 %** — the motor and the wings stay separable instead of saturating
together. The cost is honest and stated: display level no longer means one thing across the frame,
so two pixels of equal DN in different tiles can display differently, and none of this touches the
radiometric branch. Tile seams are the obvious failure and the blend removes them completely — on a
ramp, a tile boundary is no bigger a step than any other column, where assigning each pixel its own
tile's mapping with no blending puts a **255-code** cliff there.

**First image.** `make check` writes `outputs/first_image_boson_hot_patch.png` (the golden test): a
295 K background with a 305 K block and a 600 K patch through the Boson configuration, noise on,
plateau equalisation. The dark corners are the cos⁴ vignetting that plateau equalisation stretches
on an un-flat-fielded camera. `flat_field_enabled=True` (M9.12) removes it -- that claim used to be here and was untrue: `TwoPointNuc` existed from M5 and nothing applied it. The atmosphere (stage 2) is still identity in this golden.

**First light in Isaac Sim.** `python.sh scripts/render_aerial_demo.py --frames 4` renders the
phase-1 aerial stage -- quadrotors at 120 m / 500 m / 1500 m, an aircraft at 2.5 km, a bird and a
hot motor pod against sky -- through the whole camera model and writes the four outputs per frame
in physical units (M10.9a-ii, M10.19, M10.10a). There is no sky *geometry* and no ground plane: a
ray that hits nothing takes `T_sky(θ)` at its own elevation or `T_ground` below the horizon
(ADR 0060), because an emissive dome would push the sky through a float16 colour AOV. The lens is
verified against the renderer to 0.2 px. Still open for a full M10.19: the ME.6 real-vs-synthetic
comparison, and the τ(R)/R² SCR law, which needs MS.6's analytic injection rather than renderer
geometry.

**The companion visible frame has a real sky (ADR 0073).** `--rgb` used to write a flat grey void
with six grey squares in it, which told a reader nothing about where the camera pointed or what
hour it was -- and the visible frame is the only half of the pair a human can check by eye. The
stage now carries a generated lat-long environment map on its dome light: a Preetham daylight sky
above the horizon, Lambertian terrain hazed into it by Koschmieder's law below, plus a distant
light with a 0.53° cone for the solar disc. Every input is the scene's own -- NOAA sun position at
the scene's site and clock, turbidity from the shared weather's visibility against the atmosphere
preset's Rayleigh coefficient (a *column* ratio: the ground-level one reads T = 14 on a clear
23 km day), irradiance from the same weather. **The infrared frame is bit-identical with the dome
and without it**, asserted in `tests/integration`, because a light is not geometry and never
reaches the temperature plane. The dome's lat-long pole axis had to be *measured* -- on this build
the RTX dome light samples with its pole on the stage's **+Z**, not the stage up axis -- and the
sun is re-measured against the renderer: pointed down the sun's own azimuth, the disc lands within
8 px of `f_px·tan(tilt − elevation)`. No cloud is painted and no disc is baked into the sky
texture, both so the pair does not show what the infrared frame does not have.

**A drone you can actually see (ADR 0074).** `python.sh scripts/render_quad_flight.py --frames 300
--rgb` films a heavy-lift quadrotor at 20 m -- 105 px across its span, 6 px per motor bell -- flying
a 30-minute mission, and writes an MP4. The airframe is built **parametrically from primitives**,
not imported, so every part is a separate prim with its own material and thermal node: four motor
bells, four speed controllers, a battery and a carbon frame. ADR 0072's law then takes the motors
from ambient to +45 K and back as the throttle moves, with the speed controllers and the pack
following at their own ΔT_max, so motor > ESC > battery > airframe holds at every instant of the
flight by construction. Making that reachable needed a scene-schema bump: §6.6's aerial node model
had existed since M6.6 and **no scene config could ask for a motor**, so v3 adds `heat_source` and
`airframe` solvers that name a *throttle profile* and derive the temperatures.

Two framing decisions in that video are physics, not taste. It is a **time-lapse** -- one frame per
6 s of a 1800 s flight -- because the node law is a steady-state relation with no thermal time
constant, so it is only defensible while the throttle moves slowly against a motor's minutes-scale
response; every stage is told the truth about the interval, so the FFC fires on its real schedule
and the noise decorrelates as it really would. And the main video uses a **fixed display span**
rather than the camera's AGC -- in the sensor config's own white-hot grayscale, not a false-colour
palette, because a presentation video that picks its own colours is a second display path that can
drift from the camera's -- because both §11.3 AGC modes rescale from the current frame and
cancel exactly the change being filmed -- plateau equalisation additionally gives a sub-1 % target
almost no display codes, measured at 1217 of the object's pixels landing in the top ten, so the
whole airframe is one flat white shape. The span is taken from the **target's own nodes**, not
from ambient: spanning ±50 K about air spends half of 256 levels on sky-to-ambient and leaves the
target 76 display codes of spread where the target-spanned version gives 128. Sky clips to black,
deliberately; `--span-c` restores a scene-context span. The camera's own AGC output is filmed alongside, because that
difference is the lesson. Propellers are still absent from *this stage*, but no longer for want of
a model: ADR 0081 supplies one, and mounting it is glue (see below).

**An aircraft is the opposite problem (ADR 0075).** `python.sh scripts/render_aircraft_pass.py
--frames 300 --rgb` flies a light jet past the sensor at 150 m/s and films ten seconds in real
time. Where a quadrotor's heat is four motors a sixteenth of its span across that you resolve and
watch warm, a jet's is an exhaust nozzle a *fortieth* of its span across but hundreds of kelvin
hot — a near-point source — and it is **hidden behind its own nacelle from the front**. So nothing
about the target changes during the pass; **aspect** does, which is why a real aircraft's measured
signature varies by a large factor around the clock and why rear-aspect detection ranges are the
ones quoted. Counted off the instance-id plane at the same range either side of closest approach,
a rear aspect shows several times more nozzle and a hottest pixel tens of kelvin warmer.

Two pieces of physics had to be added for it. **Aerodynamic heating**: `airframe_solver` puts an
unpowered skin at air temperature and justifies it by forced convection at flight speed, which is
true at 20 m/s and false at 200 — a stagnating boundary layer takes the skin to its recovery
temperature, 0.18 K above ambient at multirotor speed and 10 K at 150 m/s. Schema v4's `ram_skin`
node names an **airspeed** and derives the rest against the shared weather's own air temperature.
And **a mount that slews**: the aircraft flies its true track while the pedestal re-aims at it, for
which `IrCamera.refresh_pose()` is not optional — the pose behind every pixel's ray is cached at
`open()`, so without it the geometry follows the new aim while the sky stays at the old one, across
a pass whose true elevation sweeps 11° → 37° → 11°.

**Cloud is in the rendered background now (ADR 0076).** Against a sky background the dominant
false alarm is a cloud edge, not sensor noise — warm, target-sized, and the *same polarity* as a
drone, since both read warmer than a cold clear zenith. MS.3's field has existed since M7 and only
the engine-free scene generator used it. The bridge now takes a `cloud_seed`, and the field is
fixed to the **sky** rather than the image plane: an image-plane field travels with the sensor, so
a slewing mount carries its clouds along and a tracked target never crosses an edge. It also
corrects ADR 0073, which called the background "clear-sky only" — `SkyModel.radiance` always
carried the uniform blend, the *expectation* over the field, so this replaces a mean with a
realisation and **the mean is preserved** (~0.2 % across seeds). Coverage is exact over the sky and
not over a frame: one elevation ring measured 0.0195 against a whole-sky 0.0500, which is precisely
what makes cloud clutter rather than texture.

**Moving targets smear now (ADR 0077).** `mtf_motion` had been in the MTF cascade since M5 and
**nothing ever called it**, so every frame was sharp however fast the scene crossed it — with the
aircraft stage sweeping the boresight at 34 °/s against a 0.049° pixel, that is 11 pixels of
unmodelled blur per frame, and it is one of the clearest tells separating real thermal video of a
moving target from synthetic. The smear is **spatially varying**, because under a tracking mount
the target is still on the focal plane while the sky sweeps past and one kernel serves neither.
The integration duty is where the detector families part: a microbolometer has no integration
window (`integration_time_ms` is `None` for one) so it smears over the whole frame, while a cooled
photon detector integrates briefly and is sharper — §16's "lateral motion smears LWIR, not cooled
MWIR". Held to the cascade term it implements: within **0.015** of `|sinc(s·f)|` across the sweep.

**A spinning rotor is a veil, not geometry (ADR 0081).** ADR 0077 claimed propellers would be the
smear operator's job. They are not: a blade tip at 3000 rpm does **112 m/s**, sweeping 109 pixels
of *circular* arc per bolometer frame and wrapping the disc 1.7 times, where that operator averages
along a straight segment with at most 65 taps. So the disc is computed as what the detector
actually reports — the **time average of an intermittent opaque occluder**, composited in radiance
(`alpha L_blade + (1-alpha) L_behind`), because blending apparent temperatures instead reads a 3 %
veil of 290 K blade over a 230 K sky as 231.9 K where the radiance blend gives 233.1 K, hiding the
disc rather than showing it. **One formula, both detector families:** coverage is a running mean
over the angle swept during the integration, so a bolometer's whole-frame window (300° = 1.67 blade
spacings) draws a smooth banded annulus and a 2 ms cooled integration (36°) resolves two arcs five
times as bright — with **the same total**, because a running mean cannot move the mean of a periodic
function. Tilt enters only as the projected area of a pitched plate, which makes coverage *exactly*
tilt-invariant until the disc is within `pitch` of edge-on; the first version asserted the opposite
and the rasteriser caught it. The disc reaches the focal plane through the **lens oracle** rather
than an `f·R/Z` stand-in — `disc_ellipse` measures both semi-axes from projected rim points, so an
8 m off-axis barrel lens shrinking the disc 4.3 % and pulling it 17 px inward is seen, not missed;
the residual (the conic is assumed centred with perpendicular axes) is **measured at 0.079 px** at
20 m, falling as range². On the frame the veil **composites rather than injects an excess**: MS.6's
point-target form carries a `sky_beyond` term because a sub-pixel target occults a sky column the
plane does not separately hold, but by stage 2c the plane already has the right background at every
pixel — sky over some of the disc, the aircraft's own arm over the rest — so one blend handles both,
lifting over cold sky and *dipping* over a warm arm in the same pass. **The quadrotor now carries
four**, above its motor bells, and authors nothing to do it — no prim, no mesh, no material.
Occlusion is a plane intersection rather than a range comparison, because the disc spans 0.36 m in
depth at 20 m and the shortcut would draw 18 px of arm on the wrong side of the aircraft. Two traps
worth naming: the sweep is taken over the **detector's** frame, not the time-lapse's six-second
capture interval, which would otherwise turn three hundred revolutions into a plausible annulus
with the banding physics quietly deleted; and rpm follows throttle by a **square root**, since
thrust goes as rpm² and the profile's `u` is a fraction of maximum thrust — reading it linearly
spins a hovering aircraft 40 % slow and moves which regime its discs are filmed in. One honest
asymmetry: the discs are in the infrared frame and **not** in the companion visible frame, which
RTX renders from stage geometry there is none of. Measured in sim: **6444 pixels change, peaking at
+10.1 K**, and the disc is brightest at its *root* because local solidity rises inward as the
circumference shrinks while the chord does not — 3.2 % at three-quarter radius, 19 % just outside
the bell. The first render found what no unit test had: sky carries `distance_m = 0` in the
G-buffer, not `inf`, so reading it as a distance put a surface at the camera and occluded every
pixel of the frame. The test used `inf` — what the AOV reports before the adapter translates it —
and passed. Note too that at the demo's target-spanned display (18–62 °C) a 273 K disc clips to
black with the sky: **real in radiance, absent from the picture**. `--span-c -15 40` shows it, 45
display codes deep.

**Targets below one pixel are injected, not rendered.** A 0.35 m quadrotor at 500 m is 0.82 of a
Boson pixel, and a rasteriser gives a phase-dependent fraction of its flux (ADR 0071), so those
prims are hidden and MS.6's analytic excess is injected in their place -- one path or the other,
never both, guarded by `IrCamera.check_no_double_count()`. Measured in sim: the excess survives
the whole chain to 5 %, lands within half a pixel of where the renderer draws the same object, and
shows that the usual `tau(R)/R^2` shorthand under-predicts longer ranges by 33 % over 400-3200 m,
because what a target occults is the sky column beyond it.

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
  (ADR 0003). **Measured, not assumed (ME.2b):** 8 bits alone put a 0.289-code floor under every
  noise statistic, which on a clip at σ_TVH = 1.5 codes leaves only the temporal white term
  measurable; and x264 at CRF 18 — a high-quality setting — removes 95 % of that term. Noise
  numbers from a lossy public set are therefore lower bounds and are reported as such, never as
  targets the simulator is tuned to hit.
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
