# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Warp stage 5 (roadmap M10.7a, ADR 0022 addendum): the seven §10.2 noise components, M9.4's OU
  drift of the device-resident fixed fields, and M9.6's NUC residual, as Warp kernels. This is the
  first stage that **cannot** be held to bit-equality with the CPU oracle -- Warp's generator is not
  NumPy's -- so it is held to statistical equivalence instead, and is deliberately not registered in
  `EQUIVALENCE_STAGES`, whose whole point is bit-level agreement.
- Two things make that comparison mean something. The device fixed pattern is **seeded from the
  CPU's own realisation** rather than redrawn, so what is measured is the two generators and not
  two different cameras -- a redrawn pattern would have passed even with the wrong spatial
  structure. And the NUC residual's ξ fields are uploaded too, keeping the one part of the chain
  with an exact CPU answer bit-comparable to 2e-6; the residual is reset by an FFC *event*, not by
  a frame, so redrawing it on device would mean a second epoch counter to keep in step across a
  shutter.
- The **stage boundary differs between the two paths and the composition does not**: on the CPU the
  per-pixel TVH term belongs to the detector (stage 4) and `NoiseStage` adds only the six correlated
  ones, while the Warp detector stage is the ideal transfer alone, so the device stage 5 supplies
  all seven. Comparing the stages alone therefore compares a six-term image with a seven-term one,
  which produces a PSD ratio of about ten and looks exactly like a broken kernel. The oracle is the
  detector and the stage together.
- Measured on an RTX A6000 and on Warp `cpu`, over 200 frames: NETD within 5 % of the CPU path and
  10 % of the configured σ_TVH; every 3-D component within 15 % of its configured sigma or within
  three times its own `estimate_floors` floor; NETD(373)/NETD(300) = 0.576, the derivative ratio, so
  non-negotiable #3 holds on the device as it does on the host; `compare_psd` ≤ 1.5. Exact where it
  can be: the same frame id is bit-identical, τ = ∞ freezes the pattern bit-identically and launches
  nothing, and the residual at ΔT_FPA = 0 is exactly the identity.
- The original M10.7 row bundled the noise stage, the drift, defects, replacement, the residual and
  the FFC. It is **split**: M10.7a is the noise half and carries every exit criterion the row
  listed; M10.7b is the new row for the device-side defect map, the iterated replacement stencil and
  the FFC hold, which are a different kind of work -- an iterated stencil and a piece of control
  flow, not a seeded draw.
- The M9 sensor chain wired into `run_frame` (roadmap M9.8, ADR 0058): `irsim.pipeline.SensorChain`
  owns the housing and FPA nodes, the breathing fixed pattern, the defect map and its state, the
  NUC residual and the FFC controller, and advances them together so nothing runs on a stale clock.
  Per-frame order follows §11.1: nodes and drift first (the housing temperature is what stage 3's
  self-emission needs), then defects on the quantised plane, replacement, the residual, the
  temporal filter and the freeze. `attach_sensor_chain` is the one place ∂DN/∂T is evaluated, so
  ADR 0056's "converted once" is enforced by there being a single site that can do it.
- The assembly's real hazard is double-counting, so the headline test is a budget:
  post-correction spatial noise is `√(σ_V² + σ_H² + σ_VH² + σ_residual²)` within 10 % -- **two
  mechanisms in quadrature, not four added up**. Three things had to be right for that to mean
  anything. The flat-field reference is subtracted, because `vignetting_cos4` alone puts a 0.9 %
  cos⁴ falloff across the array -- a larger spatial std than the whole noise budget, so a raw
  frame std measures the lens. Frames are averaged, because a single frame's spatial std also
  contains temporal per-pixel noise that is indistinguishable from fixed pattern. And it runs at
  two drift rates: at the roadmap's 0.05 K/s the residual is nearly thirty times the 3-D term and
  could hide a spurious third mechanism, so a second rate makes the two equal, where a linear sum
  would be 41 % high against a 10 % tolerance.
- ADR 0058 also records the two things the chain deliberately omits. §11.1's unnamed "temporal
  filter" stays an **identity** until ME.5's temporal PSD on flat sky shows whether real cores
  low-pass at all: the stage sits directly on σ_TVH, the quantity NETD is defined from, so an
  invented coefficient would change measured NETD by √(α/(2−α)) in a way nobody could later
  distinguish from a detector-model error. And the chain applies the NUC residual but **not** M9.2's
  raw gain/offset(T_FPA) polynomials, which are what the NUC removes -- applying both would form a
  large number and subtract almost all of it back, in float32, to reach what the residual gives
  directly (ADR 0053).
- `chain=None` remains the default and is a meaningful configuration, not an unwired one: it is the
  ideal camera every M9 mechanism is measured against, and the one the radiometric goldens
  describe. Those goldens are therefore **unchanged** by this step -- the roadmap anticipated
  updating them, which turned out to be the wrong trade, because a golden containing both the
  radiometry and the defect map can no longer fail informatively for either. The assembled chain
  gets its own golden (`boson_sensor_chain_dn16` / `_apparent_t`) instead.
- FFC controller (roadmap M9.7, ADR 0057): `irsim.isp.FfcController` is §11.2's shutter event --
  the part the spec singles out as "worth more than another decimal place of radiometry". It owns
  the schedule, the freeze, and the ΔT_FPA that M9.6's residual is evaluated at, which is what
  actually distinguishes the three §12.2 modes: raw offset from the last calibration for
  `shuttered`, a scene-based-correction first-order lag for `shutterless`, and identically zero for
  `ideal`. Objects the shutter recalibrates attach through a one-method `Resettable` protocol, so
  the controller neither imports nor knows about them.
- The freeze **holds the last good frame** rather than blanking. A closed shutter has the FPA
  looking at the blade, and a camera emitting those frames would show the scene vanish and
  reappear -- conspicuous, and not what real cores do. Holding makes the image *stale* instead:
  for 0.7 s anything tracking through it sees motion stop dead and then jump, which is the artefact
  a perception stack has to survive. At 60 Hz / 180 s / 700 ms the shutter closes on frame 10800
  and exactly 42 bit-identical frames are emitted while the input keeps changing; the rounding rule
  is round, not ceil, so 9 Hz gives 6 frames rather than 7 -- ceil would systematically lengthen
  every freeze at low frame rates, which is where the artefact is most visible.
- `shutterless` is bounded rather than unbounded: `dΔT_eff/dt = dΔT/dt − ΔT_eff/τ` with
  τ = `nuc.shutterless_tau_s`, so under a constant drift the residual approaches `r·τ` instead of
  tracking ΔT_FPA without limit. At 0.05 K/s the 540 s residual is 1.15× its 180 s value where an
  uncorrected core would be at 3×; the test asserts the saturation *value* as well as the bound, so
  "bounded" cannot be satisfied by merely being slow.
- `irsim.noise.DriftingPattern`: the FFC-resettable holder for M9.4's breathing pattern. `reset`
  redraws rather than zeroing -- a flat-field correction re-measures and re-subtracts the pattern,
  and what is left is a fresh realisation of the same distribution, uncorrelated with the old one.
- NUC residual (roadmap M9.6, ADR 0056): `irsim.noise.NucResidual` is what survives the two-point
  correction -- `g_ij = 1 + ppm·1e-6·ΔT_FPA·ξ_g` and `o_ij = mK/K·1e-3·ΔT_FPA·(∂DN/∂T)·ξ_o` -- with
  both fields redrawn by `ffc_reset`. At ΔT_FPA = 0 the gain is exactly 1 and the offset exactly 0,
  by construction rather than cancellation, so a freshly shuttered camera is perfectly corrected.
  Per ADR 0053 this is the **only** ΔT_FPA-driven mechanism in the chain: the detector's raw
  gain/offset(T_FPA) polynomials (M9.2) are what the NUC removes, and M9.4's OU drift is stationary
  and contributes no growth, so applying more than one would count the same physics twice at a rate
  that still looks plausible.
- ADR 0056 settles the unit question the config poses. `residual_offset_mk_per_k` is authored in
  millikelvin because that is how datasheets quote it, but a residual *applied* in kelvin would be
  right only at the temperature it was tuned at: measured on the committed Boson chain ∂DN/∂T runs
  97.2 / 178.0 / 309.0 / 439.6 DN/K at 250 / 300 / 373 / 450 K. The conversion therefore happens
  once, at construction, at 300 K -- where §9.4 anchors NETD and where datasheet figures are quoted
  -- and `∂DN/∂T` is supplied by the caller from `irsim.isp.dn_per_kelvin` so `irsim.noise` stays
  independent of `irsim.isp` and the "converted once" rule is visible at the call site.
  `test_residual_not_kelvin_flat` pins the payoff: the apparent-temperature error at 373 K is
  0.576× the 300 K one, exactly the measured derivative ratio -- and the same 0.576 ADR 0026
  arrived at independently for the two-blackbody NETD bench. A kelvin-space implementation would be
  flat across that range and would still produce entirely plausible images.
- `irsim.isp.dn_per_kelvin`: ∂DN/∂T of the calibrated transfer by central difference through the
  real forward chain (LUT → optics → detector), the same object the Tier 2 SITF bench measures.
- Bad-pixel replacement (roadmap M9.5b, ADR 0055 addendum): `irsim.isp.replace_bad_pixels` is the
  mean of the valid 4-neighbours, iterated until clusters fill. §10.4 asks for the defect *and* the
  replacement because "the replacement artefact is what a detector actually sees", and the stencil
  was chosen for three testable properties rather than convenience: it is exact on a linear field
  for an isolated defect (< 1 mK, so smooth scene content takes no radiometric bias); it divides
  white-noise variance by four, where copy-one-neighbour leaves σ² and an 8-neighbour mean gives
  σ²/8 -- both measured alongside it so the assertion is shown to exclude them; and it suppresses
  the local Laplacian to under half the untouched value, which *is* §10.4's detectable smoothed
  footprint. Inside a cluster it is deliberately **not** exact: a pixel in a 2×2 never sees an
  opposing pair of neighbours, so the mean is pulled outward -- real, kept, and the concrete reason
  M9.5a's clustering is not cosmetic. Passes are synchronous (a pass reads only what was valid when
  it began), so a 3×3 centre needs two passes, an isolated defect one, and transposing the problem
  transposes the answer -- without which a 2×2 would fill differently row-major than column-major
  and goldens would not reproduce. A partial fill raises: an unreplaced stuck value entering the
  NUC silently is worse than a loud failure. `irsim.noise.active_defect_mask` builds the per-frame
  mask, so an intermittent pixel is replaced only while it is actually bad.
- Bad-pixel map and defect injection (roadmap M9.5a, ADR 0055): `irsim.noise.generate_map` draws
  one sensor's defects by a Neyman--Scott cluster process -- parents uniform, `1 + Poisson(λ)`
  offspring uniform in a 2 px disc -- because §10.4's "clustered slightly, not uniform" is the part
  that matters downstream: a 4-neighbour stencil handles an isolated defect almost perfectly while
  a 2x2 cluster defeats it, so a uniformly scattered map produces no clusters at these densities
  and understates the artefact a perception stack actually sees. Measured at 1024² and 0.0015: 1479
  defects against the 1573 expected (within the 10 % bar; the shortfall is colliding offspring
  merging, left uncompensated because inflating the parent count would distort the cluster-size
  distribution, which is the part that matters), and a mean nearest-neighbour distance 0.26× the
  uniform-Poisson expectation -- with a genuinely uniform map run through the same estimator as a
  control, reading 1.0. The four §10.4 classes split as: dead and hot pinned to the DN floor and
  ceiling bit-identically forever; **flickering** still responding to the scene but offset, which
  is why it survives a map built from one calibration frame; **blinking** stuck only while its
  state is bad. Both stateful classes run on one two-state Markov chain parameterised by its
  stationary occupancy and mean dwell, so both dwell times are geometric by construction --
  verified by a Monte-Carlo-calibrated KS test (the analytic one is invalid for integer dwells and
  reports p ≈ 1e-174 on a perfect sample), with a memoryless pixel rejected by the same estimator.
  Injection is on the raw DN plane, matching §11.1's `raw DN → bad-pixel replace`.
- Mean-reverting drift of the fixed-pattern components (roadmap M9.4, ADR 0054):
  `irsim.noise.FpnDrift` makes §10.3's "pattern breathing" real. The V, H and VH terms follow an
  Ornstein--Uhlenbeck process advanced by its exact update
  `x ← x e^{−dt/τ} + σ√(1−e^{−2dt/τ}) ξ` with τ = `noise.fpn_drift_tau_s`. **Not a random walk:**
  a random walk's variance grows without bound, so it destroys the configured 3-D ratios -- the
  sensor's identity -- a little every frame while each individual frame still looks like plausible
  thermal imagery, which is precisely the failure no single-frame test or visual check would catch.
  The OU form is stationary by construction at any step size, and using the *exact* update rather
  than Euler--Maruyama is what stops the same camera modelled at 9 Hz and 60 Hz from ending up with
  different FPN. All three fixed terms drift because they share one physical cause and because
  breathing VH alone would leave the column stripes frozen -- among the most recognisable real
  artefacts, and the ones sim-to-real transfer is most sensitive to; `components` can restrict the
  set for an ablation. Passing the global term raises: the DC level drifts *physically* through the
  housing and FPA nodes (M3.3, M9.3) and the ΔT_FPA residual is M9.6's alone (ADR 0053), so putting
  a random walk on top would count one effect three times. Measured over long runs against closed
  forms: lag-τ autocorrelation e⁻¹ ± 0.05 at τ, 2τ and 3τ (with a matched random walk run through
  the same estimator as a control, reading > 0.9), σ stationary within 5 % after 2000 frames, the
  ratio vector intact within 8 %, zero mean, and τ = ∞ frozen bit-identical. Drift is the one
  sequential stream in the chain -- the OU state cannot be drawn from `(seed, frame_index)` -- so
  `drift_rng()` is the single blessed generator and M10.7's Warp twin is held to statistical, not
  bit, equivalence.
- Housing temperature source (roadmap M9.3, ADR 0016 addendum): `irsim.optics.HousingTemperature`
  supplies the `T_housing` that §8.2's self-emission term has needed since M3.3, in the three modes
  §12.2's `housing_temp_mode` already named. `fixed` is a bench number and shows no drift at all;
  `ambient` is the air with no lag; `coupled` is the lumped node `dT/dt = (T_air + ΔT_self − T)/τ`,
  which is the physical origin of shutterless drift on the optics side -- a housing warming under a
  NUC table calibrated at a different housing temperature, at ADR 0016's 87 mK of apparent
  temperature per kelvin. The coupled node delegates to M6.6's `NewtonCoolingSolver` rather than
  carrying a second lumped-node integrator; the FPA node's RK2 (ADR 0053) stays the deliberate
  exception. Verified against closed forms, not against itself: the step response matches
  `T∞ + (T0 − T∞)e^{−t/τ}` at τ, 2τ and 5τ to 0.01 K, one 5τ leap equals 5000 sub-steps to 1e-9
  (the exact update is unconditionally stable where forward Euler would diverge), the steady state
  is `T_air + ΔT_self` and not bare `T_air`, `ambient` tracks the weather to 1e-6 K, and under a
  24 h drive the node reproduces the first-order Bode response -- amplitude `1/√(1+(ωτ)²)` within
  2 % and the peak delayed by `arctan(ωτ)/ω` within half a sample, which a second-order node or a
  moving average would fail. It registers with the `Scene` through `.weather`, so a housing built
  on a different `WeatherSeries` than the atmosphere is refused at construction (non-negotiable #6)
  -- worth the guard because a wrong housing temperature is a smooth pedestal, not a visible
  artefact.
- Warp stage 4 (roadmap M10.6): the detector transfer and the microbolometer's membrane lag on the
  device. The IIR updates a persistent `wp.array` in place and writes the frame to a separate
  output, so the state is never round-tripped to the host; `WarpPipelineState` owns the device
  buffers and lives in `PipelineState.buffers`, the same single-owner rule ADR 0052 set for the CPU
  side so that M10.7's drift, defect and NUC buffers are cleared by the same cold start. The path is
  chosen from `fpa.type`: a cooled photon FPA is memoryless and allocates no state at all. Measured
  on an RTX A6000, `cuda:0` and Warp `cpu`: a 20-frame flux step tracks the oracle to 1.74e-7
  against a 1e-5 budget, the first frame of a step covers 0.8111 of it (= alpha, and the number
  §9.2's "roughly 0.6 frames" is not -- spec issue S8), one state pointer across ten frames, and
  photon DN matches the CPU floor() code for code across a 0 -> 2x saturation sweep.
- `irsim.pipeline.detector.detector_stage`: stage 4 on the plane dict -- the ideal transfer followed
  by the membrane lag for a bolometer only, with the per-pixel state in `PipelineState.buffers`.
  This is the composition ADR 0052 fixes and the oracle the Warp twin is compared against; M9.8
  wires it into `run_frame` along with the rest of the M9 chain.
- `quantise_warp`: the ADC on the device. Floor and clip, never round and never wrap.
- Semi-transparent second ray (M7.15, ADR 0046): `irsim.materials.surface_radiance`
  (ε L_B + ρ L_env + τ L_behind, per-pixel closure to 1e-6) and `MaterialTable.properties_for`
  (τ from the packed column, ρ derived, sky pixels blackbody-equivalent); stage 1 and `run_frame`
  take an optional `radiance_behind` plane, which rides the existing G-buffer contract and is
  float16-refused. With no such plane L_behind = L_env, so the form collapses to M7.13's exactly
  and opaque scenes and goldens are bit-identical; supplying one without an environment model
  raises rather than silently rendering a transparent material as opaque. The committed
  windshield's dL/dL_behind is 0.0 in LWIR, 0.02 MWIR, 0.70 SWIR, 0.77 NIR.
- `scripts/stage_own_hunk.sh`: stage only your own edit to a file several sessions are editing at
  once. It three-way merges your change (snapshot -> worktree) onto HEAD, so another session's
  *committed* change is a no-op instead of a failed patch, and a same-line collision conflicts loudly
  rather than silently picking a side. Snapshots are keyed by `$STAGE_OWN_HUNK_ID` so two sessions do
  not share a baseline. `tests/unit/test_stage_own_hunk.py` drives the real two-session collision,
  including the one blind spot it cannot cover (an uncommitted edit made after your snapshot).
- Warp stages 2 and 3 (roadmap M10.5): the atmosphere and the optics run on the device as twins of
  `irsim.pipeline.atmosphere` and `irsim.optics.stage`. One atmosphere kernel serves the grey M8.1
  path, MS.1's multi-term exponential sum and the constant-tau L1 fallback, because the class weights
  sum to 1 and the per-term path radiance collapses to (1 - tau) L_air. The optics kernels convolve
  the supersampled buffer with the optical PSF *before* the block-mean downsample and then apply
  Omega_eff tau_opt cos^4 A_d + Phi_self, with the aperture factor and Phi_self computed by
  `irsim.optics` on the host and passed in -- non-negotiable #5 holds across the second
  implementation, and `tests/unit/test_aperture_guard.py`'s scanner is pointed at the kernel file by
  name to keep it that way. Measured on an RTX A6000, `cuda:0` and Warp `cpu`: stage 2 within 1.9e-7
  relative / 0.015 mK on every branch (d = 0 and d = inf included, sky pixels bit-identical), stage 3
  within 8.5e-7 through a 53x53 PSF on the 4x step edge against a 1e-5 budget, and a +1 K housing step
  reading +87.43 mK on both paths -- the +87 +/- 2 mK the roadmap predicted, now a test.
- `irsim.pipeline.optics.optics_stage`: the plane-dict form of stage 3 (`radiance` on the k-x grid ->
  `flux` on the detector grid), which is what the Warp twin is compared against. It delegates to
  `apply_optics`; the only thing it adds is the housing-radiance lookup `run_frame` already did.
- `irsim.atmosphere.layered.LayeredAtmosphere.air_radiance`: L_B(T_air) at the surface from the
  model's own LUT, so a fast path can take the same value the oracle uses instead of recomputing it.
- `tests/unit/test_tier3_atmosphere.py` and the solar-path stub (M8.8, ADR 0051): every preset is
  checked for the §7.2 band orderings, the humid crossover (τ_LWIR < τ_SWIR in humid clear air) and
  the fog reversal from the weather alone, a target at T_air is distance-invariant to 1e-9 on a
  horizontal path for both atmosphere models, and a 300 m path at 10° reads within 0.1 K of the same
  length horizontally while 5 km differs by more than a kelvin. New `AtmospherePreset.solar.
  zenith_transmittance` (atmosphere schema v2, all seven presets, ESTIMATED) with
  `irsim.atmosphere.extinction.solar_transmittance` / `airmass`: τ_sun(θ) = τ_zenith^{sec θ},
  exact at τ_zenith² for 60°, refused beyond 85°.
- Warp is reachable without booting Kit (ADR 0014 addendum). `irsim_isaac.env.ensure_warp_on_path`
  locates the `omni.warp.core` extension in the Isaac build's `extscache` (or `$IRSIM_WARP_PATH`) and
  puts it on `sys.path`; it never shadows an already-resolving Warp. ADR 0014 had recorded Warp as
  "importable only inside a running Kit", which was a `sys.path` artefact, not a runtime requirement.
  The M10.4 equivalence harness now runs on `cpu` and `cuda:0` in 1.4 s from a bare interpreter
  instead of behind a ~35 s Kit boot, and `gpu`-marked integration tests are selected on Warp being
  available rather than on Isaac Sim.
- `irsim.validation.aerial_scene` + the `gbuffer_aerial` fixture and `tests/unit/test_tier3_sky.py`
  (MS.8): a synthetic sky-background G-buffer -- exact per-pixel ray elevation for a pitched pinhole,
  horizon, optional 1/f^β cloud coverage, targets rasterised above one native pixel and handed to
  MS.6's analytic injection below it -- plus the Tier 3 checks it exists for: the rendered sky
  profile is an identity on MS.2 within 1 mK and scale-free to 1 %, cloud reads its base temperature
  > 20 K above clear sky with real spatial structure, a sub-pixel target's excess follows
  φ τ(R)(L_t − L_air)/R² exactly against a grey atmosphere and decays measurably slower against the
  layered one (the occulted sky dims with range), and a 2 px target's footprint aliases with
  sub-pixel phase while the optical PSF lowers the peak and conserves the flux.
- Aerial thermal bridge (M10.18, ADR 0060): `irsim_isaac.pipeline.aerial_bridge` couples the M6.6
  target solvers to the renderer through a float32 table indexed by instance id -- 317.25 K reads
  back within 10 mK and a 50 mK pair stays resolved, where the rejected fp16 emissive path would
  collapse it. Thermal tick 1 Hz with per-frame interpolation (error bound computed, ~3 uK for
  tau = 900 s); background pixels take MS.2's `T_sky(theta)` per ray, and `T_ground` below the
  horizon where the sky model is undefined and extrapolating it would invert silhouette contrast.
- Warp stage 1 (roadmap M10.4, ADR 0061): `irsim_isaac.pipeline.warp_stages` runs band radiance on the
  device as an op-for-op twin of `irsim.pipeline.radiance` (float32 LUT index/clamp/interp, ε = 1 under the
  sky mask, host guards for every CPU refusal), with the LUT and ε₀ table uploaded once and cached by
  content. `tests/integration/test_kernels_vs_reference.py` is the CPU-vs-GPU equivalence harness every
  later stage registers in (`EQUIVALENCE_STAGES`): stage × fixture × {cuda:0, Warp cpu} at ≤ 1e-4 / 5 mK;
  measured ≤ 2 ulp on CUDA and bit-identical on the Warp CPU device. The `isaac` extra is now empty: Warp
  comes from the `omni.warp.core` Kit extension and a pip copy would shadow it.
- Material-ID transport for the Isaac path (M10.2): `irsim_isaac.pipeline.material_ids` (instance
  id → prim path → M7.17 material id by exact integer lookup, UNMAPPED mask, magenta display
  overlay) and `irsim_isaac.pipeline.materials_usd` (stage walk for bindings, `class` semantics and
  the `thermal:material` override). `scripts/audit_materials.py` gains `--stage` / `--dump-prims`.
  Measured: the id channel must be `instance_id_segmentation`, because `instance_segmentation`
  gives a distinct id only to *labelled* prims and collapses the rest into one id (ADR 0014
  addendum) -- four of five test prims would have shared a material with nothing raising.
- Isaac geometry AOVs assembled into the M0.6 `GBuffer` (M10.1): `irsim_isaac.pipeline.gbuffer_isaac`
  (`AovReader` + engine-free assembly of `distance_m`, `normal_dot_view` against the per-pixel ray,
  `normal_dot_up`, V_s = occlusion·(1+n·up)/2, `sky_mask`), `irsim_isaac.geometry_probe` and
  `scripts/probe_isaac_geometry.py`. Sphere cos θ matches the closed form to 1e-5 engine-free and
  0.01 in-sim; tilted-quad ray length to 1 cm; plates read V_s 1.0/0.5/0.0 ± 0.05.
- Four-material aerial library and target thermal signature (MS.7, ADR 0072): `configs/materials/`
  gains `painted_composite`, `carbon_fibre`, `aircraft_aluminium_painted` and `propeller_rubber`
  (`source: literature`, scalar `emissivity_per_band`, opaque, ρ derived -- CLAUDE.md #4 closure
  verified in all four bands before and after the M7.18 float32 packing).
  `irsim.thermal.aerial`: `HeatSource` with ΔT = ΔT_max u^n and the MOTOR/ESC/BATTERY presets
  (45/30/15 K at full throttle, n = 2 ohmic, magnitudes ESTIMATED), `heat_source_solver` and
  `airframe_solver` returning M6.6 `PrescribedSolver`s over the scene's shared `WeatherSeries`,
  and `refine_nodes` bisecting the schedule grid until linear interpolation holds the analytic law
  to 1 mK. `irsim.validation.aerial`: `AerialTarget`, `target_contrast` and
  `zero_contrast_elevation` -- for T_air = 300 K, ε = 0.9, V_s = 1, R = 1 km under
  `us_standard_clear` the contrast is +64 K at zenith and crosses zero at 1.26° (1.97° at ε = 0.8,
  0.78° at ε = 0.95, and **never at ε = 1**, which is what shows the inversion is the reflected
  cold sky rather than a path-radiance artefact).
- `irsim.atmosphere.cloud` + `SkyModel` cloud methods (MS.3): LCL base from the shared weather (Espy,
  `dew_point_k` in humidity), T_base by the preset lapse rate, ε_cloud = 1 − τ_cloud (τ authored in the
  environment preset's new `clouds:` block, schema v2), seeded 1/f^β structure with an exact-coverage
  threshold and a PSD-slope self-test; `radiance_field` / `apparent_temperature_field`. ADR 0070.
- `irsim.pipeline.point_target` (MS.6): `PointTarget`, `fill_fraction`, `excess_radiance` (per-class
  τ_k(R)[L_t − L_beyond,k] with the layered atmosphere; grey and no-atmosphere forms), `excess_power` via
  the single aperture factor, bilinear `splat`, `run_frame(..., point_targets=)`;
  `LayeredAtmosphere.class_transmittances / sky_beyond(_per_class)` and `ExponentialSum.path_radiance_per_class`.
  ADR 0071 records the measured rasteriser flux error vs size behind the 1 px handoff.
- Stage 1 reflected environment term (M7.13): `band_radiance(..., l_env=)`, `irsim.pipeline.environment`
  (`sky_view_factor = occlusion·(1+n·up)/2`, `environment_radiance` from the SkyModel tilt LUT and the
  ground mode), `PipelineConfig.sky`; scene schema v2 adds `environment_preset` and the Scene builds the
  layered atmosphere and per-band sky models on the same weather. ADR 0045.
- `irsim.atmosphere.sky.SkyModel` (MS.2): clear-sky elevation LUT over the layered column emission
  (fast path within 0.5 K), tilt LUT with an analytic azimuth kernel (`effective_radiance(β)`,
  `effective_radiance_from_sky_view(V_s)`), cloud blend to L_B(T_air), `broadband_downwelling` via M6.5,
  `fit_cos_q` deriving the §5.3(a) form and its error (ADR 0044).
- `irsim.atmosphere.layered` (MS.1): `LayeredAtmosphere` -- exponential sum over spectral classes per
  band (Planck-weighted class weights from the sensor response; water/air scale heights; horizontal
  200 m anchored to the grey preset; opaque CO₂/H₂O cores on top), analytic slant-path transmittance,
  per-class optical-depth quadrature for path radiance, `sky_radiance = L_path(∞, θ)`,
  `apparent_sky_temperature_k`; `fit_exponential_sum`, `exponential_sum_from_piecewise`;
  `scripts/validate_sky_r13.py`. Stage 2 accepts a `LayeredAtmosphere` (per-term horizontal form).
  Preset profile gains `aerosol_scale_height_m`, `air_scale_height_m`, `tropopause_m` defaults. ADR 0071.
- `irsim.materials.mapping` (M7.17): `MaterialResolver` (override → semantic → name pattern → loud
  miss with id 0), `configs/materials/mapping.yaml`, `audit()` and `scripts/audit_materials.py`
  (coverage %, misses grouped, non-zero exit below the 95 % threshold; ADR 0047).
- `irsim.config.environment` (M7.11): `EnvironmentConfig` (sky ΔT_clear per band + q, ground mode,
  solar glint model, night airglow with an explicit unit key / k_cloud / moon), ranges validated per
  regime against §5.3/§5.5, weather-like keys refused; presets `configs/environments/{clear_dry,
  humid, overcast}.yaml`.
- `MaterialTable.from_library` / `save` / `load` (M7.18): packed float32 per-band columns (ε₀, ρ, τ,
  Level-B (a, p) or (0, 4) placeholders, roughness, thermal), id 0 = UNMAPPED, ids stable across loads,
  `.npz` + sidecar with the library hash, `StaleMaterialTableError`; float16 refused.
- `irsim.config.materials` (M7.2): `MaterialConfig` -- one YAML per material with a `material:` block,
  `source` required, exactly one of spectral/scalar ε or ρ authored, optional τ/roughness per band,
  angular model union; `irsim.materials.library` derives the third quantity per band (ADR 0010 band
  average for spectra) and refuses ε + τ > 1; `irsim.materials.spectra` property-spectrum loader;
  `configs/materials/` six §16.2 materials; the CLAUDE.md #4 closure library walk. ADR 0040.
- `irsim.pipeline.atmosphere` + stage 2 in `run_frame` (M8.6): τ(d)L + (1−τ)L_B(T_air) on the k× grid
  from `PipelineConfig.atmosphere` at `PipelineState.t_s`; sky pixels bit-identical; `tau_override`
  L1 fallback; identity without an Atmosphere (goldens unchanged). ADR 0050.
- `irsim.config.scene` (`SceneConfig`: weather file, atmosphere preset, site, aware start, newton/
  prescribed targets) and `irsim.scene.Scene` (M6.17): loads the weather once, injects the same
  `WeatherSeries` into the `Atmosphere` and every target solver, refuses a consumer holding another
  weather object; `configs/scenes/sky_target_clear_day.yaml` sample.
- `irsim.atmosphere.Atmosphere` (M8.5): preset + the shared `WeatherSeries` (+ band LUTs) →
  `state(t)` with T_air, w, V, γ per band, L_air per band and a regime-mismatch flag; `transmittance`,
  `air_radiance`, `apply` (the per-pixel Beer–Lambert kernel at time t). A path raises TypeError.
- `irsim.thermal.solvers` (M6.6): `TemperatureSolver` protocol (`advance(t, dt)`, `temperature()`,
  `state`), `PrescribedSolver` (schedule, nodes exact, no extrapolation) and `NewtonCoolingSolver`
  (exact exponential update, midpoint ambient; accepts the shared `WeatherSeries` as ambient and
  exposes it as `.weather`). Scope limit documented: scripted actors and residual heat only.
- `irsim.thermal.longwave` (M6.5): Q_LW↓ = V_s·ε_sky·σT_air⁴ + (1−V_s)·σT_surround⁴ with Brunt (default)
  or Idso clear-sky emissivity and a linear cloud blend to 1; fed by the shared `WeatherSample` (ADR 0035).
- `irsim.thermal.solar` (M6.4): NOAA sun position (elevation, azimuth, declination, equation of time,
  solar noon; vectorised over Julian days), ENU sun vector, `solar_loading = S·max(0,n·s)·DNI + V_s·DHI`,
  `absorbed_solar = α·Q` (ADR 0034).
- `irsim.thermal.convection` (M6.3): h = max(c|ΔT|^{1/3}, a + b v_rel^n) with v_rel = |wind| + |vehicle|
  (ADR 0033); h(28 m/s) = 62.5 vs 5.0 parked.
- `irsim.thermal.weather_io` (M6.2): project weather CSV (`# irsim weather v1`, unit-suffixed columns,
  ISO-8601 UTC; `t_air_c`/`rh_percent` converted once at load), bit-exact write→read,
  `synthetic_clear_day`; `data/weather/clear_midlat_summer_48h.csv` (SYNTHETIC) regenerated by
  `scripts/generate_weather.py` and pinned by a test.
- `irsim.thermal.weather.WeatherSeries` / `WeatherSample` (M6.1): immutable validated hourly weather
  (T_air K, RH fraction, wind, cloud, DNI/DHI, visibility, precip), linear `at(t)`, extrapolation
  refused, `content_hash`; never opens a file (ADR 0032). `SOLAR_CONSTANT_W_M2 = 1361` in constants.
- `irsim.config.atmosphere` (`AtmospherePreset`: per-band γ₀/β/aerosol ratio, regime, profile,
  provenance; rejects weather-like keys at any depth — CLAUDE.md #6), seven presets in
  `configs/atmospheres/` fitted to the §7.2 table, `irsim.atmosphere.library`,
  `irsim.atmosphere.extinction` (Koschmieder `3.912/V` on total visible extinction, per-band ratios,
  droplet regime for fog; `KOSCHMIEDER` in constants). ADR 0049 resolves the haze row (its τ values
  imply 1.5 km visibility, not 5 km; spec issue T18).
- G-buffer contract: optional bool `sky_mask` plane (renderer hit nothing). Stage 1 treats masked
  pixels as blackbody-equivalent (ε = 1, the apparent sky temperature is what the plane carries) and
  ignores their material id, so the renderer's background id 0 is no longer confused with UNMAPPED;
  stage 2 (M8.6) will pass them through. Agreed with the Isaac lane (DistanceToCamera = +inf, id 0).
- `irsim.atmosphere.beer_lambert` (τ = e^{−γd}, path radiance, `apply_atmosphere`, the L1 `tau_override`),
  `irsim.atmosphere.humidity` (Magnus/Bolton e_s, absolute humidity with the 216.7 factor derived from
  `R_V_WATER`, γ_mol = γ₀ + βw; RH is a fraction), `irsim.atmosphere.spectral` + `scripts/
  validate_atmosphere_band_average.py` (exact spectral τ_B, curve of growth, grey-fit error: the ADR 0048
  evidence).
- `irsim.optics.mtf` (diffraction, detector sinc, motion, Gaussian, cascade, cut-off and Nyquist) and
  `irsim.optics.psf` (`optical_psf` = diffraction·Gaussian at the supersampled pitch, `apply_psf` FFT
  convolution); the PSF is now the first step of `apply_optics` and `PipelineConfig` builds it from the
  band's R-weighted mean wavelength (`psf_enabled=False` skips it). `irsim.validation.mtf.slant_edge_mtf`
  (ISO 12233-style) and the Tier 2 MTF bench: 0.31 ± 0.05 at Nyquist for the Boson, supersampled path
  aliases while a native blur does not (ADR 0059). Goldens regenerated with the PSF in the chain.
- README: tier-promotion rules (what evidence moves a row to T2/T3/T4; T5 is external), the first-image
  note, and status rows for the first LWIR camera through the ISP (isp 🟢, pipeline T2–T3).
- `run_frame` emits `display8` (RGBA8 through the isp block) and `isp_hash`; `irsim.io.png` (stdlib PNG
  writer for the human look); end-to-end goldens `boson_ramp_*` and `boson_hot_patch_*` (radiance 1e-5,
  T_app 1 mK, DN16 ±1, DN8 ±1) keyed on config hash + NumPy version; Tier 3 phenomenology through the
  pipeline: a 600 K patch entering the frame halves the pedestrian's 8-bit contrast under linear AGC
  while DN16 contrast is unchanged, and plateau 0.012 keeps > 50 % of the background std.
- `irsim.isp.display.run_display_branch`: AGC (linear / plateau / `none` = exact bit shift) → gamma → DDE →
  polarity → palette, pure per-frame, float32 rounding points R1–R4 documented, isp config hash in the
  output (ADR 0031). `irsim.isp.nuc.TwoPointNuc`: §11.2 calibrate/apply, ideal mode, corrected cold
  blackbody = 0 (ADR 0021 level).
- `irsim.isp.agc`: `agc_linear` (histogram percentiles with in-bin interpolation, gamma; ADR 0027) and
  `agc_plateau` (P = plateau·N per bin over 2^bits bins, exclusive CDF normalised to the occupied range,
  P→0 = rank map of occupied bins; ADR 0028). Hot exhaust at 5 % of pixels collapses the linear-AGC
  background std to < 20 % while plateau 0.012 keeps > 50 %.
- `irsim.isp.dde`: 3×3-box unsharp mask (±gain/6 step overshoot; transfer |1+g(1−K̂)|²; ADR 0029).
- `irsim.isp.palette`: gray/ironbow/rainbow/lava/arctic tables, polarity, `to_display8` → RGBA8 (ADR 0030).
- `irsim.noise.stage.NoiseStage` (correlated V/H/VH/TV/TH/T terms scaled from the detector's σ_TVH by the
  configured ratios, unit fixed patterns per sensor) and `measure_from_uniform_scene`; `run_frame` now runs
  the seeded detector response and the noise stage before the ADC (`PipelineConfig.from_sensor(...,
  sensor_seed, noise_enabled)`). Tier 2 benches: DN-domain two-blackbody NETD within 10 % of the anchor,
  3-D ratios recovered within the ADR 0023 floors, bit-identical replay, frame 500 reachable directly,
  sensors uncorrelated. Golden `boson_noise_cube_8x64x64` keyed on config hash + NumPy version.
- `irsim.validation.noise` (ME.2a): leakage-corrected NVESD `decompose_3d` (random-effects mean squares,
  uint16 promoted, negatives clipped with raw variances kept), `spatial_psd` with radial profile and the
  k_v = 0 / k_h = 0 striping lines, `temporal_psd`, `compare_psd`. Boson ratios recovered from 200
  frames of 64×64 within the stated sampling floors; white cube shows no directional terms (ADR 0023).
- `PhotonDetector.response` (Poisson shot + dark + background and hashed read noise in electron space,
  then DN) with Arrhenius `dark_current_a` (InSb/InGaAs band gaps in `constants.py`), and
  `MicrobolometerDetector.response` (static transfer + anchored σ in signal space, then DN);
  `DetectorFrame(signal_dn, dn, sigma_dn)`; `measured_netd_k` two-blackbody bench; golden
  `boson_sitf_dn` (ADR 0026). NETD(300 K) within 10 % of the anchor; bolometer NETD(373)/NETD(300) =
  0.576 while the DN noise std is scene-independent.
- `irsim.detector.netd` / `figures_of_merit`: NETD predictor NETD(T) = σ_total/(∂S/∂T) for both detector
  classes (∂S/∂T through the transfer, +1 aperture form), NEP, D*, the 4F² datasheet form as a labelled
  conversion (1.25 at F/1), ENBW = 1/(4τ_th) and the sampled-IIR ENBW, Johnson and temperature-
  fluctuation floors reported (ADR 0024). NETD(373)/NETD(300) = 0.576 for 7.5–13.5 µm.
- `irsim.detector.anchor.anchor_noise`: solves the scene-independent Gaussian σ so the predicted NETD at
  300 K and `netd_ref_f_number` equals `netd_mk_at_300k`; Poisson terms never rescaled; unattainable
  targets raise; NETD(F/1.4)/NETD(F/1.0) = 1.768 (ADR 0025).
- `irsim.noise.three_d`: `Sigmas7` (unit-agnostic, from ratios), `FixedPattern.generate` (V, H, VH once
  per sensor) and `synthesize_frame` = T + V + H + TV + TH + VH + TVH from the counter-based streams.
  Axis conventions tested; pooled variance = Σσ² within 3 %; row-mean variance identity within 10 %.
- `irsim.noise.seeding`: counter-based per-element hashing (`stream_key`, `hash_u64`, `hash_uniform`,
  `hash_normal`, `field_normal`: splitmix64 over (sensor seed, sensor frame index, stream, pixel index),
  Box–Muller in float32) so any pixel is regenerable alone, traversal-independent and reproducible bit-
  for-bit by a Warp kernel; `NoiseStream` values frozen; PCG64 `noise_rng` / `sensor_rng` kept for Poisson
  draws and the bad-pixel map (ADR 0022).
- Noise schema (v4): `Ratios3D.as_vector()` / `total_over_tvh()` (1.0604 for the Boson ratios),
  `noise.netd_ref_f_number`, `noise.bad_pixel_type_mix` (sums to 1).
- Isaac Sim gate spike (roadmap M2.1–M2.4, ADR 0014): `irsim_isaac.probe` (environment report; 64-quad
  emissive temperature ramp; AOV dtype, resolution, exposure and distance semantics; segmentation-id and
  float32 position transport) and `irsim_isaac.spg_probe` (SPG pass-through, cross-frame state, LUT
  delivery, checkpointed per experiment) with `scripts/probe_isaac_environment.py` and
  `scripts/probe_isaac_spg.py`. `tests/integration` boots one headless Kit per session (argv hidden from
  Kit's parser, app closed at interpreter exit); `test_environment.py` and `test_isaac_transport.py` pin
  the measured facts, including the negative one. **Outcome:** every colour AOV is float16 →
  temperature is transported as instance ids + float32 geometry, never as emission; SPG holds no state →
  stateful stages stay in Warp; LUTs are baked into the `.cu`. `docs/spec-issues.md` gains T16 (the build
  is 6.1.0-rc.26) and T17 (§13.1/§13.3 emission transport not viable).
- `irsim.radiometry.band_average`: `band_average(response, s(λ), T_ref, form)` = ∫R s B dλ / ∫R B dλ, the
  only sanctioned route from ε(λ)/ρ(λ)/τ(λ) to per-band scalars; linear so Kirchhoff closure survives
  (1e-9); the accepted grey-in-band error (~2e-3 for a 0.1 slope, 300→600 K) is measured (ADR 0010).
- `irsim.radiometry.band_integration`: the reference oracle — `band_radiance`, `band_photon_radiance`,
  `d_band_radiance_dT`, `d_band_photon_radiance_dT` by composite Simpson on an odd, edge-aligned 0.01 µm
  grid (NumPy only, float64, vectorised over T). Top-hat vs closed form: 7.5-13.5 um @ 300 K: 3.3e-11 rel, 0.000 mK; 3.0-5.0 um @ 300 K: 3.1e-11 rel, 0.000 mK; 3.0-5.0 um @ 500 K: 5.7e-11 rel, 0.000 mK; 0.9-1.7 um @ 300 K: 1.5e-06 rel, 0.016 mK.
  Derivative vs FD < 1e-5 over 200–1000 K; dLb/dT(373)/dLb/dT(300) = 1.736 for 7.5–13.5 µm.
- `irsim.radiometry.spectral_response`: R(λ) file contract and loader (two-column CSV, `#` provenance,
  µm strictly increasing, R in [0, 1], peak == 1 ± 1e-6 asserted never renormalised, zero outside support,
  half-power points, resampling) and `irsim.radiometry.band.Band` (config edges must agree with the file's
  half-power points to 0.5 µm). `data/spectra/responses/boson_vox.csv`: ESTIMATED Boson VOx curve,
  7.5–13.5 µm with ±0.25 µm raised-cosine edges (ADR 0009).
- `SIGMA_Q` (4π ζ(3) k³/(h³c²), derived in `constants.py`), `fractional_photon_exitance` and
  `band_photon_radiance_tophat` (§3.2 a): the closed-form oracle for the photon band table. Photon
  Stefan–Boltzmann closes to 1e-6; `Lb_q(7.5–13.5 µm, 300 K) = 2.915e21` known answer; `Lb/Lb_q = hc/λ_eff`
  with λ_eff inside the band.
- `d_spectral_photon_radiance_dT` (§3.4, §9.4): photon-form thermal derivative sharing the energy form's
  expm1/clip guards. Identity `dL_q/dT · hc/λ == dL/dT` to 1e-12; central FD < 1e-6; finite over
  200–2000 K × 0.4–20 µm.
- GPU-free CI: `.github/workflows/check.yml` runs `make check` on plain CPython 3.10 and 3.12; `make ci`
  reproduces it locally in a `python3.10` venv. Interpreter matrix recorded in ADR 0002.
- `docs/spec-issues.md`: the 37 physics and 15 tooling issues found in the spec, with proposed
  resolutions and applied status, for the spec owner (roadmap open question 10).
- `irsim.config.bands`: canonical band ids with §12.1 nominal ranges and per-band defaults,
  `band_id_for` by maximal overlap (< 50 % rejected), optional `band.id` validated against the edges,
  and `enabled_illumination_terms(regime)` so kernels never switch on a band name (§5.2, S17).
- `irsim.config.loader`: `load_sensor_config` (data paths resolved against `data_dir` /
  `$IRSIM_DATA_DIR` / `<repo>/data`, missing files named), `dump_sensor_config`, `config_hash` (canonical
  JSON, files by content hash) and `band_hash` (band block + spectral bytes: the LUT key). Every numeric
  and enumerated leaf of the Boson file is tested to move the hash (ADR 0008).
- `irsim.config.sensor.SensorConfig`: pydantic v2 schema for §12.2 (frozen, unknown keys rejected,
  units-in-names, discriminated bolometer/photon FPA, regime-vs-wavelength and 3-D ratio checks, derived
  pixel area / Nyquist / HFOV / frame period / DN max, deliberately no aperture factor). Boson file loads
  exactly; HFOV 30.7° vs datasheet 32° recorded (ADR 0007). `types-PyYAML` added to dev extras.
- `irsim.config.gbuffer.GBuffer`: the frozen G-buffer contract (`temperature_k`, `normal_dot_view`,
  `distance_m`, `material_id`, `sky_view_factor`; optional `encoded_t`, `motion_px`, `semantic_id`,
  `radiance*`). float16 refused on temperature/encoded/distance/radiance planes, upcast elsewhere;
  `encoded_t` checked against `temperature_k` to 10 mK; id 0 reserved as UNMAPPED.
- Synthetic G-buffer fixtures: `gbuffer_two_material`, `gbuffer_sphere` (analytic cos θ to 1e-6, reaches
  grazing), `gbuffer_step_edge` (1024², 5.5° tilt, ideal two-level), `gbuffer_moving_edge` (8 frames,
  2 px/frame with `motion_px`); all fixtures now carry `encoded_t` and use material id 1+.
  `tests/unit/test_gbuffer_schema.py` freezes the key set the Isaac adapter must emit.
- `irsim.radiometry.encoding`: float32 temperature encode/decode `c = (T − 200)/800` with the constants
  defined once in `constants.py` (plus the LUT grid constants); float16 and integer inputs refused; fp16
  coarse/fine pair codec as the §13.3 fallback. Round trip 0.05 mK; negative controls show 125 mK (raw
  kelvin) and ≥ 40 mK (encoded) through fp16 (ADR 0006).
- Stefan–Boltzmann identity test at 1e-6 relative via Simpson quadrature plus the closed-form tail
  (measured 3e-11); `fractional_exitance` gains a small-x Bernoulli branch so it is double-precision
  accurate in the Rayleigh–Jeans tail, and it and `band_radiance_tophat` now reject metre-valued
  wavelengths and Celsius temperatures (ADR 0005).
- Layering guard hardened in both directions: `src/irsim` may not import engine modules (now including
  `carb`), `irsim_isaac`, or the ML/imaging stack; default-gate tests may not import engines; glue files
  must keep engine imports inside functions. Scanner self-tested on synthetic offending modules.
- `irsim_isaac.env` (`has_isaac`, `has_warp`, `require_*`, force-off env flags) and
  `tests/integration/conftest.py`, which auto-marks and skips the directory without Isaac Sim.
- `isaac` optional extra (`warp-lang`, floor provisional until M2.1); mypy now covers `src/irsim_isaac`.
- Golden fixture helper (`tests/golden/conftest.py`, ADR 0004): `.npy` + JSON sidecar with config hash;
  STALE (inputs changed) reported distinctly from FAILING (values changed); actual array dumped to
  `outputs/golden/` on failure; float16/float64 refused on disk; `--update-golden` registered so
  `make golden-update` works. `tests/golden` now runs in `make test`.
- Project scaffold: physics specification (`docs/physics-model.md`), `CLAUDE.md`, skills, build tooling.
- `irsim.radiometry.constants` — physical constants with sources.
- `irsim.radiometry.planck` — spectral radiance (energy and photon forms), fractional exitance,
  thermal derivative. Analytic identity tests passing.
- Layering test enforcing that `src/irsim` imports no engine modules.
- ADR 0002: NumPy ≥ 2.0 floor and Isaac Sim's bundled Python as the project interpreter.
- `docs/roadmap.md`: phased implementation roadmap — 15 milestones, 157 one-commit steps with
  verification tolerances, risk register, ADR backlog, non-negotiable enforcement map, spec issue list.
- ADR 0003: sky targets first, validated against public anti-UAV thermal data (no camera); adds the
  evaluation-harness (ME) and sky/clouds/MTF/point-target (MS) milestones; revised after a four-critic
  adversarial review (72 findings applied); the twelve subsystem maps under `docs/maps/`.
- `irsim.detector.lowpass` (M9.1, ADR 0052): `BolometerLowPass` -- the §9.2 membrane thermal lag as a
  stateful per-pixel float32 IIR, `S_n = S_{n-1} + (S_ideal - S_{n-1})(1 - e^{-dt/tau_th})`, applied to
  the ideal signal *before* noise so the M4.6 NETD anchor survives; starts settled (the first frame
  adopts its input) with `reset()` for a cold start. `alpha_for`, `responsivity_rolloff(f, tau)` (the
  continuous `1/sqrt(1 + (2 pi f tau)^2)`, not the sampled IIR's transfer function) and
  `trailing_decay_length_px` = `v tau_th/dt` -- the moving-edge tail length that spec issue S8's
  "0.6 frames" phrasing conflates with `tau_th/dt`. Verified against the closed-form step response to
  1e-6, against RK4 of the membrane ODE to 1e-4 over 100 varying frames, and by measuring the tail of
  a rendered moving edge to within 2 %; the photon path is asserted memoryless (§15 T3).
- `irsim.detector.fpa_thermal` (M9.2, ADR 0053): `FpaThermalModel` -- the §9.2 FPA node
  `C dT/dt = P - h(T - T_amb)` in its identifiable form (tau = C/h and DeltaT_self = P/h are authored;
  C, h and P separately are not observable), RK2 (Heun) on the thermal tick, with the housing node's
  three modes -- `fixed` (TEC-pinned, no drift), `ambient` (zero-tau), `coupled`. Ambient comes from
  the scene's shared `WeatherSeries` *or* an injected provider, never both and never neither
  (CLAUDE.md #6). `gain_of_t`/`offset_of_t` are the **raw, uncorrected** response, coefficients in
  ascending powers of (T - T_cal) from order 1 so `g(T_cal) == 1` and `o(T_cal) == 0` structurally;
  what survives correction stays `nuc.residual_*` (M9.6), so the drift is not counted twice.
  Steady state and the 63.2 % time constant verified to 1 mK / 1 %, per-step energy conservation to
  1e-6 against the C/h/P form, and a 5 K ambient rise shown to shift DN uniformly and not at all
  when TEC-pinned. Sensor schema -> v5 (`fpa_temp_mode`, `fpa_t_cal_k`, `fpa_gain_coeffs_per_k`,
  `fpa_offset_coeffs_dn_per_k`; the node is opt-in, so existing configs are unchanged). Golden
  sidecars regenerated for the new `config_hash` -- all twelve arrays bit-identical (ADR 0004).

### Changed
- Sensor schema v8: `nuc.shutterless_tau_s`, the scene-based-correction time constant that bounds a
  shutterless core's residual (M9.7, ADR 0057). Defaults to 120 s (ESTIMATED -- no published
  convergence time was available) and is unused outside `mode: shutterless`, so no existing config
  changes behaviour.
- Sensor schema v7: `noise.bad_pixel_rts_occupancy`, `bad_pixel_rts_dwell_frames` and
  `bad_pixel_rts_amplitude_dn` for the §10.4 flickering and blinking classes (M9.5a, ADR 0055).
  All three default, so existing configs are unchanged; all three are ESTIMATED, with ME.3's
  bad-pixel extractor the thing that should eventually set them.
- Sensor schema v6: `housing_temp_mode: coupled` now **requires** `housing_tau_s`, mirroring the
  rule ADR 0053 set for `fpa_temp_mode: coupled`. A lumped node with no time constant cannot be
  integrated, and defaulting the lag would put a number nobody authored into the drift the whole
  M9 chain rests on. `configs/sensors/flir_boson_640_lwir.yaml` declared `coupled` and authored
  neither parameter, so it gains `housing_tau_s: 900 s` and `housing_self_heating_k: 4 K` -- both
  ESTIMATED, no published figure for either, both candidates for ME.3's measured drift band to
  constrain. Golden references were regenerated for the config-hash change only: every stored
  array is bit-identical, because nothing reads the new fields until M9.8 wires the node in.
- `CLAUDE.md` now matches the repository it describes: the layout block gains `tests/conftest.py`
  (the synthetic G-buffer fixtures), `configs/environments/`, `docs/roadmap.md`,
  `docs/spec-issues.md`, `docs/maps/`, `.github/workflows/` and the `$IRSIM_DATA_DIR` override, and
  the command table gains `make ci` and `make golden-update`, records that `make test` runs the
  golden suite too and that `make typecheck` covers `src/irsim_isaac`, and states that every target
  honours `PYTHON=` (ADR 0002). The guidance was describing an earlier tree, which is the failure
  mode a project instruction file cannot afford.
- The reflected-environment tests no longer assume a fully overcast sky reads exactly `T_air`. The
  isothermal-enclosure test derives its enclosure temperature from the sky model (and pins the
  ground to it with `ground.mode: fixed`), and the cold-roof test computes a reference from each
  sky's own `effective_radiance`, so both assert the reflected term itself rather than a particular
  cloud-base convention. They pass under the pre-MS.3 and post-MS.3 cloud semantics alike.
- `docs/physics-model.md` spec fixes raised before coding (M0.10): §5.3(a) sky temperature now
  `cos^q(θ_zen)` (coldest at zenith, S1); §6.1 absorbed solar written `α_sol Q_sol` (S2); §12.2
  spectral-response path `spectra/responses/boson_vox.csv` relative to `data/` (T4).
- Boson YAML `spectral_response` now `spectra/responses/boson_vox.csv`, the canonical layout under the
  data root (spec issue T4); the file itself arrives with M1.3.
- Golden helper implementation moved to `tests/golden/golden_store.py` (conftest keeps the fixture) so
  its self-tests import it unambiguously.
- Makefile targets run through a `PYTHON` variable (`make check PYTHON=.../python.sh`); `make test`
  prints the ten slowest tests so the 30 s budget stays visible.
- NumPy floor raised from 1.24 to 2.0 (the Planck tests use `np.trapezoid`).
- ruff ignores `N802` alongside `N803`/`N806` (physics notation such as `d_spectral_radiance_dT`).

### Fixed
- `import irsim.thermal` failed outright on a clean interpreter: `irsim.thermal.weather` imported
  `irsim.atmosphere.humidity` at module scope, which runs `irsim/atmosphere/__init__` and lands back
  in the partially-initialised `weather` module. The whole suite passed only because pytest collects
  alphabetically and something imported `irsim.atmosphere` first, so the cycle was invisible until a
  single test file was run alone. The two humidity helpers are now imported inside the two
  `WeatherSample` properties that use them, breaking the cycle at its source; no atmosphere module
  needed changing. `tests/unit/test_import_order.py` imports each subpackage in its own subprocess
  so collection order can never hide this again.
- `SpectralResponse.resampled` snaps grid points within 1e-9 µm of the support edges: a grid built as
  `lo + k·dl` lands 2e-16 µm past the last sample and lost the endpoint (a 5 % error for SWIR at 300 K).
- `make check` is green on the scaffold: three files reformatted, one `Any` return in
  `irsim.radiometry.planck` typed explicitly.
