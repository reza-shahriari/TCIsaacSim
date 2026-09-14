# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **The quadrotor flies with propellers** (ADR 0081). `irsim_isaac.pipeline.rotor_isaac` turns a
  prim's transform into veils and `QuadrotorSpec.rotor_mounts` puts four above the motor bells;
  `scripts/render_quad_flight.py` grows `--no-rotors` for the before/after. **Nothing is authored**
  — no prim, no mesh, no material — because a spinning rotor is not geometry. That buys the
  sub-pixel and occlusion story for free and costs one honest asymmetry: the discs are in the
  infrared frame and *not* in the companion visible frame, which RTX renders from stage geometry
  there is none of.
- **Occlusion is a plane intersection, not a range comparison.** Each pixel's ray meets the disc
  *plane* and is compared with the renderer's depth. Using the disc centre's range would be wrong
  by up to a disc radius across the ellipse — 0.36 m at 20 m, about 18 px of arm drawn on the wrong
  side of the aircraft. The mask is meaningless outside the ellipse and does not need to be; a test
  asserting otherwise failed on arithmetic that was correct, and says so.
- **Two traps, both of which make a plausible picture out of deleted physics.** The sweep is taken
  over the *detector's* frame, not the time-lapse's six-second capture interval — that would turn
  three hundred revolutions into a perfectly uniform annulus with no banding at all. And rpm
  follows throttle by a **square root**, because thrust goes as rpm² and the profile's `u` is a
  fraction of maximum thrust (the same `u` ADR 0072's `ΔT_max u²` motor law reads); linearly, a
  hovering aircraft turns 40 % slow and its discs are filmed in the wrong regime.
- ESTIMATED: the blade's sky-view factor, taken as 0.5 for a blade seen mostly edge-on.
  `propeller_rubber` is ε = 0.95 in LWIR, so the reflected term is a twentieth of what leaves the
  blade and the choice is worth about a kelvin (21 tests).
- **What the first render found that no unit test did:** sky carries `distance_m = 0` in the
  G-buffer, not `inf` — the sentinel is zero so a consumer ignoring `sky_mask` still sees τ = 1.
  Read as a *distance*, zero is a surface at the camera, and the first `occlusion_mask` therefore
  masked **every pixel of the frame** and erased all four discs; the rendered frames were
  bit-identical with and without rotors. The test that should have caught it used `inf`, which is
  what the AOV reports *before* the adapter translates it — a plausible fixture rather than the
  contract, testing the one value that could not fail. Now parameterised over both, plus a sky-only
  frame, which is the demo's actual configuration.
- Measured in sim with the fix: **6444 native pixels change, peaking at +10.1 K**, and occlusion
  falls from the whole frame to the 15 786 airframe pixels really in front of a disc. The disc is
  brightest at its **root**, because local solidity `N c(r)/(2πr)` rises inward as the circumference
  shrinks while the chord does not — 3.2 % at three-quarter radius, 19 % just outside the bell.
- And it is **invisible at the demo's own display span**: ADR 0074 spans the target's nodes
  (18–62 °C), so a 273 K disc clips to black along with the 263 K sky, and plateau equalisation
  gives it no codes either. At `--span-c -15 40` the same frames differ by up to **45 display
  codes** and the four annuli are plainly there. Real in radiance, absent from the picture — the
  AGC lesson again, and which one you measure decides whether you believe the feature works.

- **Rotor veils reach the frame** (ADR 0081, stage 2c). `irsim.pipeline.rotor_veil` composites
  projected discs onto the k× radiance plane and `run_frame` takes `rotor_veils` alongside
  `point_targets`; a frame without them is bit-identical to before.
- It **composites rather than injecting an excess**, and the reason is not efficiency. MS.6's
  point-target form adds `φ τ (L_t − L_beyond)` because a sub-pixel target occults a *sky column*
  the plane does not separately carry. A rotor does not: by stage 2c the plane already holds the
  right background at every pixel — sky beyond the disc over part of it, the aircraft's own arm or
  motor bell over the rest. So `L = L_plane + α (L_blade,at-sensor − L_plane)` is exact for both at
  once, where the excess form would subtract a sky column that is not behind the airframe pixels.
  A test puts one rotor half over cold sky and half over a 300 K arm: the veil **lifts on one side
  and dips on the other**, from one operator in one pass.
- The blade takes stage 2's own atmospheric path (`blade_radiance_at_sensor` calls the same
  `apply_layered_gbuffer` / `apply_atmosphere` on a one-element array), so the veil and the pixels
  under it cannot disagree. Range pulls the disc **towards ambient**, which is not the same as
  fainter: against a 230 K sky in 288 K air, a 290 K blade at 3 km reads dimmer than at 20 m and a
  270 K blade reads brighter. That test was written the first way round only, and corrected.
- Coverage is built on the ellipse's bounding box — a full-frame float64 map per rotor is 40 MB on
  a 4× supersampled Boson frame, 160 MB for four — with a test asserting the windowed result is
  bit-identical to a full-frame composite, not merely close. Flux through the whole chain is
  conserved with the PSF on and off (23 tests).

- **The sea as a background** (MM.2, MM.3, ADR 0078). `irsim.atmosphere.sea` gives apparent sea
  temperature against depression angle the way `SkyModel` gives it against elevation: Cox–Munk slope
  statistics from the shared weather's wind, and a facet-tilt quadrature in which the emissivity and
  the reflected sky elevation move **together** — a tilted facet both presents less grazing incidence
  and swings the reflected ray.
- **The isothermal identity is exact (0 mK)** across depression 0.15–90°, wind 0–20 m/s and SST
  275–305 K. It holds for any emissivity, so it is the one test that catches a mis-weighted
  reflection — the failure a plausible-looking gradient would hide.
- Horizon geometry is spherical, not flat-earth: at 20 m the horizon is 0.1436° down and the range
  to it is **twice** what `h/sin δ` gives at the same angle. That factor of two lands directly on the
  atmospheric path length to every long-range maritime target.

### Fixed
- **Two predictions written into the roadmap for MM.3 were wrong, and the model disproved them.**
  (a) The sea profile is **not** a monotone ramp from SST at nadir to sky at the horizon. Emissivity
  rising with depression and the reflected sky cooling with elevation pull in opposite directions, so
  the surface radiance turns over at an **interior minimum 2–15° below the horizon** — 283.7 K against
  289.5 K at nadir for a 290 K sea. The cold band, not a ramp, is what a maritime target is seen
  against. (b) Nadir reads 0.53 K under SST, not the predicted 0.2 K, and that offset checks out
  against (1−ε)(Lb(T_sea) − L_sky(90°))/dLb_dT.
- The atmospheric path then pulls the far field back toward air temperature — at 0.3° depression
  (4.1 km) the observed sea is over 1.5 K warmer than its own surface radiance. So **range, not angle
  alone, sets maritime background contrast**, and the profile cannot be tabulated against angle for a
  moving camera. The roadmap row and the Tier 3 checklist entry are corrected rather than quietly
  re-scoped.
- The phenomenology-checklist claim that overcast collapses the sea gradient below 20 % of clear was
  also wrong: it closes only ~30 %, bounded by how far the cloud base sits below the SST. An overcast
  sea is not an overcast ground scene.
- Cox & Munk's component slope fits sum to 3.0e-3 + 5.08e-3 U, not their separately fitted isotropic
  total 3.0e-3 + 5.12e-3 U — two independent regressions on the same data, 0.8 % apart. A test pins
  the gap so nobody later "fixes" the components to add up and silently substitutes an isotropic
  assumption for the measured anisotropy.

- **Rotor discs reach the focal plane through the lens oracle** (ADR 0081).
  `irsim.optics.rotor.disc_ellipse` projects a disc pose to its image ellipse by projecting rim
  points with `irsim.optics.projection.project`, so the authored distortion and the off-axis scale
  come from the forward model ADR 0015 already calls the oracle — a barrel lens 8 m off axis at
  20 m shrinks the disc 4.3 % and pulls it 17 px back towards the axis, which an `f·R/Z` stand-in
  sees as nothing. The major axis is taken along `axis × view`, the only diameter square to the
  line of sight and therefore unforeshortened.
- The residual is **measured, not argued**: the projected conic is assumed centred with
  perpendicular axes, which is exact only orthographically, and a 0.71 m rotor seen 75° off its
  axis at 20 m has rim points up to 0.079 px off the ellipse — falling quadratically to 0.013 px at
  50 m, with the axis ratio within 3e-4 of `cos(tilt)`. The docstring first claimed "under a
  hundredth of a pixel"; the measurement said 0.079 and the measurement is what is recorded.

- **A spinning rotor as a time-averaged veil** (ADR 0081, §8.3/§9.2). ADR 0074 left propellers off
  the quadrotor and ADR 0077 claimed the motion-smear operator would be their home. **It is not:**
  a blade tip at 3000 rpm does 112 m/s, sweeping 109 pixels of *circular* arc per bolometer frame
  and wrapping the disc 1.7 times, where `apply_motion_smear` averages along a straight segment
  with at most 65 taps — a different operator, not a coarse one.
- `irsim.optics.rotor` computes what the detector actually reports: the **time average of an
  intermittent opaque occluder**, composited in radiance (`α L_blade + (1−α) L_behind`). The blade
  is not semi-transparent; α is the fraction of the window it stood in the way. Blending apparent
  temperatures instead reads a 3 % veil of 290 K blade over a 230 K sky as 231.9 K against the
  radiance blend's 233.1 K — under-reporting the disc in the direction that hides it.
- **One formula, both detector families, selected by the window alone.** Coverage is a running mean
  of blade passage over the angle swept during the integration: a shutterless bolometer's whole
  frame (300° = 1.67 blade spacings) draws a smooth annulus banded exactly 2:1 by one whole blade
  pass, and a 2 ms cooled integration (36°) resolves two arcs five times as bright. **The totals
  agree**, because a running mean cannot move the mean of a periodic function — verified from a
  frozen shutter to ten revolutions. That invariant is what makes the two pictures comparable.
- Viewing tilt enters only as the azimuthal mean of a pitched plate's projected area (`cos β`
  face-on, `(2/π) sin β` edge-on), so a feathered blade is invisible edge-on and a coarse-pitch one
  is not. A consequence worth stating: coverage per unit image area is **exactly tilt-invariant**
  until the disc is within `pitch` of edge-on. The module was first written asserting the opposite
  and the rasteriser contradicted it — the peak at 0° and 45° was the same number to three digits.
  The demo stage sits just past that crossover (75° of tilt against 18° of pitch, threshold 72°).
- A two-blade 28-inch prop is 3.2 % solid at three-quarter radius, lifting a 230 K sky by 3.1 K:
  the faint annulus ADR 0074 predicted, and what a solid disc would have buried. Engine-free and
  not yet wired to a stage — mounting four needs the pose-to-ellipse projection and an occlusion
  mask (69 tests).

- **Real water optical constants and band-effective Fresnel emissivity** (M7.5 water half, MM.1).
  `data/nk/water.csv` is Segelstein 1981's complex refractive index, 2.0–15.6 µm, 365 rows, fetched
  from omlc.org with the citation in the file header. `load_nk_table` treats that header as **data,
  not lint**: a table with no `# source:` is refused at load, because every radiometric result
  computed from an unsourced table is uncheckable.
- `band_directional_emissivity(table, response, cos θ)` reduces Fresnel ε(λ, θ) to the scalar the
  pipeline uses, through the one sanctioned band-average route. Over the Boson band water reads
  **0.9879 at nadir, 0.9532 at 60°, 0.6726 at 80°, 0.1085 at 89°**.
- **The band average earns its keep.** Against Fresnel evaluated at a "representative" 10 µm, the
  band-effective value differs by 0.004 at nadir and **0.039 at 80°** — worst exactly where a sea
  surface lives, and worth more than 2 K of apparent temperature against a 60 K sea-to-sky contrast.
  Over 7.5–13.5 µm water's n runs 1.28 → 1.16 and its k spans an order of magnitude, so no single
  wavelength represents the band.
- n and k are interpolated **separately**; a test pins that this differs from interpolating the
  derived emissivity, so a refactor cannot quietly switch to the sampling-dependent version.
  Extrapolation raises rather than holding end values.
- Water's hemispherical emissivity measures 0.945 against a normal 0.988, with §16.2's tabulated
  0.96 sitting between them — the scalar ambiguity ADR 0043 exists to resolve, now with numbers.

### Changed
- `data/nk/water.csv` uses **Segelstein 1981, not Hale & Querry 1973**, whose n = 1.218 at 10 µm the
  physics model quotes and the M7.5 roadmap row originally required. Hale & Querry's n is not freely
  available as a table (only its absorption coefficient is), and mixing one compilation's n with
  another's k is worse practice than using one whole dataset. Their k agree to 0.02 %; the n differ
  by 2 %, worth 0.002 in emissivity — five times under MM.1's tolerance. Recorded in the file header
  and ADR 0079.

- **Maritime lane planned as phase 1b** (ADR 0078, roadmap milestone **MM**, 8 steps). Sky → sea →
  ground, amending ADR 0003's two-phase split. The sea is an **analytic background**, not displaced
  water geometry: at 3 km a Boson pixel spans 2.6 m and contains thousands of independent wave
  facets, so one normal per pixel is not a coarse version of the right answer, it is a different
  quantity. The reflected sky is integrated over a Cox–Munk slope distribution instead.
- The geometry is the reason the lane exists: a camera 20 m up sees the sea at 5 km at **0.23°
  depression = 89.8° incidence**, where water's emissivity has fallen from 0.99 to under 0.15. Nearly
  all visible sea is a sky mirror, so a vessel reads **dark against near water and bright against far
  water**, with a contrast null in between. A scalar `T_ground` below the horizon — what the code does
  today — cannot produce that at all.
- Phase 1b promotes **M7.3** and **M7.5** out of phase 2; M7.4 already landed early for the same
  reason. Three phenomenology-checklist rows, three ADR backlog entries (0078 written, 0079/0080
  pending) and the dependency graph updated to match.
- ADR 0078 states one error it does **not** bound: wave shadowing and inter-facet reflection are
  neglected, and that approximation is weakest in exactly the near-horizon band a low camera cares
  about most.

- **Branch-safe complex Fresnel reflectance** (M7.4, §4.2 Level A [R1]).
  `irsim.materials.fresnel_reflectance(n, k, cos θ) → (R, R⊥, R∥)` and
  `directional_emissivity_fresnel = 1 − R`, written with the §4.2 A/B reparameterisation rather
  than a complex `sqrt`. The branch is not cosmetic: the rejected root of the same Z returns
  R > 1 — a test asserts exactly that — and picking it produces reflectance discontinuities
  across wavelength and instability near grazing incidence, which in the IR is not a corner case
  because k is large for many materials.
- Water at 10 µm (n = 1.218, k = 0.0508) reproduces the [R1] validation numbers: R(0) = 0.010180
  (published 0.01018), ε(60°) = 0.9612, ε(80°) = 0.6972. **That 0.26 collapse between 60° and 80°
  is the physics a sea surface is made of** — a Level C constant emissivity reports 0.99 at both,
  and §4.2 refuses Level C for water by name. It is the first step of the maritime lane.
- Confirms [R1]'s counter-intuitive result along both axes: with k > 0 there is **no total
  internal reflection knee** at any angle (R < 1 − 1e-4 for n = 0.5, k = 0.5, bounded slope), and
  an index sweep across the lossless critical angle is a square-root cusp, not a jump — tested by
  grid refinement, since a fixed tolerance there either fails on real physics or is too loose to
  catch a real discontinuity.

- **Within-frame motion smear** (ADR 0077, §8.3/§9.2). `mtf_motion` — |sinc(v·t_int·ξ)| — has been
  in the MTF cascade since M5 and **nothing ever called it**, so every frame this simulator has
  produced was sharp however fast the scene crossed it. The aircraft stage sweeps the boresight at
  34 °/s at closest approach and a Boson pixel is 0.049°, so the scene crosses **11 pixels in one
  frame period**. It is also one of the most obvious signatures separating real thermal video of a
  moving target from synthetic video of one.
- `irsim.optics.smear.apply_motion_smear` is **spatially varying**, which the scenes here require:
  under a tracking mount the target is stationary on the focal plane while the sky sweeps past, so
  a single convolution serves neither. Each pixel is averaged along its own motion vector. Applied
  between the PSF and the box filter — both are convolutions laid down during the integration, and
  both must precede the detector sampling the result.
- **The duty is where the two detector families part.** A microbolometer has no shutter and no
  integration window — `integration_time_ms` is `None` for one on purpose — so it smears over the
  whole frame period; a cooled photon detector integrates briefly and comes out sharper. Reading a
  missing integration time as zero would have made every uncooled camera in the repo sharper than
  it is, which is the flattering direction. This is §16's "lateral motion smears LWIR, not cooled
  MWIR", as arithmetic.
- Verified against the cascade term it implements: `|sinc(s·f)|` over 3–20 px smears and
  1/48–1/12 cyc/px, worst deviation **0.015**, mostly under 0.006.

### Fixed
- **The cross-check against sinc caught a real error before it shipped.** Taps placed at the smear
  segment's *endpoints* look natural and implement a boxcar of length s + s/(N−1): measured MTF
  0.7182 where sinc said 0.7842, which is exactly the Dirichlet kernel of the longer smear. The
  operator was self-consistent and describing the wrong smear — the failure mode no amount of
  "does it look blurred" would have caught. Taps are now at sub-interval midpoints.
- A test of mine compared picowatt-scale flux with `np.allclose`, whose default `atol` of 1e-8 is
  larger than the entire signal; it called a 15 % change "close". Now compared relatively.

### Added
- **Structured cloud in the rendered infrared background** (ADR 0076; MS.3/ADR 0070 into the M10.18
  bridge). For a sky-background sensor the dominant false alarm is not sensor noise, it is a cloud
  edge — a warm, target-sized, high-contrast feature with the *same polarity* as the thing being
  looked for, since a drone and a cloud base both read warmer than a cold clear zenith. The field
  has existed since M7 and was reachable only from the engine-free scene generator; nothing that
  rendered used it. `AerialThermalBridge` now takes a `cloud_seed`.
- **The field is fixed to the sky, not to the image plane.** A per-frame field flickers; an
  image-plane field is stable and *travels with the sensor*, so a slewing mount carries its clouds
  along and a tracked target never crosses an edge — removing the clutter the field exists to
  provide. A sky-fixed field is stable and stationary in the world, so slewing sweeps across it.
  `SkyFixedCloud` samples an (elevation, azimuth) grid per ray, with `azimuth_from_rays` supplying
  the second coordinate.
- Sampling by elevation alone is refused rather than approximated: it would band the sky in
  horizontal stripes, which is worse than no cloud because it looks like a deliberate atmospheric
  layer. A caller without azimuth gets the uniform blend — less detailed, not wrong.

- **The visible dome samples the same field**, through the same `sky_angles` convention — not a
  second cloud that looks similar, the same object, so the two halves of a frame pair cannot drift.
  Whether a cloud base reads brighter or darker than the sky beside it is left to the arithmetic:
  at the midday scene's 61° sun it comes out at ~15 900 cd/m² against a ~10 800 cd/m² sky, i.e.
  brighter, as a sunlit cumulus is. It need not agree with the infrared, where a cloud base is
  always the warmer feature — that divergence between bands is a real discriminator, not an
  artefact.
- The **continuous field is stored and thresholded after interpolation**, rather than a boolean
  mask being stored and sampled. A grid cell is 0.5° and a Boson pixel is 0.049°, so sampling a
  mask nearest-neighbour gave cloud edges that were ten-pixel rectangular steps — which is what
  the first render looked like, and not cosmetic, since edge sharpness is what a detector keys on.
  The threshold is estimated on a 2× upsampling because cutting it on the grid covers ~25 % less
  of a densely sampled sky, a bias that is *flat* across grid resolutions (a 1/fᵝ field is
  scale-invariant, so bilinear averaging smooths it equally at every scale). Residual: 8 % at
  c = 0.05, under 3 % by c = 0.2, against an input quantised in oktas.
- `--cloud-seed` on both flight scripts turns it on in **both bands** at once.

### Changed
- **Correction to ADR 0073.** It recorded the infrared background as "the clear-sky profile only".
  That overstated the gap: `SkyModel.radiance` has always applied the uniform blend
  (1 − cε)·L_clear + cε·L_base, which is the *expectation* over the structured field. The mean
  cloud effect was in every rendered frame; only the structure was missing. So this replaces a mean
  with a realisation — and **the mean is preserved**, measured at ~0.2 % across seeds against a 1 %
  bound, which is what keeps the rendered background consistent with the elevation LUT, the tilt
  integral and ADR 0045's reflected term.
- Coverage is exact over the *sky*, not over any frame: one elevation ring measured 0.0195 against
  a whole-sky 0.0500. A camera pointed at a gap sees no cloud and one pointed at a bank sees only
  cloud, which is what makes cloud clutter rather than texture — and means no single frame can be
  used to check the cloud fraction.
- Opt-in by seed: without one the background is bit-identical to before, so no committed frame or
  golden moves silently.

### Added
- **Image-plane motion, synthesised rather than rendered** (roadmap M10.1b, §13.3/§9.2).
  `irsim.optics.motion.image_plane_motion` computes the G-buffer's `motion_px` plane from rigid
  per-object transforms and the camera pose instead of from a motion AOV — ADR 0014's addendum
  measured that this build transports no motion at all (`motion_vectors` sits at a ~6e-5 floor
  after a 180 px displacement; `Motion2d` returns nothing), which left `motion_px` empty and with
  it everything that reads it: motion MTF and in-sim bolometer smear.
- Computing it is *exact* for rigid bodies rather than approximate, so the tests are held to it: a
  plane translating at 3 px/frame reports 3.0000 with **zero spread** across the frame, a static
  scene reports exactly 0, and an object and camera moving together report exactly 0 — the last
  being the case that catches the two transform pairs composed in the wrong order, which the other
  two cannot see because the camera is identity in both.
- `irsim_isaac.pipeline.motion_isaac.MotionTracker` is the USD half: it reads per-prim and camera
  transforms off the stage each frame and keeps a **stack of 4x4s indexed by instance id**, exactly
  as the thermal bridge keeps a float32 temperature table indexed the same way — one transport
  mechanism, two payloads. Motion is a difference, so the first frame of any sequence reports zero
  rather than a guess, and a prim that is untracked, absent or newly appeared is treated as static.
  `IrCamera` now keeps the camera-space position plane on its frame record, since re-projecting a
  surface point needs the position and the G-buffer only carries distance along the ray.
- In sim, against the renderer rather than made-up transforms: a bar at 3 px/frame reports 3.0
  within a tenth of a pixel, a static scene reports 0, and a prim moving with the camera reports 0.
  That last one caught a real mismatch — the position AOV arrives as `(H, W, 4)` with a padded
  fourth component, where the core wanted exactly three. The rest of the adapter already takes the
  first N of any float AOV; the core now does too, with a test asserting the padding cannot change
  the answer.
- **The background is treated as being at infinity**, so it does not translate with the camera,
  only rotate. That is why a tracking mount takes smear off the target and puts it on the sky, and
  it is now a computed quantity rather than an assertion: a yawing mount sweeps the sky by
  f·tan(δ) to 0.1 %.

### Added
- **What the ISP left in the picture** (ME.3b, ADR 0068 addendum). `irsim.validation.display_signature`
  reads the AGC's fingerprint, DDE's ringing and the camera's replaced pixels off finished frames.
  The recorder conversion is a **required argument** of the first two, so a Y16-derived set is
  refused inside the function and not only in the dataset index -- a histogram measured on a
  recorder's own 16-to-8-bit conversion describes the recorder. Plateau equalisation separates from
  a linear stretch by twenty times on every sky-like frame, and a scene whose histogram is already
  uniform is reported `indeterminate` rather than guessed at. DDE overshoot is the analytic
  `gain/3` of a 3x3 unsharp mask, so the gain reads straight off a picture. A replaced pixel is the
  mean of its four neighbours, so its Laplacian vanishes: 69 of 69 injected ones found on float
  frames, 93 % on 8-bit codes, no false positives -- and none at all through a codec, which takes
  the signature away entirely and is reported as "not measurable here" rather than as zero defects.
- **Reading the shutter off the frames** (ME.3a, ADR 0068). `irsim.validation.shutter` finds the
  freezes a flat-field correction leaves in the video, reports the interval only from a clip long
  enough to mean one, and fits what grows between events. Two distinctions carry the module: **a
  freeze is not an FFC** -- a run of repeated frames is equally a dropped chunk of recording, so
  every run is reported with `pattern_change`, the step in the time-averaged frame across it in
  units of what the noise alone would give (a shutter lands at 8-20, a stall at 1); and the
  between-shutter growth is fitted in the **variance**, `1 - e^(-2t/tau)`, because fitting it with
  the amplitude's law returns twice the true correlation time and looks perfectly reasonable.
- ADR 0068 -- the evaluation-data ADR, owed since ME.1a: what the public sets are, and the three
  gates (licence and signal path, 8-bit canonical form, and an analyser that refuses what its data
  cannot support) that stand between a public clip and a number the simulator gets tuned to.
- **What a statistic measured on somebody else's clip is allowed to claim** (ME.2b, ADR 0023
  addendum). `irsim.validation.flat` chooses the windows -- `robust_noise_scale` (MAD of the first
  differences about their own median, so a gradient cancels and striping does not inflate it) is
  the ruler, and flatness is scored against it rather than in DN, so the same thresholds work on a
  clip whose recorder stretched 16 bits into 8 and on one that did not. Striped windows stay flat
  on purpose: column noise is the thing being measured. A single-pixel target is caught by the
  outlier rule alone -- in a 4x4 block average it is a quarter of the block noise -- and on a clip
  the judgement runs on the temporal median, which deletes a moving target outright.
- `irsim.validation.codec` -- the floor the storage path puts under every one of those windows.
  The lattice step is read off the data, the floor is `step/sqrt(12)` (0.289 codes at 8 bits),
  Sheppard's correction is applied *and reported as limited* within two floors of it, and
  `flag_codec_limited` marks each of the seven 3-D components measurable or not. On a clip at
  sigma_TVH = 1.5 codes only the temporal white term survives that test, which is the honest state
  of the public sets.
- `irsim.validation.temporal_shape` -- a flat sky's temporal spectrum is flat unless something
  filtered it. Fits the sampled one-pole whose time constant in frames is exactly tau/dt and
  recovers the 10 ms membrane at 60 Hz to 1 %; reports drift separately so a wandering clip does
  not become a longer membrane.
- `irsim_eval.transcode.h264_round_trip` -- puts a synthetic cube through libx264 to calibrate that
  floor. Full-range flags are pinned so a CRF 0 round trip is bit-exact; at CRF 18, a *high
  quality* setting, 95 % of the temporal noise of a Boson-ratio cube is gone and the 8-pixel
  blocking score barely moves. Lossy sets give lower bounds, not measurements.
- **An aircraft flying a low pass, filmed in LWIR** (ADR 0075). `scripts/render_aircraft_pass.py`
  flies a light business jet past a ground sensor at 150 m/s and films ten seconds of it in real
  time. The variable is **aspect**, not throttle: the exhaust nozzles are hidden behind their own
  nacelles from the front and fully exposed from the rear, which is why a real aircraft's measured
  infrared signature varies by a large factor around the clock. Measured off the instance-id plane
  at the *same range* either side of closest approach, a rear aspect shows several times more
  nozzle than a head-on one and its hottest pixel is tens of kelvin warmer.
- `irsim_isaac.aircraft` — a parametric light jet (16 m span, rear-mounted engines so both nozzles
  sit on the centreline and occlusion is a clean function of one aspect angle). A jet's hot part is
  a *fortieth* of its span across, so unlike the quadrotor's motors you never resolve it: it is a
  near-point source, bright because it is hundreds of kelvin hot rather than because it is large.
- `irsim_isaac.aircraft_pass` — the track, the mount and the stage. The pass sweeps range 2:1 from
  the geometry alone, so it exercises the inverse-square fall-off and the atmospheric path against
  a target of known size and temperature for free.

### Changed
- **Aerodynamic heating: the `ram_skin` node, and scene schema v3 → v4.** `airframe_solver` puts an
  unpowered skin *at* air temperature and justifies it by forced convection at flight speed — true
  at 20 m/s, false at 200. A stagnating boundary layer heats a skin to its recovery temperature,
  `T_r = T_air (1 + r (γ−1)/2 M²)` with `r = Pr^⅓ ≈ 0.892`: 0.18 K at multirotor speed, 10 K at
  150 m/s, 35 K at M 0.8. A config names an **airspeed** and the Mach number is taken against the
  shared weather's own air temperature, so one true airspeed is correctly a different Mach number
  on a cold day. The two skin models are deliberately not interchangeable — `airframe` refuses an
  airspeed, because it *assumes* a slow one and silently ignoring a stated 200 m/s is the failure
  this exists to prevent.
- `IrCamera.refresh_pose()`, so a camera can be **moved between frames**. The pose that every
  pixel's ray direction is built from is cached at `open()`; without a refresh the geometry AOVs
  follow a re-aimed prim while the elevations, the sky temperature and every view cosine stay at
  the opening aim — a completely normal looking frame describing two different cameras. The
  aircraft mount slews through 20° of elevation across a pass, and the sky is tens of kelvin colder
  at the top of that.
- `irsim_isaac.airframe` — `Part` and `author_parts` extracted from `quadrotor.py`, so the
  `thermal:material` override and the prim→thermal-node map have exactly one implementation.
- Both flight scripts share the burnt-in readout via `irsim_eval.video.overlay_readout`, and gain
  `--no-flat-field`.

- **The fixed-span videos now span the *target*, not the scene.** ±50 K about ambient spends half
  of 256 display levels on sky-to-ambient, which is two flat regions, and squeezes every part of
  the target into the rest. The span is now taken from the scene's own node temperatures across
  the sequence, with the floor a quarter of the node spread *below* the coldest node — on it, and
  a target that starts at ambient is invisible in the opening frames. Measured on the quadrotor at
  full throttle, like for like over the airframe's own 1000 pixels: 76 display codes of spread
  from the old rule, 128 from the new, with the cool parts starting near black rather than
  mid-grey. Sky clips to black, deliberately. `--span-c` restores a scene-context span.

### Fixed
- **The flight videos' AGC companion carried cos⁴ vignetting.** `PipelineConfig.from_sensor`
  defaults `flat_field_enabled=False` and neither flight script passed it, so the 8-bit picture
  showed the full 21 % centre-to-corner falloff that M9.12 exists to remove, stretched into dark
  corners by plateau equalisation — measured at 2.19× centre-to-edge before the fix and 0.93×
  after, the residual being genuine sky structure. The radiometric branch was never affected — it
  divides cos⁴ out analytically, and `tests/unit/test_flat_field.py` already asserted that
  `radiance`, `apparent_t` and `dn16` are bit-identical with the flat field on or off while
  `display8` changes — so the main fixed-span videos were always correct and only the companion
  was wrong.

### Added
- **A resolved quadrotor flying a mission, filmed in LWIR** (ADR 0074). `scripts/render_quad_flight.py`
  puts one heavy-lift multirotor at 20 m -- 105 px across the span, 6 px per motor bell -- against
  sky and flies a 30-minute profile while ADR 0072's heat sources take the motors from ambient to
  +45 K and back. The four hot bells, the warm speed controllers and the warm battery are separate
  objects in the image, which is what a thermal sensor actually keys on in a drone.
- `irsim_isaac.quadrotor` builds the airframe **parametrically from primitives** rather than
  importing a mesh. Every part is separate by construction, so the thermal nodes attach
  themselves; its dimensions are stated rather than measured off a bounding box, which is what
  lets a test assert an angular size; and there is no licence or unit-scale bug to carry. Motor
  bells default to a high-emissivity anodised material -- **bare aluminium is ε = 0.09 in this
  library** and would render a 70 °C motor as barely above the reflected sky.
- **Propellers are deliberately absent.** A 28-inch prop modelled as a solid disc is 33 px across
  at 20 m and hides the motors, which are the subject; physically a prop at flight rpm smears into
  a faint annulus within a 60 Hz integration period. Doing that properly needs the motion path
  (M10.1b).
- `irsim_isaac.quad_flight` is the stage. The camera tracks the aircraft, because a drone on a
  30-minute mission covers kilometres and nothing that flies realistically stays in a 33° field;
  attitude is driven by the **same throttle** the thermal model reads, so the picture and the
  physics cannot disagree about what the aircraft is doing.

### Changed
- **Scene schema v2 → v3: `heat_source` and `airframe` target solvers.** §6.6's aerial node model
  (ADR 0072) has existed since M6.6 and was reachable *only from its own unit tests* -- no scene
  could ask for a motor, because `irsim.config.scene` had no solver kind that named one. A scene
  now names a **throttle profile** and the temperature is derived through ΔT = ΔT_max·u^n, refined
  until piecewise-linear interpolation reproduces the law to 1 mK. Authoring a temperature
  schedule alongside a throttle is refused: two answers and no way to pick. `configs/scenes/
  sky_target_clear_day.yaml` is migrated in the same commit.
- An `airframe` node's schedule spans the whole weather file, so its solver is explicitly advanced
  to the scene start; without that its first reading is the air temperature hours earlier -- here
  the difference between a 26 °C midday and an 18 °C dawn, and plausible either way.
- `IrCamera` takes an optional `frame_period_s`, which makes it a **time-lapse camera**. Not a
  fast-forward: every stage is told the truth about the interval, so the FFC fires on its real
  schedule, the fixed pattern drifts by a real amount, and the temporal noise decorrelates exactly
  as it would between two captures that far apart.
- The environment dome and the visible-band looks moved from `aerial_demo` into a shared
  `irsim_isaac.stage`, so a second stage cannot re-derive either and drift from the first.
- The flight video's palette now comes from the **sensor's own ISP config** instead of being
  hardcoded. ADR 0074 already said the fixed-span video maps "through the ISP's own palette" and
  the code did not: it used `ironbow` while `flir_boson_640_lwir.yaml` says `palette: gray`,
  `polarity: white_hot`, which is how a real Boson ships. A presentation video that picks its own
  colours is a second display path, and two pictures of one frame then disagree for a reason that
  is nowhere in the physics. `--palette` still overrides for a one-off.

### Notes
- **The video is a time-lapse, and that is a physics decision.** ADR 0072's node law is a
  *steady-state* relation -- T = T_air + ΔT_max·u², with no thermal time constant -- so it is only
  defensible while the throttle moves slowly against a real motor's minutes-scale response. The
  flight is authored over 1800 s and captured one frame every 6 s. Running the same profile at
  60 Hz would look smoother and would show 45 K of swing no metal could follow. Every transient in
  the video is a *throttle* transient.
- **The main video is a fixed display span, not the camera's AGC.** Both §11.3 AGC modes rescale
  from the current frame's own histogram, which cancels exactly the change the video exists to
  show; plateau equalisation additionally allocates display codes by population, so an aircraft
  covering under 1 % of the frame saturates to flat white with the motors indistinguishable from
  the arms. That is measured, not predicted -- it is what the first render looked like. The
  camera's own AGC output is filmed alongside as a second video, because the difference is the
  lesson.
- ΔT_max stays **ESTIMATED** (45 / 30 / 15 K for motor / ESC / battery, ADR 0072). The ordering is
  the defensible part; the magnitudes are what a Tier 4 fit against public aerial IR should move
  first.

### Added
- **A real sky for the companion visible frame** (ADR 0073). `--rgb` on the aerial demo used to
  write a flat grey void with six grey squares in it: no sky, no horizon, no sun, nothing a reader
  could use to check where the camera was pointing or what hour it was. Since the infrared frame
  cannot be eyeballed for correctness at all, the visible frame is the only half of the pair a
  human can apply ordinary judgement to, and it was not doing that job.
- `irsim_isaac.visible_sky` generates a float32 lat-long environment map -- a Preetham-Shirley-Smits
  (SIGGRAPH 1999) analytic daylight sky above the horizon, a Lambertian terrain faded into the
  horizon sky by Koschmieder's contrast transmittance below it -- and `aerial_demo` binds it to the
  stage's dome light with a 0.53° distant light for the solar disc.
- Every input is the scene's own, so the two frames cannot describe different days: NOAA sun
  position at the scene's site and clock, DNI/DHI from the shared `WeatherSeries` (CLAUDE.md #6),
  and turbidity derived from that weather's visibility against the atmosphere preset's visible
  Rayleigh coefficient. Turbidity is a ratio of **column** optical depths -- taking the
  ground-level ratio instead gives T = 14.2 on a clear 23 km day, a dense industrial haze, where
  the column ratio gives 2.98.
- **The infrared path is untouched, asserted rather than argued.** A `UsdLux.DomeLight` is a light
  and not geometry, so a ray that sees it still reports instance id 0 and an infinite
  `DistanceToCameraSD` and the background keeps coming from the sky model (ADR 0060). The radiance
  and apparent-temperature planes come out **bit-identical** with the dome and without it, which is
  an integration test and not a remark.
- **The renderer's lat-long convention had to be measured.** On this build the RTX dome light
  samples the texture with its polar axis on the stage's **+Z** and its azimuth running from +X
  toward -Y -- not the stage up axis, and not what the USD documentation implies. The first
  implementation authored the map in elevation and azimuth and rendered a frame filled entirely
  with ground, because the whole camera field fell inside one texture pole. The map is now authored
  in direction space, `DOME_POLE_AXIS` records the measurement, and an integration test re-measures
  it: pointed down the sun's own azimuth, the solar disc must land within 8 px of
  `f_px·tan(tilt − elevation)`.
- Targets get a `UsdPreviewSurface` in the visible band, so the companion frame shows a white
  airframe and a black carbon one rather than six identical grey squares. Appearance only -- the
  infrared properties still come from the material library through the `thermal:material` override,
  and the looks live under `/World/Looks` so they never reach the material resolver's prim walk.
- Two absences are deliberate, so the pair does not show what the infrared frame does not have:
  **no cloud** on the dome (MS.3's cloud field is not wired into `aerial_bridge` yet) and **no
  solar disc in the sky texture** (Preetham carries the aureole, the distant light carries the
  disc, so the two do not double-count).
- The dome light's `intensity` carries an **auto-exposure** on the median daylight sky, because the
  sky's absolute level moves by two decades between dawn and noon. That is a display choice and is
  why the visible frame carries no absolute brightness information; only its *geometry* is a
  calibrated claim. The twilight fade is divided back out of the reference, so a night scene still
  renders dark instead of being exposed up to look like noon.
- `scripts/render_aerial_demo.py` gains `--heading-deg` (the compass bearing the camera looks
  along, which is what places the sun in the frame), `--camera-height-m` and `--no-dome`.

### Fixed
- **The camera's flat-field correction was never applied** (roadmap M9.12). `irsim.isp.TwoPointNuc`
  has existed since M5 with its own tests, and nothing in `irsim.pipeline` referenced it -- so the
  8-bit picture carried every fixed spatial structure the optics put into the DN plane. Most
  visibly cos⁴ vignetting: measured at 21 % from centre to corner on the Boson's 14 mm lens, which
  plateau equalisation then stretched into black corners. No real thermal clip looks like that,
  because every real camera flat-fields before its AGC.
- The README claimed "the NUC stage (M9) removes it". It did not. That line is corrected rather
  than quietly deleted, because it is the kind of claim that was ahead of the code and got
  repeated.
- The correction lives in the **display branch alone**, and that is a decision. §11.1 forks the
  branches after the ADC and `invert_optics` already divides cos⁴ out per pixel analytically --
  which is why apparent temperature was flat across a row while the picture was not. Applying a
  measured correction to the radiometric side as well would remove the same term twice, so `dn16`
  and the radiometric outputs are **bit-identical** with the flat field on or off, and there is a
  test asserting exactly that.
- Coefficients come from two synthetic blackbody frames through the real forward chain with the
  noise off -- what a bench calibration does with two real blackbodies -- rather than from the
  analytic cos⁴ field. A correction built from the formula would remove exactly the term the
  formula describes and stay silent about any other fixed structure the chain grows later.
- `TwoPointNuc` gains an optional `pedestal`: §11.2's convention makes the corrected cold blackbody
  read 0, which is right for the radiometric reference and wrong as an AGC input, since a
  flat-fielded frame reading zero on a cold scene moves the histogram for reasons that have nothing
  to do with the scene. Default 0 keeps the spec convention and every existing test.
- Measured after the fix, on a scene whose temperature varies with row only so that any
  left-to-right difference is the camera: uncorrected the edges read 20+ display codes below the
  centre; corrected the difference is under a fifth of that. A uniform blackbody comes out flat to
  0.1 %, at temperatures it was not calibrated at as well as at them.

### Added
- `--rgb` on `scripts/render_aerial_demo.py` and `capture_rgb` on `IrCamera`: the visible-light
  frame from the **same** camera prim, pose and lens, box-filtered onto the IR pixel grid, so an
  RGB/IR pair is registered by construction rather than by calibration. It carries no infrared
  information and nothing in the radiometric chain reads it -- the sidecar's unit string says so.
- Getting it took a measurement (ADR 0014 addendum). On 6.1.0-rc.26 the **lit** colour AOVs are
  silent in both render modes a script gets by default: `rgb` and `LdrColor` come back all zero
  under `RealTimePathTracing` (the build default) and `RaytracedLighting`, `HdrColor` comes back
  all NaN, and only the un-lit `DiffuseAlbedo` delivers. Under `PathTracing` the colour AOV
  delivers -- and every geometry AOV the IR chain needs still does, checked rather than assumed,
  which is what makes switching the mode safe instead of needing a second render pass. The mode is
  set only when `capture_rgb` is on.
- An all-zero colour buffer is **rejected with a reason** rather than saved: written to disk it is
  a black PNG, indistinguishable from a night scene. That is the same trap ADR 0014 already
  records for `PtWorldNormal` and ambient occlusion.
- The demo stage gains a distant light and a dome light *for the visible render only*. Neither is
  geometry, so no ray hits them: `instance_id` stays 0 and `DistanceToCameraSD` stays infinite on
  those pixels, and the IR background still comes from the sky model. Verified by the IR apparent
  temperature being identical with the lights present and absent (250.60 K to 291.65 K). The
  resulting RGB is a uniform grey field, because the stage has no sky texture or terrain -- it has
  no sky dome on purpose (ADR 0060) -- so the pair is correct and registered, and its visible
  content is as rich as the stage is.
- `--no-flat-field` as the ME.8-style ablation, so the artefact can be reproduced deliberately.

### Added
- Warp stage 5's remainder on device (roadmap M10.7b): the bad-pixel map and its RTS chain, the
  iterated 4-neighbour replacement, and the FFC hold. Unlike M10.7a's noise these are mostly
  **exact**, and the tests say so in those terms -- 34 of them, on `cuda:0` and Warp `cpu`.
- **The defect map is uploaded, not redrawn.** A bad-pixel map is a property of one physical focal
  plane, drawn once per sensor and never per frame (§10.4), so two independent draws would be two
  different cameras rather than two implementations of one. The device map is therefore
  bit-identical to the CPU's by construction. The RTS state is seeded from the CPU's own starting
  realisation for the same reason M10.7a seeded the fixed pattern that way: what is measured
  afterwards is the two generators, not two unrelated defect populations.
- The RTS chain is launched over the whole plane rather than a compact list of stateful pixels.
  The CPU restricts it because drawing 327k uniforms to move a few dozen bits is most of the noise
  chain's cost; on a GPU that argument does not apply. It stays a **chain**: a pixel that was bad
  leaves with probability 1/dwell, and redrawing the state independently each frame would give
  white dwell statistics where the geometric ones are the defining signature of RTS.
- The replacement kernel accumulates in **float64**, as `replace_bad_pixels` does, so "matches the
  CPU" means every bit rather than a tolerance -- a float32 mean of four neighbours differs in the
  last bit, and a tolerance there would hide an arithmetic difference instead of measuring one.
  Passes ping-pong two buffers because the CPU builds its neighbour sums before writing any of
  them; in place on a GPU that ordering is not merely different, it is non-deterministic.
- The pass loop lives on the host and reads back two counters per pass. That is two tiny transfers
  for a stencil that finishes in two or three passes on any map ADR 0055 can produce, and it keeps
  the CPU's loud failures: a region with no valid neighbour on any side raises rather than being
  filled, and so does a cluster still unfilled after the pass cap.
- The FFC hold is one device buffer, written on every unfrozen frame and read on every frozen one,
  so a freeze emits the same bytes for its whole length -- which is the fingerprint ME.3 looks for
  in real video. A freeze that begins before anything is held passes the frame through rather than
  inventing one the camera never saw. The *schedule* stays on the host: when the shutter fires is
  frame arithmetic, and a second `FfcController` on the device would be another thing to keep in
  step.
- Verified, exactly: the map reaches the device bit-identically; replacement matches
  `irsim.isp.replace_bad_pixels` bit for bit on 1x1, 2x2 and 3x3 clusters and on the real defect
  map; defect injection matches `apply_defects` for a given chain state, including the detail that
  an already-saturated hot pixel keeps its sub-LSB float value; a frozen frame is bit-identical
  across a six-frame freeze; and the **composition** -- inject then replace, in order -- matches
  `SensorChain.finish_frame` bit for bit. That last one is there because M10.7a's lesson was that
  two individually correct stages can still disagree by a factor of ten if the seam between them
  sits in different places.
- Verified, statistically: the chain holds its configured occupancy within 25 % and its mean dwell
  within 35 % over 2500 frames, and the dwell is more than two frames -- a memoryless redraw would
  give about one. The population comes from area rather than a raised defect fraction, because the
  schema caps that at one percent and it is right to: a focal plane with more than one percent of
  its pixels dead is scrap, not a sensor.
- `irsim_eval.data` (roadmap ME.1b): the `Sequence` / `Frame` / `Box` form every validation
  analyser reads, plus a canonical on-disk layout (a JSON index and one array per frame) and its
  writer. The indexed sets agree on nothing -- Halmstad ships MATLAB labels beside mp4, Anti-UAV410
  per-sequence JSON, the single-frame sets folders of images -- so a per-dataset converter writes
  this layout once and no analyser ever reads a publisher's format. Frames load on demand, since a
  box-size histogram over a whole set should not pay for pixels it never looks at.
- **The reader insists frames are 8-bit.** Every indexed set stores 8-bit frames, and that
  quantisation is a floor under every statistic measured on them (ADR 0003); a dtype that quietly
  widened would hide the floor rather than remove it, and invite a radiometric claim the data
  cannot support. Boxes are top-left `(x, y)` plus `(w, h)` in the project's own pixel-edge
  convention, pinned by a test because half a pixel here changes every size-versus-range number.
- `irsim_eval.motion`: the static-camera clip detector that gates every per-pixel temporal
  statistic. On flat sky a per-pixel temporal standard deviation is the closest thing these public
  clips offer to a laboratory blackbody -- but only if the scene stayed on the same pixels. Pan by
  a pixel a frame and the same number measures the sky gradient crossing the detector, which is
  larger, perfectly smooth, and indistinguishable from a noisier camera.
- The discriminator is **cumulative displacement from the first frame**, not per-frame motion,
  because that is the distinction that matters: a shaken mount wobbles sub-pixel and goes nowhere
  (static, the statistic still describes the sensor), while a slow drift takes equally small steps
  that all point the same way and crosses many pixels (moving, however small each step was). The
  two synthetic controls are built with the *same* per-frame step size so the pair tests
  accumulation rather than amplitude.
- Shifts come from phase correlation, which ignores the amplitude and so survives the brightness
  changes an AGC makes between frames. Sub-pixel accuracy uses **Foroosh's ratio, not a parabolic
  fit**: a phase-only peak is a Dirichlet kernel that splits linearly between its two nearest
  samples, and the parabola everyone reaches for under-reads it by about 30 % at a third of a
  pixel -- small enough to pass for noise, biased enough to drag a drifting clip under the
  threshold. Caught by the estimator's own known-shift test, which the parabolic version failed at
  (-0.4, 0.6) and passed at every integer shift.
- New `validation` extra (imageio) for sets stored as images; sequences written as `.npy` need no
  decoder, which is why the default gate installs nothing new and every test here runs without it.
- The validation-data index (roadmap ME.1, first half; ADR 0003). `data/validation/datasets.yaml`
  records, per public dataset: licence, access mode, sensor, **signal path**, bit depth, codec,
  frame counts, and which ME.2-ME.4 analysers it may and may not support. `irsim_eval.manifest`
  makes licence, signal path and the analyser list *required* fields, so a set that cannot say
  what its frames are cannot be added.
- The index is the thing that decides whether a statistic means anything. The same flat-sky patch
  is a laboratory noise measurement on a Y16-derived clip and a measurement of somebody's unknown
  AGC on a display-output one. `usable_for("noise_3d")` returns exactly one set; the Halmstad
  clips never went through an AGC, so `agc_signature` is in their *excluded* list and only
  Anti-UAV410 can carry it; only LRDDv3 has range labels, so only it can do `size_vs_range`; and
  the single-frame sets are marked `prior`, able to bound a distribution but never to be a target
  the simulator is tuned to hit.
- **Every field was checked against the publisher's own page, and the findings changed the plan.**
  Only the Halmstad set (Svanström et al. 2021, Zenodo 10.5281/zenodo.5500576) states a licence at
  all -- CC0-1.0 -- and it is also the only one whose sensor and signal path are fully documented
  (FLIR Breach PTQ-136 / Boson 320x256, 24 deg x 19 deg, 60 fps, Y16 to 8-bit in the recorder,
  mp4). It is therefore the **primary** set, and the schema permits exactly one.
- The other five state no licence. `unstated` is recorded as a value, not left blank: it means the
  terms are unknown, not that they are permissive. Anti-UAV410 and CST Anti-UAV publish no licence;
  LRDDv3 is behind an access request citing US export control; IRSTD-1k and NUAA-SIRST have no
  canonical licensed source. CST Anti-UAV is not released yet.
- `scripts/fetch_validation_data.py` plans, reports and mostly refuses. A set with no stated
  licence is **skipped** unless `--accept-unstated-licence` is passed, so using data on unknown
  terms is always a deliberate act; a `manual` set (a drive link, a Zenodo record, an access form)
  is never downloaded, because pretending a script can do that fails in a way that looks like a
  network error; `http` is refused rather than silently upgraded. What it always does is hash what
  is on disk, so "measured on these exact bytes" stays checkable (ADR 0004's rule applied to
  someone else's data).
- `data/validation/README.md` is **generated** from the YAML and a test fails when it is stale --
  an index that disagrees with itself is worse than no index, because the prose is what people
  read and the fields are what the code reads. Datasets stay out of git (`data/validation/*/`).
- New package `src/irsim_eval/`, the home for everything that reads somebody else's imagery, kept
  out of the core so that verifying Planck's law never requires an image decoder (CLAUDE.md #1).
  `make typecheck` now covers it.
- MS.6 analytic point targets wired into `IrCamera` (roadmap M10.19, ADR 0071). Below one native
  pixel the renderer is the wrong instrument -- it samples geometry, so a target covering a
  quarter of a pixel is drawn or not drawn depending where the sample landed. `AnalyticTarget`
  carries such a target's world position, projected area, material and thermal node;
  `IrCamera.point_targets()` turns it into an MS.6 `PointTarget` each frame, rebuilt rather than
  cached because the temperature is still moving.
- **A target is rendered or injected, never both.** `aerial_demo.analytic_targets` hides the
  sub-pixel prims in the same call that produces their specs, so the two paths cannot both claim
  one, and `IrCamera.check_no_double_count()` reports any that still reached the id plane. Both
  halves are tested, including the failure: with the prim left visible the guard names it.
- The leaving radiance comes from `irsim.validation.aerial.target_leaving_radiance`, the same
  function the engine-free aerial work uses, so the in-sim and CPU paths cannot disagree about
  eps L_B(T) + (1 - eps) L_env.
- In-sim verification (`tests/integration/test_point_targets_isaac.py`, 7 tests): the injected
  excess **survives the whole chain to 5 %** -- splat, optical PSF, box downsample, detector, ADC
  and the radiometric inversion all compose back to the excess that went in; an injected target
  lands within half a native pixel of where the renderer draws the same object, so the handover at
  one pixel introduces no jump; and the fill fraction carries the entire geometric range law
  (phi R^2 constant to 1e-12).
- **The "signal falls as tau/R^2" shorthand is not exact, and now there is a test saying so.**
  Measured over 400-3200 m it drifts by 33 %, always in the direction of under-predicting the
  longer range. The excess is phi * sum_k w_k tau_k (L_t - L_beyond,k): the bracket depends on
  range too, because what a target occults is the sky column beyond it and there is less of that
  column left at 3200 m. A trade study using the shorthand would over-estimate detection range.
- `scripts/render_aerial_demo.py` uses the split: the 500 m, 1500 m and motor-pod targets are now
  injected analytically and marked ANALYTIC in the report, with `--no-point-targets` as the
  ADR 0071 ablation that renders them as geometry instead.

### Fixed
- `world_to_camera` returns **USD** camera space (+Y up, -Z forward), so the projection has to be
  `project_usd`; `project` reads -Z as behind the camera and returns NaN for every target in front
  of it. Caught by that NaN guard rather than by a wrong picture, which is what it is for.
- Aerial demo stage and a one-command render (roadmap M10.19, stage half):
  `irsim_isaac.aerial_demo` authors six targets against sky -- a resolved quadrotor at 120 m, one
  at the pixel limit at 500 m, one well below it at 1500 m, an aircraft at 2.5 km, a bird and a
  hot motor pod -- and `scripts/render_aerial_demo.py` renders them through `IrCamera` and writes
  the frames with M10.10a. Sizes are the real ones, so the demo shows what the range problem
  actually looks like: a 0.35 m quadrotor at 500 m is 0.82 of a Boson pixel, and the stage says
  which targets are in MS.6's sub-pixel territory rather than leaving it to be discovered.
- There is **no sky dome and no ground plane** in the stage. A ray that hits nothing takes
  `T_sky(theta)` at its own elevation, or `T_ground` below the horizon (ADR 0060). An emissive
  dome would push the sky through a colour AOV -- all float16 on this build, ~100 mK against a
  50 mK NETD -- and would be worst at the horizon, where the gradient is steepest and the targets
  are.

### Fixed
- **`Camera3dPositionSD` is in camera space, not world space** (ADR 0014 addendum). ADR 0014
  recorded it as world, which was true of every scene that measured it: all of them had an
  unrotated camera at the origin, where the two frames coincide. Tilt the camera up and they
  separate -- measured at 8 degrees about X, the AOV still reports the frame centre's ray as
  `(0, 0, -1)`.
- Read as world it produces no error, just a different camera: the boresight reads 0 degrees
  instead of 8, the horizon moves 164 rows to the middle of the picture, the upper half of the sky
  falls below the model's horizon and is painted with `T_ground` 30 K too warm, and
  `normal_dot_view`, `normal_dot_up` and the sky-view factor all tilt with it. The result is
  smooth, monotonic and entirely plausible. `IrCamera` now defaults to `position_frame="camera"`
  and passes the prim's own local-to-world rotation.
- It was caught by asking where the horizon *should* be: `cy + f_px tan(tilt)` has no free
  parameters. With the fix, elevation crosses zero at native row 221.5 against a predicted 221.7
  and the frame centre reads 7.979 degrees for an 8 degree tilt. Nothing else in the repo was
  affected, because nothing else had rotated a camera -- which is exactly why it survived until a
  scene needed to look up.
- The in-sim aerial phenomenology suite (10 tests) states the background claims, since on a
  sky-target camera the background is most of the image and is where a wrong model hides: the
  boresight elevation, the horizon row, a monotone sky gradient over every row above it, a single
  flat ground temperature below it, and the sky reading 25 K below ambient while the ground reads
  at it.
- One of those is worth its own line because it is counter-intuitive and was a wrong assumption in
  the first draft of the test: **the horizon is not a visible edge**. A grazing clear sky has
  unbounded path length, so its emissivity approaches one at about the air temperature -- which is
  what the ground is at. The step across the horizon is smaller than the sky gradient over the
  frame, and an algorithm hunting the strongest edge finds a place in the sky instead.
- Float32-preserving dataset writers (roadmap M10.10a): `irsim.io.write_frame` puts one frame's
  four §12.2 outputs on disk with a JSON sidecar. radiance and apparent temperature go to float32
  `.npy` or float32 EXR; DN16 to a uint16 PNG (an ADC code is an integer and loses nothing);
  display8 to an RGBA8 PNG. A plane in physical units never reaches an integer or half-float
  container, which is CLAUDE.md #2 at the disk boundary.
- The arithmetic is now a test rather than an assertion in prose: a 16-bit PNG over the 233-473 K
  radiometric range quantises to 3.66 mK, and half-float at 300 K to 250 mK -- **exactly** five
  times a 50 mK NETD, the figure CLAUDE.md #2 quotes. Both would produce a frame that opens and
  looks right with the sensitivity already gone, so `write_frame` refuses a float16 plane and
  `write_exr` refuses `half=True` even when it is asked for.
- `irsim.io.exr`: a minimal single-part, scanline, uncompressed **float32** OpenEXR writer and a
  matching narrow reader, in stdlib `struct` and `zlib`. EXR exists because no image tool reads
  `.npy` and the practical alternative people reach for is a 16-bit PNG. No dependency is added
  (CLAUDE.md #1 forbids an imaging stack in the core), and the header is checked against the
  format specification -- magic, version, FLOAT pixel type, little-endian samples -- rather than
  only round-tripped through our own reader, which would pass with the byte order wrong in both
  directions. Multi-channel files come back in alphabetical channel order, as EXR stores them;
  pinned by a test so it is documented rather than surprising.
- `irsim.io.png` grows 16-bit grayscale. The DN16 round trip is decoded from the raw IHDR/IDAT
  chunks in the test rather than through our own encoder twice.
- The sidecar carries the config and band hashes (ADR 0008), the ISP hash, the frame index, the
  scene time and the UTC it corresponds to on the weather axis, and for every file its dtype,
  shape and **unit**. A directory of arrays nobody can trace to a configuration is a pile of
  images, not a dataset. Frame names are zero-padded so a sequence sorts lexicographically, and an
  output turned off in `sensor.outputs` leaves no file rather than a plane of zeros -- zeros are
  indistinguishable from a real dark frame.
- `IrCamera` (roadmap M10.9a-ii, ADR 0015 addendum): the object that turns a USD stage into an
  infrared frame. It authors the camera prim from the sensor YAML, creates the render product at
  `supersample x native`, attaches the M10.1 annotators, assembles the G-buffer from the geometry
  AOVs plus the M10.2 material ids and the M10.18 temperature table, and runs
  `irsim.pipeline.run_frame`. **This is first light from Isaac Sim through the whole camera
  model.**
- The constructor takes the `Scene` rather than the atmosphere preset and environment the roadmap
  sketched. The scene already holds all three bound to its single `WeatherSeries`; taking them
  separately would let one atmosphere run in the transmittance, another in the sky and a third in
  the thermal solvers (CLAUDE.md #6), and the camera refuses a pipeline whose sky or atmosphere is
  on a different weather object.
- **It runs the CPU reference, not the Warp stages, and that is deliberate.** The M10.4--M10.8
  twins cover stages 1--6 but not the M9 chain's post-ADC half -- device defects, the iterated
  bad-pixel replacement and the FFC hold are M10.7b. Running the device path today would produce a
  frame from a *different camera* and label it the same. `IrCamera.planes()` exposes the assembled
  G-buffer so the Warp path can be driven from the same scene and compared; the all-device frame
  lands with M10.7b.
- **Every attribute of the distortion schema is written, never a subset.** Measured on
  6.1.0-rc.26: the schemas default to `fx = 900`, `cx = 1024`, `imageSize = (2048, 1024)` and, on
  the fisheye, a non-zero `k1`. An attribute left unwritten is not "no distortion", it is a lens
  for somebody else's camera, and it renders without complaint. `imageSize` is a `GfVec2i`; a
  Python tuple is coerced to `GfVec2d` and the set is rejected outright, so it is built explicitly.
- In-sim verification (`tests/integration/test_ir_camera_isaac.py`, 12 tests): a grid of quads at
  known world positions reprojects through the config model to **< 0.2 px** both undistorted and
  under a barrel lens; the barrel coefficients move the outer quads 2.1 px while the optical axis
  stays put; and reprojecting the barrel render through a *zero-coefficient* model fails by more
  than 1.5 px, so neither half could pass on a build that discarded the schema.
- Measuring to a fifth of a pixel off a binary id mask needed a correction. A rectangle's
  included-pixel set snaps to whole pixels, so its centroid lands on a multiple of half a pixel
  however large the rectangle is -- averaging over more pixels does not help. Every quad came back
  a flat 1/3 px off, which looks exactly like a lens error and is not one. The grid is rendered 8x
  supersampled and the centroids divided down, putting the quantisation at 1/16 of a native pixel.
- Whole-frame verification on the five-prim material stage: float32 / float32 / uint16 / RGBA8 at
  the native grid, no float16 plane anywhere, the high-emissivity prims reading back their
  authored temperatures within 8 K, and -- the sharpest check that emissivity reaches the kernel --
  the two `bare_aluminium` prims, 27.15 K apart at the surface, collapsing to under a quarter of
  that in apparent temperature because at eps = 0.09 both show the same sky and ground. Note the
  direction: a mirror does not read cold, it reads *its environment*, so the 268 K window reads
  warmer than it is.
- An UNMAPPED prim has no emissivity and `MaterialTable` refuses to invent one (ADR 0047). In
  `debug_unmapped` mode those pixels are handed to stage 1 as blackbody-equivalent -- the eps = 1
  ADR 0047 specifies, on the prim's own temperature -- and painted magenta at the native grid,
  over-reporting rather than hiding a forgotten prim; without the debug mode the frame raises.
- `scripts/probe_isaac_camera.py` and `irsim_isaac.camera_probe` record the distortion-schema
  survey the mapping is built on, so the next build is re-measured rather than argued about.
- Lens projection (roadmap M10.9a, ADR 0015 addendum): `irsim.optics.projection` answers "given
  this `optics.distortion` block, where should a ray at this angle land?". ADR 0015 is unchanged --
  the engine still applies the lens and the imaging path stays rectilinear -- but without a
  **forward** model there is no way to tell whether the coefficients written onto a camera prim
  produced the lens that was configured, and a barrel term is a smooth radial stretch, exactly the
  error a human eye accepts in a picture.
- Two conventions are pinned by tests rather than by comments, because both are silent when wrong:
  USD camera space (+Y up, -Z forward) versus OpenCV (+Y down, +Z forward), flipped in exactly one
  function; and the principal point at the format corner with pixel centres at `i + 0.5`, tied to
  `irsim.optics.vignetting`'s independently written geometry -- half a pixel of disagreement is
  3.8e-4 rad at the Boson's corner, invisible in an image and fatal to a reprojection check.
- Measured on 6.1.0-rc.26, not assumed: `brown_conrady` maps to
  `OmniLensDistortionOpenCvPinholeAPI`, whose twelve attributes are in OpenCV's own
  `[k1, k2, p1, p2, k3, k4, k5, k6, s1..s4]` order, so a five-term block maps **positionally**;
  `kannala_brandt` maps to `OmniLensDistortionOpenCvFisheyeAPI`.
- **`ftheta` is refused, not approximated.** Nothing this build exposes determines whether its
  polynomial returns pixels or normalised units, or whether `k0` is a constant term -- under one
  reading a one-coefficient block is a lens, under the other a constant radius, which is not.
  Either guess renders a plausible fisheye that disagrees with the engine by tens of pixels at the
  field edge, so `project` raises and M10.9b's renderer audit measures it. No configured camera
  uses f-theta.
- Verified against arithmetic decided outside the module: hand-evaluated OpenCV radial and
  tangential terms, the equidistant limit of the fisheye model (r = theta, which differs from the
  rectilinear r = tan theta by 66 % at 60 deg, so a silent pinhole fallback cannot pass), the
  cos^4 identity against `irsim.optics.vignetting`, and a distort/undistort round trip over the
  whole Boson frame at < 1e-6 px against the row's 0.2 px in-sim budget.
- Warp stage 6 (roadmap M10.8): the display branch on device -- an atomic histogram, the ADR 0028
  plateau clip with `wp.utils.array_scan` for the exclusive CDF, the AGC applied as a
  2^bit_depth lookup table, an edge-clamped 3x3 DDE and the palette to RGBA8. Both §11.3 AGC modes
  are monotone functions of DN alone, so each *is* exactly a table; that is what makes them
  portable to a kernel at all, and why the port can be held to a display code rather than to a
  resemblance. The O(1) reductions over the 65536-entry histogram -- the percentile positions, the
  occupied-bin span -- are finished on the host, where each is a line rather than a kernel.
- Measured on an RTX A6000 and Warp `cpu`: ≥ 99.9 % of pixels within ±1 display code of the CPU
  branch for linear, plateau-equalisation and none, on a ramp, a hot-exhaust patch and a constant
  frame; the float image before quantisation agrees to 2e-3. DN16 is bit-identical under every AGC
  mode -- §11.1 forks the branches after the ADC, and an AGC that reached back into the linear
  output would be invisible in the picture and fatal to the validation that consumes it. The
  ADR 0028 exhaust collapse is reproduced: plateau equalisation retains more than three times the
  background contrast linear AGC leaves.
- `agc: none` is a **bit shift** (the top 8 bits over 255, ADR 0031), not a linear rescale of the
  full-scale range. The device table had it as the latter and was half a display code out on every
  pixel -- caught by comparing the AGC *tables* rather than the images, which localises a
  disagreement to the AGC instead of leaving it somewhere in the DDE or the palette.
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
- `make ci` -- the plain-CPython gate that mirrors GitHub Actions -- now typechecks clean. It was
  red for 19 mypy errors that `make check` on the Isaac Sim interpreter does not see, because the
  two resolve different numpy versions (2.2.6 against 2.3.1) and numpy 2.x made `ndarray` generic
  over **shape** as well as dtype. Twelve were bare `np.ndarray` annotations in the probes, now
  `NDArray[Any]` -- the deliberate choice there, since probe code handles whatever an annotator
  returns and pinning a dtype would assert what the probe exists to measure. The rest were
  shape-parameter widening (`np.roll`, `np.tensordot`, fancy indexing) restated at the point it
  happens, and two `.shape` values narrowed to the 2-tuple their callee declares.
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
