# ADR 0014 — Isaac Sim 6.1 findings: transport ids and geometry, not temperature

**Status:** Accepted
**Date:** 2026-09-10

## Context

`docs/physics-model.md` §13.1/§13.3, the `isaac-sim-spg` skill and roadmap M2 assumed the renderer
could carry an encoded surface temperature `c = (T − 200)/800` in an emissive colour AOV with a
**float32** channel (`PtSelfIllumination`, or `HdrColor` at zero bounces), and that the RTX Sensor
Processing Graph (`omni.rtx.spg`) could hold cross-frame state and read a LUT file. None of that had
been checked on a real build. Roadmap M2 is the time-boxed gate spike that checks it before any
pipeline code depends on it. This ADR records what was measured (`src/irsim_isaac/probe.py`,
`src/irsim_isaac/spg_probe.py`, `scripts/probe_isaac_environment.py`, `scripts/probe_isaac_spg.py`;
reports under the git-ignored `outputs/isaac_probe/`) and the design that follows.

### The build every number below comes from

| item | measured |
|---|---|
| Isaac Sim | **6.1.0-rc.26** (`isaacsim.core.version.get_version()`), source build at `/home/hunter/IsaacSim`; CLAUDE.md, README and §13 say "6.0" (spec-issues T16) |
| Kit | 110.3.0+feature.371399.00c488ae.gl; real-time render mode is named `RealTimePathTracing` (the setting value `RaytracedLighting` is silently ignored) |
| `omni.rtx.spg` | 0.4.0, present and enabled by default; docs and Lua stubs live in the extension folder (`docs/Overview.md`, `.luarc/cuda.lua`) |
| `isaacsim.sensors.experimental.rtx` | 1.9.0: `RtxCamera`, `CameraSensor`, `TiledCameraSensor`, `SPGNode`, `author_spg` |
| deprecated `isaacsim.sensors.camera` / `.rtx` | present in `extsDeprecated/`, enabled, not imported by `irsim_isaac` (guarded by `test_environment.py`) |
| Warp | 1.16.0 from the `omni.warp.core` Kit extension; **importable only inside a running Kit** (`has_warp()` is False from plain `python.sh`); one CUDA device (RTX A6000 48 GB, driver 580.173.02) |
| Replicator | `omni.replicator.core` 1.13.36, `omni.syntheticdata` 0.6.17 |
| Kit boot | 14–35 s headless; `SimulationApp.close()` ends in `os._exit`, so the pytest session fixture closes the app from an `atexit` handler after pytest has reported |

## What was measured

### M2.2 — the gate: temperature in emission (fails as specified)

Scene: 64 flat OmniPBR quads (UsdShade authoring: `Shader` with `OmniPBR.mdl`, `enable_emission`,
`emissive_color = (c, c, c)` in raw colour space, `diffuse_color_constant = (id/255, 0, 0)`), a
temperature ramp 200–1000 K including 300.000 / 300.050 / 300.100 K, camera at the origin looking
−Z, no lights, 256², DLSS off (`/rtx/post/aa/op = 0`). Read back through Replicator annotators in
real-time mode and in path tracing at 0 / 1 / 4 bounces, emissive intensity 1 and 40.

| AOV | dtype | resolution | result |
|---|---|---|---|
| `HdrColor` | **float16** ×4 | 256² | linear in colour × intensity with slope **≈ 2.8 × 10⁻⁴** (0.00028 at intensity 1, 0.0112 at 40): an exposure scale, not a unit gain; at intensity 1 the whole ramp sits in the fp16 subnormal/denormal range |
| `LdrColor` | uint8 ×4 | 256² | auto-exposed (ramp reads 0 at intensity 1, 0–16 at 40) |
| `PtSelfIllumination`, `PtDirectIllumation`, `PtGlobalIllumination`, `PtBackground` | **float16** ×4 | 256² | 8.8 × 10⁻⁴ floor with a ≈ 2.4 × 10⁻⁴ per unit-c signal on top at 4 bounces; constant at 0 bounces |
| `GroundTruthEmission`, `GroundTruthEmissionAndForegroundMask`, `GroundTruthDiffuseAlbedo`, `GroundTruthAmbientOcclusion` | float32 (registered) | — | **no data returned** |
| `EmissionAndForegroundMask`, `DiffuseAlbedo`, `BumpNormal` | fp16 / uint8 / float32 | **128²** even with DLSS off | not usable at full resolution |
| `AmbientOcclusion`, `SmoothNormal`, `Motion2d` (real-time) | — | — | no data on the unlit, static ramp scene |

fp16 has 11 significant bits: at c = 0.125 (300 K) its spacing is 1.2 × 10⁻⁴ → **≈ 100 mK**, ten
times the 10 mK bound, and the measured scale factor puts the values far from where the pair codec
of ADR 0006 was designed to run. A control quad with grey 0.5 read 6 × 10⁻⁸ at intensity 1. The
materials authored through `CreateMdlMaterialPrimCommand` + `omni.usd.create_material_input`
silently emitted nothing; only the UsdShade path works.

### What does transport exactly

| channel | dtype / shape | measured |
|---|---|---|
| `instance_segmentation`, `semantic_segmentation`, `instance_id_segmentation` (`colorize=False`) | uint32, 256² | **64 / 64 quad centres carry distinct non-background ids, every 5 × 5 centre window is a single id** (no edge blending); `idToLabels` maps id → prim path or `{"class": label}`; 0 = background, 1 = unlabelled |
| `Camera3dPositionSD`, `PtWorldPos` | float32 ×4, 256² | world position; mean error vs the authored quad centres (3.45, 3.37, 0.0) mm in (x, y, z), max 6.5 mm — one pixel at 2 m is 6.8 mm |
| `DistanceToCameraSD` | float32, 256² | **Euclidean ray length** (1.0 mm mean error), `inf` where nothing is hit |
| `DistanceToImagePlaneSD` | float32, 256² | **z-depth** (0.0 mm error); `PtZDepth` is float16 and reads 1.0 |
| `PtWorldNormal` | float32 ×4 | (0, 0, 1) at 4 bounces, zeros at 0–1 bounces |

### M2.3 — SPG capabilities (`RtxCamera.author_spg(SPGNode(...))`, one render product per graph)

| experiment | result |
|---|---|
| (a) float32 pass-through | **bit-identical**: `DistanceToCameraSD`, `DistanceToImagePlaneSD`, `Camera3dPositionSD` (`inf` pattern identical, max diff 0.0); `LdrColor` identical; `HdrColor` identical on the fp16 grid. Read through `register_annotator_from_aov(output_data_type=np.float32)`. Two nodes on one product work |
| (b) cross-frame state | **none**: a `cuda.static` buffer from `cuda.array` or `cuda.zeros` reads 1 every frame after an in-kernel `atomicAdd` (re-initialised per frame); `cuda.empty` is an output descriptor, not a kernel argument (`args[3] missing 'name' field`); wiring the node's output AOV back into its own input yields no output at all, silently. Lua file-scope variables **do** persist (the launch function runs once per rendered frame) and `rtx.frameId` exists, advancing 6–7 per `rep.orchestrator.step()` |
| (c) LUT delivery | `io.` is a forbidden token: the sandbox validator rejects the **whole file** textually (`LuaSandbox: Lua source rejected - forbidden pattern 'io.' found`). A 16 001-entry Lua literal at file scope **crashes the Kit process** under the default `/rtx/spg/lua/instructionLimit = 1000` (`lua_error` inside `librtx.spg.fabric`), a 256-entry literal works, a Lua loop fails cleanly (`Lua script execution limit exceeded`); with the limit raised to 10⁷ (settable at runtime) both forms work and match NumPy exactly. A **`__device__ const float[16001]` baked into the `.cu`** (205 kB source) compiles under NVRTC in ≈ 1 s and matches NumPy exactly under default limits |
| hazard | a kernel that fails to load (wrong symbol name: SPG resolves the CUDA function by the node's `sub_identifier`) still produces a **zero-filled output with status ok** |

### Things that cost time and are worth knowing

`rep.create.render_product(path, (w, h))` and `CameraSensor(cam, resolution=(h, w))` give the same
pixels; `CameraSensor` also applies `OmniRtxCameraExposureAPI_1` / `OmniRtxCameraAutoExposureAPI_1`
to the camera — that schema is where the colour scale above lives. In one run a camera prim
pre-authored with `UsdGeom.Camera.Define` and then wrapped by `RtxCamera` rendered all-`inf` depth;
letting `RtxCamera(path)` create the prim and setting focal length / apertures / clipping afterwards
worked in every later run (not root-caused). Prim names may not contain `.`. Path tracing with
`maxBounces = 0` degenerates every `Pt*` AOV to a constant. `SimulationApp` parses its own
command line with `parse_known_args`, so runner scripts may add arguments.

## Options considered

1. **Temperature in emission through the fp16 colour path** — ≈ 100 mK at best; fails CLAUDE.md #2
   and the gate. Rejected.
2. **fp16 coarse/fine pair (ADR 0006 fallback) through the exposure scale** — possible in principle
   (the fp16 error after scaling is about half the coarse-bin spacing), but it depends on an
   undocumented per-camera exposure constant and every tonemap, colour-management or exposure
   setting change would break it silently — exactly the failure mode CLAUDE.md #2 warns about.
   Rejected; kept as the documented fallback only if a colour path is ever unavoidable.
3. **A float32 emission AOV** — `GroundTruthEmission*` return no data and there is no documented
   way to add a float32 renderer AOV from Python in this build. Rejected until it appears.
4. **Transport ids and geometry, look the temperature up** — the renderer emits exact integer
   instance/semantic ids and float32 position / distance at full resolution; the per-facet
   temperature lives in a float32 table on the Warp side, filled by the thermal bridge. No
   quantisation at all, and the material-id transport question (M2.4, R4) is answered by the same
   channel. **Chosen.**

## Decision

- **The G-buffer's `temperature_k` plane is assembled in Warp from `instance_segmentation` and a
  float32 facet table**, not decoded from a colour AOV. The `GBuffer` contract (M0.6) is unchanged;
  `encoded_t` becomes optional and is never produced by the Isaac adapter. `material_id` comes from
  the same id channel through the resolver (M10.2). Sky pixels are background id 0 and take their
  radiance from the per-pixel ray direction (`Camera3dPositionSD`) and the MS.2 sky model; no
  emissive dome is needed.
- **Geometry AOVs:** `DistanceToCameraSD` for path length, `DistanceToImagePlaneSD` only where
  z-depth is wanted, `Camera3dPositionSD` for world position; all read at full resolution with
  `/rtx/post/aa/op = 0`. Normals, ambient occlusion and motion vectors are re-probed by M10.1 on a
  lit, moving, tilted-geometry scene (they returned nothing on the unlit static ramp).
- **SPG** hosts stateless stages only. Stages 4–5 (bolometer IIR, drift, RTS) stay in Warp
  (M10.13c/d deferred; ADR 0061 records it). Band LUTs reach SPG kernels **baked into the generated
  `.cu` as `__device__` arrays** (M10.12); the Lua launch script stays small and the sandbox
  limits are never raised as a dependency. `FileCapture`-style tests compare values, never just
  presence, because of the zero-filled-on-failure hazard.
- **Version pin:** the facts above hold for Isaac Sim 6.1.0-rc.26 / Kit 110.3.0 / `omni.rtx.spg`
  0.4.0 / `isaacsim.sensors.experimental.rtx` 1.9.0 / Warp 1.16.0. No `warp-lang` extra is added
  to `pyproject.toml`: Warp is a Kit extension and a pip copy would shadow it.
- `tests/integration/test_environment.py` and `test_isaac_transport.py` pin the build facts, the
  exact ids, the float32 geometry, the bit-exact SPG pass-through **and the fp16 finding itself**,
  so a build that changes any of them fails a test instead of silently changing the physics.

## Consequences

- Temperature is exact at **facet (prim / instance) granularity** and constant within a facet.
  Sub-facet gradients (an engine bay hot spot on one mesh) are not represented until meshes are
  split per thermal facet or an id-per-face channel exists — a future ADR under M10.3. The error
  is bounded by the thermal model's own facet size, which is where the temperature is solved
  anyway.
- Ids do not blend, so object edges are hard at the id level; edge radiance must be anti-aliased by
  supersampling the id/geometry buffers and filtering **radiance**, never ids (M3 / M10 kernels).
- Position and distance carry one pixel of quantisation (≈ 3.4 mm at 2 m and 256²): negligible for
  path length (1.7 × 10⁻³ relative here, far less at range).
- The phase-1 thermal bridge (M10.18) writes a table, not USD emissive colours: cheaper per frame
  and no `Sdf.ChangeBlock` traffic.
- The Warp reference pipeline is the production path until `omni.rtx.spg` gains state; SPG stays
  an optimisation for stages 1–3.
- §13.1/§13.3 and the skill's AOV table are now wrong for this build (spec-issues T17); the spec
  owner rewrites them.

## Revisit when

- `test_colour_aovs_are_float16_in_this_build` fails — a float32 colour or custom AOV appeared.
- `omni.rtx.spg` documents persistent buffers or feedback edges (then M10.13c/d come back).
- A target needs sub-facet temperature structure (id-per-face or a UV temperature texture).
- Isaac Sim moves off the 6.1 line; re-run `scripts/probe_isaac_environment.py` and
  `scripts/probe_isaac_spg.py` and diff the reports against this ADR.

---

## Addendum 2026-09-12 — geometry AOVs on a lit, tilted, moving scene (M10.1)

The decision above was measured on an unlit, static, front-parallel ramp with the camera at the
world origin. On that scene the normals, ambient-occlusion and motion annotators returned nothing,
and a world-space and a camera-space channel were numerically identical, so four questions were
left open. `scripts/probe_isaac_geometry.py` answers them on the M10.1 scene
(`src/irsim_isaac/geometry_probe.py`): lit by a distant light, a sphere and a 24° tilted quad,
plates facing up / sideways / down, a translating bar, and the **camera at (2, 1, 5)** rather than
the origin. Same build as above (6.1.0-rc.26 / Kit 110.3.0 / Replicator 1.13.36), 384² render
product, `/rtx/post/aa/op = 0`.

| channel | annotator | dtype | shape | verdict |
|---|---|---|---|---|
| distance | `DistanceToCameraSD` | float32 | 384² | **use.** Euclidean ray length; `inf` where nothing is hit. `DistanceToImagePlaneSD` (z-depth) also delivers and must not be substituted |
| position | `Camera3dPositionSD` | float32 ×4 | 384² | **use.** **World space** (misses carry a −1000 sentinel). `PtWorldPos` delivers at 192² only |
| normal | `normals` | float32 ×4 | 384² | **use.** **World space** (see the pitch test below) |
| normal | `PtWorldNormal` | float16 ×4 | **192²** | **do not use.** Attaches, returns data, and is **all zero** |
| normal | `BumpNormal` | float32 ×4 | 192² | unusable: half resolution, values up to ±3 × 10³⁸ |
| normal | `SmoothNormal` | — | — | no data; `NormalSD` / `PtSmoothNormal` are not registered |
| occlusion | `AmbientOcclusion` | — | — | **no data** on a lit scene either; no alternative name is registered |
| motion | `motion_vectors` | float32 ×4 | 384² | **carries no motion.** A constant ≈ 6 × 10⁻⁵ floor after displacements of 3 px **and 180 px** |
| motion | `Motion2d` | — | — | no data (`SdPostRenderVarToHost: invalid input resource`) |
| instance / semantic | `instance_segmentation`, `instance_id_segmentation`, `semantic_segmentation` | uint32 | 384² | **use.** 7 distinct ids for 6 targets plus background, as ADR 0014 found |

**The normals frame.** With an axis-aligned camera a world-space and a camera-space normal are the
same numbers, so the scene cannot distinguish them — and getting it wrong silently corrupts every
angular emissivity and sky-view factor downstream. Pitching the camera 20° breaks the degeneracy:
the up-facing plate's normal must stay exactly `(0, 1, 0)` in world space, where a camera-space
normal would read `cos 20° = 0.940` on the up axis. Measured `max |n·up| > 0.99`
(`test_normals_are_world_space_not_camera_space`), so the AOV is world space.

**Resolution is a trap, not a detail.** Several AOVs come back at the renderer's internal
resolution (192² for a 384² product) whether or not AA is off, and a plane at the wrong resolution
misaddresses every pixel lookup without raising anything. `configure_renderer()` exists so no
caller forgets `/rtx/post/aa/op = 0`, and `AovReader` rejects a required channel whose shape does
not match the render product.

### Decisions that follow

- **The adapter's annotator preference order is a measurement.** `normals` before `PtWorldNormal`,
  `motion_vectors` before `Motion2d`. `irsim_isaac.pipeline.gbuffer_isaac.AOV_NAMES` is the single
  place it is written down, and `scripts/probe_isaac_geometry.py` regenerates the evidence.
- **A required channel that returns an all-zero buffer is a failure, not data.** This is the same
  hazard the SPG lane already records (a kernel that fails to load still produces a zero-filled
  output with status ok). `PtWorldNormal` would otherwise have produced a G-buffer that passes the
  M0.6 schema and contains no surface orientation at all.
- **`sky_view_factor` uses the unoccluded geometric form** `V_s = (1 + n·up)/2`, because no ambient
  occlusion AOV delivers. This is exact for the open-sky aerial scenes of phase 1 (ADR 0003) and
  optimistic for a cluttered ground scene, where a wall in a street sees less sky than the formula
  says. Phase 2 needs either a working AO AOV or the §5.3(b) irradiance cubemap.
- **`motion_px` is omitted from the Isaac G-buffer.** It is optional in the M0.6 contract, so
  nothing downstream breaks today; what does not work today is motion MTF and bolometer smear
  *in-sim* (M10.11, M10.19), which need per-pixel image-plane velocity. The cheap fix, when those
  steps arrive, is to synthesise it analytically from the per-prim transforms and the camera pose
  rather than from an AOV — exact for rigid targets, and it removes a renderer dependency. That is
  a step, not a silent addition to M10.1; it is on the roadmap as M10.1b.
- **Temperature and material id are still not renderer-transported.** Unchanged from the main
  decision: `to_gbuffer()` takes them as float32 facet-table lookups keyed by instance id.

### Revisit when

- `test_motion_aov_does_not_transport_motion_on_this_build` fails — the channel started working;
  determine its units and wire `motion_px` instead of synthesising it.
- `AmbientOcclusion` returns data — then `V_s` picks up real occlusion for ground scenes.
- `PtWorldNormal` stops being all-zero, or any of these AOVs changes resolution or dtype; the
  integration tests pin each of those facts and will fail first.

---

## Addendum 2026-09-12 — instance ids are only per-prim on `instance_id_segmentation` (M10.2)

The main decision above rests on "the renderer emits exact integer instance/semantic ids", measured
as 64 distinct ids at 64 quad centres. That measurement is correct but its scene was special: the
ramp's `label_quads()` applied a `class` semantic to **every** quad. On the M10.2 stage, where only
one prim of five carries a semantic, the same annotator behaves very differently.

| annotator | ids present | `idToLabels` |
|---|---|---|
| `instance_segmentation` | `0, 1, 2` | `{0: BACKGROUND, 1: UNLABELLED, 2: /World/Targets/Road}` |
| `semantic_segmentation` | `0, 1, 2` | `{0: {class: BACKGROUND}, 1: {class: UNLABELLED}, 2: {class: road}}` |
| **`instance_id_segmentation`** | `0, 1, 2, 3, 4, 5` | **one prim path per prim**, semantics irrelevant |

`instance_segmentation` gives a distinct id only to prims that carry a semantic label and collapses
every unlabelled prim into a single `UNLABELLED` id. Four of the five test prims therefore shared
one id, and the material transport built on it painted all four with the first material it
resolved — a G-buffer that passes the M0.6 schema, renders a plausible image, and is made of the
wrong substances. Nothing raised.

### Decisions that follow

- **`material_id` transport uses `instance_id_segmentation`**, with `instance_segmentation` kept
  only as a fallback. `AOV_NAMES` records the order and the reason.
- **`colorize=False` is passed explicitly** on both segmentation channels (`AOV_INIT_PARAMS`). The
  default happened to be False here; a colorized id is an RGBA palette entry that cannot be looked
  up, only guessed at, so it is not left to a default.
- **Requiring semantics for correct ids would have been the wrong fix.** Semantics are for dataset
  labels and for the mapping's precedence rule 2 (ADR 0047); making radiometric correctness depend
  on an artist remembering to label a prim reintroduces exactly the silent-default failure the
  UNMAPPED sentinel exists to prevent.
- `scripts/audit_materials.py` gains `--stage` (walk a USD asset directly) and `--dump-prims`.
  The USD path boots Kit because `pxr` is not importable outside a running Kit application, so the
  two-step flow — dump once inside Kit, audit the JSON anywhere — stays the fast path. Both paths
  were measured to agree on the five-prim stage: `miss=1, override=1, pattern=2, semantic=1`.
- `omni.usd.get_context().open_stage()` returns a **bare bool** on this build, not the `(ok, error)`
  pair the older documentation shows; unpacking it raises *inside Kit*, which then reports exit
  status **0**. A script that boots Kit must therefore print its result before `SimulationApp.close()`
  (which ends in `os._exit`) and must not trust Kit to surface a Python failure as a non-zero exit.

### Revisit when

- `test_stage_walk_sees_every_binding_semantic_and_override` or the id-decode tests fail — the
  segmentation semantics changed.
- A build makes `instance_segmentation` per-prim regardless of labelling; the fallback order can
  then be simplified.
