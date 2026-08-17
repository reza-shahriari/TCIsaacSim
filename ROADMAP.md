# Roadmap

File-and-test-mapped version of `docs/isaacsim_checklist.md` — same 7 phases, same order, pointing at real paths in this repo. Read the checklist for the *why* and the full verification reasoning behind each check; use this to track *where* and confirm *done*.

Work top to bottom. Don't start a phase's Isaac-Sim-dependent work until the phase above it is green.

## Phase 1 — Physics only (no Isaac Sim)

- [ ] `thermal_physics/emission.py` — `planck_radiance`, `band_radiance`, `surface_leaving_radiance`
- [ ] `thermal_physics/atmosphere.py` — `transmittance`, `path_radiance`, `apply_atmosphere`
- [ ] `thermal_physics/optics.py` — `focal_plane_irradiance`
- [ ] `thermal_physics/detector.py` — `linear_detector_signal` first; `bolometer_*` functions once that's solid
- [ ] `thermal_physics/blur.py` — `diffraction_mtf`, `detector_mtf`, `apply_system_blur`
- [ ] `thermal_physics/noise.py` — `netd_noise_sigma`, `add_temporal_noise`, `add_fixed_pattern_noise`
- [ ] `thermal_physics/quantize.py`, `agc.py`, `colormap.py`
- [ ] `thermal_physics/pipeline.py` — wire all of the above into `render_frame()`

```bash
pytest tests/test_emission.py tests/test_atmosphere.py tests/test_optics.py \
       tests/test_detector.py tests/test_noise.py tests/test_pipeline.py -v
```

**Done when:** every test above passes (not skips), and `render_frame()` on a synthetic temperature gradient produces a believable image. Nothing here should import `omni` or `isaacsim`.

## Phase 2 — Coarse Isaac Sim prototype

- [ ] Isaac Sim installed (Quick Install), `scripts/run_standalone.py` launches and exits clean
- [ ] Trivial scene (plane + cube) + camera, one `rgb` frame captured with the built-in `BasicWriter`
- [ ] Semantic labels via Semantics Schema Editor, `semantic_segmentation` annotator confirmed
- [ ] `isaac_ext/thermal.camera/thermal/camera/writer_prototype.py` — fill in `ThermalWriterPrototype.write()`, calling `thermal_physics.pipeline.render_frame`
- [ ] `scripts/capture_frame.py` — wire up a `RenderProduct` + the prototype writer

**Done when:** a real Isaac Sim scene, with per-material (not yet per-point) temperatures, produces a thermal image through `thermal_physics.pipeline.render_frame` — not a stub.

## Phase 3 — Time-varying dynamics

- [ ] `thermal_physics/dynamics/lumped_capacitance.py` — `LumpedThermalNode`
- [ ] Wire a `LumpedThermalNode` into the Phase 2 writer's per-material lookup, `.step()` once per sim tick

```bash
pytest tests/test_dynamics.py::test_lumped_capacitance_matches_closed_form -v
```

**Done when:** that test passes, and a captured frame sequence visibly shows an object warming up.

## Phase 4 — Continuous field, on GPU (SPG)

- [ ] `omni.rtx.spg` enabled, NVIDIA's own grayscale-conversion tutorial runs unmodified on a test scene
- [ ] `primvars:temperature` round-trips correctly on a test mesh (per-vertex, not one constant)
- [ ] OmniPBR material with emission enabled, driven by the primvar
- [ ] `PtSelfIllumination` AOV requested (fallback: `HdrColor` + bounces=0) and confirmed clean
- [ ] `spg/ThermalKernel.cu` / `.cu.lua` / `.usda` — start as an unchanged pass-through of the emission AOV
- [ ] Port the real pipeline (`thermal_physics/` SS3-SS11 math) into `ThermalKernel.cu`
- [ ] Swap this in for the "ground" mesh from Phase 2, replacing its flat per-material lookup

**Done when:** at least one mesh shows genuine per-point (not per-object) temperature in the final image, computed entirely on GPU — the original point-wise requirement, actually working. This phase has the highest risk of the whole roadmap; see `docs/isaacsim_implementation_plan.md` SS8 before you start.

## Phase 5 — Special materials

- [ ] `thermal_physics/dynamics/water.py` — `evaporative_loss_w`, plus a water-parameterized `LumpedThermalNode`
- [ ] `thermal_physics/dynamics/ice.py` — `EnthalpyThermalNode`
- [ ] `thermal_physics/dynamics/fire.py` — `flicker_signal`, `volumetric_emission`

```bash
pytest tests/test_dynamics.py -v
```

**Done when:** all of `test_dynamics.py` passes, and water/ice/fire each visibly behave differently from a plain object under the same engine.

## Phase 6 — ROS2

- [ ] `ros2 topic list` works standalone, independent of Isaac Sim
- [ ] Isaac Sim's built-in RGB camera -> ROS2 tutorial confirmed working, unmodified
- [ ] Add an `rclpy` publish step to the writer/SPG hand-off
- [ ] Rate control via `omni:sensor:tickRate` (Isaac Sim 6.0+) or the Isaac Simulation Gate node

**Done when:** `ros2 topic hz <your topic>` reports a stable rate and `rqt_image_view`/`rviz2` shows the colorized thermal image correctly.

## Phase 7 — Calibration

- [ ] Reference image gathered (real thermal camera capture, or a comparable DIRSIG render)
- [ ] Scene geometry/temperatures roughly reproduced
- [ ] NETD/gain/AGC tuned until noise `σ` in a flat region is in the same ballpark as the reference

**Done when:** you have one concrete data point showing the simulator's output is in the right neighborhood of a real (or high-fidelity reference) thermal image.
