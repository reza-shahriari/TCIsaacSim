# Isaac Sim Step-by-Step Checklist

Companion to **isaacsim_implementation_plan.md**. Same 7 phases, same order — broken into steps small enough to finish and verify in one sitting each. Every step has a concrete "Check" — something you can actually look at or measure, not "does it seem right."

## Phase 1 — Physics-Only Validation (no Isaac Sim yet)

- [ ] **1.1** Set up a clean Python environment (venv) with numpy and matplotlib. *Check: `python -c "import numpy, matplotlib"` runs with no errors.*
- [ ] **1.2** Implement `L_bb(λ, T)` (Planck's law, thermal_camera_model.md §1). *Check: at T=300 K the curve peaks near λ≈9.7 μm (Wien's law: λ_max ≈ 2898/T μm) — plot it and confirm.*
- [ ] **1.3** Band-integrate to get `L_band(T)` over one band (e.g. LWIR 8–14 μm) as a lookup table over 250–400 K. *Check: plot `L_band(T)` — smooth and strictly increasing, no kinks or negative values.*
- [ ] **1.4** Build a tiny synthetic scene: a 2D numpy array of `T` (e.g. a 280–320 K gradient) plus constant `ε`, `R`. *Check: `imshow()` the raw array and confirm it looks like the gradient you intended.*
- [ ] **1.5** Implement the emission + reflection equation (§1). *Check: with `ε=1`, output exactly equals `L_bb(T)`; with `ε=0`, output exactly equals `L_bb(T_env)` regardless of `T`.*
- [ ] **1.6** Implement atmospheric transmission + path radiance (§3) as a function of `R`. *Check: at `R=0`, output exactly equals the input; as `R→∞`, output approaches `L_bb(T_atm)` regardless of the source.*
- [ ] **1.7** Implement the optics equation (§4). *Check: doubling the f-number `N` should quarter the irradiance, all else equal — verify numerically.*
- [ ] **1.8** Implement a simple linear detector response (§5, the `K_resp·E + Offset` shortcut). *Check: monotonically increasing in irradiance, sensible magnitude (not all zeros, not saturating) across your scene's temperature range.*
- [ ] **1.9** Implement NETD-based noise injection (§8). *Check: run 1,000+ noise realizations at a fixed input, measure the empirical std dev, confirm it's close to `NETD · dSignal/dT`.*
- [ ] **1.10** Implement quantization to N-bit integers (§9). *Check: output contains only integers in `[0, 2^bits−1]`, even for extreme inputs.*
- [ ] **1.11** Implement AGC (percentile-clipped linear scaling, §10). *Check: output spans close to the full 0–255 range for a normal scene, without one outlier pixel crushing the rest of the contrast.*
- [ ] **1.12** Implement the colormap LUT (§11) and run the full pipeline end to end on the synthetic gradient. *Check: visually reads as a plausible thermal image, warm/cool ends clearly distinguishable, and re-running gives a slightly different (noisy) result each time.*

**Phase 1 done when:** you can hand this pipeline any `T(x,y)`, `ε(x,y)`, `R(x,y)` array and get back a believable false-color image — with zero dependency on Isaac Sim.

## Phase 2 — Coarse End-to-End in Isaac Sim

- [ ] **2.1** Install Isaac Sim via Quick Install (not pip) for this first pass — pip installs have known intermittent failures reported on the NVIDIA forums, and Quick Install is what NVIDIA itself recommends starting with. *Check: Isaac Sim launches, empty stage visible.*
- [ ] **2.2** If/when you set up standalone scripting, confirm your Python version matches (3.11 for Isaac Sim 5.x, 3.10 for 4.x). *Check: `python --version` matches.*
- [ ] **2.3** Run the bare `SimulationApp({"headless": True})` launch-then-close pattern as its own script, no scene content. *Check: runs to completion, no errors.*
- [ ] **2.4** Build a trivial scene: one ground plane, one cube. *Check: both visible in the viewport.*
- [ ] **2.5** Add a Camera prim, position it to frame the scene. *Check: switching the viewport to that camera shows the plane and cube.*
- [ ] **2.6** Capture a single frame with the built-in `BasicWriter`, `rgb` annotator only. *Check: an image file is written and opens correctly, showing the scene.*
- [ ] **2.7** Label the plane and cube with different semantic classes via the Semantics Schema Editor (e.g. "ground", "metal"). *Check: the editor panel lists both labels correctly assigned.*
- [ ] **2.8** Re-capture with `semantic_segmentation` added. *Check: output shows two distinct regions/colors matching the plane and cube.*
- [ ] **2.9** Add `distance_to_image_plane` and `normals` to the same capture. *Check: depth output shows a sensible near/far gradient; normals output is color-coded and varies across the cube's faces.*
- [ ] **2.10** Write your own custom `Writer` subclass (skeleton in isaacsim_implementation_plan.md §1) that does nothing but re-save the `rgb` annotator untouched. *Check: pixel-identical to step 2.6's output — proves your writer plumbing works before any real logic goes in.*
- [ ] **2.11** In that writer, hardcode a `{"ground": 290, "metal": 340}` label→temperature dict and output a flat color per segment (skip the full pipeline for now). *Check: output shows exactly two flat colors, correctly matching which object you set hotter.*
- [ ] **2.12** Wire in your real Phase 1 pipeline: semantic lookup for `T`, real `distance_to_image_plane` for `R`, full atmosphere → optics → detector → noise → colormap. *Check: output looks like a real noisy thermal image; editing the temperature dict and re-running visibly changes that object's color.*

**Phase 2 done when:** a real Isaac Sim scene, with per-material (not yet per-point) temperatures, produces a thermal image through your actual physics — not a stub.

## Phase 3 — Time-Varying Dynamics

- [ ] **3.1** Implement the lumped-capacitance ODE (thermal_object_dynamics.md §1) as a standalone class with a `.step(dt)` method. *Check: unit test against the closed-form `T(t) = T_ambient + (Q_run/G_th)(1−exp(−t/τ))` for constant `Q_in` — should match within a small tolerance.*
- [ ] **3.2** Replace the hardcoded "metal" temperature from step 2.11 with this object's current `T`. *Check: printing `T` once per loop shows it changing, not stuck constant.*
- [ ] **3.3** Call `.step(dt)` once per sim tick using the real sim `dt`. *Check: log `(sim_time, T)` and confirm the shape matches the closed-form warm-up curve when plotted.*
- [ ] **3.4** Capture a short frame sequence while it warms up. *Check: played back in order, the object visibly gets warmer-colored over time.*
- [ ] **3.5** Add the cool-down case — flip `Q_in` to 0 partway through, keep stepping. *Check: temperature turns around and decays back toward ambient, matching the cool-down closed form.*

**Phase 3 done when:** at least one object's temperature is driven by a real ODE stepped every frame, numerically checked against its closed-form solution — not a static value.

## Phase 4 — Continuous Field, on GPU (via SPG)

- [ ] **4.1** Enable the `omni.rtx.spg` extension (Extension Manager, or launch Kit with `--enable omni.rtx.spg`). *Check: it shows up in the enabled extensions list.*
- [ ] **4.2** Run NVIDIA's own grayscale-conversion SPG tutorial, unmodified, on a test scene. *Check: switching the viewport's display render var to the tutorial's output AOV shows a grayscale-converted image — proves your SPG toolchain (NVRTC compile, Lua launch, USD wiring) works before any thermal-specific code goes in.*
- [ ] **4.3** Add a `primvars:temperature` float attribute to a single test mesh (per-vertex, not one constant) via a small USD Python script. *Check: read it back in a separate call and confirm the values match what you set.*
- [ ] **4.4** Assign that mesh an OmniPBR material with emission enabled, driven by the primvar rather than a flat constant. *Check: the mesh renders as a varying (not flat) emissive color pattern, unaffected by moving scene lights.*
- [ ] **4.5** Request the `PtSelfIllumination` AOV on your RenderProduct (fall back to `HdrColor` with bounces=0 if `PtSelfIllumination` isn't available on your install), no SPG shader yet. *Check: capturing that AOV directly shows just the emissive pattern — background and non-emissive geometry read as black.*
- [ ] **4.6** Write your own minimal SPG shader (copy NVIDIA's grayscale example, rename) that passes the emission AOV through unchanged. *Check: output is pixel-identical to step 4.5's raw capture — proves your shader plumbing works before real physics logic goes in. This is the hardest single checkpoint in the whole plan — the person who originally requested this feature got stuck exactly here, so budget real time for it.*
- [ ] **4.7** Add your real per-pixel pipeline (atmosphere → optics → detector → noise → colormap, thermal_camera_model.md §3–§11) into the CUDA kernel. *Check: output looks like a real noisy thermal image; changing the mesh's temperature primvar and re-running visibly changes its rendered color.*
- [ ] **4.8** Swap this in for the "ground" mesh from Phase 2, replacing its flat per-material lookup. *Check: final output now shows genuine gradient variation across the ground, not one flat color — and it's running entirely on GPU, no per-frame CPU round-trip.*

**Phase 4 done when:** at least one mesh has real per-point (not per-object) temperature flowing all the way to the final image through your own SPG shader — the original point-wise requirement, actually working, on GPU.

## Phase 5 — Special Materials

- [ ] **5.1** Water: assign a much larger `C_th` (thermal_object_dynamics.md §3) to a test region, run it through a simulated day/night ambient cycle. *Check: that region's temperature swing is visibly smaller and phase-delayed vs. a nearby "ground" region under the same cycle.*
- [ ] **5.2** Ice: implement the enthalpy-tracking function (§4) standalone, as a unit test. *Check: constant heat input → `T` plateaus at 0°C for the expected duration (`m·L_f / heat rate`) before rising.*
- [ ] **5.3** Wire the ice enthalpy state into a test region the same way as the engine in Phase 3. *Check: the region's rendered color visibly holds constant during the plateau, then changes afterward — not a smooth ramp.*
- [ ] **5.4** Fire: implement the flicker function (noise + ~10–20 Hz modulation, §5) standalone, plot several seconds of output. *Check: oscillates in the expected frequency range, doesn't look smooth/static.*
- [ ] **5.5** Wire the flicker function into one region's temperature (a flat billboard/texture region is fine — not a true 3D volume yet). *Check: a captured sequence visibly flickers frame to frame.*

**Phase 5 done when:** water, ice, and fire each behave differently from a plain object under the same engine — because their underlying dynamics genuinely differ, not because you special-cased their rendering.

## Phase 6 — ROS2 + Rate Control

- [ ] **6.1** Verify ROS2 Humble is installed and sourced, independent of Isaac Sim. *Check: `ros2 topic list` runs with no errors in a plain terminal.*
- [ ] **6.2** Run Isaac Sim's own built-in RGB camera → ROS2 tutorial, unmodified, before touching your custom writer. *Check: `rqt_image_view` or `ros2 topic echo` shows the standard camera feed.*
- [ ] **6.3** Modify your custom writer to also publish a `sensor_msgs/Image` via `rclpy`, alongside (or instead of) saving to disk. *Check: `ros2 topic list` shows your new topic.*
- [ ] **6.4** Check the publish rate. *Check: `ros2 topic hz <your topic>` reports a stable, sane frequency.*
- [ ] **6.5** View the image content in `rqt_image_view` or `rviz2`. *Check: the colorized thermal image appears correctly, matching what you saw saved to disk earlier.*
- [ ] **6.6** Add rate control (`omni:sensor:tickRate` or the Isaac Simulation Gate node) to decouple publish rate from render rate. *Check: changing the setting changes the measured `ros2 topic hz`.*

**Phase 6 done when:** the thermal camera behaves like an ordinary ROS2 sensor — discoverable, subscribable, rate-controllable — from outside Isaac Sim entirely.

## Phase 7 — Calibration

- [ ] **7.1** Gather one or two reference images — a real thermal camera photo if accessible, or a comparable DIRSIG-rendered scene. *Check: you have the image(s) plus a rough idea of the true scene temperatures.*
- [ ] **7.2** Reproduce the reference scene's rough geometry and temperatures in your simulator. *Check: side by side, the general layout of hot/cold regions corresponds.*
- [ ] **7.3** Tune NETD, gain, and AGC until the noise texture and contrast visually match. *Check: measure noise `σ` in a flat region of both images — yours should be in the same ballpark, not off by an order of magnitude.*

**Phase 7 done when:** you have at least one concrete data point showing your output is in the right neighborhood of a real (or high-fidelity reference) thermal image — not just internally self-consistent.
