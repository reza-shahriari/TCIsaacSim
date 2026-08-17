# Isaac Sim Implementation Plan

Companion to **thermal_camera_model.md** (the radiometric pipeline) and **thermal_object_dynamics.md** (how `T` changes over time). This is the plan for building both inside Isaac Sim specifically — grounded in what's actually documented and working right now (Isaac Sim 5.0/6.0, open source), not assumed.

**Hard prerequisite:** Isaac Sim's renderer is RTX-based (real ray-traced light transport) — you need an NVIDIA RTX-capable GPU. There's no software-rendering fallback worth using for this project.

**Revised:** §1 and §2 below originally proposed a hand-rolled two-tier design (Replicator annotators + a Python post-process Writer), reasoned from Isaac Sim's general extensibility patterns since no one had built this yet. Since then, Isaac Sim 6.0 GA shipped `omni.rtx.spg` (Sensor Processing Graph), and an Isaac Sim maintainer confirmed directly, in the GitHub discussion that originally requested this feature, that SPG is the intended mechanism. Both sections are updated below to match — verified against the official SPG documentation and the actual discussion thread, not taken at face value.

**The one finding that shapes everything below:** there is no native thermal sensor, but there is now a confirmed, NVIDIA-endorsed mechanism for building one — `omni.rtx.spg` — plus the RTX Lidar/Radar sensors' own extensibility framework (custom non-visual materials, custom annotators) as a second, complementary set of seams. You're not fighting the platform here; you're using the tools NVIDIA itself pointed a user to for this exact request.

## 1. Core Architecture: Sensor Processing Graph (SPG), GPU-Native

**`omni.rtx.spg`** runs custom GPU code as a post-processing pass directly on an RTX-rendered AOV — no CPU round-trip. Confirmed by an Isaac Sim maintainer as "the right tool for implementing thermal/IR cameras," and documented in full at [docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg](https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/latest/Overview.html). This is a better fit than a Python post-process for exactly the reason the rest of this pipeline is GPU-shaped: atmosphere → optics → detector → noise → colormap is a per-pixel numeric transform, which is what a CUDA kernel is built for.

**Every SPG shader is three files:**

| File | Role |
|---|---|
| `.cu` | CUDA kernel — the actual per-pixel math, `extern "C" __global__`, compiled at runtime via NVRTC |
| `.cu.lua` | Lua launch script (same base filename + `.lua`) — validates input shapes/dtypes, allocates outputs, returns the kernel launch config. Runs every frame; Lua was chosen for a small, fast-to-evaluate, sandboxed (no filesystem access) footprint — it does *not* do the pixel math itself |
| `.usda` | USD shader definition — declares inputs/outputs, points at the `.cu` file via `info:spg:sourceAsset`, names the function via `info:spg:sourceAsset:subIdentifier` (must match the CUDA function name and the Lua function name) |

Minimal skeleton, adapted from NVIDIA's own tutorial (verified function/argument names — not approximated):

```cuda
// ThermalKernel.cu
extern "C" __global__ void thermal_ir(
    int width, int height,
    float netdSigma,
    cudaTextureObject_t inputSignal,       // the emission-encoded temperature AOV — see §2
    cudaSurfaceObject_t outputThermalIR)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 px = tex2D<float4>(inputSignal, x, y);
    float signal = px.x;  // decode your encoded temperature here
    // ... atmosphere, optics, detector, noise, colormap from thermal_camera_model.md §3–§11 ...

    uchar4 out = { (unsigned char)(signal * 255.0f), 0, 0, 255 };
    surf2Dwrite<uchar4>(out, outputThermalIR, x * sizeof(uchar4), y);
}
```

```lua
-- ThermalKernel.cu.lua
function thermal_ir(inputs, outputs)
    local height = inputs["Signal"].shape[1]
    local width  = inputs["Signal"].shape[2]
    outputs["ThermalIR"] = cuda.image(width, height, cuda.uchar4)
    return cuda.kernel({
        args = {
            cuda.int(width), cuda.int(height),
            cuda.float(inputs["netdSigma"]),
            cuda.TextureObject(inputs["Signal"]),
            cuda.SurfaceObject(outputs["ThermalIR"]),
        },
        block = { 32, 32 },
        grid  = { math.ceil(width / 32), math.ceil(height / 32) },
    })
end
```

```usda
# ThermalKernel.usda
#usda 1.0
(defaultPrim = "ThermalKernel")

def Shader "ThermalKernel"
{
    uniform token info:implementationSource = "sourceAsset"
    uniform asset info:spg:sourceAsset = @./ThermalKernel.cu@
    uniform token info:spg:sourceAsset:subIdentifier = "thermal_ir"

    float inputs:netdSigma = 0.01
    opaque inputs:Signal
    opaque outputs:ThermalIR
}
```

Note the exact names: `cuda.TextureObject(...)` / `cuda.SurfaceObject(...)` (not `cuda.texture()`/`cuda.surface()`), and launch-config keys `block` / `grid` (not `blockDim`/`gridDim`) — a couple of specifics that show up differently elsewhere don't match the current official reference and would fail at runtime as written.

**A thin hand-off still exists** for getting the final AOV off the GPU when you actually need it outside the renderer — a `FileCapture`, a Replicator `Writer`, or a ROS2 publish step (§4) — but that only runs once per published frame, not once per pixel per frame, since the per-pixel math never leaves the GPU.

**Before any of the above (Phase 2 in §7): prototype in plain Python first.** A custom Replicator `Writer` doing the physics in numpy — pulling annotator arrays to the CPU each frame — is *not* the final architecture, but it's a legitimate, fast way to prove thermal_camera_model.md's pipeline against real Isaac Sim data before you've written a line of CUDA:

```python
import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

class ThermalWriterPrototype(Writer):
    def __init__(self, output_dir):
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),                     # stand-in for encoded T(u,v) until §2's emission pass exists
            AnnotatorRegistry.get_annotator("distance_to_image_plane"), # R(u,v)
            AnnotatorRegistry.get_annotator("semantic_segmentation"),
        ]
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})

    def write(self, data: dict):
        # data["rgb"], data["distance_to_image_plane"], data["semantic_segmentation"] arrive as numpy arrays —
        # run thermal_camera_model.md §3–§11 on them here, once per frame
        ...

WriterRegistry.register(ThermalWriterPrototype)
```

Retire this once Phase 4 lands — it's a stepping stone, not something to keep maintaining in parallel with the SPG version.

## 2. Getting a Continuous Temperature Field Into the Renderer

**Confirmed, not a guess** — an Isaac Sim maintainer laid out this exact approach in the same GitHub discussion:

1. **Encode temperature as emission.** Assign each surface an OmniPBR material with emission enabled, using the emissive color channel to carry the temperature value. For genuinely continuous point-wise resolution rather than one value per material, drive that emissive input from a USD primvar (`primvars:temperature`, per-vertex or `faceVarying` — standard OpenUSD, not Isaac-specific) instead of a flat constant.
2. **Capture the signal cleanly.** Request the **`PtSelfIllumination`** AOV — it isolates only the emission component, so reflected/bounced light doesn't contaminate the per-pixel temperature readout. The maintainer names `HdrColor` with max bounces set to 0 as a fallback if `PtSelfIllumination` isn't available on your install. I could independently confirm `HdrColor` (it's a long-standing, widely-documented AOV); I could *not* independently confirm `PtSelfIllumination` beyond this one reply — though `Pt`-prefixed AOV names (e.g. `PtZDepth`) are a real, established pattern in the RTX Interactive/path-tracing renderer, which is at least consistent with it being genuine. Check both against your actual install rather than assuming.
3. **Process it in SPG** (§1), not a hand-rolled MDL shader graph.

**Where the real friction is:** not primvar-to-emissive encoding (step 1 is confirmed and simple) — it's step 3. The person who originally requested this feature completed steps 1 and 2 but [got stuck writing the actual SPG shader](https://github.com/isaac-sim/IsaacSim/discussions/298#discussioncomment-17247645), and as of the most recent reply in that thread (Jul 21, 2026) hadn't reported success. §1 above exists specifically to close that gap with verified-correct code rather than the "figure out SPG yourself" position that thread was left in.

One inconsistency worth flagging directly to NVIDIA if you engage with that thread: the maintainer's reply says SPG "lets you chain processing stages using Python or Warp kernels," but the current official SPG documentation only describes CUDA + Lua — no Python or NVIDIA Warp authoring path is mentioned. CUDA + Lua is what's actually documented and what §1 is built on; treat "Python or Warp" as unconfirmed until you see it in the docs or try it yourself.

**Where RTX Sensor Non-Visual Materials fits:** a second, complementary framework — the one Isaac Sim's RTX Lidar/Radar use (custom USD attributes on a Material prim → the renderer computes a material ID AOV → `isaacsim.sensors.rtx` APIs map material ID to a physical response). Better suited to discrete material classes with custom physical attributes than to a continuous per-vertex field — worth it if you want material properties to live in the USD file itself rather than being encoded through the emission channel.

## 3. Feeding In Time-Varying State (thermal_object_dynamics.md)

Each dynamic entity (an engine, a fire, a body of water) is a small Python object holding its own ODE state (§1 lumped-capacitance `T`, or §4 enthalpy `H`, from the other document), stepped once per physics tick, writing the result into the primvar/material input driving that entity's temperature.

For anything stepped every frame (fire flicker especially, at the ~10–20 Hz range from thermal_object_dynamics.md §5), use **NVIDIA Fabric / USDRT** for the read/write rather than the plain USD Python API — Fabric is built specifically for high-frequency scene-data access without a full USD-stage traversal each time, which plain USD attribute-setting isn't optimized for at simulation-rate frequencies.

## 4. Wiring It As an Actual Sensor (ROS2)

The pattern already exists for standard sensors and extends cleanly to a custom one:
- `omni.syntheticdata` + Replicator writers (`rep.writers.get(...)`) is the same mechanism the built-in `ROS2PublishImage`-style writers use — your `ThermalWriter.write()` can construct and publish a `sensor_msgs/Image` directly via `rclpy`, so you don't need to learn OmniGraph node authoring just to get data out.
- Rate control is separate from render rate: the **Isaac Simulation Gate** node (accessed via `omni:sensor:tickRate` on the camera prim as of Isaac Sim 6.0 — the older `frameSkipCount` input is deprecated) lets you publish slower than you render, useful once you know how expensive the full radiometric pass is per frame.
- Use **ROS2 Humble** — Foxy is no longer tested/supported with current Isaac Sim.
- If raw throughput matters more than ROS2 ecosystem compatibility, the **ZMQ bridge** (added in Isaac Sim 4.5) is a lighter-weight alternative for streaming the same data.

## 5. What to Reuse vs. What to Build

Isaac Sim earns its place in this project specifically for these — don't rebuild them:
- **`omni.rtx.spg`** for the radiometric pipeline itself (§1) — GPU-native AOV post-processing is exactly what this project needs, and it's the mechanism NVIDIA itself pointed a user to for this exact request.
- **RTX ray-traced rendering** for the geometric ground truth (range, normals, occlusion/visibility) — genuinely better than a hand-rolled rasterizer would give you for free, and it's exactly the "point-wise, per-pixel range" architecture from thermal_camera_model.md §2.
- **PhysX** if a moving platform is involved (vehicle, drone, legged robot) — don't hand-roll dynamics.
- **URDF import** if the camera mounts on a robot.
- **Replicator randomizers** for domain randomization (scene temperatures, atmospheric parameters, material assignment per episode) if you're generating a training dataset rather than a single scenario — mature and built-in, not something to reinvent.
- **The "Incident" extension** (new in Isaac Sim 5.0) is described by NVIDIA as generating "incident-based data such as fires." I couldn't verify its depth — whether it's physically-based flame radiance or just scenario/training-data generation — so treat it as a 30-minute scouting spike before building thermal_object_dynamics.md §5's fire model from scratch. It may save real work; it may not be what you need. Worth checking before, not instead of, building your own.

## 6. Project Structure

- **Extension workflow** for the thermal camera itself — package it the way `isaacsim.sensors.rtx` is packaged: a reusable, hot-reloadable Isaac Sim extension, not a one-off script. This is also what makes it feel like "a sensor" rather than "a script that happens to run in Isaac Sim."
- **Standalone Python** (`SimulationApp({"headless": True})`) for running scenarios, batch data generation, and anything you want scripted/repeatable rather than click-driven.
- **Validate Tier 2 completely standalone first**, with fabricated numpy input arrays (no Isaac Sim running) — a uniform-temperature scene, a two-material checkerboard, a known range gradient. This isolates "is my physics wrong" from "is my Isaac Sim integration wrong," which matters a lot given how much of this stack is genuinely under-documented.

## 7. Phased Roadmap

*(For a small-steps, checkbox-level breakdown of each phase below, see `isaacsim_checklist.md`.)*

1. **Physics-only validation** — thermal_camera_model.md's pipeline as pure Python/numpy, tested against synthetic (non-Isaac-Sim) inputs.
2. **Coarse end-to-end in Isaac Sim** — semantic-segmentation-based per-material temperature, full pipeline running as a plain Python Replicator `Writer` (deliberately not SPG yet — prove the physics and the Isaac Sim wiring separately before adding CUDA/Lua to the mix), visualized output.
3. **Time-varying dynamics** — wire one entity's ODE (engine warm-up) into the per-frame update loop (§3).
4. **Continuous field, on GPU** — §2's emission-encoding approach plus §1's SPG shader, replacing the Phase 2 Python Writer for at least one ground mesh.
5. **Special materials** — water's larger effective `C_th`, ice's enthalpy tracking, fire's volumetric approximation (thermal_object_dynamics.md §3–§5). Fire is the likely long pole — a true RTX volumetric render in a custom band is its own under-documented rabbit hole, so a cheap noise-driven texture/billboard approximation is a reasonable first target rather than full participating-media rendering.
6. **ROS2 + rate control**, and domain randomization if you need a dataset rather than a live sensor.
7. **Calibration** — if you can get your hands on a real thermal camera capture or two, or want to cross-check against DIRSIG output, use it to tune your NETD/noise parameters rather than trusting the illustrative values in thermal_camera_model.md §13 blindly.

## 8. Where the Documentation Is Genuinely Thin (go in expecting this)

- Authoring your first real SPG shader (§1) is the confirmed hard part, not a hypothesized one — the person who originally requested thermal camera support got stuck exactly here. NVIDIA's own tutorial (which §1's skeleton is adapted from) is the best available de-risking; lean on it before improvising.
- Whether `PtSelfIllumination` is actually present on your specific install (§2): unverified beyond one maintainer reply. Check early, fall back to `HdrColor` + bounces=0 if it's missing.
- SPG itself is explicitly flagged by NVIDIA as "under active development... API surface... may evolve across releases" — don't be surprised if function names or behavior shift between Isaac Sim point releases.
- True volumetric/participating-media rendering for a non-visible band (fire, if you want more than a texture trick): sparse, and SPG's stdlib nodes as of this writing operate on 2D textures only, not volumes.
- Real-time performance headroom for a full custom radiometric pass at your target resolution/frame rate: no public numbers exist for this specific workload — profile early (end of Phase 2), don't assume.

## References

- "Infrared Camera Support," isaac-sim/IsaacSim GitHub Discussion #298 — the original feature request; an Isaac Sim maintainer's May 16, 2026 reply is the primary source for §1–§2's approach, and the thread's unresolved ending (Jul 21, 2026) is the primary source for where the real difficulty sits. https://github.com/isaac-sim/IsaacSim/discussions/298
- "RTX Sensor Processing Graphs [omni.rtx.spg]," Omniverse Kit Documentation — the authoritative SPG reference; §1's code skeleton and file-role table are adapted directly from its grayscale-conversion tutorial. https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/latest/Overview.html
- NVIDIA, "Advanced Sensor Physics, Customization, and Model Benchmarking Coming to NVIDIA Isaac Sim and NVIDIA Isaac Lab," developer blog — Fabric, the ZMQ bridge, and the Incident extension are all described here. https://developer.nvidia.com/blog/advanced-sensor-physics-customization-and-model-benchmarking-coming-to-nvidia-isaac-sim-and-nvidia-isaac-lab
- "RTX Sensor Non-Visual Materials," Isaac Sim Documentation — the custom-material-attribute framework §2 builds on. https://docs.isaacsim.omniverse.nvidia.com/5.1.0/sensors/isaacsim_sensors_rtx_materials.html
- "Synthetic Data Recorder" / custom Writer pattern, Isaac Sim Documentation — source of the `Writer` skeleton in §1. https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_recorder.html
- "Publishing Camera's Data," ROS 2 Tutorials, Isaac Sim Documentation — the writer-based ROS2 image-publishing pattern referenced in §4. https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera_publishing.html
- "Workflows," Isaac Sim Documentation — GUI vs. Extension vs. Standalone Python, referenced in §6. https://docs.isaacsim.omniverse.nvidia.com/6.0.0/introduction/workflows.html
