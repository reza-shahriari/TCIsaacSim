---
name: isaac-sim-spg
description: Isaac Sim 6.0 integration for the IR camera simulator — AOV/render-product setup, temperature encoding, the RTX Sensor Processing Graph (omni.rtx.spg) CUDA+Lua+USD structure, the Warp/annotator alternative, and reflection strategies. Use this skill for ANY task that touches Isaac Sim, Omniverse, Kit, USD render products, RenderVars, AOVs, OmniPBR emission, PtSelfIllumination, SPG kernels, Warp kernels, Replicator annotators, or getting simulated imagery out of the renderer — including vague requests like "hook the sensor up to the sim" or "render a thermal frame". Isaac Sim 6.0 is recent and its SPG API differs from what you may recall, so consult this before writing engine code.
---

# Isaac Sim 6.0 integration

Isaac Sim has **no built-in IR camera**. We build one on top of the RTX renderer. This skill records what
is actually true about the 6.0 API, because it is newer than most training data and the public guidance
is partly contradictory.

Reference: `docs/physics-model.md` §13.

## The strategy in one paragraph

Do not try to make the RTX renderer produce infrared radiance. Make it produce a **G-buffer of physical
quantities** — encoded temperature, material ID, normals, distance, sky visibility — and do all the
radiometry ourselves in GPU post-processing. The renderer handles what it is good at: visibility,
geometry, and material lookup. We handle what it knows nothing about: Planck's law.

## Two implementation paths — start with the second

### Path A: Sensor Processing Graph (`omni.rtx.spg`)

NVIDIA's recommended route. Runs custom GPU code as post-processing passes on RTX AOVs, entirely on the
GPU with no CPU round trip. Requires Kit launched with `--enable omni.rtx.spg`.

Each SPG shader is **three co-located files**:

| File | Role |
|---|---|
| `Kernel.cu` | CUDA kernel, an `extern "C" __global__` function. Compiled at runtime via NVRTC. |
| `Kernel.cu.lua` | Lua launch script. Called every frame: validates inputs, allocates outputs, returns launch config. Function name must match the CUDA function name. |
| `Kernel.usda` | USD shader definition. Declares `opaque inputs:X` / `opaque outputs:Y`, points at the `.cu` via `info:spg:sourceAsset`, names the entry point via `info:spg:sourceAsset:subIdentifier`. |

SPG finds the Lua script by appending `.lua` to the source asset path. The three names —
CUDA function, Lua function, `subIdentifier` — must all be identical.

Wiring in the scene file:
`RenderVar.omni:rtx:aov` → `Shader.inputs:X` → `Shader.outputs:Y` → `RenderVar.omni:rtx:aov.connect`

Shaders chain output-to-input directly with no intermediate RenderVar. Execution order comes from the
connection dependency graph, **not** from the order of `orderedVars`. Every RenderVar consumed or
produced must appear in `orderedVars`.

Useful API details:

- `cuda.static(fn, ...)` caches a computation, recomputing only when its arguments change. **This is how
  the Planck LUT gets uploaded once instead of per frame.** Use it for every constant table.
- `rtx.frameId` is a Lua global — use it to seed temporal noise so runs are reproducible.
- `cuda.image(w, h, dtype)` for image outputs; `cuda.empty(shape, dtype)` for LUTs and statistics.
- `cuda.TextureObject(...)` for read-only input, `cuda.SurfaceObject(...)` for writable output,
  `cuda.array(...)` for raw device pointers.
- Value inputs (typed USD attributes like `float inputs:strength`) arrive in the same `inputs` table as
  resource inputs but carry a `.value` field. Distinguish with `inputs["x"].value ~= nil`.
- Stdlib nodes exist for trivial ops (`spg:rtx.spg.stdlib/Add`, `/Multiply`, `/Scale`, `/Swizzle`),
  authored inline in USD with `info:implementationSource = "id"`. Not useful for our physics, but handy
  for debug taps.

Known limitations to design around:

- Only **local** `.cu`, `.cu.lua`, `.usda` files are supported.
- SPG shader prims **must not be nested under a Material prim**.
- Stdlib nodes handle 2D textures only; integer texture formats (`SINT`/`UINT`) and buffer-backed
  resources are not supported by them.
- `display_render_var` only works for RGBA unorm textures. Our float32 outputs will not show in the
  viewport — use `FileCapture` to write them to disk, or add a separate 8-bit display AOV.

### Path B: Annotators + Warp (start here)

`isaacsim.sensors.experimental.rtx` provides `RtxCamera` (authoring — wraps a USD Camera prim and
exposes optical parameters) and `CameraSensor` (runtime — creates a Replicator render product, attaches
annotators, and provides `get_data()` returning numpy or **warp** arrays). `TiledCameraSensor` batches
many cameras for dataset generation.

Because `get_data()` can return warp arrays, the data is already on the GPU. Run the same six pipeline
stages as Warp kernels and there is no round trip. Cost: Python-side per-frame dispatch, well under a
millisecond at 640×512.

**Use Path B until the physics is verified against `irsim`'s CPU reference, then port hot stages to
Path A.** Debugging radiometry inside a runtime-compiled CUDA kernel wired through USD is the worst
possible place to discover that your emissivity model has a sign error.

Note: some community guidance describes SPG as being scripted "in Python or Warp". It is not — SPG is
CUDA + Lua + USD. Path B is where Python and Warp live. If you find yourself confused about which one
you are writing, that mismatch is why.

Note the deprecations: `isaacsim.sensors.camera` and `isaacsim.sensors.rtx` are deprecated as of 6.0 in
favour of `isaacsim.sensors.experimental.rtx`. Do not write new code against the old extensions.

## Temperature encoding — verify this before anything else

The renderer transports temperature to us through an emissive channel. Emissive channels are commonly
fp16 or 8-bit unorm. At 300 K:

- fp16 spacing → **0.25 K** (5× coarser than a 50 mK NETD)
- 8-bit unorm over 200–400 K → **0.78 K**

Either destroys the sensor completely while still producing a plausible-looking image.

Encode as:

```
c = (T - T_REF) / T_SPAN        # T_REF = 200.0 K, T_SPAN = 800.0 K → c in [0, 1]
T = c * T_SPAN + T_REF
```

into a **float32** AOV. If float32 is unavailable on a given channel, split coarse/fine across two fp16
channels rather than accepting the precision loss.

**First engine task in the project:** render a ramp of known temperatures, decode, assert round-trip
error < 10 mK. This is `tests/integration/test_temperature_encoding.py`. Nothing downstream matters if
it fails, so do not build anything else until it passes.

## AOV set

| AOV | Carries | Format | Notes |
|---|---|---|---|
| `PtSelfIllumination` (or custom) | encoded temperature | **float32** | emission-encoded; bounces must not contaminate it |
| material ID / `DiffuseAlbedo` | emissivity lookup key | uint / float | index into the material table |
| `Normal` | surface normal | fp16 ok | for ε(θ) and sky-view factor |
| `DistanceToCamera` | path length d | **float32** | atmospheric attenuation |
| `AmbientOcclusion` (or custom) | sky-view factor V_s | fp16 ok | reflected-sky term |
| `MotionVectors` | image-plane velocity | fp16 ok | motion MTF, bolometer smear |
| `SemanticSegmentation` | ground truth labels | uint | dataset export |

`PtSelfIllumination` captures only the self-illumination component, giving a clean per-pixel temperature
readout without contamination from reflected or bounced light. The alternative is `HdrColor` with max
bounces set to 0.

## Pipeline stages

Six kernels, each independently testable against the `irsim` CPU reference:

1. `band_radiance` — decode T, look up `Lb(T)`, apply ε(θ), add reflected environment
2. `atmosphere` — `τ = exp(-γ·d)`; `L' = τ·L + (1−τ)·B(T_air)`
3. `optics` — `π·τ_opt/(4F²+1)`, cos⁴θ vignetting, optics self-emission
4. `detector` — bolometer IIR (thermal time constant) or photon → electrons → well → ADC
5. `noise` — 3D noise components, NUC residual, bad pixels; seeded from `rtx.frameId`
6. `isp` — AGC/plateau equalisation, DDE, gamma, polarity, palette

Outputs: `IrRadiance` (float32), `IrApparentT` (float32), `IrDN16` (uint16), `IrDisplay8` (RGBA8).
Publish all four — perception consumes the 8-bit, validation needs the linear ones.

MTF and aliasing: render at 3–4× the sensor resolution and box-filter down to the pixel footprint in
stage 3. This gives correct detector MTF *and* correct aliasing for free. Rendering at native resolution
and blurring afterwards gives you the blur without the aliasing, and aliasing is exactly what corrupts
small-target detection at range.

## The reflection problem

`PtSelfIllumination` deliberately excludes bounced light — which is what gives a clean temperature
buffer, and which also means **no reflections at all**. But painted bodywork and glass are near-mirrors
in LWIR reflecting a very cold sky; that is why car roofs read cold at night. Options, cheapest first:

1. **Sky-view factor blend.** Use the AO AOV as `V_s`, blend sky and ground band radiance. No specular
   structure, but gets the broad behaviour and the cold-roof effect. **Implement this first.**
2. **Low-res T-cubemap.** A second small render product (64×64 per face) of the encoded-T field from a
   probe near the camera, converted to band radiance and used as a reflection probe. Good value.
3. **Second render product with bounces**, where emission encodes band radiance rather than temperature.
   Most correct available; costs a full extra render and requires keeping the two encodings from mixing.
4. **Warp ray cast** against the T-field for mirror directions on flagged specular materials only.
   Surgical and cheap when only a few materials need it.

Whichever is in use must be recorded in an ADR, because it is the largest remaining approximation in
the system and reviewers will ask.

## Verification discipline

Every SPG or Warp kernel gets a companion test that runs the same inputs through the `irsim` CPU
reference and asserts agreement. Tolerance: 1e-4 relative on radiance, or 5 mK on apparent temperature.

Build a synthetic G-buffer fixture (flat temperature field, known normals, known distances) so kernels
can be tested without launching Isaac Sim at all. Most kernel bugs are findable this way in seconds.

## Performance notes

The kernels are element-wise and trivial — six passes over 640×512 is microseconds. The cost is the
renderer: 4× supersampling means 2560×2048, and a second render product for reflections doubles it.
Budget there, not in the physics. Use `TiledCameraSensor` when generating datasets with many cameras.

## When something is unclear

The SPG documentation is explicitly incomplete and states the API may evolve across releases. If the
docs do not answer a question, say so and propose a minimal experiment to resolve it — do not write code
that assumes the ambiguity resolved conveniently, and do not silently invent an API surface.
