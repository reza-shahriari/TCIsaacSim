---
name: thermal-camera-sim
description: Project-specific architecture, conventions, and known gotchas for the thermal-camera-sim repo -- a point-wise, physically-based thermal/IR camera simulator built for Isaac Sim. Consult this before writing or editing any code anywhere in this repository -- thermal_physics/, tests/, isaac_ext/, spg/, or scripts/ -- even for a small edit, and before answering any question about this project's design or build order. Also use when the user asks about thermal cameras, IR sensors, radiometry, Isaac Sim SPG, or anything that sounds like this project even if they don't name it directly.
---

# thermal-camera-sim

A thermal/IR camera simulator where every pixel comes from a real surface point's own temperature, emissivity, and range -- never one flat value per object or per material zone as a design endpoint (coarse per-material is an allowed *intermediate* step, see Build order below, but the finished sensor must be point-wise). If a proposed change would make temperature a per-object scalar again, it's wrong -- stop and reread `docs/thermal_camera_model.md` SS0.

## Before you write anything

Read the relevant doc section, don't re-derive from memory -- every stub function's docstring names its source file and section (e.g. "SS5a"). The docs are the source of truth; this skill is just the map to them and the conventions layered on top:

| If you're touching... | Read first |
|---|---|
| `thermal_physics/` (anything except `dynamics/`) | `docs/thermal_camera_model.md`, the section named in the function's docstring |
| `thermal_physics/dynamics/` | `docs/thermal_object_dynamics.md`, the section named in the function's docstring |
| `isaac_ext/`, `spg/`, or anything Isaac-Sim-specific | `docs/isaacsim_implementation_plan.md` in full, then the relevant `docs/isaacsim_checklist.md` phase |
| Not sure what to work on next | `ROADMAP.md` at the repo root -- work top to bottom, don't skip ahead into Isaac-Sim-dependent phases before the phase above is green |

## Non-negotiable conventions

- **`thermal_physics/` has zero Isaac Sim dependency.** No `import omni`, no `import isaacsim`, anywhere under `thermal_physics/`. It must run and test with nothing but numpy installed. If you find yourself wanting to import `omni` there, the code belongs in `isaac_ext/` or `spg/` instead.
- **Every implemented function needs a passing test before it counts as done.** The test files already exist with the right names (`tests/test_*.py`) and currently `pytest.skip(...)`. Remove the skip and write real assertions when you implement the function -- don't add new test files with different names.
- **Follow phase order.** `ROADMAP.md` phases build on each other on purpose (Phase 1 physics validated standalone before Phase 2 touches Isaac Sim; Phase 2's Python writer is a deliberate throwaway prototype before Phase 4's real SPG version). Don't start Phase 4 SPG work to "save time" before Phase 1's tests are green -- that just moves debugging into a much harder environment.
- **SPG shaders are always a matched triple.** `.cu` + `.cu.lua` + `.usda`, same base filename, and the CUDA function name / Lua function name / USD `subIdentifier` must all match exactly. The Lua file validates and configures the launch; it does not do pixel math.

## Known gotchas (already paid for -- don't rediscover them)

- Isaac Sim `pip install` has documented intermittent failures on the NVIDIA forums. Use Quick Install for the first setup pass; pip is fine later for CI/scripting once the environment is known-good.
- ROS2 **Humble**, not Foxy -- Foxy is untested/unsupported with current Isaac Sim.
- SPG's Lua API: `cuda.TextureObject(...)` / `cuda.SurfaceObject(...)` (capitalized, not `cuda.texture()`/`cuda.surface()`), and launch-config table keys `block` / `grid` (not `blockDim`/`gridDim`). These specific wrong forms show up in circulating notes about SPG -- if you see them, they're wrong.
- `PtSelfIllumination` (the AOV that isolates emission for SS2's temperature-encoding trick) is confirmed by exactly one direct Isaac Sim maintainer reply, not an independent AOV reference doc. Verify it exists on the actual install before depending on it; `HdrColor` + `maxBounces=0` is the documented fallback.
- The person who originally requested Isaac Sim thermal camera support (the GitHub discussion `docs/isaacsim_implementation_plan.md` cites) got through temperature-as-emission encoding fine but got stuck writing the SPG shader itself. That's the highest-real-risk step in this whole project -- budget accordingly, and see `docs/isaacsim_implementation_plan.md` SS8 before attempting it.
- SPG itself is explicitly under active development per NVIDIA's own docs -- function names/behavior may shift between Isaac Sim point releases. Note the exact Isaac Sim version you're building against if something stops matching this skill.

## When something in this skill turns out to be wrong

Isaac Sim and SPG are moving fast and were only partially documented as of this project's last research pass. If you find a gotcha above is stale, or discover a new one, update this file in the same change -- don't just fix the code and leave the next session to rediscover the same problem.
