# thermal-camera-sim

A physically-based, point-wise thermal (infrared) camera simulator, built for Isaac Sim.

Every pixel maps back to a real surface point with its own temperature, emissivity, and range — not one flat value per object. The full derivation lives in `docs/`; this repo is the code structure built from it.

## Status

**Phase 1 complete** (`thermal_physics/`, pure Python, no Isaac Sim): every function implemented, 33/33 tests passing, zero skips. `python scripts/demo_render.py` renders a synthetic scene end to end — see `captures/demo_frame.png`.

**Phases 2-7 (everything Isaac-Sim-shaped) are still stubs.** They need a real Isaac Sim install and an RTX GPU to write against and test honestly, which the environment this was built in doesn't have — signatures/docstrings/TODOs are in place, same as before. See `ROADMAP.md` for build order and what's left.

## Project layout

```
docs/                          the design docs — read these first
  thermal_camera_model.md          the radiometric pipeline: Planck's law -> atmosphere ->
                                    optics -> detector -> blur -> noise -> quantize -> AGC -> colormap
  thermal_object_dynamics.md       how temperature changes over time: engines, water, ice, fire
  isaacsim_implementation_plan.md  the SPG-based Isaac Sim architecture (verified against
                                    NVIDIA's own docs and a real GitHub discussion)
  isaacsim_checklist.md            the granular, checkbox-level build checklist

thermal_physics/                pure Python, zero Isaac Sim dependency -- Phase 1
  emission.py, atmosphere.py, optics.py, detector.py, blur.py,
  noise.py, quantize.py, agc.py, colormap.py, pipeline.py
  dynamics/                        time-varying temperature: lumped_capacitance, water, ice, fire

tests/                          pytest, one file per thermal_physics module

isaac_ext/thermal.camera/       the Isaac Sim extension -- Phase 2+

spg/                            Sensor Processing Graph shader (.cu / .cu.lua / .usda) -- Phase 4

scripts/                        standalone entry points (headless launch, frame capture)

.claude/skills/thermal-camera-sim/SKILL.md   project-specific skill for coding agents working in this repo
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest                           # everything currently skips -- that's the expected state of a fresh scaffold
```

Isaac Sim itself is a separate install, not a pip dependency of this project — see `docs/isaacsim_implementation_plan.md`'s prerequisites (RTX-capable NVIDIA GPU, Quick Install recommended over pip for the initial install).

## Build order

Work through `ROADMAP.md` phase by phase — it maps each phase in `docs/isaacsim_checklist.md` to the actual files and tests in this repo. Each phase is sized to finish and verify before starting the next one; Phase 1 needs nothing but Python.

## Known gotchas (see docs/ for full context)

- Isaac Sim `pip install` has known intermittent failures — use Quick Install for your first pass.
- ROS2 Foxy is unsupported with current Isaac Sim — use Humble.
- The SPG Lua API is `cuda.TextureObject` / `cuda.SurfaceObject` and launch-config keys `block` / `grid` — a couple of plausible-looking alternatives (`cuda.texture`, `blockDim`/`gridDim`) do not match the current docs and will fail at runtime.
- `PtSelfIllumination` (the recommended AOV for isolating emission) is confirmed by one direct maintainer reply, not an independent reference doc — check it against your actual install before depending on it.

## License

MIT (`LICENSE`) — included as a permissive default; change it if you want something else.
