# AGENTS.md

## Project Context

Robotics/simulation project integrating an infrared (IR) camera sensor into
NVIDIA Isaac Sim.

- Simulator: NVIDIA Isaac Sim (Omniverse Kit based)
- Task: implement/configure an infrared camera sensor inside Isaac Sim
- OS: Linux
- Scripting: Python (Isaac Sim's `omni.isaac.*` / `isaacsim.*` Python API)

## Environment

- Isaac Sim version: 6.0.0 (early developer release era — API surface may
  still shift; verify names against the installed build)
- Python: use the Python bundled with Isaac Sim's `python.sh` (e.g., `<isaac_sim_root>/python.sh`), not system Python
- GPU: CUDA-capable GPU required for rendering; confirm driver/CUDA compatibility
  with the installed Isaac Sim version before running
- Sensor API namespace: `isaacsim.sensors.*` (6.0 namespace, not the older
  `omni.isaac.sensor.*`)

## Task: Infrared Camera Sensor (Thermal)

- Goal: add a thermal (heat/emissivity-based) IR camera to a robot/drone
  prim in the USD stage
- See `.agents/skills/isaacsim-thermal-camera/SKILL.md` for the detailed
  implementation approach — load it for any work on this task
- Isaac Sim 6.0 has no native thermal-radiation sensor type; thermal must be
  built as a custom sensor (false-color post-process on segmentation output,
  or a custom MDL thermal material) — do not search for a built-in
  ThermalCamera class
- Camera should be attached as a child prim under the robot/drone's USD
  hierarchy, with correct local transform (position/orientation) relative to
  the mount point
- Expose configurable parameters: resolution, focal length/FOV, clipping range,
  update rate
- Camera ISP is configured via USD schema attributes on the prim in 6.0, not
  a separate Python config object

## Code Standards

- Follow Isaac Sim's USD/prim-path conventions; do not hardcode absolute
  stage paths where avoidable — pass them as parameters
- Any new sensor should be encapsulated in its own class/module, not inlined
  into scene-setup scripts
- Include type hints
- Log sensor initialization (prim path, resolution, mode) at startup for
  debugging
- Each step should be managed in Git as separate commits and should not add Big objects(USD, etc) as well as debugging codes / images to the git

## Constraints

- Do not commit large binary assets (USD, textures, checkpoints) to git;
  reference external asset paths instead
- Confirm Isaac Sim version and sensor API names before generating code —
  the sensor API has changed across Isaac Sim releases (`omni.isaac.sensor`
  vs newer `isaacsim.sensors.*` namespaces)

## Reference

- Isaac Sim documentation: <https://docs.omniverse.nvidia.com/isaacsim/latest/index.html>
