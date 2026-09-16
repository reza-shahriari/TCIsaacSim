#!/usr/bin/env python3
"""Roadmap M2.3: SPG capability experiments (float32 pass-through, persistent state, LUT delivery).

One Kit session per invocation; the report is saved after every experiment because a Lua error in
omni.rtx.spg crashes the process. Examples:

    python.sh scripts/probe_isaac_spg.py --out outputs/isaac_probe/spgA \\
        --experiments pass_colour,pass_geometry,state,feedback,lut_baked,lut_io,camera_sensor
    python.sh scripts/probe_isaac_spg.py --out outputs/isaac_probe/spgB \\
        --experiments lut_literal --lut-n 256
    python.sh scripts/probe_isaac_spg.py --out outputs/isaac_probe/spgC \\
        --experiments lut_loop --raise-limits
"""

from __future__ import annotations

import argparse
import json
import os
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/isaac_probe/spg")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--settle", type=int, default=20)
parser.add_argument("--experiments", default="", help="comma list; default all")
parser.add_argument("--lut-n", type=int, default=16001)
parser.add_argument("--raise-limits", action="store_true")
args = parser.parse_args()

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())

from irsim_isaac.probe import build_ramp_scene  # noqa: E402
from irsim_isaac.spg_probe import ALL_EXPERIMENTS, run_spg_experiments, save  # noqa: E402

os.makedirs(args.out, exist_ok=True)
scene = build_ramp_scene(emissive_intensity=40.0)
print(f"SCENE: {len(scene.quad_temperatures)} quads, errors={scene.errors}")
selected = tuple(e for e in args.experiments.split(",") if e) or ALL_EXPERIMENTS
report_path = os.path.join(args.out, "spg.json")
report = run_spg_experiments(
    args.out,
    experiments=selected,
    lut_n=args.lut_n,
    raise_limits=args.raise_limits,
    resolution=(args.resolution, args.resolution),
    settle_frames=args.settle,
    save_path=report_path,
)
report["boot_and_total_s"] = round(time.time() - t0, 1)
save(report, report_path)
print("SPGREPORT", json.dumps(report, default=str))
app.close()
