#!/usr/bin/env python3
"""Roadmap M2.1 + M2.2 gate spike: boot Isaac Sim headless, record the environment, author the
temperature-ramp scene, read back every candidate AOV in the real-time and path-traced modes.

Run with the Isaac interpreter (docs/decisions/0002):
    python.sh scripts/probe_isaac_environment.py --out outputs/isaac_probe/runN --dump

Writes environment.json, then per emissive-intensity value a sub-directory with scene.usda,
scene.json, <mode>.json and (with --dump) every AOV array as .npy; prints a summary. Never raises
on a probe failure — failures are data (ADR 0014).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/isaac_probe")
parser.add_argument("--resolution", type=int, default=256)
parser.add_argument("--settle", type=int, default=20)
parser.add_argument("--modes", default="rt,pt_b0,pt_b4", help="comma list of rt, pt_b0, pt_b4")
parser.add_argument("--emissive-intensity", default="1", help="comma list; scene + modes per value")
parser.add_argument("--keep-aa", action="store_true", help="leave DLSS/AA on (default off)")
parser.add_argument("--dump", action="store_true", help="save every AOV array as .npy")
parser.add_argument(
    "--ids", action="store_true", help="also probe segmentation-id and float32 position transport"
)
parser.add_argument("--skip-aovs", action="store_true", help="skip the emission AOV sweep")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
boot_s = time.time() - t_boot

from irsim_isaac.probe import (  # noqa: E402
    build_ramp_scene,
    label_quads,
    probe_environment,
    probe_id_transport,
    probe_render_mode,
    save_report,
    summarize,
)

os.makedirs(args.out, exist_ok=True)
env = probe_environment()
env["boot_seconds"] = round(boot_s, 1)
save_report(env, os.path.join(args.out, "environment.json"))
keys = ("kit_version", "isaac_version", "warp", "sensor_api", "errors")
print("ENVIRONMENT:", json.dumps({k: env[k] for k in keys}, default=str))
print("EXTENSIONS:", json.dumps(env["extensions"], default=str))

for intensity in [float(v) for v in args.emissive_intensity.split(",") if v.strip()]:
    tag = f"ei{intensity:g}"
    out_dir = os.path.join(args.out, tag)
    scene = build_ramp_scene(
        emissive_intensity=intensity, export_path=os.path.join(out_dir, "scene.usda")
    )
    save_report({"scene": scene}, os.path.join(out_dir, "scene.json"))
    print(f"SCENE {tag}: {len(scene.quad_temperatures)} quads, errors={scene.errors}")

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        bounces = {"pt_b0": 0, "pt_b1": 1, "pt_b4": 4}.get(mode)
        if args.skip_aovs:
            continue
        try:
            rep = probe_render_mode(
                scene,
                mode,
                resolution=args.resolution,
                settle_frames=args.settle,
                rt_subframes=32 if mode.startswith("pt") else 1,
                pt_max_bounces=bounces,
                disable_aa=not args.keep_aa,
                dump_dir=os.path.join(out_dir, "arrays") if args.dump else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"MODE {tag} {mode}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        save_report(rep, os.path.join(out_dir, f"{mode}.json"))
        print("REPORT", tag, mode)
        print(summarize(rep))

    if args.ids:
        label_errors = label_quads(scene)
        print(f"LABELS {tag}: errors={label_errors}")
        for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
            bounces = {"pt_b0": 0, "pt_b1": 1, "pt_b4": 4}.get(mode, 1)
            try:
                idrep = probe_id_transport(
                    scene,
                    mode,
                    resolution=args.resolution,
                    settle_frames=args.settle,
                    rt_subframes=32 if mode.startswith("pt") else 1,
                    pt_max_bounces=bounces,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"IDS {tag} {mode}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            save_report(idrep, os.path.join(out_dir, f"ids_{mode}.json"))
            print("IDREPORT", tag, mode, json.dumps(idrep, default=str))

app.close()
