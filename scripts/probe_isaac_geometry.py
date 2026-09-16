#!/usr/bin/env python3
"""Roadmap M10.1: survey the geometry AOVs on the lit, tilted, moving probe scene.

ADR 0014 measured transport on an unlit static ramp and left the normals, ambient-occlusion and
motion channels open, because none of them returned anything on that scene. This script answers
those questions: it attaches every plausible annotator name in turn and records the dtype, the
shape (against the requested render-product resolution -- several AOVs come back at the renderer's
internal resolution instead, which silently misaddresses every pixel lookup), and whether the
values are all zero.

Run with the Isaac interpreter (docs/decisions/0002):
    python.sh scripts/probe_isaac_geometry.py --out outputs/isaac_probe/geometry

Never raises on a probe failure -- failures are data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/isaac_probe/geometry")
parser.add_argument("--resolution", type=int, default=384)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--export-stage", action="store_true", help="write the scene as scene.usda")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot

from irsim_isaac.geometry_probe import build_geometry_scene, survey_channels  # noqa: E402

os.makedirs(args.out, exist_ok=True)


def main() -> int:
    scene = build_geometry_scene(
        resolution=args.resolution,
        export_path=os.path.join(args.out, "scene.usda") if args.export_stage else None,
    )
    report = survey_channels(scene, settle_frames=args.settle)
    report["boot_s"] = round(boot_s, 2)
    report["scene_errors"] = scene.errors
    report["targets"] = {
        name: {
            "centre": list(t.centre),
            "pixel": list(scene.pixel_of(t.centre)),
            "distance_m": t.distance_m,
            "normal": None if t.normal is None else list(t.normal),
            "sky_view_factor": t.sky_view_factor,
        }
        for name, t in scene.targets.items()
    }
    report["bar"] = {"step_m": scene.bar_step_m, "depth_m": scene.bar_depth_m}

    path = os.path.join(args.out, "geometry.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nrender product {args.resolution}^2   settings {report['settings']}")
    print(f"scene errors: {scene.errors or 'none'}")
    for channel, entries in report["channels"].items():
        print(f"\n{channel}:")
        for name, e in entries.items():
            status = str(e.get("status"))
            if status != "ok":
                print(f"  {name:28s} {status[:80]}")
                continue
            flag = "FULL" if e.get("full_resolution") else "PART"
            extra = ""
            if e.get("all_zero"):
                extra = " all-zero"
            elif "min_max" in e:
                extra = f" range {e['min_max'][0]:.4g}..{e['min_max'][1]:.4g}"
            elif "unique_count" in e:
                extra = f" {e['unique_count']} unique"
            print(
                f"  {name:28s} ok   {e['dtype']:8s} {str(e['shape']):20s} {flag}"
                f" finite={e.get('finite_fraction', 1.0):.3f}{extra}"
            )
    print(f"\nwrote {path}")
    return 0


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a probe crash is still a result worth reporting
    print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    app.close(exit_code=status if isinstance(status, int) else 1)
