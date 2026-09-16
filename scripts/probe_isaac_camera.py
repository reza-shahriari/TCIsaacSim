#!/usr/bin/env python3
"""Roadmap M10.9a: record which lens-distortion schemas this build carries, and their defaults.

ADR 0015 said the engine applies distortion but not *how* the config reaches it. This measures
that: the schema each `optics.distortion.model` maps to, the attribute names it creates, and --
the part that matters -- what those attributes default to. They do not default to an identity
lens (fx = 900, imageSize = 2048x1024, and a non-zero k1 on the fisheye), so the mapping writes
every attribute every time rather than only the coefficients.

Run with the Isaac interpreter (docs/decisions/0002):
    python.sh scripts/probe_isaac_camera.py --out outputs/isaac_probe/camera

Never raises on a probe failure -- failures are data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/isaac_probe/camera")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())
boot_s = time.time() - t_boot

from irsim_isaac.camera_probe import survey_distortion_schemas  # noqa: E402

os.makedirs(args.out, exist_ok=True)


def main() -> int:
    import omni.usd

    omni.usd.get_context().new_stage()
    report = survey_distortion_schemas()
    report["boot_s"] = round(boot_s, 2)

    path = os.path.join(args.out, "camera.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nregistered distortion schemas: {report['registered']}")
    for schema, entry in report["schemas"].items():
        if "error" in entry:
            print(f"\n{schema}: {entry['error']}")
            continue
        print(f"\n{schema}  applied={entry['applied']}")
        for name, meta in entry["attributes"].items():
            print(f"  {name:52s} {meta['type']:8s} default={meta['default']}")
    print(f"\nwrote {path}")
    return 0


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a probe crash is still a result worth reporting
    print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    app.close(exit_code=status if isinstance(status, int) else 1)
