#!/usr/bin/env python3
"""Roadmap ME.7: score a detector's output with the §15 Tier 5 metrics.

    python scripts/eval_detector.py --predictions preds.json --truth truth.json

Reads two JSON files -- predictions as ``[{frame, x, y, width, height, score}, ...]`` and truth as
``[{frame, x, y, width, height, scr?, size_px?}, ...]`` -- and reports AP@0.5, AP on small targets,
P_d against SCR and against apparent size, and false alarms per frame.

**No torch here.** Scoring is arithmetic and lives in :mod:`irsim_eval.detection`, so a detector
trained anywhere -- or a set of boxes written by hand -- can be scored in the default environment.
The metric that carries the weight for this project is **P_d against SCR**: a drone at 2 km subtends
two pixels at an SCR of 2 and is a rounding error in an AP dominated by close, large targets, so two
sets can agree on AP and disagree completely about the regime the simulator exists to model.

docs/physics-model.md §15 T5; roadmap ME.7
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim_eval.detection import (
    Detection,
    average_precision,
    detection_rate_by_bin,
    false_alarms_per_frame,
    match_detections,
)

#: §15's small-target cut. 32x32 is the COCO convention and is far larger than anything this
#: project cares about; the second cut is where a sky target actually lives.
SMALL_AREA_PX = 32.0 * 32.0
TINY_AREA_PX = 8.0 * 8.0


@dataclass(frozen=True)
class Truth:
    x: float
    y: float
    width: float
    height: float
    frame: int = 0
    scr: float | None = None

    @property
    def area(self) -> float:
        return float(self.width * self.height)


def _load(path: pathlib.Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path}: expected a JSON list of boxes")
    return payload


def evaluate(detections: list[Detection], truths: list[Truth], n_frames: int) -> dict[str, Any]:
    found = np.zeros(len(truths), dtype=bool)
    for frame in sorted({t.frame for t in truths}):
        idx = [i for i, t in enumerate(truths) if t.frame == frame]
        _, hit = match_detections(
            [d for d in detections if d.frame == frame], [truths[i] for i in idx]
        )
        for i, was in zip(idx, hit, strict=True):
            found[i] = was
    areas = np.array([t.area for t in truths], dtype=np.float64)
    out: dict[str, Any] = {
        "ap50": average_precision(detections, truths),
        "n_truths": len(truths),
        "n_detections": len(detections),
        "false_alarms_per_frame": false_alarms_per_frame(detections, truths, n_frames=n_frames),
        "recall": float(found.mean()) if found.size else 0.0,
    }
    for label, cut in (("ap50_small", SMALL_AREA_PX), ("ap50_tiny", TINY_AREA_PX)):
        subset = [t for t in truths if t.area <= cut]
        out[label] = average_precision(detections, subset) if subset else None
    size_bins = [0.0, TINY_AREA_PX, SMALL_AREA_PX, float(max(areas.max(), SMALL_AREA_PX) + 1.0)]
    by_size = detection_rate_by_bin(truths, found, areas, size_bins)
    out["pd_by_area_px2"] = {
        "edges": by_size.edges.tolist(),
        "counts": by_size.counts.tolist(),
        "rate": [None if np.isnan(v) else float(v) for v in by_size.rate],
    }
    if all(t.scr is not None for t in truths) and truths:
        scr = np.array([float(t.scr) for t in truths], dtype=np.float64)  # type: ignore[arg-type]
        by_scr = detection_rate_by_bin(truths, found, scr, [0.0, 2.0, 5.0, 10.0, 20.0, 1e6])
        out["pd_by_scr"] = {
            "edges": by_scr.edges.tolist(),
            "counts": by_scr.counts.tolist(),
            "rate": [None if np.isnan(v) else float(v) for v in by_scr.rate],
        }
    else:
        out["pd_by_scr"] = None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=pathlib.Path, required=True)
    parser.add_argument("--truth", type=pathlib.Path, required=True)
    parser.add_argument("--frames", type=int, default=None, help="frames in the clip")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    detections = [
        Detection(
            x=float(b["x"]),
            y=float(b["y"]),
            width=float(b["width"]),
            height=float(b["height"]),
            score=float(b.get("score", 1.0)),
            frame=int(b.get("frame", 0)),
        )
        for b in _load(args.predictions)
    ]
    truths = [
        Truth(
            x=float(b["x"]),
            y=float(b["y"]),
            width=float(b["width"]),
            height=float(b["height"]),
            frame=int(b.get("frame", 0)),
            scr=None if b.get("scr") is None else float(b["scr"]),
        )
        for b in _load(args.truth)
    ]
    if not truths:
        raise SystemExit("no ground truth: every metric here is undefined without it")
    frames = args.frames or (max({d.frame for d in detections} | {t.frame for t in truths}) + 1)
    report = evaluate(detections, truths, frames)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
