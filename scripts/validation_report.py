#!/usr/bin/env python3
"""Roadmap ME.6: the Tier 4 acceptance report. Exits non-zero when a target is exceeded.

    python scripts/validation_report.py --real <dir> --synthetic <dir>
    python scripts/validation_report.py --self-test

Frames are 8-bit ``.npy`` or ``.png`` files -- the display domain, because ADR 0068 established
that is the only domain the public sky-target data exists in. Every DN8 target from
:class:`irsim.validation.compare.Tier4Targets` is checked, plus the discriminator's gap score, and
**everything that cannot be measured is printed as `untestable` rather than omitted**: on this
project's data most of Tier 4 is untestable, and a report that hid that would be the single most
misleading thing it could produce.

``--self-test`` runs the report on synthetic-versus-itself, which is the control: two halves of one
distribution must pass every check and give an AUC near 0.5. It needs no data and is what the unit
test drives.

Needs no GPU, no Isaac Sim and no network.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

from irsim.validation.compare import Tier4Report, Tier4Targets, compare_frames
from irsim_eval.discriminator import gap_score

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_frames(directory: pathlib.Path, limit: int | None) -> list[np.ndarray]:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".npy", ".png"})
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"no .npy or .png frames under {directory}")
    frames: list[np.ndarray] = []
    for path in paths:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            from irsim_eval.data import _decode_png  # the validation extra lives there

            array = _decode_png(path)
        array = np.asarray(array)
        if array.ndim == 3:
            array = array[..., 0]  # a display frame stored RGBA; the channels are equal in grey
        if array.dtype != np.uint8:
            raise SystemExit(f"{path.name}: Tier 4 is measured on 8-bit frames, got {array.dtype}")
        frames.append(array)
    return frames


def _patches(frames: list[np.ndarray], size: int, per_frame: int, seed: int) -> list[np.ndarray]:
    """Fixed-size patches on a deterministic grid, so a re-run compares the same pixels."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape
        if h < size or w < size:
            raise SystemExit(f"a {size}x{size} patch does not fit in a {h}x{w} frame")
        for _ in range(per_frame):
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            out.append(frame[y : y + size, x : x + size].astype(np.float64))
    return out


def _mosaic(frames: list[np.ndarray]) -> np.ndarray:
    """One frame standing for the set, for the whole-frame statistics.

    The temporal median rather than the mean: it is what ME.2b's window finder already uses, and a
    target that moves disappears from it, so the histogram and the spectrum describe the
    *background* the two sets are really being compared on.
    """
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise SystemExit(f"frames are not all the same size: {sorted(shapes)}")
    return np.median(np.stack(frames).astype(np.float64), axis=0)


def _self_test_frames(
    seed: int = 20260915, n: int = 24
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two halves of one distribution: the control every acceptance report needs.

    If the report cannot pass this, its thresholds are wrong and nothing it says about a real
    comparison means anything.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (64, 64))
    smooth = np.zeros_like(base)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            smooth += np.roll(np.roll(base, dy, 0), dx, 1)
    smooth /= smooth.std()
    scene = 110.0 + 12.0 * smooth
    frames = [
        np.clip(scene + rng.normal(0.0, 3.0, scene.shape), 0, 255).astype(np.uint8)
        for _ in range(2 * n)
    ]
    return frames[:n], frames[n:]


def _run(
    real: list[np.ndarray],
    synthetic: list[np.ndarray],
    *,
    patch: int,
    per_frame: int,
    seed: int,
    targets: Tier4Targets,
) -> Tier4Report:
    result = gap_score(
        _patches(real, patch, per_frame, seed),
        _patches(synthetic, patch, per_frame, seed + 1),
        seed=seed,
    )
    report = compare_frames(_mosaic(real), _mosaic(synthetic), targets=targets, auc=result.auc)
    for check in report.checks:
        if check.name == "discriminator AUC":
            report.checks[report.checks.index(check)] = type(check)(
                name=check.name,
                measured=check.measured,
                target=check.target,
                verdict=check.verdict,
                note=(
                    f"{result.n_real} vs {result.n_synthetic} patches, {result.folds}-fold "
                    f"cross-validated; {result.z:+.2f} null sigma from chance "
                    f"(sigma = {result.null_sigma:.3f}); heaviest features: {result.top()}"
                ),
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=pathlib.Path)
    parser.add_argument("--synthetic", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true", help="synthetic vs itself: the control")
    parser.add_argument("--limit", type=int, default=None, help="frames per set")
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--patches-per-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=pathlib.Path, default=None, help="write markdown + json here")
    args = parser.parse_args(argv)

    if args.self_test:
        real, synthetic = _self_test_frames(args.seed)
        label = "self-test (synthetic vs itself)"
    else:
        if not (args.real and args.synthetic):
            parser.error("--real and --synthetic are required unless --self-test is given")
        real = _load_frames(args.real, args.limit)
        synthetic = _load_frames(args.synthetic, args.limit)
        label = f"{args.real} vs {args.synthetic}"

    report = _run(
        real,
        synthetic,
        patch=args.patch,
        per_frame=args.patches_per_frame,
        seed=args.seed,
        targets=Tier4Targets(),
    )
    print(f"# Tier 4 acceptance -- {label}\n")
    print(report.as_markdown())
    print()
    if report.untestable:
        print(f"{len(report.untestable)} check(s) untestable on this input:")
        for check in report.untestable:
            print(f"  - {check.name}: {check.note}")
        print()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "tier4-acceptance.md").write_text(
            f"# Tier 4 acceptance -- {label}\n\n{report.as_markdown()}\n", "utf-8"
        )
        payload: dict[str, Any] = {
            "label": label,
            "checks": [c.as_dict() for c in report.checks],
            "passed": report.passed,
        }
        (args.out / "tier4-acceptance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), "utf-8"
        )
        print(f"wrote {args.out / 'tier4-acceptance'}.md and .json")
    if report.failed:
        print(f"FAILED: {', '.join(c.name for c in report.failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
