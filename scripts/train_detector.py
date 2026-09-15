#!/usr/bin/env python3
"""Roadmap ME.7: train a small detector under one of §15's four transfer protocols.

    python scripts/train_detector.py --protocol real_to_real --real <dir> --out runs/r2r
    python scripts/train_detector.py --protocol synthetic_to_real --real <dir> --synthetic <dir>

**This script needs the `ml` extra (torch), which the default environment does not install.** That
is deliberate: torch is a multi-gigabyte dependency, `src/irsim` may never import it (enforced by
`tests/unit/test_layering.py`), and a Tier 4 report must not be blocked on it. Run
``pip install -e ".[ml]"`` first; the import below fails with this message rather than a traceback.

**A second blocker is in the data, not the dependency, and it is the binding one today.** ME.5
found that the reference set's annotations are MATLAB Video Labeler `groundTruth` objects -- MCOS
class instances that `scipy.io.loadmat` returns as an opaque blob, with no Python reader -- so
there are no boxes to train on. Until somebody runs the dataset's own MATLAB script and exports
them, every protocol here is blocked on labels rather than on compute. The metrics the protocols
are scored with are implemented, tested and dependency-free in :mod:`irsim_eval.detection`.

docs/physics-model.md §15 T5; roadmap ME.7
"""

from __future__ import annotations

import argparse
import sys

from irsim_eval.detection import PROTOCOLS

TORCH_MISSING = (
    "this script needs the `ml` extra: pip install -e '.[ml]'\n"
    "torch is deliberately not a default dependency -- src/irsim may never import it "
    "(tests/unit/test_layering.py), and the Tier 4 report must not be blocked on a "
    "multi-gigabyte install."
)


def _require_torch() -> object:
    try:
        import torch
    except ModuleNotFoundError:
        raise SystemExit(TORCH_MISSING) from None
    return torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS))
    parser.add_argument("--real", help="directory of real sequences (irsim_eval.data layout)")
    parser.add_argument("--synthetic", help="directory of rendered sequences")
    parser.add_argument("--out")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--protocols", action="store_true", help="print the four protocols and exit"
    )
    args = parser.parse_args(argv)

    if args.protocols:
        print("§15 Tier 5 transfer protocols:\n")
        for name, question in PROTOCOLS.items():
            print(f"{name}\n    {question}\n")
        return 0

    if not args.protocol or not args.out:
        parser.error("--protocol and --out are required (or --protocols to list them)")
    needs = {
        "real_to_real": ("real",),
        "synthetic_to_real": ("synthetic", "real"),
        "real_to_synthetic": ("real", "synthetic"),
        "mixed_to_real": ("real", "synthetic"),
    }[args.protocol]
    missing = [n for n in needs if getattr(args, n) is None]
    if missing:
        parser.error(f"protocol {args.protocol} needs --{' and --'.join(missing)}")

    print(f"protocol {args.protocol}: {PROTOCOLS[args.protocol]}", file=sys.stderr)
    _require_torch()
    raise SystemExit(
        "no annotation boxes are available to train on. ME.5 established that the reference "
        "set's labels are MATLAB Video Labeler `groundTruth` (MCOS) objects with no Python "
        "reader; export them with the dataset's own MATLAB script, write them through "
        "`irsim_eval.data.write_sequence`, and re-run. The scoring metrics are ready and tested "
        "(irsim_eval.detection)."
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
