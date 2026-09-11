#!/usr/bin/env python3
"""Audit the thermal-material coverage of an asset from an engine-free prim dump (M7.17).

    python scripts/audit_materials.py prims.json [--rules configs/materials/mapping.yaml]
                                                 [--threshold 0.95] [--band lwir]

``prims.json`` is a list of records the Isaac adapter writes without any physics:

    [{"path": "/World/Car/Body", "material_name": "Car_Paint_Red",
      "semantic_class": "car_body", "override": null}, ...]

Prints every prim that fell through to UNMAPPED, grouped and counted, plus the coverage; exits
non-zero below the threshold (CI gate, ADR 0047).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, PrimRecord, audit, load_mapping_rules
    from irsim.materials.table import MaterialTable

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("prims", type=pathlib.Path, help="JSON list of prim records")
    ap.add_argument(
        "--rules", type=pathlib.Path, default=None, help="mapping.yaml (default: configs/materials)"
    )
    ap.add_argument(
        "--threshold", type=float, default=None, help="coverage threshold (default: from the rules)"
    )
    ap.add_argument("--band", default="lwir", help="band whose packed table fixes the ids")
    ap.add_argument(
        "--materials",
        type=pathlib.Path,
        default=None,
        help="material directory (default: configs/materials)",
    )
    args = ap.parse_args(argv)

    library = MaterialLibrary.load(args.materials)
    names = MaterialTable.from_library(library, args.band).names
    rules = load_mapping_rules(args.rules, known_materials=library.names)
    records = [PrimRecord.from_dict(d) for d in json.loads(args.prims.read_text(encoding="utf-8"))]
    report = audit(records, MaterialResolver(rules, names), args.threshold)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
