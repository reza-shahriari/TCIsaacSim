#!/usr/bin/env python3
"""Roadmap ME.1: obtain, verify and hash the public validation sets (ADR 0003).

Most of what this prints is what it will *not* do. Of the six indexed sets exactly one states a
licence, one is unreleased, and the rest sit behind drive links or an access request citing export
control -- so the default is to plan, report and refuse, and the human decides.

    python scripts/fetch_validation_data.py --list
    python scripts/fetch_validation_data.py --render-readme
    python scripts/fetch_validation_data.py --plan
    python scripts/fetch_validation_data.py --set halmstad_drone_detection
    python scripts/fetch_validation_data.py --set anti_uav_410 --accept-unstated-licence

Nothing here needs Isaac Sim, a GPU or the network unless a set has a direct URL.
"""

from __future__ import annotations

import argparse
import sys

from irsim_eval.fetch import download, plan, sha256_of, verify
from irsim_eval.manifest import load_manifest, readme_path, render_readme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", dest="sets", help="index name; repeatable")
    parser.add_argument("--list", action="store_true", help="print the index and exit")
    parser.add_argument("--plan", action="store_true", help="say what would happen, do nothing")
    parser.add_argument(
        "--accept-unstated-licence",
        action="store_true",
        help="proceed with sets whose publisher stated no terms of use (read them first)",
    )
    parser.add_argument("--hash", dest="hash_path", help="print the SHA-256 of one file and exit")
    parser.add_argument(
        "--render-readme",
        action="store_true",
        help="regenerate data/validation/README.md from the index and exit",
    )
    args = parser.parse_args(argv)

    if args.hash_path:
        print(sha256_of(args.hash_path))
        return 0

    manifest = load_manifest()

    if args.render_readme:
        readme_path().write_text(render_readme(manifest), encoding="utf-8")
        print(f"wrote {readme_path()}")
        return 0

    if args.list:
        primary, _ = manifest.primary
        print(f"index checked {manifest.checked_utc}; primary set: {primary}\n")
        print(f"{'set':28s} {'licence':12s} {'access':11s} {'role':11s} analysers")
        for name, d in manifest.datasets.items():
            print(f"{name:28s} {d.licence:12s} {d.access:11s} {d.role:11s} {len(d.analysers)}")
        print("\n'unstated' means the publisher stated no terms -- not that they are permissive.")
        return 0

    actions = plan(manifest, args.sets, accept_unstated_licence=args.accept_unstated_licence)
    status = 0
    for action in actions:
        print(f"\n[{action.decision}] {action.name}")
        print(f"  {action.reason}")
        print(f"  destination: {action.destination}")
        if action.url:
            print(f"  source: {action.url}")
        if args.plan or action.needs_a_person:
            # A plan is a report, so it always succeeds. A real run exits non-zero when a set was
            # refused, so a pipeline notices rather than silently proceeding with less data.
            if action.decision == "refused" and not args.plan:
                status = 1
            continue
        if action.decision == "download":
            assert action.url is not None
            dest = action.destination / action.url.rsplit("/", 1)[-1]
            print(f"  downloading -> {dest}")
            download(action.url, dest)
            ok, digest = verify(manifest.datasets[action.name], dest)
            print(f"  sha256 {digest} ({'matches the index' if ok else 'DOES NOT MATCH'})")
            status = status or (0 if ok else 2)
        elif action.decision == "verify":
            files = sorted(p for p in action.destination.rglob("*") if p.is_file())
            print(f"  {len(files)} files on disk")
            for path in files[:10]:
                print(f"    {sha256_of(path)}  {path.name}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")

    print("\nNothing was written into the repository: data/validation/<set>/ is gitignored.")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
