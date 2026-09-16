"""RP.8 — CLAUDE.md's repository-layout block describes the repository that exists.

CLAUDE.md is the first thing every session reads and the one document nobody re-checks. Two of its
layout claims had drifted into fiction:

* `src/irsim_isaac/spg/` was advertised as holding ".cu kernels, .cu.lua launch scripts, .usda
  shader defs". It holds one README, whose own first paragraph says "**This directory is empty on
  purpose**" — the four steps that would fill it are blocked (DC.1).
* `docs/maps/` was advertised as "per-module JSON maps used for fast navigation of the physics
  core". The JSON is there, but its own README calls it frozen 2026-09-10 snapshots, and it has one
  commit in its history. A reader sent there for navigation gets a six-day-old picture of a
  fast-moving tree and no warning.

Neither is a policy question — they are statements of fact about paths, and they were false. The
wording that *is* contested (line 13 on Unreal, the commit-scope list, the Isaac version string)
belongs to open questions 2 and 9 and is deliberately untouched here.

The parser is the point: a claim about the tree that a test can check stops being something someone
has to remember.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
CLAUDE_MD = REPO / "CLAUDE.md"

LAYOUT_HEADING = "## Repository layout"
#: `src/irsim/radiometry/    # Planck, band integration, ...` — a path, then an optional comment.
ENTRY = re.compile(r"^(?P<indent> *)(?P<path>[\w./@-]+/?)\s*(?:#\s*(?P<comment>.*))?$")
#: `.cu`, `.cu.lua`, `.usda` — an extension claimed inside a comment.
EXTENSION = re.compile(r"(?<![\w/])\.(?:[a-z][a-z0-9]*)(?:\.[a-z][a-z0-9]*)?\b")
#: A directory that says of itself that it is not a live artefact.
FROZEN_WORDS = ("frozen", "not maintained", "empty on purpose")


def _layout() -> list[tuple[str, str]]:
    """``(path, comment)`` for every entry of the layout block, with nesting resolved."""
    text = CLAUDE_MD.read_text()
    block = text.split(LAYOUT_HEADING, 1)[1].split("```", 2)[1]
    out: list[tuple[str, str]] = []
    parent = ""
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = ENTRY.match(line.rstrip())
        if not m:
            continue
        path, comment = m.group("path"), (m.group("comment") or "")
        if m.group("indent"):
            path = parent + path
        else:
            parent = path if path.endswith("/") else ""
        out.append((path, comment))
    return out


# --- the block parses at all ------------------------------------------------------------------


def test_the_layout_block_parses() -> None:
    """A parser that silently matched nothing would pass everything below for ever."""
    entries = _layout()
    assert len(entries) >= 25, f"only {len(entries)} layout entries parsed"
    paths = {p for p, _ in entries}
    for expected in ("src/irsim/radiometry/", "src/irsim_isaac/spg/", "docs/maps/", "scripts/"):
        assert expected in paths, f"{expected} not parsed out of the layout block"


# --- the claims ---------------------------------------------------------------------------------


def test_every_path_named_in_the_layout_exists() -> None:
    missing = [path for path, _ in _layout() if not (REPO / path).exists()]
    assert not missing, f"CLAUDE.md names paths that do not exist: {missing}"


def test_a_directory_claiming_file_types_contains_them() -> None:
    """`spg/` claimed .cu / .cu.lua / .usda and held one README.

    A directory may legitimately be empty — but then it says so, and CLAUDE.md says so too, rather
    than describing the intention in the present tense.
    """
    wrong = []
    for path, comment in _layout():
        target = REPO / path
        if not target.is_dir():
            continue
        claimed = {m.group(0) for m in EXTENSION.finditer(comment)}
        if not claimed:
            continue
        if any(word in comment.lower() for word in FROZEN_WORDS):
            continue
        for suffix in sorted(claimed):
            if not any(target.rglob(f"*{suffix}")):
                wrong.append(f"{path} is described as holding {suffix} files and holds none")
    assert not wrong, "; ".join(wrong)


def test_a_directory_that_calls_itself_frozen_is_not_advertised_as_live() -> None:
    """`docs/maps/` is a 2026-09-10 snapshot with one commit; it was sold as a navigation aid.

    The directory's own README is the source of truth about itself. Where it disclaims currency,
    CLAUDE.md has to carry the same disclaimer, or a reader is sent to stale data unwarned.
    """
    wrong = []
    for path, comment in _layout():
        target = REPO / path
        readme = target / "README.md"
        if not target.is_dir() or not readme.is_file():
            continue
        own = readme.read_text().lower()
        if not any(word in own for word in FROZEN_WORDS):
            continue
        if not any(word in comment.lower() for word in FROZEN_WORDS):
            wrong.append(
                f"{path}'s own README disclaims currency but CLAUDE.md describes it as {comment!r}"
            )
    assert not wrong, "; ".join(wrong)
