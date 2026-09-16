"""RP.9 — the row-length lint the roadmap says enforces it.

Revision 3 of the plan held 224,508 of its 298,272 characters inside markdown table cells, averaging
1,234 per row with a 4,958-character maximum. Rows that size are a second changelog: unrenderable,
unreviewable as a diff, and unmergeable between two sessions without a manual rewrite — which is how
four subsystems came to be over-reported and how a whole-file write silently reverted committed
work.

Revision 4 states the fix in its own prose: "**Step rows are capped at 600 characters and ledger
rows
at 200**, enforced by a parser in `make check` (RP.9)." This is that parser. Until it existed the
cap
was a sentence, and a sentence is what revision 3 had too.

**Scope, because a lint that creeps is a lint that gets disabled.** Step tables and the shipped
ledger
only — never prose, never the generated *Do next* block, never the phase or risk tables. A paragraph
is allowed to be a paragraph; the cap exists because a *table cell* is the thing that does not diff.

This is a lint, not a physics verification, and is not counted as one.

The ledger's own 200-character cap has an owner already (`test_shipped_ledger.py`, RP.6) and is not
duplicated here; this file owns the 600 and the self-test that proves the parser catches rather than
merely passes.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = REPO / "docs" / "roadmap.md"

#: The cap the document states for itself, in the paragraph this test exists to make true.
STEP_ROW_CHARS = 600
#: A step row: `| RP.9 | what | verification | deps | size | phase |` -- lane, dot, number.
STEP_ROW = re.compile(r"^\|\s*([A-Z]{2}\.[0-9]+)\s*\|")
#: Where step tables live. Everything before the first lane heading is prose, phase and risk tables.
LANE_HEADING = re.compile(r"^## [A-Z]{2} — ")


def _step_rows() -> list[tuple[int, str, str]]:
    """``(line number, step id, line)`` for every row of every lane table."""
    out: list[tuple[int, str, str]] = []
    in_lane = False
    for number, line in enumerate(ROADMAP.read_text().splitlines(), start=1):
        if line.startswith("## "):
            in_lane = bool(LANE_HEADING.match(line))
        if not in_lane:
            continue
        if m := STEP_ROW.match(line):
            out.append((number, m.group(1), line))
    return out


def _over(rows: list[tuple[int, str, str]], cap: int) -> list[str]:
    return [f"{sid} (line {n}): {len(line)} characters" for n, sid, line in rows if len(line) > cap]


# --- the lint ---------------------------------------------------------------------------------


def test_the_parser_finds_the_step_tables() -> None:
    """A lint that matched nothing would pass for ever, which is the failure mode of a lint."""
    rows = _step_rows()
    assert len(rows) >= 80, f"only {len(rows)} step rows found; the table shape has changed"
    ids = {sid for _, sid, _ in rows}
    for expected in ("RP.9", "IG.1", "PT.4"):
        assert expected in ids, f"{expected} was not parsed out of a lane table"


def test_no_step_row_is_over_the_cap() -> None:
    over = _over(_step_rows(), STEP_ROW_CHARS)
    assert not over, (
        f"step rows past {STEP_ROW_CHARS} characters: "
        + "; ".join(over)
        + ". A row that needs more space needs an ADR instead."
    )


def test_the_document_still_states_the_cap_this_test_enforces() -> None:
    """If the prose and the parser disagree, the prose is what a reader will believe."""
    # Whitespace-normalised: the sentence wraps across two lines in the source, and a reader sees
    # it wrapped or not depending on their viewport. The claim is what matters, not the line break.
    text = " ".join(ROADMAP.read_text().split())
    assert f"capped at {STEP_ROW_CHARS} characters" in text


# --- the self-test the step asks for ------------------------------------------------------------


def test_the_parser_catches_a_synthetic_over_length_row(tmp_path: pathlib.Path) -> None:
    """Proof the lint catches rather than merely passes, without making this document red."""
    fat = "| ZZ.1 | " + "x" * STEP_ROW_CHARS + " | v | — | S | 0 |"
    rows = [(1, "ZZ.1", fat)]
    assert _over(rows, STEP_ROW_CHARS) == [f"ZZ.1 (line 1): {len(fat)} characters"]
    # And one character under the cap is accepted, so the boundary is the stated number.
    assert _over([(1, "ZZ.1", "|" * STEP_ROW_CHARS)], STEP_ROW_CHARS) == []


@pytest.mark.parametrize(
    "line",
    [
        "| **0 — Repair** | `RP.1`–`RP.9` | a phase row, not a step row |",
        "| R28 | a risk-register row | not a step row |",
        "| 1 | **`RP.9`** | RP | 0 | S | — | ready |",
        "| M0 build, layering | done | `e0c6249` | a ledger row, capped at 200 elsewhere |",
        "| S13 | a spec-issue row in another file entirely | open |",
        "Prose about `RP.9` that happens to mention a step id.",
    ],
)
def test_the_row_pattern_matches_nothing_outside_a_step_table(line: str) -> None:
    """Phase, risk, queue, ledger and spec-issue rows and prose are all out of scope.

    The queue row is the one worth naming: the generated *Do next* block lives inside the file this
    parser reads, and its rows begin with a position number rather than a step id -- the same
    property `test_roadmap_queue.py` relies on. If that ever changed, the block would start linting
    itself.
    """
    assert STEP_ROW.match(line) is None


def test_no_step_row_is_found_before_the_first_lane_heading() -> None:
    """The section filter, checked rather than assumed.

    Everything above the first `## XX — ` heading is prose, the phase plan and the shipped ledger.
    A parser that ignored headings would pull the ledger's rows in and lint them at the wrong cap.
    """
    lines = ROADMAP.read_text().splitlines()
    first_lane = next(n for n, line in enumerate(lines, start=1) if LANE_HEADING.match(line))
    rows = _step_rows()
    assert rows, "no step rows parsed"
    assert min(n for n, _, _ in rows) > first_lane


def test_the_ledger_is_not_linted_at_the_step_cap() -> None:
    """Its rows are capped at 200 and owned by `test_shipped_ledger.py`; double ownership rots."""
    ledger_ids = {sid for _, sid, _ in _step_rows() if sid.startswith("M")}
    assert not ledger_ids, f"ledger rows leaked into the step lint: {sorted(ledger_ids)}"
