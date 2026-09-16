"""RP.6 — every shipped-ledger row points at a commit that exists.

Revision 3 of the roadmap prescribed `done YYYY-MM-DD <hash>` on its shipped rows and **zero of its
162 done rows carried one**, M0.1 included — the row it used as the example of the format. A rule
that only applies going forward leaves the state unrecoverable after the next whole-file write,
which is the failure this revision of the plan exists to fix.

So the hash is checked rather than prescribed, and checked the only way that means anything: the
7-hex string has to **resolve in this repository**. A typo'd hash is worse than `pending`, because
`pending` at least tells the reader it has nothing for them.

The test degrades rather than lies: outside a git checkout it skips the resolution half and still
enforces the shape.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = REPO / "docs" / "roadmap.md"

LEDGER_HEADING = "## Shipped ledger"
#: `| M7 materials | done | `1a0f13c` | note |`
ROW = re.compile(
    r"^\|\s*((?:M[0-9]{1,2}|ME|MS|MM|MP|IU)\b[^|]*?)\s*\|\s*(\w+)\s*\|\s*([^|]*?)\s*\|"
)
HASH = re.compile(r"^`([0-9a-f]{7,40})`$")
#: The cap the section states for itself.
ROW_CHARS = 200


def _ledger() -> list[tuple[str, str, str]]:
    """``(milestone, state, hash cell)`` for every row of the shipped ledger."""
    text = ROADMAP.read_text()
    section = text.split(LEDGER_HEADING, 1)[1].split("\n## ", 1)[0]
    return [
        (m.group(1), m.group(2), m.group(3))
        for line in section.splitlines()
        if (m := ROW.match(line))
    ]


def _in_git() -> bool:
    return (REPO / ".git").exists()


def _resolves(sha: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


# --- the invariant ------------------------------------------------------------------------


def test_the_ledger_is_found_and_not_empty() -> None:
    """A parser that silently matched nothing would pass every test below for ever."""
    rows = _ledger()
    assert len(rows) >= 15, f"only {len(rows)} ledger rows parsed; the table's shape has changed"


def test_no_row_is_still_pending() -> None:
    """`pending` was the placeholder RP.6 existed to remove."""
    pending = [name for name, _, cell in _ledger() if "pending" in cell.lower()]
    assert not pending, f"rows still say pending: {pending}"


def test_every_shipped_row_carries_a_hash_and_every_open_row_does_not() -> None:
    wrong = []
    for name, state, cell in _ledger():
        if state in {"done", "partial"}:
            if not HASH.match(cell):
                wrong.append(f"{name} is {state} but its hash cell is {cell!r}")
        elif state == "open":
            if cell not in {"—", "-", ""}:
                wrong.append(f"{name} is open but carries {cell!r}")
        else:
            wrong.append(f"{name} has an unknown state {state!r}")
    assert not wrong, "; ".join(wrong)


@pytest.mark.skipif(not _in_git(), reason="not a git checkout")
def test_every_ledger_hash_resolves_to_a_commit() -> None:
    """The half that matters. A hash of the right shape and the wrong value reads as evidence."""
    dangling = [
        f"{name} → {m.group(1)}"
        for name, state, cell in _ledger()
        if state in {"done", "partial"} and (m := HASH.match(cell)) and not _resolves(m.group(1))
    ]
    assert not dangling, "ledger hashes that are not commits: " + "; ".join(dangling)


@pytest.mark.skipif(not _in_git(), reason="not a git checkout")
def test_a_wrong_hash_would_be_caught() -> None:
    """The negative control for the test above, so it cannot pass by never resolving anything."""
    assert not _resolves("0000000")
    some_real = next(
        m.group(1) for _, state, cell in _ledger() if state != "open" and (m := HASH.match(cell))
    )
    assert _resolves(some_real)


# --- the section's own rules ----------------------------------------------------------------


def test_ledger_rows_stay_under_the_cap_the_section_states() -> None:
    """200 characters is what keeps this table mergeable; revision 3 averaged 1,234 and was not."""
    text = ROADMAP.read_text()
    section = text.split(LEDGER_HEADING, 1)[1].split("\n## ", 1)[0]
    over = [
        f"{line[:48]}… ({len(line)})"
        for line in section.splitlines()
        if ROW.match(line) and len(line) > ROW_CHARS
    ]
    assert not over, f"ledger rows past {ROW_CHARS} characters: " + "; ".join(over)


def test_each_milestone_appears_once() -> None:
    names = [name.split()[0] for name, _, _ in _ledger()]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"milestones listed twice: {sorted(duplicates)}"
