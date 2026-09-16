"""RP.5 — every ADR carries a status in one spelling, and supersessions are actually recorded.

Two problems, both of which make the decision record less trustworthy than its own size suggests.

**Three spellings.** `**Status:** Accepted`, `Status: accepted` and `- Status: accepted` were all in
use across 88 files, so no single grep answered "what is the state of the record". They are now one
label, `**Status:**`, with the bullet preserved where a file's metadata is a list.

**Nothing was ever superseded.** ADR 0001's template offers `Superseded by ADR NNNN` and not one
file had used it — while at least three decisions had been overtaken in part:

* **0059** ruled that "bolometer smear is the inter-frame IIR of §9.2, never a second blur".
  ADR 0077 then made `smear_duty(None)` mean *a bolometer and therefore 1*, so
  `optics/stage.py` applies exactly the second blur that clause forbade. The two are different
  mechanisms and a bolometer has both, so 0077 is right and the clause was wrong.
* **0006** designed a temperature encoding for a render G-buffer; ADR 0014 measured that no
  colour AOV can carry it, so the transport it was written for does not exist.
* **0048** documented the grey band model as valid to ~500 m; ADR 0071's layered slant path replaced
  that clause for non-horizontal geometry.

None of the three is superseded *entirely*, which is why the status line names the clause. A record
that can only say "Accepted" or "dead" cannot describe what actually happens to decisions.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DECISIONS = REPO / "docs" / "decisions"

#: The one label. A leading `- ` is kept where a file's metadata block is a markdown list.
STATUS = re.compile(r"^(?:- )?\*\*Status:\*\* (.+?)\s*$")
#: The states the template sanctions, each optionally qualified after a `;` or in parentheses.
STATES = ("Accepted", "Proposed", "Superseded")
#: ADR 0001 quotes the blank template inside a fenced block; that line is not a status.
TEMPLATE_LINE = "**Status:** Proposed | Accepted | Superseded by ADR NNNN"


def _adrs() -> list[pathlib.Path]:
    return sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))


def _status_of(path: pathlib.Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.strip() == TEMPLATE_LINE:
            continue
        if m := STATUS.match(line):
            return m.group(1)
    return None


# --- one spelling -------------------------------------------------------------------------------


def test_there_are_adrs_to_check() -> None:
    assert len(_adrs()) >= 80, "the decision record has changed shape"


def test_every_adr_carries_a_status() -> None:
    missing = [p.name for p in _adrs() if _status_of(p) is None]
    assert not missing, f"ADRs with no `**Status:**` line: {missing}"


def test_no_adr_uses_an_unbolded_status_label() -> None:
    """The three spellings this step collapsed. A grep has to answer for the whole record."""
    stray = []
    for path in _adrs():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if re.match(r"^(?:- )?Status:", line):
                stray.append(f"{path.name}:{number}")
    assert not stray, f"unbolded `Status:` labels remain: {stray}"


def test_every_status_starts_with_a_sanctioned_state() -> None:
    wrong = [
        f"{p.name}: {s!r}" for p in _adrs() if (s := _status_of(p)) and not s.startswith(STATES)
    ]
    assert not wrong, "statuses outside the template's vocabulary: " + "; ".join(wrong)


# --- a supersession points somewhere --------------------------------------------------------------


def test_every_adr_a_status_supersedes_to_exists() -> None:
    """A status that names a replacement which was never written is worse than no status."""
    have = {p.name[:4] for p in _adrs()}
    dangling = []
    for path in _adrs():
        status = _status_of(path) or ""
        for number in re.findall(r"superseded by ADR (\d{4})", status, flags=re.IGNORECASE):
            if number not in have:
                dangling.append(f"{path.name} → ADR {number}")
    assert not dangling, "statuses point at ADRs that do not exist: " + "; ".join(dangling)


def test_no_adr_declares_itself_superseded_by_itself() -> None:
    for path in _adrs():
        status = _status_of(path) or ""
        assert path.name[:4] not in re.findall(
            r"superseded by ADR (\d{4})", status, flags=re.IGNORECASE
        )


# --- the pairs this step was seeded on -------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "supersedes_to", "clause"),
    [("0059", "0077", "motion-MTF"), ("0006", "0014", "G-buffer"), ("0048", "0071", "range")],
)
def test_the_known_supersessions_are_recorded(number: str, supersedes_to: str, clause: str) -> None:
    """The self-test the step asks for: these were live contradictions in the record."""
    path = next(DECISIONS.glob(f"{number}-*.md"))
    status = _status_of(path) or ""
    assert f"superseded by ADR {supersedes_to}" in status, f"ADR {number}'s status: {status!r}"
    assert clause in status, f"ADR {number}'s status does not name the clause: {status!r}"


def test_the_superseded_clause_in_0059_is_marked_in_the_body_too() -> None:
    """A reader who lands mid-document must not act on the clause the header retired."""
    body = next(DECISIONS.glob("0059-*.md")).read_text()
    assert "never a second blur" in body, "the clause should be struck through, not deleted"
    assert "~~" in body and "Superseded by ADR 0077" in body


def test_a_status_the_parser_would_reject(tmp_path: pathlib.Path) -> None:
    """Negative control, so the vocabulary check is known to be able to fail."""
    bad = tmp_path / "0999-a-decision.md"
    bad.write_text("# 0999 — x\n\n**Status:** Probably fine\n")
    assert not (_status_of(bad) or "").startswith(STATES)
