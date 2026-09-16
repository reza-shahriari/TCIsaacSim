"""RP.4 — every `ADR NNNN` citation in the repository resolves to a file that exists.

CLAUDE.md's reason for ADRs is that "a reader who was not in the conversation" can recover the
reasoning behind a decision. A citation pointing at a file that was never written is worse than no
citation: it tells that reader the reasoning exists and sends them to look for it. Two shipped
decisions were in exactly that state until this step — the angular-emissivity level policy
(ADR 0042, cited by two source modules, five test files and two spec issues) and the sea-water
optical constants and slope model (ADR 0079, cited by two source modules, one test, and by ADR 0078
for a bound it deferred).

The parser handles the citation forms the repository actually uses, which are not only `ADR 0042`:
`ADRs 0031, 0058`, `ADR 0053/0054` and `ADR 0025 and 0023` all appear, and a parser that reads only
the first number in each would have silently passed twenty citations it never checked.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DECISIONS = REPO / "docs" / "decisions"
SEARCH_ROOTS = ("src", "tests", "docs", "scripts", "configs")
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh", ".cu", ".lua", ".usda"}

#: This file holds example citations (`ADR 9999`) and names the phantom it allowlists, so
#: scanning it would make the guard fail on its own fixtures.
SELF = pathlib.Path(__file__).resolve()

#: `ADR 0042`, `ADRs 0031, 0058`, `ADR 0053/0054`, `ADR 0025 and 0023`.
CITATION = re.compile(r"ADRs?\s+(\d{4}(?:\s*(?:,|/|,?\s+and)\s*\d{4})*)")

#: Citations that are *deliberately* unresolvable, each with the reason. An entry here is a claim
#: that the text names a non-existent ADR on purpose, so `test_every_allowlisted_phantom_is_still
#: _needed` makes an entry fail once the text or the ADR changes — an allowlist that cannot rot.
PHANTOMS: dict[tuple[str, str], str] = {
    ("docs/roadmap.md", "0069"): (
        "The roadmap's post-mortem of the rejected revision 4 records that that revision "
        "'cited ADR 0069 as an existing record' when no such ADR existed. Deleting the mention "
        "to satisfy this test would delete the record of the mistake it exists to prevent."
    ),
}


def _numbers(blob: str) -> list[str]:
    return re.findall(r"\d{4}", blob)


def _cited() -> dict[str, set[str]]:
    """``{adr number: {relative paths that cite it}}`` across the whole repository."""
    out: dict[str, set[str]] = {}
    for root in SEARCH_ROOTS:
        for path in sorted((REPO / root).rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES or path.resolve() == SELF:
                continue
            for match in CITATION.finditer(path.read_text(errors="ignore")):
                for number in _numbers(match.group(1)):
                    out.setdefault(number, set()).add(str(path.relative_to(REPO)))
    return out


def _existing() -> dict[str, pathlib.Path]:
    return {p.name[:4]: p for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")}


# --- the invariant ------------------------------------------------------------------------------


def test_every_adr_citation_resolves_to_a_file() -> None:
    existing = _existing()
    dangling = [
        (number, path)
        for number, paths in _cited().items()
        for path in sorted(paths)
        if number not in existing and (path, number) not in PHANTOMS
    ]
    assert not dangling, "citations point at ADRs that do not exist: " + "; ".join(
        f"ADR {number} cited by {path}" for number, path in sorted(dangling)
    )


def test_every_allowlisted_phantom_is_still_needed() -> None:
    """An allowlist entry must still describe reality, or it is hiding a citation nobody checks."""
    existing = _existing()
    cited = _cited()
    for (path, number), reason in PHANTOMS.items():
        assert reason.strip(), f"the {path}/{number} allowlist entry has no reason"
        assert number not in existing, (
            f"ADR {number} now exists, so the {path} allowlist entry is stale — delete it"
        )
        assert path in cited.get(number, set()), (
            f"{path} no longer cites ADR {number}; delete the allowlist entry"
        )


# --- the two this step was written for ----------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "must_mention"),
    [
        ("0042", ("Level A", "Level B", "Level C", "S12", "S18")),
        ("0079", ("Segelstein", "Cox", "salinity")),
    ],
)
def test_the_adrs_shipped_code_cites_say_what_it_cites_them_for(
    number: str, must_mention: tuple[str, ...]
) -> None:
    """A file with the right name and the wrong contents would pass the invariant above.

    Each subject is named by the citing code or by `docs/spec-issues.md`, so an ADR that does not
    mention it is not the record those citations promise.
    """
    path = _existing().get(number)
    assert path is not None, f"ADR {number} is missing"
    text = path.read_text()
    missing = [m for m in must_mention if m not in text]
    assert not missing, f"ADR {number} never mentions {missing}"


# --- the decision record's own housekeeping -----------------------------------------------------


def test_each_adr_heading_matches_its_filename() -> None:
    """A renumbered file with an unrenumbered heading sends a reader to the wrong decision."""
    wrong = []
    for number, path in sorted(_existing().items()):
        first = path.read_text().splitlines()[0]
        if not re.match(rf"#\s+(ADR\s+)?{number}\b", first):
            wrong.append(f"{path.name} opens with {first!r}")
    assert not wrong, "ADR headings disagree with their filenames: " + "; ".join(wrong)


def test_no_two_adrs_share_a_number() -> None:
    seen: dict[str, str] = {}
    clashes = []
    for path in sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")):
        number = path.name[:4]
        if number in seen:
            clashes.append(f"{number}: {seen[number]} and {path.name}")
        seen[number] = path.name
    assert not clashes, "duplicate ADR numbers: " + "; ".join(clashes)


# --- the parser itself --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see ADR 0042", ["0042"]),
        ("ADRs 0031, 0058", ["0031", "0058"]),
        ("ADR 0053/0054", ["0053", "0054"]),
        ("ADR 0025 and 0023", ["0025", "0023"]),
        ("ADR 0077/0082/0084/0089", ["0077", "0082", "0084", "0089"]),
        ("ADR 0044, 0050, 0070, 0071", ["0044", "0050", "0070", "0071"]),
    ],
)
def test_the_parser_reads_every_number_in_a_citation(text: str, expected: list[str]) -> None:
    """The negative control for the whole file: a parser reading only the first number would let
    a dangling `ADR 0053/9999` through, and twenty real citations use these compound forms."""
    found = [n for m in CITATION.finditer(text) for n in _numbers(m.group(1))]
    assert found == expected


def test_a_dangling_citation_is_actually_caught() -> None:
    """Without this, a parser that silently matched nothing would pass the invariant for ever."""
    found = [
        n
        for m in CITATION.finditer("cites ADR 9999 which is not a thing")
        for n in _numbers(m.group(1))
    ]
    assert found == ["9999"]
    assert "9999" not in _existing()
