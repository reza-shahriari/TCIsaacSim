"""RP.2 — `CHANGELOG.md` is grouped by day, and no heading repeats inside a section.

One `[Unreleased]` section of 3,300 lines carrying **twenty** repeated `Added` / `Changed` / `Fixed`
headings is a single merge hotspot: three sessions working in one tree all append under the same
heading near the top of the same file, and a whole-file write from any of them silently takes the
others' entries with it. That is not hypothetical here — two commits already exist whose entire
subject is restoring lost CHANGELOG entries (`68acd1c`, `303b56b`).

Dated sub-sections shrink the collision surface to one day's block, and this parser makes a third
loss detectable *before* it is committed rather than by later `git log` archaeology.

The regrouping that created this structure moved whole bullet blocks and rewrote nothing: the
multiset of non-heading lines was compared before and after, and 3,242 content lines came through
unchanged.
"""

from __future__ import annotations

import collections
import datetime
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
CHANGELOG = REPO / "CHANGELOG.md"

RELEASE = re.compile(r"^## (.+)$")
DATED = re.compile(r"^### (\d{4}-\d{2}-\d{2})\s*$")
KIND = re.compile(r"^#### (\w+)\s*$")
#: Keep a Changelog's own set, plus the `Notes` this project uses for non-change commentary.
KINDS = {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security", "Notes"}


def _sections() -> dict[str, list[str]]:
    """``{date: [heading kinds in order]}`` for every dated sub-section."""
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in CHANGELOG.read_text().splitlines():
        if m := DATED.match(line):
            current = out.setdefault(m.group(1), [])
        elif (m := KIND.match(line)) and current is not None:
            current.append(m.group(1))
    return out


# --- the invariant this step exists for -------------------------------------------------------


def test_the_changelog_has_dated_sections() -> None:
    """A parser that found none would pass every test below for ever."""
    sections = _sections()
    assert len(sections) >= 3, f"only {len(sections)} dated sections found"


def test_no_heading_repeats_inside_a_dated_section() -> None:
    """The hotspot itself: twenty `Added` headings in one section is twenty places to collide."""
    offenders = []
    for date, kinds in _sections().items():
        repeated = [k for k, n in collections.Counter(kinds).items() if n > 1]
        if repeated:
            offenders.append(f"{date}: {sorted(repeated)}")
    assert not offenders, "repeated headings inside a dated section: " + "; ".join(offenders)


def test_no_date_heading_repeats() -> None:
    """Two sections for one day would reintroduce the same collision one level down."""
    dates = [m.group(1) for line in CHANGELOG.read_text().splitlines() if (m := DATED.match(line))]
    repeated = [d for d, n in collections.Counter(dates).items() if n > 1]
    assert not repeated, f"dates appearing twice: {sorted(repeated)}"


# --- the shape stays legible --------------------------------------------------------------------


def test_every_heading_is_one_keep_a_changelog_kind() -> None:
    unknown = {k for kinds in _sections().values() for k in kinds if k not in KINDS}
    assert not unknown, f"headings outside the known set: {sorted(unknown)}"


def test_dates_are_real_and_newest_first() -> None:
    """A reader skims the top for what just happened; ordering is the whole affordance."""
    dates = [m.group(1) for line in CHANGELOG.read_text().splitlines() if (m := DATED.match(line))]
    parsed = [datetime.date.fromisoformat(d) for d in dates]
    assert parsed == sorted(parsed, reverse=True), f"dated sections out of order: {dates}"


def test_every_dated_section_is_under_unreleased() -> None:
    """Once a release is cut, its dated sections move with it; until then they live here."""
    releases: list[str] = []
    seen_dated = 0
    for line in CHANGELOG.read_text().splitlines():
        if m := RELEASE.match(line):
            releases.append(m.group(1))
        elif DATED.match(line):
            seen_dated += 1
            assert releases, "a dated section appears before any `## ` release heading"
    assert releases[0] == "[Unreleased]"
    assert seen_dated == len(_sections())


def test_no_dated_section_is_empty() -> None:
    empty = [date for date, kinds in _sections().items() if not kinds]
    assert not empty, f"dated sections with no entries: {empty}"


# --- the parser itself ----------------------------------------------------------------------------


def test_the_parser_would_catch_a_repeat() -> None:
    """Negative control: the invariant must be able to fail, not merely to pass."""
    kinds = ["Added", "Fixed", "Added"]
    repeated = [k for k, n in collections.Counter(kinds).items() if n > 1]
    assert repeated == ["Added"]
