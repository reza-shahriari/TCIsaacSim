"""The validation-data index: schema, loader, and the README it generates.

roadmap ME.1; ADR 0003 (no camera -- every comparison with reality is against public imagery).

A directory of downloaded video is worthless without a record of **what each file is**. The same
flat-sky patch supports a three-dimensional noise decomposition if it came off a Y16 stream and
supports nothing at all if it came out of an unknown ISP and an unknown codec; a box size converts
to a range only if the sensor's pitch and focal length are known. So the index is not
documentation, it is a precondition: :class:`Dataset` makes ``licence``, ``signal_path`` and
``analysers`` required fields, and a set that cannot fill them cannot be added.

**``licence: unstated`` is a value, not a gap.** Most of these publishers simply never said. The
schema records that as a fact rather than defaulting to something comfortable, and
``scripts/fetch_validation_data.py`` refuses such a set unless it is asked explicitly, so the
decision to use data on unknown terms is made by a person on purpose.

``data/validation/README.md`` is **generated** from the YAML
(``scripts/fetch_validation_data.py --render-readme``) and a test asserts it is current, because an
index that disagrees with itself is worse than none: the prose is what people read and the fields
are what the code reads.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Access",
    "Role",
    "UNSTATED",
    "Dataset",
    "Manifest",
    "manifest_path",
    "readme_path",
    "load_manifest",
    "render_readme",
]

#: What a licence field says when the publisher did not state one. Not a guess that it is
#: permissive -- a record that the terms are unknown.
UNSTATED = "unstated"

#: How the data is obtained. ``direct`` is the only one a script can do unattended.
Access = Literal["direct", "manual", "unreleased"]

#: What the set is for. ``primary`` is the reference set; ``prior`` sets can bound a distribution
#: but can never be a target the simulator is tuned to hit.
Role = Literal["primary", "supplement", "prior"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Dataset(_Frozen):
    """One indexed set. Every field that decides whether a measurement is *allowed* is required."""

    title: str = Field(min_length=1)
    role: Role
    why: str = Field(min_length=1)
    licence: str = Field(min_length=1)
    access: Access
    signal_path: str = Field(min_length=1)
    analysers: list[str] = Field(min_length=1)

    licence_note: str | None = None
    excluded_analysers: list[str] = Field(default_factory=list)
    notes: str | None = None

    doi: str | None = None
    paper_doi: str | None = None
    paper_arxiv: str | None = None
    code_url: str | None = None
    site_url: str | None = None
    download_url: str | None = None

    sensor: str | None = None
    fov_deg: list[float] | None = None
    resolution: list[int] | None = None
    frame_rate_hz: float | None = None
    bit_depth_native: int | None = None
    bit_depth_stored: int | None = None
    codec: str | None = None
    bitrate_kbps: float | None = None

    clips: int | None = None
    clip_seconds: float | None = None
    frames: int | None = None
    thermal_images: int | None = None
    annotated_frames: int | None = None
    annotated_boxes: int | None = None
    classes: list[str] | None = None
    label_format: str | None = None

    sha256: str | None = None

    @field_validator("analysers", "excluded_analysers")
    @classmethod
    def _no_duplicates(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("analyser lists must not repeat an entry")
        return value

    @field_validator("sha256")
    @classmethod
    def _hash_shape(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be 64 hex characters")
        return value.lower()

    @property
    def licence_known(self) -> bool:
        """False when the publisher stated no terms -- the fetch script's gate."""
        return self.licence != UNSTATED

    @property
    def fetchable(self) -> bool:
        """Whether a script can obtain this set without a human."""
        return self.access == "direct" and self.download_url is not None

    def may_run(self, analyser: str) -> bool:
        """Whether ``analyser`` is allowed on this set (the exclusions win over the inclusions)."""
        if analyser in self.excluded_analysers:
            return False
        return analyser in self.analysers


class Manifest(_Frozen):
    """The whole index."""

    schema_version: int
    checked_utc: str
    datasets: dict[str, Dataset]

    @field_validator("schema_version")
    @classmethod
    def _version(cls, value: int) -> int:
        if value != 1:
            raise ValueError(f"unknown manifest schema_version {value}")
        return value

    @field_validator("datasets")
    @classmethod
    def _exactly_one_primary(cls, value: dict[str, Dataset]) -> dict[str, Dataset]:
        primaries = [k for k, v in value.items() if v.role == "primary"]
        if len(primaries) != 1:
            raise ValueError(
                f"the index needs exactly one primary set, found {primaries}: the primary is the "
                "one whose licence and signal path are known well enough to state a result "
                "against, and having two invites a comparison that averages them"
            )
        return value

    @property
    def primary(self) -> tuple[str, Dataset]:
        return next((k, v) for k, v in self.datasets.items() if v.role == "primary")

    def usable_for(self, analyser: str) -> list[str]:
        """Names of the sets ``analyser`` may be run on."""
        return sorted(k for k, v in self.datasets.items() if v.may_run(analyser))


def manifest_path(root: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """``data/validation/datasets.yaml``, from the repository root by default."""
    if root is not None:
        return pathlib.Path(root) / "data" / "validation" / "datasets.yaml"
    return pathlib.Path(__file__).resolve().parents[2] / "data" / "validation" / "datasets.yaml"


def load_manifest(path: str | os.PathLike[str] | None = None) -> Manifest:
    """Load and validate the index."""
    p = pathlib.Path(path) if path is not None else manifest_path()
    return Manifest.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))


def _row(label: str, value: Any) -> str:
    return f"| {label} | {value} |\n"


def render_readme(manifest: Manifest) -> str:
    """The human-readable index, generated so it cannot drift from the fields the code reads."""
    primary_name, primary = manifest.primary
    lines = [
        "<!-- GENERATED FILE - do not edit.",
        "     Source: data/validation/datasets.yaml",
        "     Regenerate: python scripts/fetch_validation_data.py --render-readme",
        "     tests/unit/test_validation_manifest.py fails if this file is stale. -->",
        "",
        "# Validation data",
        "",
        "No IR camera is available to this project (ADR 0003), so every comparison against reality",
        "is made against imagery somebody else published. What each set *is* therefore decides",
        "what may be measured on it:",
        "a three-dimensional noise decomposition means something on a Y16",
        "stream and nothing on an unknown ISP's output through a lossy codec, and a box size",
        "converts to a range only when the sensor's pitch and focal length are known.",
        "",
        "The datasets themselves are **not** in git. `scripts/fetch_validation_data.py` obtains or",
        "verifies them into `data/validation/<set>/`.",
        "",
        f"**Primary set: `{primary_name}`** -- {primary.title}.",
        "",
        primary.why.strip(),
        "",
        "## Licences",
        "",
        "`unstated` means the publisher did not state terms. That is not a guess that they are",
        "permissive: it is a record that they are unknown, and the fetch script refuses such a set",
        "unless asked explicitly, so using data on unknown terms is always a deliberate act.",
        "",
        "| set | licence | access | role |",
        "|---|---|---|---|",
    ]
    for name, d in manifest.datasets.items():
        lines.append(f"| `{name}` | {d.licence} | {d.access} | {d.role} |")
    checked = f"Fields last checked against the publishers' own pages: **{manifest.checked_utc}**."
    lines += ["", checked, ""]

    for name, d in manifest.datasets.items():
        lines += [f"## `{name}`", "", f"**{d.title}**", "", d.why.strip(), ""]
        lines += ["| field | value |", "|---|---|"]
        block = ""
        block += _row("licence", d.licence)
        if d.licence_note:
            block += _row("licence note", d.licence_note.strip().replace("\n", " "))
        block += _row("access", d.access)
        for label, value in (
            ("doi", d.doi),
            ("paper", d.paper_doi or (f"arXiv:{d.paper_arxiv}" if d.paper_arxiv else None)),
            ("code", d.code_url),
            ("site", d.site_url),
            ("sensor", d.sensor),
            ("resolution", d.resolution),
            ("field of view (deg)", d.fov_deg),
            ("frame rate (Hz)", d.frame_rate_hz),
            ("bit depth (native / stored)", f"{d.bit_depth_native} / {d.bit_depth_stored}"),
            ("codec", d.codec),
            ("bitrate (kbps)", d.bitrate_kbps),
            ("clips", d.clips),
            ("clip length (s)", d.clip_seconds),
            ("frames", d.frames),
            ("thermal images", d.thermal_images),
            ("annotated frames", d.annotated_frames),
            ("annotated boxes", d.annotated_boxes),
            ("classes", None if d.classes is None else ", ".join(d.classes)),
            ("label format", d.label_format),
            ("sha256", d.sha256 or "not downloaded"),
        ):
            if value is not None and value != "None / None":
                block += _row(label, value)
        lines.append(block.rstrip("\n"))
        lines += [
            "",
            "**Signal path.** " + d.signal_path.strip().replace("\n", " "),
            "",
            "**May be used for:** " + ", ".join(f"`{a}`" for a in d.analysers) + ".",
        ]
        if d.excluded_analysers:
            lines += [
                "",
                "**Must not be used for:** "
                + ", ".join(f"`{a}`" for a in d.excluded_analysers)
                + " -- see the signal path above.",
            ]
        if d.notes:
            lines += ["", d.notes.strip().replace("\n", " ")]
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def readme_path(root: str | os.PathLike[str] | None = None) -> pathlib.Path:
    return manifest_path(root).with_name("README.md")
