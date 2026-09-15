"""Obtain, verify and hash the indexed validation sets.

roadmap ME.1; ADR 0003.

Most of this module is refusal, which is the point. Of the six indexed sets exactly one states a
licence, one is not released yet, and the rest are behind drive links or an access request that
cites export control. A fetcher that cheerfully pulled all of them would be doing something nobody
decided to do, so the decision is surfaced instead:

* a set whose ``licence`` is ``unstated`` is **skipped** unless ``accept_unstated_licence`` is
  passed -- not because using it is necessarily wrong, but because "the terms are unknown" is a
  fact a person should see before the bytes land;
* a set whose ``access`` is ``manual`` is never downloaded. A Google Drive interstitial or a
  university access form is a human's job, and a script that pretended otherwise would fail in a
  way that looks like a network error;
* a set that is ``unreleased`` is reported as such.

What the script *can* always do, and what is worth more than the download, is **verify**: hash
whatever is on disk and compare it with the index, so that "the numbers were measured on these
exact bytes" is checkable a year later (ADR 0004's rule, applied to somebody else's data).

:func:`plan` decides all of this and touches nothing, so the policy is testable without a network
and without a single byte of anyone's dataset.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import Literal

from irsim_eval.manifest import Dataset, Manifest

__all__ = [
    "Decision",
    "Action",
    "CHUNK_BYTES",
    "target_dir",
    "sha256_of",
    "plan",
    "download",
    "verify",
]

#: Read size for hashing and downloading. Datasets here run to tens of gigabytes; nothing is ever
#: held in memory whole.
CHUNK_BYTES = 1 << 20

#: What will happen to a set, and why. ``refused`` and ``manual`` are outcomes, not errors.
Decision = Literal["download", "verify", "manual", "unreleased", "refused"]


@dataclass(frozen=True)
class Action:
    """One set's plan: what to do, where, and the reason a human should read."""

    name: str
    decision: Decision
    reason: str
    destination: pathlib.Path
    url: str | None = None

    @property
    def needs_a_person(self) -> bool:
        return self.decision in ("manual", "unreleased", "refused")


def target_dir(name: str, root: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """``data/validation/<set>/`` -- gitignored; the datasets never enter the repository."""
    from irsim_eval.manifest import manifest_path

    return manifest_path(root).parent / name


def sha256_of(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def plan(
    manifest: Manifest,
    names: list[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    accept_unstated_licence: bool = False,
) -> list[Action]:
    """Decide what to do for each named set (default: all), without doing any of it.

    The licence gate is checked **before** the access mode, deliberately: a set that is both
    unlicensed and manual should report the licence, because that is the fact that decides whether
    to go and get it at all.
    """
    wanted = list(manifest.datasets) if names is None else names
    unknown = [n for n in wanted if n not in manifest.datasets]
    if unknown:
        raise KeyError(f"not in the index: {unknown}; known sets are {sorted(manifest.datasets)}")

    actions: list[Action] = []
    for name in wanted:
        dataset = manifest.datasets[name]
        destination = target_dir(name, root)
        if not dataset.licence_known and not accept_unstated_licence:
            actions.append(
                Action(
                    name,
                    "refused",
                    "the publisher states no licence, so the terms of use and redistribution are "
                    "unknown. Read them at the source, then pass accept_unstated_licence to "
                    "proceed deliberately.",
                    destination,
                )
            )
            continue
        if dataset.access == "unreleased":
            actions.append(
                Action(name, "unreleased", "the data has not been published yet", destination)
            )
            continue
        if dataset.fetchable:
            assert dataset.download_url is not None  # `fetchable` is exactly this check
            size = (
                ""
                if dataset.download_bytes is None
                else f" ({dataset.download_bytes / 1e9:.2f} GB)"
            )
            actions.append(
                Action(name, "download", f"direct URL{size}", destination, dataset.download_url)
            )
            continue
        if destination.is_dir() and any(destination.iterdir()):
            actions.append(
                Action(name, "verify", "already on disk: hash it and compare", destination)
            )
            continue
        actions.append(
            Action(
                name,
                "manual",
                _manual_reason(dataset),
                destination,
                dataset.site_url or dataset.code_url or dataset.doi,
            )
        )
    return actions


def _manual_reason(dataset: Dataset) -> str:
    where = dataset.site_url or dataset.code_url or (f"doi:{dataset.doi}" if dataset.doi else None)
    return (
        "no direct URL: fetch it by hand"
        + (f" from {where}" if where else "")
        + f", into the destination below, then re-run to hash it ({dataset.access})"
    )


def download(url: str, destination: str | os.PathLike[str]) -> pathlib.Path:
    """Stream a URL to a file. Only ``https`` and ``file`` are accepted.

    ``http`` is refused rather than upgraded: these archives are tens of gigabytes and a silent
    downgrade to an unauthenticated transport is not something to do on someone's behalf.
    """
    if not url.startswith(("https://", "file://")):
        raise ValueError(f"refusing to fetch over a non-https scheme: {url!r}")
    out = pathlib.Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(out, "wb") as fh:  # noqa: S310 - scheme checked
        while chunk := response.read(CHUNK_BYTES):
            fh.write(chunk)
    return out


def verify(dataset: Dataset, path: str | os.PathLike[str]) -> tuple[bool, str]:
    """``(matches, digest)`` for a file against the index.

    A set with no recorded hash returns ``True`` with the freshly computed digest: there is nothing
    to contradict yet. Recording it is a separate, deliberate step -- writing a hash automatically
    from whatever happened to be on disk would make the field describe this machine rather than
    the dataset.
    """
    digest = sha256_of(path)
    if dataset.sha256 is None:
        return True, digest
    return dataset.sha256 == digest, digest
