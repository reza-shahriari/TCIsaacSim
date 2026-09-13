"""Evaluation stack: everything that reads somebody else's imagery.

Separate from ``irsim`` on purpose (CLAUDE.md non-negotiable #1, enforced by
``tests/unit/test_layering.py``). The physics core is NumPy-only and must stay importable and
testable with no image decoder, no detector and no GPU; image decoding, the real-vs-synthetic
discriminator and the detector trainers all live here, behind the ``validation`` and ``ml``
extras.

The split is not tidiness. The core's iteration loop has to run in seconds on any machine, and the
moment verifying Planck's law needs OpenCV installed, nobody verifies Planck's law.

roadmap ME.1; ADR 0003 (validation against public data, no camera).
"""

from irsim_eval.manifest import (
    Dataset,
    Manifest,
    load_manifest,
    manifest_path,
    readme_path,
    render_readme,
)

__all__ = [
    "Dataset",
    "Manifest",
    "load_manifest",
    "manifest_path",
    "readme_path",
    "render_readme",
]
