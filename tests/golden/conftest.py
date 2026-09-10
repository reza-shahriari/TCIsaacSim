"""Golden / regression fixture helper.

For outputs with no analytic answer (a full pipeline render, an AGC output) we store a reference
array and compare against it. The rules that keep golden tests from becoming a liability are the
ones in the ir-sim-testing skill and ADR 0004:

* the **config hash** that produced the array is stored beside it, so a golden whose inputs
  changed is reported as STALE, not as a physics failure;
* comparison always uses a stated tolerance in the array's own units, never exact equality;
* on failure the actual array is written next to the expected one and both paths are printed;
* arrays on disk are float32, uint16 or uint8 -- never float16 (CLAUDE.md non-negotiable #2
  applies to files) and never float64 (a golden should be what the pipeline actually emits);
* regeneration is deliberate: ``make golden-update`` passes ``--update-golden``.

Usage in a test::

    def test_band_radiance_stage(golden, gbuffer_ramp):
        out = run_stage(gbuffer_ramp, cfg)
        golden.check("band_radiance_ramp", out, config_hash=cfg_hash, atol=1e-3, units="W/m2/sr")

The reference files live in ``tests/golden/data/<name>.npy`` + ``<name>.json``.
"""

from __future__ import annotations

import pytest
from golden_store import ACTUAL_DIR, GOLDEN_DIR, GoldenStore


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenStore:
    """The repo golden store, in compare mode unless ``--update-golden`` was passed."""
    return GoldenStore(
        root=GOLDEN_DIR,
        actual_dir=ACTUAL_DIR,
        update=bool(request.config.getoption("--update-golden")),
    )
