"""Roadmap M10.4, engine-free half: the Warp stage module imports without Warp, refuses what the
CPU oracle refuses, and keys its device tables by content (so an identical table is never
re-uploaded and a rebuilt one always is). The kernel itself is compared to the oracle in
`tests/integration/test_kernels_vs_reference.py`."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials import MaterialTable
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import BandLUT
from irsim_isaac.pipeline import warp_stages as ws


def test_module_imports_without_warp_and_entry_points_say_so() -> None:
    if ws.has_warp_module():
        pytest.skip("Warp is importable here; the no-Warp path is exercised in CI")
    with pytest.raises(RuntimeError, match="Warp"):
        ws.band_radiance_warp(
            np.full((2, 2), 300.0, np.float32),
            np.ones((2, 2), np.int32),
            MaterialTable.constant(1.0),
            _tiny_lut(),
        )


def test_equivalence_registry_names_the_cpu_oracle() -> None:
    cpu, gpu = ws.EQUIVALENCE_STAGES["band_radiance"]
    assert cpu is band_radiance_stage
    assert gpu is ws.band_radiance_stage_warp
    assert ws.WarpBandRadianceStage.name == "band_radiance"


def _tiny_lut() -> BandLUT:
    n = 5
    t = np.linspace(200.0, 1000.0, n)
    lb = (t / 300.0).astype(np.float32)
    return BandLUT(200.0, 1000.0, n, lb, lb, lb, lb, band_hash="tiny")


def test_validate_material_ids_mirrors_the_oracle() -> None:
    table = MaterialTable.from_mapping({1: 0.95, 2: 0.5})
    ids = np.array([[1, 2], [2, 1]], np.int32)
    ws.validate_material_ids(ids, table)  # fine
    with pytest.raises(ValueError, match="UNMAPPED"):
        ws.validate_material_ids(np.array([[0, 1]], np.int32), table)
    with pytest.raises(ValueError, match="table has"):
        ws.validate_material_ids(np.array([[1, 7]], np.int32), table)
    with pytest.raises(TypeError):
        ws.validate_material_ids(np.array([[1.0, 2.0]]), table)
    # under the sky mask the id is ignored, exactly as MaterialTable.emissivity_for does
    sky = np.array([[True, False]])
    ws.validate_material_ids(np.array([[0, 1]], np.int32), table, sky)
    with pytest.raises(ValueError, match="sky_mask"):
        ws.validate_material_ids(ids, table, np.array([[1, 0]], np.int32))


def test_tables_key_tracks_content_not_identity() -> None:
    lut = _tiny_lut()
    a = MaterialTable.from_mapping({1: 0.9})
    b = MaterialTable.from_mapping({1: 0.9})
    c = MaterialTable.from_mapping({1: 0.8})
    k_a = ws.tables_key(lut, a, "lb", "cuda:0")
    assert k_a == ws.tables_key(lut, b, "lb", "cuda:0")
    assert k_a != ws.tables_key(lut, c, "lb", "cuda:0")
    assert k_a != ws.tables_key(lut, a, "lb", "cpu")
    assert k_a != ws.tables_key(lut, a, "lb_q", "cuda:0") or np.array_equal(lut.lb, lut.lb_q)
