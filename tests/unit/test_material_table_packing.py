"""Packed per-band MaterialTable (M7.18): closure survives float32 packing, the float16 guard is
live, columns are float32, id 0 is UNMAPPED, ids are stable across loads, a stale table is
refused, the kernel path still works, and the G-buffer fixtures never use id 0."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.bands import BAND_IDS
from irsim.materials import (
    UNMAPPED_MATERIAL_ID,
    UNMAPPED_NAME,
    MaterialLibrary,
    MaterialTable,
    StaleMaterialTableError,
)
from irsim.materials.table import ANGULAR_A_PLACEHOLDER, ANGULAR_P_PLACEHOLDER, THERMAL_COLUMNS


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.mark.parametrize("band", BAND_IDS)
def test_closure_survives_float32_packing(library: MaterialLibrary, band: str) -> None:
    t = MaterialTable.from_library(library, band)
    assert t.reflectance is not None and t.transmittance is not None
    for col in (
        t.emissivity,
        t.reflectance,
        t.transmittance,
        t.angular_a,
        t.angular_p,
        t.roughness,
    ):
        assert col is not None and col.dtype == np.float32
    for key in THERMAL_COLUMNS:
        assert t.thermal[key].dtype == np.float32
    closure = t.emissivity[1:].astype(np.float64) + t.reflectance[1:] + t.transmittance[1:]
    assert np.max(np.abs(closure - 1.0)) < 1e-6
    assert t.is_unmapped[UNMAPPED_MATERIAL_ID] and not t.is_unmapped[1:].any()
    assert np.isnan(t.emissivity[0]) and t.names[0] == UNMAPPED_NAME
    assert t.name_for(t.id_for("glass_windshield")) == "glass_windshield"


def test_float16_pack_is_refused() -> None:
    with pytest.raises(TypeError, match="float16"):
        MaterialTable(emissivity=np.array([np.nan, 0.9], dtype=np.float16))
    with pytest.raises(TypeError, match="float32"):
        MaterialTable(emissivity=np.array([np.nan, 0.9], dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        MaterialTable(
            emissivity=np.array([np.nan, 0.9], dtype=np.float32),
            reflectance=np.array([0.1], dtype=np.float32),
        )


def test_ids_stable_and_kernel_path(library: MaterialLibrary) -> None:
    a = MaterialTable.from_library(library, "lwir")
    b = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    assert a.names == b.names and np.array_equal(a.emissivity[1:], b.emissivity[1:])
    assert a.names[1:] == tuple(sorted(library.names))
    ids = np.array([[a.id_for("asphalt_dry"), a.id_for("bare_aluminium")]], dtype=np.int32)
    eps = a.emissivity_for(ids)
    assert (
        eps.dtype == np.float32
        and eps[0, 0] == pytest.approx(0.94)
        and eps[0, 1] == pytest.approx(0.09)
    )
    with pytest.raises(ValueError, match="UNMAPPED"):
        a.emissivity_for(np.array([[0, 1]], dtype=np.int32))
    with pytest.raises(KeyError):
        a.id_for("unobtainium")
    black = a.id_for("car_paint_black")
    assert a.angular_a is not None and a.angular_p is not None
    # (a, p) fitted against Level A on the M7.5 paint proxy, not the estimate that preceded it.
    assert (a.angular_a[black], a.angular_p[black]) == (np.float32(0.75), np.float32(4.0))
    alu = a.id_for("bare_aluminium")
    assert (a.angular_a[alu], a.angular_p[alu]) == (ANGULAR_A_PLACEHOLDER, ANGULAR_P_PLACEHOLDER)
    assert a.thermal["heat_capacity_j_m2_k"][black] == pytest.approx(7800 * 470 * 0.0012, rel=1e-6)


def test_save_load_round_trip_and_stale_rejection(
    library: MaterialLibrary, tmp_path: pathlib.Path
) -> None:
    t = MaterialTable.from_library(library, "mwir")
    npz, side = t.save(tmp_path / "materials_mwir")
    assert npz.exists() and side.exists()
    back = MaterialTable.load(tmp_path / "materials_mwir", library)
    assert back.names == t.names and back.band_id == "mwir" and back.library_hash == t.library_hash
    assert back.reflectance is not None and t.reflectance is not None
    np.testing.assert_array_equal(back.emissivity[1:], t.emissivity[1:])
    np.testing.assert_array_equal(back.reflectance[1:], t.reflectance[1:])
    for key in THERMAL_COLUMNS:
        np.testing.assert_array_equal(back.thermal[key][1:], t.thermal[key][1:])
    side.write_text(side.read_text().replace(t.library_hash, "0" * 64))
    with pytest.raises(StaleMaterialTableError, match="regenerate"):
        MaterialTable.load(tmp_path / "materials_mwir", library)
    assert MaterialTable.load(tmp_path / "materials_mwir").library_hash == "0" * 64


@pytest.mark.parametrize(
    "fixture",
    [
        "gbuffer_ramp",
        "gbuffer_uniform",
        "gbuffer_two_material",
        "gbuffer_sphere",
        "gbuffer_step_edge",
    ],
)
def test_fixture_ids_never_use_the_sentinel(fixture: str, request: pytest.FixtureRequest) -> None:
    planes = request.getfixturevalue(fixture)
    assert int(planes["material_id"].min()) >= 1, (
        "spec issue T7: fixtures must not collide with UNMAPPED"
    )
