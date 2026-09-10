"""LUT bundles on disk (M1.8, ADR 0012): bitwise round trip, float32 enforced on disk, stale
detection that names the changed input, and the raw .f32 export for non-NumPy consumers."""

from __future__ import annotations

import copy
import json
import pathlib
import shutil

import numpy as np
import pytest
import yaml

from irsim.config.loader import band_hash, config_hash
from irsim.config.sensor import SensorConfig
from irsim.radiometry.lut import QUANTITIES, BandLUT
from irsim.radiometry.lut_files import (
    StaleLUTError,
    build_band_lut_for_config,
    load_band_lut,
    load_band_lut_for_config,
    lut_paths,
    save_band_lut,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
RESPONSE_REL = pathlib.Path("spectra/responses/boson_vox.csv")


@pytest.fixture
def data_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """A private copy of the data root so tests can mutate the response file."""
    root = tmp_path / "data"
    (root / RESPONSE_REL).parent.mkdir(parents=True)
    shutil.copy(REPO / "data" / RESPONSE_REL, root / RESPONSE_REL)
    return root


@pytest.fixture
def boson_cfg() -> SensorConfig:
    return SensorConfig.model_validate(yaml.safe_load(BOSON_YAML.read_text()))


def test_save_load_is_bitwise(boson_lut: BandLUT, tmp_path: pathlib.Path) -> None:
    paths = save_band_lut(
        boson_lut,
        tmp_path,
        band_hash="a" * 64,
        config_sha256="b" * 64,
        spectral_sha256="c" * 64,
        sensor_name="x",
        band={"id": "lwir"},
    )
    back = load_band_lut(tmp_path, "a" * 64, spectral_sha256="c" * 64)
    for q in QUANTITIES:
        assert np.array_equal(back.table(q), boson_lut.table(q))
        assert back.table(q).dtype == np.float32
        raw = np.frombuffer(paths.f32[q].read_bytes(), dtype="<f4")
        assert np.array_equal(raw, boson_lut.table(q)), f"{q}: .f32 differs from .npy"
    assert (back.t0_k, back.t1_k, back.n) == (boson_lut.t0_k, boson_lut.t1_k, boson_lut.n)


def test_sidecar_grid_lands_integer_kelvin_on_integer_indices(
    boson_lut: BandLUT, tmp_path: pathlib.Path
) -> None:
    paths = save_band_lut(
        boson_lut,
        tmp_path,
        band_hash="a" * 64,
        config_sha256="b",
        spectral_sha256="c",
        sensor_name="x",
        band={},
    )
    meta = json.loads(paths.sidecar.read_text())
    assert (meta["t0_k"], meta["t1_k"], meta["n"], meta["dtype"]) == (
        200.0,
        1000.0,
        16001,
        "float32",
    )
    for t in (200.0, 300.0, 373.0, 1000.0):
        i = (t - meta["t0_k"]) / meta["dt_k"]
        assert abs(i - round(i)) < 1e-9
    assert meta["schema_version"] == 1 and meta["quantities"] == list(QUANTITIES)


@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_hand_written_non_float32_table_rejected(
    boson_lut: BandLUT, tmp_path: pathlib.Path, dtype: type
) -> None:
    save_band_lut(
        boson_lut,
        tmp_path,
        band_hash="a" * 64,
        config_sha256="b",
        spectral_sha256="c",
        sensor_name="x",
        band={},
    )
    paths = lut_paths("a" * 64, tmp_path)
    np.save(paths.npy["lb"], boson_lut.lb.astype(dtype), allow_pickle=False)
    with pytest.raises(TypeError, match="float32"):
        load_band_lut(tmp_path, "a" * 64)
    # the constructor itself refuses non-float32, so a corrupted in-memory table cannot be saved
    with pytest.raises(TypeError):
        BandLUT(200.0, 1000.0, 3, *(np.zeros(3, dtype),) * 4)


def test_build_and_load_for_config_and_netd_grades_share_bundle(
    boson_cfg: SensorConfig, data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "lut"
    lut, paths = build_band_lut_for_config(boson_cfg, out, data_dir)
    assert paths.sidecar.is_file()
    meta = json.loads(paths.sidecar.read_text())
    assert meta["band_hash"] == band_hash(boson_cfg, data_dir) == lut.band_hash
    assert meta["config_sha256"] == config_hash(boson_cfg, data_dir)
    assert meta["band"]["id"] == "lwir" and meta["sensor_name"] == "flir_boson_640_lwir"
    again = load_band_lut_for_config(boson_cfg, out, data_dir)
    assert np.array_equal(again.lb, lut.lb)
    # an industrial-grade (40 mK) variant of the same detector reuses the bundle
    d = yaml.safe_load(BOSON_YAML.read_text())
    d["sensor"]["noise"]["netd_mk_at_300k"] = 40.0
    industrial = SensorConfig.model_validate(d)
    assert np.array_equal(load_band_lut_for_config(industrial, out, data_dir).lb, lut.lb)


def test_one_byte_csv_change_is_stale_and_named(
    boson_cfg: SensorConfig, data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "lut"
    build_band_lut_for_config(boson_cfg, out, data_dir)
    csv = data_dir / RESPONSE_REL
    csv.write_text(csv.read_text().replace("10.00,1.000000", "10.00,0.999999"))
    with pytest.raises(StaleLUTError, match="spectral response file .* changed"):
        load_band_lut_for_config(boson_cfg, out, data_dir)


def test_band_block_change_is_stale_and_named(
    boson_cfg: SensorConfig, data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "lut"
    build_band_lut_for_config(boson_cfg, out, data_dir)
    d = yaml.safe_load(BOSON_YAML.read_text())
    d["sensor"]["band"]["lambda_max_um"] = 13.6
    with pytest.raises(StaleLUTError, match="band block"):
        load_band_lut_for_config(SensorConfig.model_validate(d), out, data_dir)


def test_missing_bundle_says_make_luts(
    boson_cfg: SensorConfig, data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(FileNotFoundError, match="make luts"):
        load_band_lut_for_config(boson_cfg, tmp_path / "empty", data_dir)


def test_sidecar_schema_version_mismatch_is_stale(
    boson_lut: BandLUT, tmp_path: pathlib.Path
) -> None:
    paths = save_band_lut(
        boson_lut,
        tmp_path,
        band_hash="a" * 64,
        config_sha256="b",
        spectral_sha256="c",
        sensor_name="x",
        band={},
    )
    meta = json.loads(paths.sidecar.read_text())
    meta["schema_version"] = 99
    paths.sidecar.write_text(json.dumps(meta))
    with pytest.raises(StaleLUTError, match="schema_version"):
        load_band_lut(tmp_path, "a" * 64)


def test_regeneration_is_bitwise_deterministic(
    boson_cfg: SensorConfig, data_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    a, pa = build_band_lut_for_config(boson_cfg, tmp_path / "a", data_dir)
    b, pb = build_band_lut_for_config(boson_cfg, tmp_path / "b", data_dir)
    for q in QUANTITIES:
        assert pa.npy[q].read_bytes() == pb.npy[q].read_bytes()
        assert pa.f32[q].read_bytes() == pb.f32[q].read_bytes()
    assert copy.deepcopy(json.loads(pa.sidecar.read_text())) == json.loads(pb.sidecar.read_text())
