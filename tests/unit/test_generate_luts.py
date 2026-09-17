"""scripts/generate_luts.py (M1.9): a synthetic top-hat band reproduces the closed form, an
invalid response file exits non-zero with nothing written, and the CLI entry point works."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml

from irsim.radiometry.lut_files import load_band_lut_for_config
from irsim.radiometry.planck import band_radiance_tophat

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "generate_luts.py"
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


def _load_script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("generate_luts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Minimal LWIR sensor pointing at an exact 7.5-13.5 um top-hat response."""
    data = tmp_path / "data"
    (data / "spectra" / "responses").mkdir(parents=True)
    (data / "spectra" / "responses" / "tophat.csv").write_text(
        "# exact top-hat\n7.5,1.0\n13.5,1.0\n"
    )
    cfg = yaml.safe_load(BOSON_YAML.read_text())
    cfg["sensor"]["name"] = "synthetic_tophat_lwir"
    cfg["sensor"]["band"]["spectral_response"] = "spectra/responses/tophat.csv"
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "synthetic.yaml").write_text(yaml.safe_dump(cfg))
    return {
        "data": data,
        "configs": configs,
        "out": tmp_path / "lut",
        "yaml": configs / "synthetic.yaml",
    }


def test_tophat_sensor_reproduces_closed_form(synthetic: dict[str, pathlib.Path], capsys) -> None:  # type: ignore[no-untyped-def]
    mod = _load_script()
    rc = mod.main(
        [
            "--configs",
            str(synthetic["configs"]),
            "--out",
            str(synthetic["out"]),
            "--data",
            str(synthetic["data"]),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "1/1 bundle(s) written" in out and "Lb(300 K)" in out and "mK equivalent" in out
    from irsim.config.loader import load_sensor_config

    cfg = load_sensor_config(synthetic["yaml"], synthetic["data"])
    lut = load_band_lut_for_config(cfg, synthetic["out"], synthetic["data"])
    lb300 = float(lut.lookup(300.0)[()])
    assert abs(lb300 / band_radiance_tophat(7.5, 13.5, 300.0) - 1.0) < 1e-3
    assert lb300 == pytest.approx(55.49, rel=1e-3), "the roadmap's top-hat anchor"
    sidecars = list(synthetic["out"].glob("*_lut.json"))
    assert len(sidecars) == 1
    meta = json.loads(sidecars[0].read_text())
    assert meta["sensor_name"] == "synthetic_tophat_lwir" and meta["dtype"] == "float32"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("7.5,0.8\n13.5,0.8\n", "peak != 1"),
        ("7500,1.0\n13500,1.0\n", "nanometres"),
    ],
)
def test_invalid_response_exits_nonzero_and_writes_nothing(
    synthetic: dict[str, pathlib.Path], body: str, reason: str, capsys
) -> None:  # type: ignore[no-untyped-def]
    (synthetic["data"] / "spectra" / "responses" / "tophat.csv").write_text(body)
    mod = _load_script()
    rc = mod.main(
        [
            "--configs",
            str(synthetic["yaml"]),
            "--out",
            str(synthetic["out"]),
            "--data",
            str(synthetic["data"]),
        ]
    )
    assert rc == 1, reason
    err = capsys.readouterr().err
    assert "error:" in err and "synthetic.yaml" in err
    assert not synthetic["out"].exists() or not list(synthetic["out"].iterdir()), "nothing written"


def test_rerun_is_bitwise_deterministic(synthetic: dict[str, pathlib.Path]) -> None:
    mod = _load_script()
    args = [
        "--configs",
        str(synthetic["configs"]),
        "--out",
        str(synthetic["out"]),
        "--data",
        str(synthetic["data"]),
    ]
    assert mod.main(args) == 0
    first = {p.name: p.read_bytes() for p in synthetic["out"].iterdir()}
    assert mod.main(args) == 0
    second = {p.name: p.read_bytes() for p in synthetic["out"].iterdir()}
    assert first == second


def test_no_configs_found_exits_2(tmp_path: pathlib.Path, capsys) -> None:  # type: ignore[no-untyped-def]
    mod = _load_script()
    empty = tmp_path / "none"
    empty.mkdir()
    assert mod.main(["--configs", str(empty), "--out", str(tmp_path / "o")]) == 2


def test_cli_entry_point(synthetic: dict[str, pathlib.Path]) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--configs",
            str(synthetic["configs"]),
            "--out",
            str(synthetic["out"]),
            "--data",
            str(synthetic["data"]),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "bundle(s) written" in proc.stdout
    assert np.load(next(synthetic["out"].glob("*_lb.npy"))).dtype == np.float32
