"""Roadmap M2.1 (AST part): the Isaac glue never imports the deprecated sensor modules.

`isaacsim.sensors.camera` and `isaacsim.sensors.rtx` are deprecated in Isaac Sim 6.x in favour of
`isaacsim.sensors.experimental.rtx` (ADR 0014). Kit still ships and enables them, so a `sys.modules`
check inside a running Kit says nothing about our code; this walks `src/irsim_isaac` with `ast`
instead and needs no GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

GLUE = Path(__file__).resolve().parents[2] / "src" / "irsim_isaac"
DEPRECATED = ("isaacsim.sensors.camera", "isaacsim.sensors.rtx")


def _imported_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _is_deprecated(name: str) -> bool:
    return any(name == d or name.startswith(d + ".") for d in DEPRECATED)


def test_glue_package_exists() -> None:
    assert (GLUE / "__init__.py").exists(), GLUE


def test_no_deprecated_sensor_imports_in_glue() -> None:
    offenders = {
        str(py.relative_to(GLUE.parent)): [n for n in _imported_names(py) if _is_deprecated(n)]
        for py in sorted(GLUE.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert offenders == {}, offenders


def test_scanner_catches_a_deprecated_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("from isaacsim.sensors.camera import Camera\nimport isaacsim.sensors.rtx.foo\n")
    assert [n for n in _imported_names(bad) if _is_deprecated(n)] == [
        "isaacsim.sensors.camera",
        "isaacsim.sensors.rtx.foo",
    ]
    ok = tmp_path / "ok.py"
    ok.write_text("from isaacsim.sensors.experimental.rtx import RtxCamera\n")
    assert [n for n in _imported_names(ok) if _is_deprecated(n)] == []
