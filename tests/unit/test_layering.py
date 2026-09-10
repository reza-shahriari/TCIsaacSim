"""Enforce non-negotiable #1: the physics core imports no engine modules.

If this fails, the physics can no longer be tested without launching a simulator,
and the Unreal port stops being a glue-only rewrite. Fix the import, not the test.
"""

import ast
import pathlib

import pytest

FORBIDDEN = {"omni", "pxr", "isaacsim", "warp", "carb"}
CORE = pathlib.Path(__file__).resolve().parents[2] / "src" / "irsim"


def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted(CORE.rglob("*.py")), ids=lambda p: p.name)
def test_core_has_no_engine_imports(path: pathlib.Path) -> None:
    leaked = _imported_roots(path) & FORBIDDEN
    assert not leaked, (
        f"{path.relative_to(CORE.parent)} imports {sorted(leaked)}. "
        "Engine code belongs in src/irsim_isaac/ -- see CLAUDE.md non-negotiable #1."
    )
