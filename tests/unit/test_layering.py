"""Enforce non-negotiable #1: the physics core imports no engine modules -- in either direction.

If this fails, the physics can no longer be tested without launching a simulator, and the Unreal
port stops being a glue-only rewrite. Fix the import, not the test.

Three guards:
  * nothing under src/irsim imports an engine module, the glue package, or the ML/imaging stack
    (that lives in src/irsim_eval, so the core stays NumPy-only);
  * nothing under tests/unit or tests/golden imports an engine module, so the default gate stays
    runnable without a GPU;
  * every file under src/irsim_isaac keeps its engine imports inside functions, so
    `import irsim_isaac` succeeds without Isaac Sim.
The scanner itself is self-tested on synthetic offending modules.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CORE = REPO / "src" / "irsim"
GLUE = REPO / "src" / "irsim_isaac"
GATE_TESTS = [
    *sorted((REPO / "tests" / "unit").rglob("*.py")),
    *sorted((REPO / "tests" / "golden").rglob("*.py")),
    REPO / "tests" / "conftest.py",
]

ENGINE = frozenset({"omni", "pxr", "isaacsim", "warp", "carb"})
# The evaluation stack (image decoding, detectors) belongs in src/irsim_eval, never in the core.
ML_AND_IMAGING = frozenset({"torch", "torchvision", "cv2", "PIL", "imageio", "sklearn"})
FORBIDDEN_IN_CORE = ENGINE | ML_AND_IMAGING | {"irsim_isaac"}


def _root_of(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {a.name.split(".")[0] for a in node.names}
    if node.module and node.level == 0:
        return {node.module.split(".")[0]}
    return set()


def imported_roots(path: pathlib.Path) -> set[str]:
    """Top-level package names imported anywhere in the file (module level or nested)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots |= _root_of(node)
    return roots


def module_level_imported_roots(path: pathlib.Path) -> set[str]:
    """Package names imported unconditionally at module level (executed by `import pkg`).

    Imports nested in a function, class, `try`, `if` or `with` block are not counted: those are
    the sanctioned places for an engine import in the glue package.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots |= _root_of(node)
    return roots


@pytest.mark.parametrize("path", sorted(CORE.rglob("*.py")), ids=lambda p: p.name)
def test_core_has_no_engine_or_glue_imports(path: pathlib.Path) -> None:
    leaked = imported_roots(path) & FORBIDDEN_IN_CORE
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)}. Engine code belongs in "
        "src/irsim_isaac/, evaluation/ML code in src/irsim_eval/ -- CLAUDE.md non-negotiable #1."
    )


@pytest.mark.parametrize("path", GATE_TESTS, ids=lambda p: p.name)
def test_default_gate_tests_are_engine_free(path: pathlib.Path) -> None:
    leaked = imported_roots(path) & ENGINE
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)}; tests needing an engine go in "
        "tests/integration with @pytest.mark.isaac or @pytest.mark.gpu."
    )


@pytest.mark.parametrize("path", sorted(GLUE.rglob("*.py")), ids=lambda p: p.name)
def test_glue_keeps_engine_imports_inside_functions(path: pathlib.Path) -> None:
    leaked = module_level_imported_roots(path) & ENGINE
    assert not leaked, (
        f"{path.relative_to(REPO)} imports {sorted(leaked)} at module level, so "
        "`import irsim_isaac` would fail without Isaac Sim. Import inside the function, "
        "after an irsim_isaac.env probe."
    )


def test_glue_is_importable_without_engine() -> None:
    import irsim_isaac
    import irsim_isaac.env
    import irsim_isaac.pipeline

    assert isinstance(irsim_isaac.env.has_isaac(), bool)
    assert isinstance(irsim_isaac.env.has_warp(), bool)


def test_env_force_flags_and_require(monkeypatch: pytest.MonkeyPatch) -> None:
    from irsim_isaac import env

    monkeypatch.setenv("IRSIM_FORCE_NO_ISAAC", "1")
    monkeypatch.setenv("IRSIM_FORCE_NO_WARP", "1")
    assert env.has_isaac() is False and env.has_warp() is False
    with pytest.raises(RuntimeError, match="Isaac Sim"):
        env.require_isaac()
    with pytest.raises(RuntimeError, match="Warp"):
        env.require_warp()


# --- scanner self-tests: would the guard actually catch a leak? -----------------------------

OFFENDERS = {
    "plain.py": ("import omni\n", {"omni"}),
    "from_sub.py": ("from omni.replicator import core\n", {"omni"}),
    "aliased.py": ("import warp as wp\n", {"warp"}),
    "torch.py": ("import numpy as np\nimport torch\n", {"torch"}),
    "glue.py": ("from irsim_isaac.pipeline import aovs\n", {"irsim_isaac"}),
    "nested.py": ("def f():\n    import carb\n    return carb\n", {"carb"}),
    "multi.py": ("import os, pxr\n", {"pxr"}),
}


@pytest.mark.parametrize(("name", "source", "expected"), [(k, *v) for k, v in OFFENDERS.items()])
def test_scanner_detects_known_leaks(
    tmp_path: pathlib.Path, name: str, source: str, expected: set[str]
) -> None:
    path = tmp_path / name
    path.write_text(source)
    assert imported_roots(path) & FORBIDDEN_IN_CORE == expected


def test_scanner_passes_a_clean_module(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "clean.py"
    path.write_text(
        "import numpy as np\nfrom .sibling import thing\nfrom irsim.radiometry import x\n"
    )
    assert not imported_roots(path) & FORBIDDEN_IN_CORE


def test_module_level_scanner_ignores_function_scope(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "glue_ok.py"
    path.write_text(
        "import numpy as np\n\ndef run():\n    import omni.replicator.core as rep\n    return rep\n"
    )
    assert not module_level_imported_roots(path) & ENGINE
    assert imported_roots(path) & ENGINE == {"omni"}
    bad = tmp_path / "glue_bad.py"
    bad.write_text("import warp as wp\n")
    assert module_level_imported_roots(bad) & ENGINE == {"warp"}


def test_integration_directory_only_skips_without_isaac() -> None:
    """With Isaac forced absent, tests/integration yields only skips: no errors, no failures."""
    env = {**os.environ, "IRSIM_FORCE_NO_ISAAC": "1", "IRSIM_FORCE_NO_WARP": "1"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(REPO / "tests" / "integration"),
            "-q",
            "-m",
            "",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env,
        check=False,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "skipped" in out and "passed" not in out and "error" not in out.lower(), out
