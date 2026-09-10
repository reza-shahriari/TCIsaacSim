"""Non-negotiable #5: π/(4F² + 1) is written once, in irsim/optics/aperture.py.

An AST walk (not a string grep) over every module in src/irsim and src/irsim_isaac flags any
expression that squares an f-number-like name (``x ** 2``, ``pow(x, 2)``, ``x * x``) or multiplies
such a term by a literal 4. The scanner is self-tested on three obfuscated variants and one clean
module so that it cannot silently stop catching things.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCAN_ROOTS = (REPO / "src" / "irsim", REPO / "src" / "irsim_isaac")
ALLOWED = REPO / "src" / "irsim" / "optics" / "aperture.py"
F_NAMES = frozenset({"f_number", "fnum", "fnumber", "f_num", "F", "f"})


def _is_f_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in F_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in F_NAMES
    return False


def _mentions_f(node: ast.AST) -> bool:
    return any(_is_f_name(n) for n in ast.walk(node))


def _is_two(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (2, 2.0)


def _is_four(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (4, 4.0)


def aperture_expressions(source: str) -> list[str]:
    """Return the offending expressions (unparsed) found in ``source``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and _is_two(node.right) and _mentions_f(node.left):
                found.append(ast.unparse(node))
            elif isinstance(node.op, ast.Mult):
                same = ast.dump(node.left) == ast.dump(node.right) and _mentions_f(node.left)
                by_four = (_is_four(node.left) and _mentions_f(node.right)) or (
                    _is_four(node.right) and _mentions_f(node.left)
                )
                if same or by_four:
                    found.append(ast.unparse(node))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pow"
            and len(node.args) == 2
            and _is_two(node.args[1])
            and _mentions_f(node.args[0])
        ):
            found.append(ast.unparse(node))
    return found


def _modules() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in SCAN_ROOTS:
        out += sorted(p for p in root.rglob("*.py") if p.resolve() != ALLOWED)
    return out


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_hand_written_aperture_factor(path: pathlib.Path) -> None:
    found = aperture_expressions(path.read_text())
    assert not found, (
        f"{path.relative_to(REPO)} squares or 4x-multiplies an f-number: {found}. "
        "Import irsim.optics.aperture_factor instead (CLAUDE.md non-negotiable #5)."
    )


@pytest.mark.parametrize(
    "source",
    [
        "omega = math.pi / (4.0 * f_number ** 2)",
        "omega = math.pi / (2 * F) ** 2",
        "omega = math.pi / (f * f * 4)",
        "omega = math.pi / pow(cfg.optics.f_number, 2)",
        "omega = 1 / (4 * self.f_number * self.f_number + 1)",
    ],
)
def test_scanner_catches_obfuscated_variants(source: str) -> None:
    assert aperture_expressions(source), source


def test_scanner_passes_clean_code() -> None:
    clean = (
        "from irsim.optics import aperture_factor\n"
        "omega = aperture_factor(cfg.optics.f_number)\n"
        "theta = math.atan(1.0 / (2.0 * f_number))\n"
        "n = 4 * width\n"
        "x = pitch ** 2\n"
    )
    assert aperture_expressions(clean) == []


def test_allowed_file_does_contain_the_definition() -> None:
    assert aperture_expressions(ALLOWED.read_text()), "aperture.py must be where the factor lives"
