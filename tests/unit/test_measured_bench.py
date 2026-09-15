"""M12.4: the layout and the arithmetic a Tier 2 bench comparison will use when a camera arrives.

No infrared camera is available to this project, so every Tier 2 number it reports today is a
self-consistency check. Fixing the file layout, the fitting rule and the tolerances **now** -- and
testing them on synthetic "measured" files -- is what stops the bench becoming a rubber stamp the
day hardware appears, because deciding a tolerance while looking at the first measurement is how
that happens.

The procedure these files come from is `docs/validation/tier2-bench-protocol.md`.
"""

from __future__ import annotations

import importlib
import pathlib

import numpy as np
import pytest

from irsim.validation.measured import (
    MEASURED_QUANTITIES,
    compare_absolute,
    compare_shape,
    load_measured_pairs,
    measured_path,
)

CAMERA = "flir_boson_640_lwir"


def test_the_layout_is_one_place_and_every_bench_uses_it() -> None:
    """A second layout invented in a test module is how a bench file ends up somewhere nobody
    looks, so each Tier 2 module's path is asserted to *be* `measured_path`'s."""
    for quantity, (subdir, columns, _) in MEASURED_QUANTITIES.items():
        path = measured_path(quantity, CAMERA)
        assert path.parent.name == subdir
        assert path.name == f"{CAMERA}.csv"
        assert len(columns) == 2 and all(columns)
    for module_name, quantity, attribute in [
        ("test_tier2_sitf", "sitf", "MEASURED_SITF"),
        ("test_tier2_netd", "netd", "MEASURED_NETD"),
        ("test_tier2_mtf", "mtf", "MEASURED_MTF"),
        ("test_tier2_3d_noise", "noise3d", "MEASURED_NOISE3D"),
    ]:
        module = importlib.import_module(module_name)
        assert getattr(module, attribute) == measured_path(quantity, CAMERA), module_name
    with pytest.raises(ValueError, match="unknown quantity"):
        measured_path("emissivity", CAMERA)


def _write(path: pathlib.Path, rows: list[tuple[float, float]], header: str = "a,b") -> None:
    lines = ["# a synthetic bench file (M12.4)", header]
    lines += [f"{x},{y}" for x, y in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_synthetic_measured_file_round_trips_through_the_loader(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "sitf.csv"
    rows = [(253.0, 1200.0), (300.0, 4800.0), (373.0, 12400.0), (453.0, 25600.0)]
    _write(path, rows, header="T_blackbody_K,DN")
    x, y = load_measured_pairs(path)
    assert np.allclose(x, [r[0] for r in rows]) and np.allclose(y, [r[1] for r in rows])
    # A one-row table cannot constrain a shape, and a silent pass on the ground truth would be the
    # worst possible outcome for a file whose whole purpose is to be the ground truth.
    _write(tmp_path / "one.csv", rows[:1])
    with pytest.raises(ValueError, match="at least two rows"):
        load_measured_pairs(tmp_path / "one.csv")
    (tmp_path / "junk.csv").write_text("1,2\nnot,numbers\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        load_measured_pairs(tmp_path / "junk.csv")
    with pytest.raises(FileNotFoundError):
        load_measured_pairs(tmp_path / "absent.csv")


def test_a_shape_comparison_ignores_gain_and_offset_but_not_curvature() -> None:
    """ADR 0019 makes a simulated SITF's absolute DN a range choice, so the bench compares the
    *curvature* -- which comes from Planck through the band and from nothing else."""
    simulated = np.array([1000.0, 2500.0, 6000.0, 13000.0])
    assert compare_shape(simulated, 3.5 * simulated + 900.0).passed
    assert compare_shape(simulated, -2.0 * simulated + 50.0).passed, "sign is a range choice too"
    bent = simulated.copy()
    bent[2] *= 1.25  # one point off the curve by a quarter
    assert not compare_shape(simulated, bent, tolerance=0.01).passed
    with pytest.raises(ValueError, match="at least two"):
        compare_shape([1.0], [1.0])


def test_an_absolute_comparison_refuses_what_a_shape_fit_would_hide() -> None:
    """The reason NETD and MTF are compared without a fit: a gain fit would let the model be wrong
    by any factor and still pass, which is the one thing those benches exist to rule out."""
    simulated = np.array([50.0, 38.0])  # mK at 300 K and 373 K
    assert compare_absolute(simulated, np.array([54.0, 41.0]), tolerance=0.15).passed
    doubled = 2.0 * simulated
    assert compare_shape(simulated, doubled).passed, "a shape fit sees nothing wrong"
    assert not compare_absolute(simulated, doubled, tolerance=0.15).passed
    with pytest.raises(ValueError, match="at least one"):
        compare_absolute([], [])


def test_a_comparison_describes_the_fit_it_needed(tmp_path: pathlib.Path) -> None:
    """The report line has to say whether a gain was fitted, or a reader cannot tell an absolute
    agreement from a shape one."""
    path = tmp_path / "netd.csv"
    _write(path, [(300.0, 54.0), (373.0, 41.0)], header="T_blackbody_K,NETD_mK")
    temps, measured = load_measured_pairs(path)
    assert list(temps) == [300.0, 373.0]
    assert measured[1] < measured[0], "NETD must fall as the scene warms (dL/dT rises)"
    absolute = compare_absolute(np.array([50.0, 38.0]), measured)
    shaped = compare_shape(np.array([50.0, 38.0]), measured)
    assert "fit" not in absolute.describe()
    assert "gain" in shaped.describe()
    assert absolute.n == 2 and shaped.n == 2


def test_the_protocol_document_exists_and_names_every_quantity() -> None:
    """A layout with no procedure behind it is a directory nobody can fill correctly."""
    repo = pathlib.Path(__file__).resolve().parents[2]
    text = (repo / "docs" / "validation" / "tier2-bench-protocol.md").read_text(encoding="utf-8")
    for quantity, (subdir, columns, _) in MEASURED_QUANTITIES.items():
        assert f"validation/{subdir}/" in text, quantity
        for column in columns:
            assert column in text, (quantity, column)
    assert "no fit" in text.lower(), "the fitting rule must be stated, not left to the reader"
