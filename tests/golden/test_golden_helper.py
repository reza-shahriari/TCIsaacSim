"""Self-tests for the golden fixture helper (tests/golden/conftest.py).

These prove the helper can tell STALE from FAILING from MISSING, refuses float16 on disk, and
leaves an inspectable actual array on failure -- the properties that make golden tests a
regression net rather than a liability (ir-sim-testing skill, ADR 0004).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from golden_store import GoldenStaleError, GoldenStore, config_hash

CFG = {"band": {"lambda_min_um": 7.5, "lambda_max_um": 13.5}, "netd_mk": 50}
HASH = config_hash(CFG)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> GoldenStore:
    return GoldenStore(root=tmp_path / "data", actual_dir=tmp_path / "actual", update=False)


@pytest.fixture
def recorded(store: GoldenStore) -> np.ndarray:
    """A golden already stored: a 300-400 K ramp in float32."""
    ramp = np.linspace(300.0, 400.0, 64, dtype=np.float32).reshape(8, 8)
    writer = GoldenStore(root=store.root, actual_dir=store.actual_dir, update=True)
    writer.check("ramp", ramp, config_hash=HASH, atol=1e-3, units="K")
    return ramp


def test_update_writes_array_and_sidecar(store: GoldenStore, recorded: np.ndarray) -> None:
    npy, sidecar = store.paths("ramp")
    assert npy.exists() and sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["config_hash"] == HASH
    assert meta["dtype"] == "float32" and meta["shape"] == [8, 8]
    assert meta["atol"] == 1e-3 and meta["units"] == "K"
    np.testing.assert_array_equal(np.load(npy), recorded)


def test_matching_array_passes_within_tolerance(store: GoldenStore, recorded: np.ndarray) -> None:
    store.check("ramp", recorded + np.float32(5e-4), config_hash=HASH, atol=1e-3)


def test_stale_is_reported_distinctly_and_names_the_hash(
    store: GoldenStore, recorded: np.ndarray
) -> None:
    """A changed config makes the golden STALE, not FAILING -- even if the values still agree."""
    new_hash = config_hash({**CFG, "netd_mk": 60})
    with pytest.raises(GoldenStaleError, match="STALE") as info:
        store.check("ramp", recorded, config_hash=new_hash, atol=1e-3)
    msg = str(info.value)
    assert HASH[:12] in msg and new_hash[:12] in msg
    assert "golden-update" in msg
    assert not (store.actual_dir / "ramp.actual.npy").exists(), "stale must not dump an actual"


def test_failure_dumps_actual_and_prints_both_paths(
    store: GoldenStore, recorded: np.ndarray
) -> None:
    """A 1 % perturbation against a 0.1 % tolerance fails, and both paths appear in the message."""
    perturbed = (recorded * np.float32(1.01)).astype(np.float32)
    with pytest.raises(AssertionError, match="FAILING") as info:
        store.check("ramp", perturbed, config_hash=HASH, rtol=1e-3)
    msg = str(info.value)
    expected_path, _ = store.paths("ramp")
    actual_path = store.actual_dir / "ramp.actual.npy"
    assert str(expected_path) in msg and str(actual_path) in msg
    assert "64/64 elements" in msg
    np.testing.assert_array_equal(np.load(actual_path), perturbed)


def test_shape_mismatch_is_failing(store: GoldenStore, recorded: np.ndarray) -> None:
    with pytest.raises(AssertionError, match="shape"):
        store.check("ramp", recorded[:4], config_hash=HASH, atol=1e-3)


def test_missing_golden_says_how_to_create_it(store: GoldenStore) -> None:
    with pytest.raises(FileNotFoundError, match="MISSING.*golden-update"):
        store.check("nothing", np.zeros(4, np.float32), config_hash=HASH, atol=1.0)


@pytest.mark.parametrize("update", [True, False])
def test_float16_golden_refused(store: GoldenStore, update: bool) -> None:
    """Non-negotiable #2 applies to files: float16 is refused before anything is written."""
    s = GoldenStore(root=store.root, actual_dir=store.actual_dir, update=update)
    with pytest.raises(TypeError, match="float16"):
        s.check("bad", np.zeros(4, np.float16), config_hash=HASH, atol=1.0)
    assert not list(store.root.glob("*")) if store.root.exists() else True


def test_float64_golden_refused(store: GoldenStore) -> None:
    writer = GoldenStore(root=store.root, actual_dir=store.actual_dir, update=True)
    with pytest.raises(TypeError, match="float64"):
        writer.check("bad", np.zeros(4, np.float64), config_hash=HASH, atol=1.0)


def test_exact_equality_is_not_allowed(store: GoldenStore, recorded: np.ndarray) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        store.check("ramp", recorded, config_hash=HASH)


def test_hand_edited_float16_file_is_rejected_on_load(
    store: GoldenStore, recorded: np.ndarray
) -> None:
    npy, _ = store.paths("ramp")
    np.save(npy, recorded.astype(np.float16), allow_pickle=False)
    with pytest.raises(TypeError, match="float16"):
        store.check("ramp", recorded, config_hash=HASH, atol=1e-3)


def test_config_hash_ignores_key_order_but_not_values() -> None:
    reordered = {"netd_mk": 50, "band": {"lambda_max_um": 13.5, "lambda_min_um": 7.5}}
    assert config_hash(reordered) == HASH
    assert config_hash({**CFG, "netd_mk": 50.0001}) != HASH


def test_repo_store_is_wired_to_the_option(
    golden: GoldenStore, request: pytest.FixtureRequest
) -> None:
    """The ``golden`` fixture points at tests/golden/data and honours --update-golden."""
    assert golden.root.name == "data" and golden.root.parent.name == "golden"
    assert golden.update is bool(request.config.getoption("--update-golden"))
