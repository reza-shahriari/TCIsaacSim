"""Golden / regression fixture helper.

For outputs with no analytic answer (a full pipeline render, an AGC output) we store a reference
array and compare against it. The rules that keep golden tests from becoming a liability are the
ones in the ir-sim-testing skill and ADR 0004:

* the **config hash** that produced the array is stored beside it, so a golden whose inputs
  changed is reported as STALE, not as a physics failure;
* comparison always uses a stated tolerance in the array's own units, never exact equality;
* on failure the actual array is written next to the expected one and both paths are printed;
* arrays on disk are float32, uint16 or uint8 -- never float16 (CLAUDE.md non-negotiable #2
  applies to files) and never float64 (a golden should be what the pipeline actually emits);
* regeneration is deliberate: ``make golden-update`` passes ``--update-golden``.

Usage in a test::

    def test_band_radiance_stage(golden, gbuffer_ramp):
        out = run_stage(gbuffer_ramp, cfg)
        golden.check("band_radiance_ramp", out, config_hash=cfg_hash, atol=1e-3, units="W/m2/sr")

The reference files live in ``tests/golden/data/<name>.npy`` + ``<name>.json``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "data"
ACTUAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "golden"

ALLOWED_DTYPES = (np.dtype(np.float32), np.dtype(np.uint16), np.dtype(np.uint8))


class GoldenStaleError(Exception):
    """The golden's recorded config hash differs from the one the test now supplies.

    Distinct from a failing comparison: the reference is out of date, not (necessarily) the code.
    Regenerate with ``make golden-update`` once the config change is intended.
    """


def config_hash(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of a plain config object (dict / list / scalars).

    Key order and whitespace do not affect the hash; any numeric leaf does. The real sensor
    config hash (with file-content substitution) lands with the YAML loader; this is enough
    for goldens produced from literal dictionaries.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _check_storable(name: str, array: np.ndarray) -> None:
    if array.dtype == np.float16:
        raise TypeError(
            f"golden {name!r}: float16 arrays are refused -- at 300 K float16 spacing is 0.25 K, "
            "five times a 50 mK NETD (CLAUDE.md non-negotiable #2 applies to files on disk)"
        )
    if array.dtype not in ALLOWED_DTYPES:
        raise TypeError(
            f"golden {name!r}: dtype {array.dtype} is not storable; goldens are float32, uint16 "
            "or uint8 -- cast deliberately at the call site so the file records what the "
            "pipeline emits"
        )


@dataclass(frozen=True)
class GoldenStore:
    """Compare-or-record helper bound to a golden directory. Obtain via the ``golden`` fixture."""

    root: pathlib.Path
    actual_dir: pathlib.Path
    update: bool

    def paths(self, name: str) -> tuple[pathlib.Path, pathlib.Path]:
        return self.root / f"{name}.npy", self.root / f"{name}.json"

    def check(
        self,
        name: str,
        actual: np.ndarray,
        *,
        config_hash: str,
        atol: float | None = None,
        rtol: float | None = None,
        units: str = "",
    ) -> None:
        """Compare ``actual`` against the stored golden ``name``, or store it under --update-golden.

        At least one of ``atol``/``rtol`` is required (both allowed); ``units`` labels
        ``atol`` in the failure message so a reader can judge whether the tolerance is sane.

        Outcomes: PASS; STALE (hash differs -> :class:`GoldenStaleError`); FAILING (values differ
        beyond tolerance -> ``AssertionError``, actual array dumped to ``actual_dir``);
        MISSING (no golden and not updating -> ``FileNotFoundError``).
        """
        if atol is None and rtol is None:
            raise ValueError(
                f"golden {name!r}: state a tolerance (atol and/or rtol) -- never exact"
            )
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"golden name must be a bare file stem, got {name!r}")
        actual = np.asarray(actual)
        _check_storable(name, actual)
        npy_path, json_path = self.paths(name)

        if self.update:
            self.root.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, actual, allow_pickle=False)
            sidecar = {
                "name": name,
                "config_hash": config_hash,
                "dtype": str(actual.dtype),
                "shape": list(actual.shape),
                "atol": atol,
                "rtol": rtol,
                "units": units,
                "numpy_version": np.__version__,
            }
            json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
            return

        if not npy_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"MISSING golden {name!r}: expected {npy_path} and {json_path}. "
                "Generate deliberately with `make golden-update` and commit both files."
            )

        sidecar = json.loads(json_path.read_text())
        expected = np.load(npy_path, allow_pickle=False)
        _check_storable(name, expected)

        if sidecar["config_hash"] != config_hash:
            raise GoldenStaleError(
                f"STALE golden {name!r}: stored config_hash {sidecar['config_hash'][:12]}… "
                f"but the test now supplies {config_hash[:12]}…. The inputs that produced the "
                "reference changed, so this is not a physics failure. If the change is intended, "
                f"regenerate with `make golden-update` (files: {npy_path}, {json_path})."
            )

        self._compare(name, actual, expected, npy_path, atol=atol, rtol=rtol, units=units)

    def _compare(
        self,
        name: str,
        actual: np.ndarray,
        expected: np.ndarray,
        npy_path: pathlib.Path,
        *,
        atol: float | None,
        rtol: float | None,
        units: str,
    ) -> None:
        problems: list[str] = []
        if actual.shape != expected.shape:
            problems.append(f"shape {actual.shape} vs golden {expected.shape}")
        elif actual.dtype != expected.dtype:
            problems.append(f"dtype {actual.dtype} vs golden {expected.dtype}")
        else:
            a = actual.astype(np.float64)
            e = expected.astype(np.float64)
            diff = np.abs(a - e)
            bound = np.zeros_like(diff)
            if atol is not None:
                bound += atol
            if rtol is not None:
                bound += rtol * np.abs(e)
            bad = diff > bound
            if bad.any():
                worst = np.unravel_index(int(np.argmax(diff - bound)), diff.shape)
                unit_str = f" {units}" if units else ""
                problems.append(
                    f"{int(bad.sum())}/{bad.size} elements exceed tolerance "
                    f"(atol={atol}{unit_str}, rtol={rtol}); worst at index "
                    f"{tuple(int(i) for i in worst)}: actual {a[worst]:.6g} vs golden "
                    f"{e[worst]:.6g}, |diff| {diff[worst]:.3g}{unit_str}"
                )
        if not problems:
            return

        self.actual_dir.mkdir(parents=True, exist_ok=True)
        actual_path = self.actual_dir / f"{name}.actual.npy"
        np.save(actual_path, actual, allow_pickle=False)
        raise AssertionError(
            f"FAILING golden {name!r}: "
            + "; ".join(problems)
            + f"\n  expected: {npy_path}\n  actual:   {actual_path}"
        )


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenStore:
    """The repo golden store, in compare mode unless ``--update-golden`` was passed."""
    return GoldenStore(
        root=GOLDEN_DIR,
        actual_dir=ACTUAL_DIR,
        update=bool(request.config.getoption("--update-golden")),
    )
