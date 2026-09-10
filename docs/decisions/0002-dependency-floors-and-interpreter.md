# ADR 0002 — Dependency floors and the project interpreter

**Status:** Accepted
**Date:** 2026-09-10

## Context

The scaffold declared `numpy>=1.24` while its own tests call `np.trapezoid`, which exists only in
NumPy ≥ 2.0; on NumPy 1.26 three physics tests error before any physics is exercised. No project
interpreter had been chosen: the system Python 3.10 has no pytest, the miniconda base (3.13) has no
NumPy. Isaac Sim 6.0 is built from source on the development machine and ships its own CPython.

## Options considered

1. Keep `numpy>=1.24` and add a `trapz`/`trapezoid` compatibility shim in the tests.
2. Raise the floor to `numpy>=2.0`.
3. Vendor a small integration helper and never call NumPy's directly.

For the interpreter: a separate venv per half of the repo, or Isaac Sim's bundled Python for both.

## Decision

- `numpy>=2.0`. Isaac Sim 6.0's bundled interpreter ships NumPy 2.3.1, so the floor costs nothing and
  removes a silent version dependency from the physics tests.
- The project interpreter is Isaac Sim's bundled Python,
  `/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh` (CPython 3.12.13, NumPy 2.3.1,
  SciPy 1.17, pydantic 2.11). It serves both the engine-free core and the `@pytest.mark.isaac`
  integration tests, so there is one environment to keep consistent. The Makefile gains a `PYTHON`
  variable so every target runs through `$(PYTHON) -m ...`; CI or a teammate without Isaac Sim can
  pass any CPython ≥ 3.10.
- ruff rule `N802` joins `N803`/`N806` in the ignore list: physics names such as
  `d_spectral_radiance_dT` follow the spec's notation on purpose.

## Consequences

- `make check PYTHON=…/python.sh` is green on the scaffold as of this ADR (25 unit tests in 0.19 s).
- `requires-python >= 3.10` stays; code must avoid 3.11+ syntax even though the project interpreter
  is 3.12.
- Warp was not importable from `python.sh` in a first check; M2.1 of the roadmap verifies the GPU
  stack before any Warp kernel is written.

## Revisit when

- Isaac Sim moves to a Python or NumPy major version the core does not support.
- The team wants a GPU-free CI job (then the `PYTHON` variable points at a plain venv there).
