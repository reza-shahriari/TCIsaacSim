"""Engine-availability probes. Importing this module never imports an engine.

`irsim_isaac` must be importable on a machine with no Isaac Sim and no GPU (the unit gate runs
there), so every engine import in this package happens inside a function, after one of these
probes. `tests/integration/conftest.py` uses them to skip the whole directory cleanly.

Detection is by `importlib.util.find_spec`, which locates a module without executing it. Whether
`isaacsim` resolves outside a running Kit application is exactly the kind of Isaac Sim 6.0 detail
the public docs leave open; roadmap M2.1 verifies it. Set `IRSIM_FORCE_NO_ISAAC=1` (or
`IRSIM_FORCE_NO_WARP=1`) to make the probes answer False, which is how the skip path is tested.
"""

from __future__ import annotations

import importlib.util
import os

__all__ = ["has_isaac", "has_warp", "require_isaac", "require_warp"]


def _spec_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_isaac() -> bool:
    """True when the Isaac Sim Python package (`isaacsim`) is importable."""
    if os.environ.get("IRSIM_FORCE_NO_ISAAC") == "1":
        return False
    return _spec_present("isaacsim")


def has_warp() -> bool:
    """True when NVIDIA Warp is importable (does not check for a CUDA device)."""
    if os.environ.get("IRSIM_FORCE_NO_WARP") == "1":
        return False
    return _spec_present("warp")


def require_isaac() -> None:
    """Raise a clear error at the call site that needs the engine, not deep inside it."""
    if not has_isaac():
        raise RuntimeError(
            "This code path needs Isaac Sim (`isaacsim` is not importable). Run it through the "
            "Isaac Sim interpreter (docs/decisions/0002) or use the engine-free path in `irsim`."
        )


def require_warp() -> None:
    if not has_warp():
        raise RuntimeError("This code path needs NVIDIA Warp (`pip install -e '.[isaac]'`).")
