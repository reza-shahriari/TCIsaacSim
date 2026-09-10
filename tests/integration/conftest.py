"""Integration tests need a running Isaac Sim. Without it every test here is skipped, not errored.

Every test collected from this directory is marked `isaac` automatically (so the default
`-m 'not isaac and not gpu'` in pyproject deselects it) and, when selected with `make test-all`
on a machine without Isaac Sim, is skipped with an explicit reason.

The `simulation_app` fixture boots one headless Kit application for the whole session (~35 s on
an RTX A6000, ADR 0014). Engine modules (`omni.*`, `pxr`, `warp`) are importable only after that
boot, so tests must import them inside the test body, never at module level, and must request the
fixture. NVIDIA Warp is provided by the `omni.warp.core` extension and is not on `sys.path` outside
Kit — `irsim_isaac.env.has_warp()` is False before the fixture runs — so `gpu`-marked tests are
skipped at collection only when Isaac Sim itself is absent.

Shutdown: `SimulationApp.close()` ends in `os._exit`, which would kill pytest before it prints its
summary and would replace its exit status with 0. The fixture therefore does not close the app in
its teardown; it registers an `atexit` handler that closes with the status pytest reported in
`pytest_sessionfinish`. Handlers run last-in-first-out, so ours runs before the one SimulationApp
registered at construction.

Command line: `SimulationApp` forwards every argument it does not recognise to Kit, and Kit's own
parser aborts the process on pytest's options (`-m ""` is an "Ill formed parameter" → segfault in
`_start_app`), so the fixture hides `sys.argv` while the app boots.
"""

from __future__ import annotations

import atexit
import sys
from collections.abc import Iterator
from typing import Any

import pytest

from irsim_isaac.env import has_isaac, has_warp

_session_exit = {"status": 0}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    del config
    no_isaac = pytest.mark.skip(reason="Isaac Sim not available (irsim_isaac.env.has_isaac())")
    no_warp = pytest.mark.skip(reason="NVIDIA Warp not available (irsim_isaac.env.has_warp())")
    for item in items:
        item.add_marker(pytest.mark.isaac)
        if not has_isaac():
            item.add_marker(no_isaac)
            if item.get_closest_marker("gpu") is not None and not has_warp():
                item.add_marker(no_warp)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del session
    _session_exit["status"] = int(exitstatus)


@pytest.fixture(scope="session")
def simulation_app() -> Iterator[Any]:
    """One headless Isaac Sim application per test session (closed at interpreter exit)."""
    if not has_isaac():
        pytest.skip("Isaac Sim not available")
    from isaacsim import SimulationApp

    argv = sys.argv
    sys.argv = argv[:1]  # Kit must not see pytest's options
    try:
        app = SimulationApp({"headless": True})
    finally:
        sys.argv = argv
    atexit.register(lambda: app.close(exit_code=_session_exit["status"]))
    yield app
