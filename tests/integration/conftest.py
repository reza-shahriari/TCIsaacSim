"""Integration tests need a running Isaac Sim. Without it every test here is skipped, not errored.

Every test collected from this directory is marked `isaac` automatically (so the default
`-m 'not isaac and not gpu'` in pyproject deselects it) and, when selected with `make test-all`
on a machine without Isaac Sim, is skipped with an explicit reason.
"""

from __future__ import annotations

import pytest

from irsim_isaac.env import has_isaac, has_warp


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
