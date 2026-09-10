"""Smoke check that the integration environment is what the rest of tests/integration assumes."""

from __future__ import annotations

import pytest


def test_isaacsim_importable() -> None:
    pytest.importorskip("isaacsim")


@pytest.mark.gpu
def test_warp_has_cuda_device() -> None:
    wp = pytest.importorskip("warp")
    wp.init()
    assert wp.get_cuda_device_count() > 0
