"""Every engine-free subpackage must import on its own, in a fresh interpreter, in any order.

This is a regression guard, not a style check. ``irsim.thermal.weather`` imported
``irsim.atmosphere.humidity`` at module scope, which runs ``irsim/atmosphere/__init__`` and lands
back in ``irsim.thermal.weather`` while it is still initialising -- so ``import irsim.thermal``
raised ImportError on a clean interpreter. The whole suite still passed, because pytest collects
test modules alphabetically and something imported ``irsim.atmosphere`` before anything imported
``irsim.thermal``; the failure only appeared when a single test file was run on its own, or when a
new test happened to import ``irsim.thermal`` first.

A cycle like that is invisible until it is load-bearing, so each import runs in its own subprocess
with a cold module table rather than relying on collection order.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

SUBPACKAGES = [
    "irsim",
    "irsim.atmosphere",
    "irsim.config",
    "irsim.detector",
    "irsim.isp",
    "irsim.materials",
    "irsim.noise",
    "irsim.optics",
    "irsim.pipeline",
    "irsim.radiometry",
    "irsim.scene",
    "irsim.thermal",
    "irsim.validation",
]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackage_imports_standalone(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`import {module}` fails in a fresh interpreter -- an import cycle that the suite's "
        f"collection order is currently hiding:\n{proc.stderr}"
    )


def test_thermal_before_atmosphere_is_the_failing_order() -> None:
    """The specific order that used to break: thermal first, atmosphere second."""
    proc = subprocess.run(
        [sys.executable, "-c", "import irsim.thermal; import irsim.atmosphere"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_weather_sample_humidity_properties_still_work() -> None:
    """The deferred import must not have broken the properties it was moved into."""
    from irsim.thermal import WeatherSample

    sample = WeatherSample(293.15, 0.5, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0)
    assert sample.vapour_pressure_hpa == pytest.approx(11.7, rel=0.05)
    assert sample.absolute_humidity_g_m3 == pytest.approx(8.6, rel=0.05)
