"""Roadmap M2.1: the integration environment is what tests/integration and ADR 0014 assume.

Runs `irsim_isaac.probe.probe_environment()` inside the session's headless Kit and checks the
facts the rest of the Isaac lane depends on. A rename in a 6.x point release fails here first,
not deep inside a pipeline test. The report also lists which deprecated `isaacsim.sensors.*`
modules Kit itself has imported (the deprecated extensions are enabled by the app profile); that
`irsim_isaac` never imports them is an engine-free AST check in
`tests/unit/test_isaac_glue_imports.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def environment(simulation_app: Any, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    del simulation_app
    from irsim_isaac.probe import probe_environment, save_report

    report = probe_environment()
    path = tmp_path_factory.mktemp("isaac") / "environment.json"
    save_report(report, str(path))
    print(f"environment report: {path}")
    return report


def test_isaac_version_is_recorded(environment: dict[str, Any]) -> None:
    version = environment["isaac_version"]
    assert version and str(version[0]).startswith("6."), version


def test_spg_extension_present(environment: dict[str, Any]) -> None:
    spg = environment["extensions"]["omni.rtx.spg"]
    assert spg["present"], "omni.rtx.spg is not in this build — every SPG step is meaningless"


def test_experimental_rtx_api_present(environment: dict[str, Any]) -> None:
    ext = environment["extensions"]["isaacsim.sensors.experimental.rtx"]
    assert ext["present"] and ext["enabled"], ext
    api = environment["sensor_api"]
    for name in ("RtxCamera", "CameraSensor", "TiledCameraSensor", "SPGNode"):
        assert api.get(name), f"{name} missing from isaacsim.sensors.experimental.rtx: {api}"


@pytest.mark.gpu
def test_warp_has_cuda_device_inside_kit(environment: dict[str, Any]) -> None:
    warp = environment["warp"]
    assert warp is not None, environment["errors"].get("warp")
    assert warp["cuda_device_count"] > 0, warp


def test_report_is_json(environment: dict[str, Any]) -> None:
    assert json.loads(json.dumps(environment, default=str))["kit_version"]
    assert Path(__file__).exists()
