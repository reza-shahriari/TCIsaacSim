"""Roadmap M2.2 / M2.4: the renderer-side transport ADR 0014 settled on, kept as regression guards.

The gate as specified — temperature in an emissive colour AOV, round trip < 10 mK — failed on this
build because every colour AOV is float16 (ADR 0014). These tests pin the facts the Isaac lane now
depends on instead: exact integer ids from the segmentation annotators, float32 geometry at full
resolution, and a bit-exact float32 pass-through of those AOVs through an SPG kernel. The fp16
finding itself is asserted too, so a build that changes it fails here first and ADR 0014 is
reopened rather than silently bypassed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

QUADS = 64
RESOLUTION = 256
PIXEL_AT_2M = 2.0 * 20.955 / 24.0 / RESOLUTION  # metres per pixel for the ramp-scene camera


@pytest.fixture(scope="module")
def ramp(simulation_app: Any) -> Any:
    del simulation_app
    from irsim_isaac.probe import build_ramp_scene, label_quads

    scene = build_ramp_scene(emissive_intensity=40.0)
    assert scene.errors == {}, scene.errors
    errors = label_quads(scene)
    assert errors == {}, errors
    return scene


@pytest.fixture(scope="module")
def transport(ramp: Any) -> dict[str, Any]:
    from irsim_isaac.probe import probe_id_transport

    return probe_id_transport(ramp, "rt", resolution=RESOLUTION, settle_frames=30)


def test_segmentation_ids_are_exact_integers(transport: dict[str, Any]) -> None:
    for name in ("instance_segmentation", "semantic_segmentation"):
        ids = transport["ids"][name]
        assert ids["status"] == "ok", ids
        assert ids["dtype"] == "uint32", ids["dtype"]
        assert ids["centre_ids_unique"], ids
        assert ids["centre_ids_nonbackground"] == QUADS, ids
        # a 5x5 window around every quad centre holds one id: no blended edge pixels
        assert ids["centre_windows_pure"] == QUADS, ids


def test_geometry_aovs_are_float32_at_full_resolution(transport: dict[str, Any]) -> None:
    pos = transport["positions"]["Camera3dPositionSD"]
    assert pos["status"] == "ok", pos
    assert pos["dtype"] == "float32", pos["dtype"]
    assert pos["shape"][:2] == [RESOLUTION, RESOLUTION], pos["shape"]
    assert max(pos["mean_abs_err_m_xyz"]) < PIXEL_AT_2M, pos["mean_abs_err_m_xyz"]
    dist = transport["positions"]["DistanceToCameraSD"]
    assert dist["status"] == "ok", dist
    assert dist["dtype"] == "float32", dist["dtype"]
    assert dist["shape"] == [RESOLUTION, RESOLUTION], dist["shape"]


def test_colour_aovs_are_float16_in_this_build(ramp: Any) -> None:
    """The negative result behind ADR 0014: reopen it if a float32 colour AOV ever appears."""
    from irsim_isaac.probe import probe_render_mode

    report = probe_render_mode(
        ramp, "rt", resolution=RESOLUTION, settle_frames=30, aovs=("HdrColor",)
    )
    hdr = report["aovs"]["HdrColor"]
    assert hdr["status"] == "ok", hdr
    assert hdr["dtype"] == "float16", hdr["dtype"]


def test_spg_passthrough_of_float32_geometry_is_bit_exact(ramp: Any, tmp_path: Path) -> None:
    del ramp
    from irsim_isaac.spg_probe import run_spg_experiments

    report = run_spg_experiments(
        str(tmp_path),
        experiments=("pass_geometry",),
        resolution=(RESOLUTION, RESOLUTION),
        settle_frames=30,
    )
    assert report["errors"] == {}, report["errors"]
    for aov in ("DistanceToCameraSD", "DistanceToImagePlaneSD", "Camera3dPositionSD"):
        res = report["experiments"]["pass_geometry"][aov]
        assert res["out"]["dtype"] == "float32", res
        assert res["finite_pattern_identical"], res
        assert res["identical_where_finite"], res
        assert res["max_abs_diff_finite"] == 0.0, res
