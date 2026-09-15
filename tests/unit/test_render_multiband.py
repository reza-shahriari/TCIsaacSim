"""The multi-band driver's bookkeeping (M10.24).

`scripts/render_multiband.py` runs three demo scenes x four bands as child processes, so almost
all of it needs Isaac Sim and a GPU. Two parts do not, and both are places where a partial run
silently misrepresents a complete output tree -- which is the failure mode this driver exists to
prevent, since its whole purpose is that the twelve renders are compared against one another.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "render_multiband.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("render_multiband", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: pathlib.Path, payload: dict) -> None:
    (root / "index.json").write_text(json.dumps(payload), "utf-8")


def test_a_partial_rerun_keeps_the_scenes_it_did_not_touch(driver, tmp_path):
    """The defect: re-rendering one scene left an index claiming the other two did not exist."""
    _write(
        tmp_path,
        {
            "renders": {"drone/lwir": "ok", "airplane/lwir": "ok", "ship/lwir": "ok"},
            "contact_sheets": {"drone": "/d.png", "airplane": "/a.png", "ship": "/s.png"},
        },
    )
    merged = driver.merged_index(tmp_path, {"ship/lwir": "ok"}, {"ship": "/s2.png"})
    assert merged["renders"] == {"drone/lwir": "ok", "airplane/lwir": "ok", "ship/lwir": "ok"}
    assert merged["contact_sheets"]["drone"] == "/d.png"
    assert merged["contact_sheets"]["airplane"] == "/a.png"


def test_this_run_wins_where_the_two_overlap(driver, tmp_path):
    """A scene that rendered last time and failed this time must read failed, not ok.

    This is the half of the merge that a plain `previous | fresh` gets right and a plain
    `fresh | previous` gets exactly backwards -- a stale success outliving the failure that
    replaced it is worse than no manifest at all.
    """
    _write(tmp_path, {"renders": {"ship/mwir": "ok"}, "contact_sheets": {"ship": "/old.png"}})
    merged = driver.merged_index(tmp_path, {"ship/mwir": "failed"}, {"ship": "/new.png"})
    assert merged["renders"]["ship/mwir"] == "failed"
    assert merged["contact_sheets"]["ship"] == "/new.png"


def test_no_index_yet_is_an_empty_one(driver, tmp_path):
    merged = driver.merged_index(tmp_path, {"drone/nir": "ok"}, {})
    assert merged == {"renders": {"drone/nir": "ok"}, "contact_sheets": {}}


@pytest.mark.parametrize("payload", ["{not json", "[]", '"a string"', "null"])
def test_an_unreadable_index_does_not_lose_this_run(driver, tmp_path, payload):
    """A convenience file any run can rebuild must never be able to discard a completed render."""
    (tmp_path / "index.json").write_text(payload, "utf-8")
    merged = driver.merged_index(tmp_path, {"drone/swir": "ok"}, {"drone": "/d.png"})
    assert merged["renders"] == {"drone/swir": "ok"}
    assert merged["contact_sheets"] == {"drone": "/d.png"}


def test_the_merge_does_not_mutate_the_arguments(driver, tmp_path):
    _write(tmp_path, {"renders": {"ship/lwir": "ok"}, "contact_sheets": {}})
    status = {"drone/lwir": "ok"}
    driver.merged_index(tmp_path, status, {})
    assert status == {"drone/lwir": "ok"}


def test_every_scene_is_long_enough_to_watch(driver):
    """Each scene is encoded to video, so each default count is a clip length.

    The ship's default was 8 frames -- a quarter of a second at 30 fps -- because it was written
    when the maritime script exported loose per-frame files and encoded nothing. The three scenes
    need not agree on a count (the aerial ones fly and the maritime camera is static), but a
    default that cannot be played is a bug in the driver rather than a choice about the scene.
    """
    for scene, (_script, _args, frames) in driver.SCENES.items():
        assert frames >= 90, f"{scene} renders {frames} frames, under 3 s at 30 fps"


def test_every_band_names_a_config_that_exists(driver):
    configs = SCRIPT.resolve().parents[1] / "configs" / "sensors"
    for band, (sensor, _args) in driver.BANDS.items():
        assert (configs / sensor).is_file(), f"{band} names a missing sensor config: {sensor}"
