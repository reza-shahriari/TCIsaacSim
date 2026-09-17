"""Every render driver writes the frame, not only a picture of it (IG.13).

ADR 0068 is explicit that an 8-bit stream cannot carry a radiometric claim: a display PNG has had
the AGC, the palette and a quantisation to 256 levels applied to it, and none of those are
invertible. Three of this project's six drivers -- `render_quad_flight`, `render_aircraft_pass`
and `render_vessel_departure`, which between them are the whole aerial-flight and
vessel-departure lanes -- wrote nothing else. Every frame those renders have produced is
unusable for anything the simulator exists to do.

:class:`~irsim.io.dataset.FrameWriter` is the fix, and its shape is the point: nine keyword
arguments of which seven are constant for a run, bound once, so "does this driver write planes"
is a one-line question instead of a search through 500 lines of staging code. The last test here
asks that question of all six drivers and is what keeps a seventh from arriving without them.

docs/physics-model.md §12.2; CLAUDE.md non-negotiable #2; ADR 0004 (hashes travel with data).
"""

from __future__ import annotations

import ast
import json
import pathlib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from irsim.io import FrameWriter, read_float_plane

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"

#: The drivers that film a scene. `render_multiband` is excluded on purpose: it renders nothing
#: itself, it runs these as child processes.
DRIVERS = sorted(p for p in SCRIPTS.glob("render_*.py") if p.name != "render_multiband.py")

#: t = 0 on the weather axis -- the *series'* start, not the scene's. A scene beginning four
#: hours in carries that offset in `t_s`, which is exactly what the sidecar has to resolve.
START = datetime(2024, 6, 21, 0, 0, 0, tzinfo=timezone.utc)


def _outputs(rows: int = 3, columns: int = 4) -> SimpleNamespace:
    """A stand-in for `irsim.pipeline.frame.Outputs`: `write_frame` only ever reads attributes."""
    rng = np.random.default_rng(20260917)
    return SimpleNamespace(
        radiance=rng.uniform(1.0, 20.0, (rows, columns)).astype(np.float32),
        apparent_t=rng.uniform(280.0, 320.0, (rows, columns)).astype(np.float32),
        dn16=rng.integers(0, 65535, (rows, columns)).astype(np.uint16),
        display8=rng.integers(0, 255, (rows, columns, 4)).astype(np.uint8),
        isp_hash="isp-cafe",
    )


def _writer(directory: pathlib.Path, **overrides: object) -> FrameWriter:
    kwargs: dict[str, object] = {
        "directory": directory,
        "config_hash": "cfg-1234",
        "band_hash": "band-5678",
        "quantity": "radiance",
        "start_utc": START,
        "metadata": {"scene": "quad_flight_clear_noon.yaml", "sensor": "boson"},
    }
    kwargs.update(overrides)
    return FrameWriter(**kwargs)  # type: ignore[arg-type]


# --- the planes reach disk in physical units -------------------------------------------------


def test_the_planes_survive_the_round_trip_in_float32(tmp_path) -> None:
    """The whole point: what comes back is what the pipeline computed, bit for bit.

    A display PNG of the same frame cannot do this, which is why it is not a substitute.
    """
    outputs = _outputs()
    record = _writer(tmp_path).write(outputs, frame_index=0, t_s=14400.0)
    assert record is not None

    for key in ("radiance", "apparent_t"):
        back = read_float_plane(record.files[key])
        assert back.dtype == np.float32
        np.testing.assert_array_equal(back, getattr(outputs, key))


def test_the_sidecar_says_what_the_frame_is_and_where_it_came_from(tmp_path) -> None:
    record = _writer(tmp_path).write(outputs := _outputs(), frame_index=7, t_s=14400.0 + 60.0)
    assert record is not None
    meta = json.loads(record.sidecar.read_text("utf-8"))

    # Traceability (ADR 0004): without these the directory is a pile of images.
    assert meta["config_hash"] == "cfg-1234"
    assert meta["band_hash"] == "band-5678"
    assert meta["isp_hash"] == outputs.isp_hash
    assert meta["quantity"] == "radiance"

    # The run's own metadata travels on every frame.
    assert meta["scene"] == "quad_flight_clear_noon.yaml"
    assert meta["sensor"] == "boson"

    # The scene time and the wall clock it lands on: 14460 s past midnight is 04:01:00Z, the
    # second frame of a scene whose own start_utc is 04:00Z.
    assert meta["t_s"] == pytest.approx(14460.0)
    assert meta["utc"].startswith("2024-06-21T04:01:00")

    # ... and each plane's unit, so nobody has to guess whether a number is kelvin or watts.
    assert meta["planes"]["apparent_t"]["unit"] == "K"
    assert meta["planes"]["apparent_t"]["dtype"] == "float32"
    assert meta["planes"]["dn16"]["dtype"] == "uint16"


def test_a_float16_plane_is_refused_rather_than_written(tmp_path) -> None:
    """CLAUDE.md #2 at the disk boundary: half-float at 300 K is 0.25 K, 5x a 50 mK NETD."""
    outputs = _outputs()
    outputs.apparent_t = outputs.apparent_t.astype(np.float16)
    with pytest.raises(TypeError, match="float16"):
        _writer(tmp_path).write(outputs, frame_index=0, t_s=0.0)


# --- stride: thinning the written sequence, never lying about having done so -------------------


def test_a_stride_thins_the_sequence_and_says_so_in_every_sidecar(tmp_path) -> None:
    """A gap in the numbering must be distinguishable from a render that died halfway."""
    writer = _writer(tmp_path, stride=4)
    written = [writer.write(_outputs(), frame_index=i, t_s=float(i)) for i in range(10)]

    kept = [r for r in written if r is not None]
    assert [r.metadata["frame_index"] for r in kept] == [0, 4, 8]
    assert all(r.metadata["plane_stride"] == 4 for r in kept)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == [
        "frame_000000.json",
        "frame_000004.json",
        "frame_000008.json",
    ]


def test_stride_zero_writes_nothing_at_all(tmp_path) -> None:
    """The only honest spelling of "this run makes no radiometric claim"."""
    writer = _writer(tmp_path, stride=0)
    assert not writer.wants(0)
    assert writer.write(_outputs(), frame_index=0, t_s=0.0) is None
    assert list(tmp_path.iterdir()) == []


def test_a_negative_stride_is_refused_at_construction(tmp_path) -> None:
    with pytest.raises(ValueError, match="stride"):
        _writer(tmp_path, stride=-1)


def test_per_frame_metadata_layers_over_the_run_s_without_replacing_it(tmp_path) -> None:
    record = _writer(tmp_path).write(
        _outputs(), frame_index=0, t_s=0.0, extra_metadata={"throttle": 0.62, "sensor": "override"}
    )
    assert record is not None
    assert record.metadata["throttle"] == 0.62
    assert record.metadata["scene"] == "quad_flight_clear_noon.yaml"  # the run's, untouched
    assert record.metadata["sensor"] == "override"  # the frame's wins where they collide


def test_a_companion_plane_is_written_beside_the_frame_with_its_own_unit(tmp_path) -> None:
    """The visible companion is an extra plane, not an output: nothing radiometric reads it."""
    rgb = np.zeros((3, 4, 4), dtype=np.uint8)
    record = _writer(tmp_path).write(_outputs(), frame_index=0, t_s=0.0, extra_planes={"rgb": rgb})
    assert record is not None
    assert record.files["rgb"].suffix == ".png"
    assert "NO infrared content" in record.metadata["planes"]["rgb"]["unit"]


# --- and the guard that would have caught the defect ------------------------------------------


def _option_strings(tree: ast.Module) -> set[str]:
    """Every ``--flag`` the script's parser accepts, read off the `add_argument` calls."""
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and str(argument.value).startswith("--"):
                options.add(str(argument.value))
    return options


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_every_render_driver_writes_float32_planes_and_a_sidecar(driver) -> None:
    """The exit bar of IG.13, asked of each driver in turn.

    Before this step it failed for `render_quad_flight`, `render_aircraft_pass` and
    `render_vessel_departure`. It is a static check because the alternative is a render, and a
    render needs Isaac Sim and a GPU -- but what it checks is not cosmetic: a driver that names
    neither writer cannot be producing planes, whatever else it does.
    """
    source = driver.read_text("utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert names & {"FrameWriter", "write_frame"}, (
        f"{driver.name} writes no float32 planes. An 8-bit display PNG has had the AGC, the "
        "palette and a 256-level quantisation applied and none of them invert (ADR 0068)"
    )
    assert "--float-format" in _option_strings(tree), (
        f"{driver.name} gives no way to choose the float container; every other driver does"
    )


@pytest.mark.parametrize("driver", DRIVERS, ids=lambda p: p.name)
def test_every_render_driver_records_the_hashes_a_frame_is_traceable_by(driver) -> None:
    """ADR 0004: a frame nobody can trace back to a configuration is not data."""
    source = driver.read_text("utf-8")
    for hash_name in ("config_hash", "band_hash"):
        assert hash_name in source, f"{driver.name} never computes {hash_name}"
