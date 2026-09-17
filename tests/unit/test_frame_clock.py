"""The time written beside a frame is the time that frame was captured at.

``IrCamera.get_outputs`` advances the render clock on its way out, so ``t_rel_s`` read *after* a
capture names the frame that has not happened yet. Two of the six render drivers were computing
their sidecar's ``t_s`` that way -- ``scene.t0_s + camera.t_rel_s``, one line below the capture --
so every frame they have ever written is labelled with the following frame's scene time, and the
sidecar's ``utc`` with it. ``render_car_ignition`` had already hit this and worked around it with
a comment; the other two had not.

A frame period is 1/60 s on a Boson and **six seconds** on this project's time-lapse drivers, where
the weather, the sun angle and every thermal node have visibly moved. A dataset whose timestamps
are one row out is not obviously broken -- it reads as a small calibration error in whatever is
fitted to it, which is the expensive kind of wrong.

The check is not that the attribute holds what it was assigned: that would pass with the bug
present. It is that the timestamp agrees with the clock the *physics* ran on -- the surface
field's own, advanced through a different call path (`point_bridge.advance_to`) inside
``planes()`` before the increment happens.

Roadmap IG.13; ADR 0087 (the field); docs/physics-model.md §6.1.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from test_camera_strictness import (  # the shared engine-free camera rig
    _aovs,
    _camera,
    _FakeReader,
    _patch,
)

from irsim.scene import Scene
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

REPO = pathlib.Path(__file__).resolve().parents[2]
SCENE_YAML = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"


#: Seconds of scene time per capture. The two drivers that carried the defect are time-lapses
#: (6 s and 60 s a frame), and that is where a one-frame slip stops being academic -- so the rig
#: runs as one, rather than at the Boson's 1/60 s where every stamp looks nearly right.
PERIOD_S = 60.0


@pytest.fixture(scope="module")
def scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_file(SCENE_YAML, {"lwir": tophat_lwir_lut})


def _field_at(patch: PlanarPatch, t0_s: float) -> PlanarThermalField:
    """A field already spun up to the scene start, so frame 0 integrates nothing."""
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(patch.n_cells, 1e9),
        emissivity=np.full(patch.n_cells, 0.95),
        solar_absorptivity=np.full(patch.n_cells, 0.9),
    )
    return PlanarThermalField(
        patch,
        props,
        lambda _t: FacetForcing(t_air_k=290.0, h_w_m2_k=0.0),
        float(t0_s),
        np.linspace(290.0, 320.0, patch.n_cells),
    )


@pytest.fixture()
def rig(aerial_sensor, scene, aerial_materials, tophat_lwir_lut):  # type: ignore[no-untyped-def]
    # The G-buffer has to be the detector grid: `get_outputs` runs the whole frame, where
    # `planes()` alone does not care. One cell per pixel keeps the field and the image 1:1.
    rows, columns = aerial_sensor.sensor.fpa_shape
    patch = _patch(n_u=columns, n_v=rows)
    field = _field_at(patch, scene.t0_s)
    camera = _camera(
        aerial_sensor,
        scene,
        aerial_materials,
        tophat_lwir_lut,
        bind_field=field,
        strict_patch_coverage=False,
    )
    camera.frame_period_s = PERIOD_S
    camera._reader = _FakeReader(_aovs(patch))  # type: ignore[assignment]
    return camera, field, scene


def test_no_frame_has_a_time_before_the_first_capture(rig) -> None:
    """``None``, not ``t0_s``: nothing has been rendered, so there is no frame to stamp."""
    camera, _, _ = rig
    assert camera.last_frame_t_s is None


def test_the_stamped_time_is_the_time_the_field_was_integrated_to(rig) -> None:
    """The oracle is the thermal field's own clock, not the attribute being tested."""
    camera, field, scene = rig
    period = camera.frame_period_s

    for index in range(3):
        camera.get_outputs(step=False)
        stamped = camera.last_frame_t_s
        assert stamped == pytest.approx(scene.t0_s + index * period, abs=1e-9)

        # Where the physics actually got to, reached through a different call path
        # (`planes()` -> `point_bridge.advance_to`). `advance_to` produces whole ticks, so the
        # field's clock sits at or just past this frame's time -- and nowhere near the next one's.
        assert stamped <= field.latest_t_s < stamped + period

        # The expression the two drivers used, kept as the negative control: read after the
        # capture it is a whole frame period ahead, and lands past the physics it claims to label.
        stale = scene.t0_s + camera.t_rel_s
        assert stale == pytest.approx(stamped + period, abs=1e-9)
        assert stale > field.latest_t_s


def test_the_stamp_is_absolute_scene_time_not_render_time(rig) -> None:
    """``t0_s`` is 4 h into the weather file here, so a relative stamp would be 14400 s wrong."""
    camera, _, scene = rig
    assert scene.t0_s > 0.0
    camera.get_outputs(step=False)
    assert camera.last_frame_t_s == pytest.approx(scene.t0_s, abs=1e-9)
    assert camera.last_frame_t_s != pytest.approx(camera.t_rel_s, abs=1.0)


def test_every_render_driver_takes_the_frame_time_from_the_camera() -> None:
    """The lint that would have caught it: no driver may do the arithmetic itself.

    `scene.t0_s + camera.t_rel_s` is only correct read *before* the capture, and a reader cannot
    tell which side of the call a line sits on at a glance. Taking the stamp off the camera
    removes the question.
    """
    offenders = []
    for driver in sorted((REPO / "scripts").glob("render_*.py")):
        text = driver.read_text("utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "t_s=" in line and "camera.t_rel_s" in line:
                offenders.append(f"{driver.name}:{number}: {line.strip()}")
    assert not offenders, (
        "a frame's time must come from `camera.last_frame_t_s`, not from arithmetic on a clock "
        "that has already advanced past it:\n  " + "\n  ".join(offenders)
    )
