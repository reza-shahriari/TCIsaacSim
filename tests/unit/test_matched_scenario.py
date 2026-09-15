"""M12.1: the camera and the scenarios a Tier 4 comparison is run against.

A comparison is only as good as the match between what was rendered and what was filmed. If the
synthetic clips are clear sky at 100 m and the real ones half cloud at 200 m, the report measures
the difference in *scenario* and the physics is never tested. These tests pin the camera against
the publication's own figures, and pin the sampler's honesty about how much of the scenario is
actually matched -- which on this set is none of it, for reasons ME.5 recorded.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.validation.scenario import (
    HALMSTAD_PRIORS,
    ScenarioParameter,
    ScenarioSampler,
    elevation_for_range,
    target_pixels,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
HALMSTAD = REPO / "configs" / "sensors" / "halmstad_boson_320.yaml"
SCRIPT = REPO / "scripts" / "generate_matched_scenario.py"


def _script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("generate_matched_scenario", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the camera ----------------------------------------------------------------------------------


def test_the_matched_camera_is_the_published_one() -> None:
    """Every number here is checkable against the publication, and is checked.

    The field of view is the trap: it is what turns a target's metres into pixels, so a camera
    with the right array and the wrong lens produces targets of the wrong size and a Tier 4
    size-vs-range comparison that fails for a reason that has nothing to do with the physics.
    """
    sensor = load_sensor_config(HALMSTAD).sensor
    assert sensor.fpa_shape == (256, 320)
    assert sensor.fpa.pitch_um == 12.0
    assert sensor.hfov_deg == pytest.approx(24.0, abs=0.1), "the PTQ-136's stated 24 deg"
    vfov = 2.0 * np.degrees(
        np.arctan(
            sensor.fpa.height * sensor.fpa.pitch_um * 1e-3 / 2.0 / sensor.optics.focal_length_mm
        )
    )
    assert vfov == pytest.approx(19.0, abs=0.4), "and its stated 19 deg"
    assert sensor.fpa.frame_rate_hz == 60.0, "the core's rate; the files are stored at 30 (ME.5)"


def test_the_isp_models_the_recorder_and_not_the_core() -> None:
    """ME.1a's signal path made concrete: these clips are a Y16 stream somebody converted to
    8 bits, so the core's DDE and palette are off -- but the conversion itself is not the
    identity, and pretending it is would render a sky-only scene into a handful of dark codes
    nothing like the published frames."""
    isp = load_sensor_config(HALMSTAD).sensor.isp
    assert isp.dde_gain == 0.0, "no DDE ran on this data"
    assert isp.palette == "gray"
    assert isp.agc == "linear", "the recorder's estimated Y16 -> 8-bit stretch, not the core's AGC"


# --- the scenarios -------------------------------------------------------------------------------


def test_a_scenario_is_reproducible_by_index_not_by_position() -> None:
    """Scenario 7 must be the same whether ten were drawn or a thousand. Without that, a failing
    case can only be reproduced by re-running the whole set, which on a slow generator means it is
    not reproduced at all."""
    few = ScenarioSampler(seed=11).draw(4)
    many = ScenarioSampler(seed=11).draw(40)
    for a, b in zip(few, many[:4], strict=True):
        assert a.index == b.index and a.values == b.values
    assert ScenarioSampler(seed=12).draw(1)[0].values != few[0].values


def test_every_prior_says_where_its_range_came_from() -> None:
    for prior in HALMSTAD_PRIORS:
        assert prior.provenance in {"measured", "stated", "estimated"}
        assert len(prior.source) > 30, prior.name
        assert prior.high >= prior.low
    with pytest.raises(ValueError, match="where its range came from"):
        ScenarioParameter("x", 0.0, 1.0, "estimated", "   ")
    with pytest.raises(ValueError, match="duplicate"):
        ScenarioSampler(priors=(HALMSTAD_PRIORS[0], HALMSTAD_PRIORS[0]))


def test_the_sampler_admits_that_none_of_this_set_is_measured() -> None:
    """The uncomfortable result, asserted so it cannot quietly stop being true.

    ME.5 could not measure the sky distributions (they need a labelled region and a visible
    horizon) or the target ones (the boxes are MATLAB MCOS objects with no Python reader), so
    every Halmstad prior is `stated` or `estimated`. A Tier 4 number from these clips is a
    comparison against an *assumed* scenario, and the report has to say so.
    """
    summary = ScenarioSampler().summary()
    assert summary["provenance_counts"]["measured"] == 0
    assert summary["provenance_counts"]["stated"] >= 2, "the range and size do come from the text"
    assert ScenarioSampler().draw(1)[0].measured_fraction == 0.0


def test_apparent_size_follows_the_pinhole_relation() -> None:
    assert target_pixels(1.0, 100.0, 9.03, 12.0) == pytest.approx(9.03e-3 / (100.0 * 12e-6))
    assert target_pixels(2.0, 100.0, 9.03, 12.0) == pytest.approx(
        2.0 * target_pixels(1.0, 100.0, 9.03, 12.0)
    )
    assert target_pixels(1.0, 200.0, 9.03, 12.0) == pytest.approx(
        0.5 * target_pixels(1.0, 100.0, 9.03, 12.0)
    )
    with pytest.raises(ValueError, match="positive"):
        target_pixels(1.0, 0.0, 9.03, 12.0)
    assert elevation_for_range(100.0, 50.0) == pytest.approx(30.0, abs=1e-9)
    assert elevation_for_range(100.0, 200.0) == pytest.approx(90.0), "clamped, not NaN"


# --- the generator -------------------------------------------------------------------------------


def test_a_generated_clip_puts_its_box_inside_the_frame(tmp_path: pathlib.Path) -> None:
    """The bug this prevents is silent and total: `build_aerial_gbuffer` reports its boxes on the
    **supersampled** grid, and the frames written out are the native one. A box off by the
    supersample factor lands outside the image, and every box-dependent statistic downstream then
    measures empty sky while reporting a number.
    """
    module = _script()
    sensor = load_sensor_config(HALMSTAD)
    from irsim.materials import MaterialLibrary, MaterialTable
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene

    band = sensor.sensor.band.band_id
    lut = BandLUT.build(load_spectral_response(sensor.sensor.band.spectral_response), n=1501)
    scene = Scene.from_file(
        REPO / "configs" / "scenes" / "sky_target_clear_day.yaml",
        {band: lut},
        quantity=sensor.sensor.quantity,
    )
    materials = MaterialTable.from_library(MaterialLibrary.load(), band)
    scenario = ScenarioSampler(seed=3).draw(1)[0]
    images, boxes = module._render_clip(
        scenario,
        sensor=sensor,
        scene=scene,
        lut=lut,
        materials=materials,
        frames=2,
        supersample=sensor.sensor.optics.supersample_factor,
    )
    height, width = sensor.sensor.fpa_shape
    assert images.shape == (2, height, width) and images.dtype == np.uint8
    for x, y, w, h in boxes:
        assert x >= 0.0 and x + w <= width, (x, w, width)
        assert y >= 0.0 and y + h <= height, (y, h, height)
        assert w > 0.0 and h > 0.0
    # The target moves across the frame: a stationary one makes the smear and the static-clip gate
    # vacuous, and both are Tier 4 statistics.
    assert boxes[1][0] > boxes[0][0]
