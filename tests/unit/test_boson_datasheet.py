"""The Boson configs against FLIR's own published figures (SC.3, ADR 0091).

[R24] is *FLIR Boson® Thermal Imaging Core Product Datasheet*, Doc. # 102-2013-40 Release 340,
March 2021 -- free, public, EAR99. This project has no camera and is not getting one, so a
datasheet is the only external anchor available for the reference core, and three committed values
disagreed with it:

* ``ffc_interval_s: 180``, which matches **no** published default (Section 5 says 300 s);
* ``thermal_time_constant_ms: 10.0``, marked ESTIMATED against "typical VOx 8-12 ms" while the
  datasheet states "Nominally 8 msec" outright;
* ``ratios_3d`` carrying no provenance marker at all, in files whose headers promise one.

The bench below reproduces the acceptance conditions Table 13 states -- lensless at f/1.0, high
gain, 20 C camera, 30 C background, averager disabled, free-running -- and asserts the three
components FLIR actually bounds. It asserts **compliance with the limits**, not equality with the
ratio between them: Table 13 publishes upper bounds, and the ratio of two upper bounds is not the
ratio of two typical values. ADR 0091 argues that at length; `SC.2` substitutes ME.5's measured
ratios, which is the only thing that can settle the number.

docs/physics-model.md §9.4, §10.2, §16.1; ADR 0025 (σ_TVH convention), ADR 0091.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import RATIO_ORDER, SensorConfig, SensorSpec
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    MicrobolometerDetector,
    anchor_noise,
    fpa_params_from_config,
)
from irsim.noise import NoiseStage, measure_from_uniform_scene
from irsim.optics import pixel_power
from irsim.radiometry.lut import BandLUT
from irsim.validation import decompose_3d

# GT.1: a validation bench, not a unit test -- it synthesises 200-frame cubes.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
SENSORS = REPO / "configs" / "sensors"
BOSON_YAML = SENSORS / "flir_boson_640_lwir.yaml"
BOSON = yaml.safe_load(BOSON_YAML.read_text())
HALMSTAD = yaml.safe_load((SENSORS / "halmstad_boson_320.yaml").read_text())

#: [R24] Table 13, "Temporal NEDT in high-gain state", in millikelvin. Acceptance-test limits, so
#: every entry is an upper bound: grade -> (tvh, th, tv).
TABLE_13_MK: dict[str, tuple[float, float, float]] = {
    "industrial": (40.0, 14.0, 14.0),
    "professional": (50.0, 18.0, 18.0),
    "consumer": (60.0, 21.0, 21.0),
}

#: The conditions Table 13 states, reproduced by the bench below.
CAMERA_C, SCENE_C, F_NUMBER = 20.0, 30.0, 1.0


# --- the three corrected values, against the document ------------------------------------------


@pytest.mark.parametrize("config", [BOSON, HALMSTAD], ids=["boson_640", "halmstad_320"])
def test_the_ffc_period_is_the_published_factory_default(config) -> None:  # type: ignore[no-untyped-def]
    """[R24] S5: "the factory-default value of 300 represents a 300 second ... maximum time".

    The same document's Table 8 says 1200 s, and the 2018 FFC/NUC application note says its
    companion value 3.0 C where S5 says 1.0 C. ADR 0091 records why S5 wins: Table 8 carries its
    own staleness note, and its pair matches the 2018 release rather than this one.
    """
    assert config["sensor"]["nuc"]["ffc_interval_s"] == 300


@pytest.mark.parametrize("config", [BOSON, HALMSTAD], ids=["boson_640", "halmstad_320"])
def test_the_membrane_is_the_published_nominal_and_not_a_range_midpoint(config) -> None:  # type: ignore[no-untyped-def]
    """[R24]: "Boson's sensor assembly has a characteristic thermal time constant, nominally 8
    msec." The configs carried 10.0, the middle of the generic VOx 8-12 ms range, as ESTIMATED --
    a literature figure standing in for a number the datasheet publishes."""
    assert config["sensor"]["fpa"]["thermal_time_constant_ms"] == 8.0


def _ratios_block(path: pathlib.Path) -> tuple[str, str]:
    """``(the `ratios_3d:` line, the whole block)`` of a sensor YAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("ratios_3d:"))
    end = next(i for i, line in enumerate(lines[start:], start) if "fpn_drift_tau_s" in line)
    return lines[start], "\n".join(lines[start:end])


def test_the_640_carries_datasheet_limits_and_says_its_ratios_are_estimated() -> None:
    """Open question 6's split, one half: the 640 has no field measurement of its own.

    It is the camera the spec assumes and the datasheet describes, so what it can carry is
    [R24]'s acceptance *limits* -- which bound three of seven components from above and settle
    none of them. The file header promises "values marked ESTIMATED are literature/typical
    figures, not measurements", and until SC.3 the least-measured block in the file was the only
    one with no marker.
    """
    header, block = _ratios_block(BOSON_YAML)
    assert "ESTIMATED" in header, "the marker belongs on the block, where a reader lands"
    assert "Table 13" in block, "the limits the ratios are and are not bounded by go with them"
    # It may *point at* the 320's measurement -- saying why this file does not get one is the
    # useful thing to record. What it must not do is carry those values, which the paired test
    # below checks on the numbers rather than on the prose.
    assert "open\n      # question 6" in block or "open question 6" in block


def test_the_320_carries_me_5s_measured_ratios_and_names_their_bounds() -> None:
    """The other half: the 320 is the camera the public Halmstad set was recorded with.

    So it has something the 640 does not -- a decomposition of 365 of its own clips (`SC.2`).
    Five of the seven components come from it. The two that do not are marked, and the three
    caveats travel with the numbers, because an upper bound presented as a calibration is worse
    than an estimate that admits to being one.
    """
    header, block = _ratios_block(SENSORS / "halmstad_boson_320.yaml")
    assert "MEASURED" in header
    assert "ME.5" in block, "a measured number has to say which measurement"
    assert "reference-stats-2026-09-15" in block, "and where to read it"
    # The caveats, each of which changes how the number should be used.
    assert "28 of 365" in block, "the sample size the ratios rest on"
    assert "upper bound" in block, "ME.5's own framing of every value in that table"
    assert "denominator" in block, "the codec biases the ratios one way; say which"
    # And the two components ME.5 never reported stay marked.
    for line in block.splitlines():
        if line.strip().startswith(("tv:", "th:")):
            assert "ESTIMATED" in line, line


def test_the_two_cameras_never_share_a_ratios_block() -> None:
    """Open question 6, stated as a check rather than as a plan.

    Datasheet limits on the 640, field-measured ratios on the 320, never mixed in one block: a
    file that claimed both would be describing a camera that does not exist, and the mixture
    would be invisible in an image.
    """
    boson = SensorConfig.model_validate(BOSON).sensor.noise.ratios_3d
    halmstad = SensorConfig.model_validate(HALMSTAD).sensor.noise.ratios_3d
    assert boson.as_vector() != halmstad.as_vector()

    measured = {"vh": 2.64, "v": 0.58, "t": 0.38, "h": 0.16}
    for name, value in measured.items():
        assert getattr(halmstad, name) == pytest.approx(value), name
        assert getattr(boson, name) != pytest.approx(value), f"{name} leaked onto the 640"


def test_the_configured_ratio_is_far_inside_the_limit_ratio_and_that_is_recorded() -> None:
    """The measurement SC.3 exists to surface, stated rather than silently corrected.

    Every grade in Table 13 gives th/tvh = tv/tvh = 0.35. Both configs carry 0.05, so the
    committed cameras are modelled **seven times more spatially uniform than FLIR guarantees**.
    That is compliant and it is not measured; raising it to 0.35 would be reading a ratio of upper
    bounds as a typical value. `SC.2` substitutes measured ratios instead.
    """
    for grade, (tvh, th, tv) in TABLE_13_MK.items():
        assert th / tvh == pytest.approx(0.35, abs=0.011), grade
        assert tv == th, f"{grade}: Table 13 bounds row and column noise identically"

    for config in (BOSON, HALMSTAD):
        ratios = config["sensor"]["noise"]["ratios_3d"]
        assert ratios["th"] == ratios["tv"] == 0.05
        assert 0.35 / ratios["th"] == pytest.approx(7.0, rel=1e-9)


# --- the bench: FLIR's acceptance conditions, and the components it bounds ----------------------


@pytest.fixture(scope="module")
def sensor() -> SensorSpec:
    """The committed 640 at Table 13's conditions: f/1.0, high gain, averager off."""
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=64, height=64)
    d["sensor"]["optics"].update(f_number=F_NUMBER)
    return SensorConfig.model_validate(d).sensor


def _flux(sensor: SensorSpec, lut: BandLUT, t_k: float) -> np.ndarray:
    phi = float(
        pixel_power(
            lut.lookup(t_k),
            sensor.optics.f_number,
            sensor.optics.transmittance,
            sensor.detector_active_area_m2,
        )
    )
    return np.full(sensor.fpa_shape, phi, dtype=np.float32)


def _detector(sensor: SensorSpec, lut: BandLUT) -> MicrobolometerDetector:
    params = fpa_params_from_config(SensorConfig(sensor=sensor))
    assert isinstance(params, BolometerParams)
    lo = float(_flux(sensor, lut, 233.15)[0, 0])
    hi = float(_flux(sensor, lut, 473.15)[0, 0])
    return MicrobolometerDetector(
        params, BolometerTransfer.from_power_range(lo, hi, 16), anchor_noise(sensor, lut)
    )


def _components_mk(sensor: SensorSpec, lut: BandLUT, *, frames: int = 200) -> dict[str, float]:
    """The bench. Returns tvh/th/tv in millikelvin at Table 13's conditions.

    DN is converted to kelvin by the camera's own SITF, measured here as a differential rather
    than assumed: two uniform scenes a degree apart, through the same chain. A component quoted in
    DN cannot be compared to a datasheet, and the DN scale is a range choice (ADR 0019).
    """
    detector = _detector(sensor, lut)
    stage = NoiseStage.from_sensor(sensor, sensor_seed=20260917)
    scene_k = SCENE_C + 273.15

    ideal = MicrobolometerDetector(detector.params, detector.transfer, detector.budget)
    sitf_dn_per_k = float(
        ideal.noiseless_signal_dn(_flux(sensor, lut, scene_k + 0.5)).mean()
        - ideal.noiseless_signal_dn(_flux(sensor, lut, scene_k - 0.5)).mean()
    )
    assert sitf_dn_per_k > 0.0

    cube, _ = measure_from_uniform_scene(detector, stage, _flux(sensor, lut, scene_k), frames)
    d = decompose_3d(cube)
    return {k: getattr(d, k) / sitf_dn_per_k * 1e3 for k in ("tvh", "th", "tv")}


def test_the_bench_meets_table_13_at_the_configured_grade(
    sensor: SensorSpec, tophat_lwir_lut: BandLUT
) -> None:
    """The committed camera is the professional grade: < 50 / < 18 / < 18 mK.

    The temporal component is the anchor's by construction (ADR 0025 solves for it), so the
    informative half is the two spatial ones -- nothing in the chain forces those to land inside
    a limit, and at a large enough `ratios_3d` they would not.
    """
    limits = TABLE_13_MK["professional"]
    assert sensor.noise.netd_mk_at_300k == limits[0]

    measured = _components_mk(sensor, tophat_lwir_lut)
    for name, limit in zip(("tvh", "th", "tv"), limits, strict=True):
        assert measured[name] < limit, f"{name} = {measured[name]:.1f} mK exceeds < {limit:g} mK"

    # ... and by how much, which is the finding rather than the pass: ~2.5 mK against an 18 mK
    # bound, because the configured ratio is 0.05 where the limits' own ratio is 0.35.
    assert measured["th"] == pytest.approx(0.05 * limits[0], rel=0.25)
    assert measured["tv"] == pytest.approx(0.05 * limits[0], rel=0.25)


def test_the_limit_check_has_teeth(sensor: SensorSpec, tophat_lwir_lut: BandLUT) -> None:
    """A compliance check that cannot fail is decoration, so here is the failing direction.

    At th/tvh = 0.5 the column noise is 25 mK against an 18 mK limit and the camera is out of
    spec. The margin is real but not large: 0.35, the ratio of the limits themselves, sits at
    17.5 mK and passes with 0.5 mK to spare -- which is what "the limits are jointly attainable"
    means and why reading 0.35 off them is not absurd, only unsupported.
    """
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=64, height=64)
    d["sensor"]["optics"].update(f_number=F_NUMBER)
    d["sensor"]["noise"]["ratios_3d"].update(th=0.5, tv=0.5)
    loud = SensorConfig.model_validate(d).sensor

    measured = _components_mk(loud, tophat_lwir_lut)
    assert measured["th"] > TABLE_13_MK["professional"][1]
    assert measured["th"] == pytest.approx(25.0, rel=0.25)


# --- SC.2: the substituted ratios drive the rendered noise --------------------------------------


def test_the_measured_ratios_are_recovered_from_a_rendered_cube(tophat_lwir_lut: BandLUT) -> None:
    """ME.5's ratios survive the round trip through the chain and the decomposition.

    Worth checking rather than assuming, because `vh = 2.64` puts this camera somewhere nothing
    in the repository had rendered: **the fixed pattern is larger than the temporal noise**. Every
    previous config had vh < 1, so the estimator, the synthesiser and the variance closure had
    only ever been exercised on the other side of that line. A real core between flat-field events
    sits on this side, which is the whole reason §11.2's shutter exists.

    Tolerances are the estimator's own sampling floors, as ADR 0023 requires -- a 64x64x200 cube
    has 4096 samples for a VH term and 200 for a T term, so the two cannot be held to the same
    number.
    """
    from irsim.detector import BolometerTransfer, MicrobolometerDetector
    from irsim.noise import NoiseStage, Sigmas7, measure_from_uniform_scene
    from irsim.validation import decompose_3d

    dumped = copy.deepcopy(HALMSTAD)
    dumped["sensor"]["fpa"].update(width=64, height=64)
    sensor = SensorConfig.model_validate(dumped).sensor

    params = fpa_params_from_config(SensorConfig(sensor=sensor))
    assert isinstance(params, BolometerParams)
    lo = float(_flux(sensor, tophat_lwir_lut, 233.15)[0, 0])
    hi = float(_flux(sensor, tophat_lwir_lut, 473.15)[0, 0])
    detector = MicrobolometerDetector(
        params,
        BolometerTransfer.from_power_range(lo, hi, 16),
        anchor_noise(sensor, tophat_lwir_lut),
    )
    stage = NoiseStage.from_sensor(sensor, sensor_seed=20260917)

    cube, sigma_tvh = measure_from_uniform_scene(
        detector, stage, _flux(sensor, tophat_lwir_lut, 300.0), 200
    )
    decomposed = decompose_3d(cube)
    recovered = dict(zip(RATIO_ORDER, decomposed.ratios(), strict=True))

    assert recovered["tvh"] == pytest.approx(1.0, abs=1e-9)
    assert decomposed.tvh / sigma_tvh == pytest.approx(1.0, rel=0.02)

    # The fixed pattern really does come back larger than the temporal noise.
    assert recovered["vh"] > 1.0
    assert recovered["vh"] == pytest.approx(2.64, rel=0.05)
    assert recovered["v"] == pytest.approx(0.58, rel=0.10)

    # And the total: 2.91x sigma_TVH against the 1.06x the estimates gave, so this camera is
    # 2.7x noisier in total than the one it replaces.
    injected = Sigmas7.from_ratios(sigma_tvh, sensor.noise.sigma_ratios())
    assert sensor.noise.total_over_tvh() == pytest.approx(2.912, rel=1e-3)
    assert decomposed.total / injected.total == pytest.approx(1.0, rel=0.05)
