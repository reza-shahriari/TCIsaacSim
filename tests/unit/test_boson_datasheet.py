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

from irsim.config.sensor import SensorConfig, SensorSpec
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


@pytest.mark.parametrize("config", [BOSON, HALMSTAD], ids=["boson_640", "halmstad_320"])
def test_the_ratios_say_that_they_are_estimated(config) -> None:  # type: ignore[no-untyped-def]
    """Both file headers promise "Values marked ESTIMATED are literature/typical figures, not
    measurements". Four of the seven ratios are bounded by nothing published at all and the other
    three only from above, so the block is the least measured thing in either file and was the
    only block with no marker."""
    path = BOSON_YAML if config is BOSON else SENSORS / "halmstad_boson_320.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("ratios_3d:"))
    end = next(i for i, line in enumerate(lines[start:], start) if "fpn_drift_tau_s" in line)
    block = "\n".join(lines[start:end])
    assert "ESTIMATED" in lines[start], "the marker belongs on the block, where a reader lands"
    assert "Table 13" in block, "the limits the ratios are and are not bounded by go with them"
    assert "SC.2" in block, "and where the number that would replace them is coming from"


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
