"""Engine-free tests for the aircraft, its aerodynamic heating and its flypast (ADR 0075).

Three claims carry this stage and each fails silently if it is wrong.

**The skin temperature.** A jet's whole skin sits above ambient because its own boundary layer
stagnates against it. Get that wrong -- or reach for the multirotor ``airframe`` node, which
assumes it away -- and the entire aircraft reads uniformly cooler than it is, in the direction that
makes it harder to detect. The formula is checked against its closed form and against the limit
where it must reproduce the model it replaces.

**The occlusion.** The nozzles must be *behind* their nacelles, or the aspect dependence the whole
stage exists to show does not exist. That is geometry, asserted here; whether the renderer agrees
is `tests/integration/test_aircraft_pass_isaac.py`'s job.

**The track.** Range and aspect are the two independent variables of the pass and both come out of
closed-form geometry, so a test can state what they should be rather than record what they were.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import TargetSpec, load_scene_config
from irsim.radiometry.constants import GAMMA_AIR, PRANDTL_AIR
from irsim.scene import Scene
from irsim.thermal.aerial import (
    RECOVERY_FACTOR_TURBULENT,
    mach_number,
    recovery_temperature_k,
    speed_of_sound_m_s,
)
from irsim_isaac.aircraft import LIGHT_JET, AircraftSpec
from irsim_isaac.aircraft_pass import (
    LOW_PASS,
    PassTrack,
    aircraft_attitude_quaternion,
    look_at_quaternion,
    minimal_rotation,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
PASS_YAML = REPO / "configs" / "scenes" / "aircraft_pass_clear_noon.yaml"
BOSON_IFOV_MRAD = 0.8571


# --- aerodynamic heating ----------------------------------------------------------------------


def test_the_speed_of_sound_is_the_standard_atmosphere_value() -> None:
    """340.29 m/s at the ISA sea-level 288.15 K -- a number with an outside reference."""
    assert float(speed_of_sound_m_s(288.15)) == pytest.approx(340.29, abs=0.01)
    assert float(speed_of_sound_m_s(223.15)) == pytest.approx(299.46, abs=0.05)


def test_recovery_temperature_matches_its_closed_form() -> None:
    """T_r = T_air (1 + r (gamma-1)/2 M^2), evaluated independently of the implementation."""
    for t_air, mach in ((300.0, 0.5), (223.0, 0.8), (288.15, 0.0)):
        expected = t_air * (1.0 + RECOVERY_FACTOR_TURBULENT * 0.5 * (GAMMA_AIR - 1.0) * mach**2)
        assert float(recovery_temperature_k(t_air, mach)) == pytest.approx(expected, rel=1e-12)


def test_the_recovery_factor_is_the_turbulent_boundary_layer_one() -> None:
    """r = Pr^(1/3) ~ 0.892: a full-scale skin at flight Reynolds number is turbulent."""
    assert pytest.approx(PRANDTL_AIR ** (1.0 / 3.0), rel=1e-12) == RECOVERY_FACTOR_TURBULENT
    assert 0.88 < RECOVERY_FACTOR_TURBULENT < 0.90


def test_ram_heating_reproduces_the_multirotor_assumption_at_multirotor_speed() -> None:
    """At 20 m/s the rise is 0.18 K, which is why `airframe_solver` may ignore it -- and at
    180 m/s it is 14 K, which is why an aircraft may not.

    This is the test that justifies having two skin models instead of one. The existing airframe
    node says a skin sits at air temperature "because forced convection pins it there"; that claim
    is true to a fifth of a kelvin at multirotor speed and wrong by fourteen at jet speed.
    """
    slow = float(recovery_temperature_k(300.0, mach_number(20.0, 300.0))) - 300.0
    fast = float(recovery_temperature_k(300.0, mach_number(150.0, 300.0))) - 300.0
    assert slow < 0.25, f"a multirotor's skin should be within a fraction of a K, got {slow:.2f}"
    assert 8.0 < fast < 13.0, f"a 150 m/s skin should be ~10 K over ambient, got {fast:.2f}"


def test_ram_heating_grows_as_the_square_of_speed() -> None:
    """Doubling the airspeed quadruples the rise: the M^2 in the law, not a fitted curve."""
    base = float(recovery_temperature_k(300.0, mach_number(100.0, 300.0))) - 300.0
    double = float(recovery_temperature_k(300.0, mach_number(200.0, 300.0))) - 300.0
    assert double / base == pytest.approx(4.0, rel=1e-9)


def test_mach_number_uses_the_air_temperature_it_is_given() -> None:
    """One true airspeed is a different Mach number on a cold day; the weather decides."""
    assert float(mach_number(180.0, 300.0)) < float(mach_number(180.0, 223.0))


def test_ram_heating_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        speed_of_sound_m_s(0.0)
    with pytest.raises(ValueError):
        mach_number(-1.0, 300.0)
    with pytest.raises(ValueError):
        recovery_temperature_k(300.0, 0.5, recovery_factor=0.0)
    with pytest.raises(ValueError):
        recovery_temperature_k(300.0, -0.1)


# --- the airframe -----------------------------------------------------------------------------


def test_the_nozzles_sit_behind_their_nacelles() -> None:
    """Aft of, narrower than, and coaxial with the nacelle: the three conditions for occlusion.

    If any of them failed the nozzle would be visible from every aspect and the signature would
    not change through the pass -- which is the one thing this stage is built to show.
    """
    parts = {p.name: p for p in LIGHT_JET.parts()}
    for index in (0, 1):
        nacelle, nozzle = parts[f"nacelle_{index}"], parts[f"nozzle_{index}"]
        assert nozzle.centre_m[2] > nacelle.centre_m[2], "the nozzle must be aft (+Z) of the cowl"
        assert nozzle.size_m[0] < nacelle.size_m[0], "and narrower than it"
        assert nozzle.centre_m[:2] == nacelle.centre_m[:2], "and on the same axis"


def test_an_airframe_whose_nozzle_is_too_wide_is_refused() -> None:
    with pytest.raises(ValueError, match="never be occluded"):
        AircraftSpec(nozzle_diameter_m=2.0, nacelle_diameter_m=1.1)


def test_every_part_has_a_material_and_a_thermal_node() -> None:
    parts = LIGHT_JET.parts()
    assert parts
    for part in parts:
        assert part.material and part.thermal_node, part.name
    assert LIGHT_JET.thermal_nodes() == ("nacelle", "nozzle", "skin")


def test_the_nozzle_material_is_high_emissivity() -> None:
    """A hot nozzle behind eps = 0.09 bare aluminium would read barely warm. It is not that."""
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.table import MaterialTable

    table = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    index = {name: i for i, name in enumerate(table.names)}
    assert float(np.asarray(table.emissivity)[index[LIGHT_JET.nozzle_material]]) > 0.8


def test_the_hot_part_is_a_near_point_source_at_realistic_range() -> None:
    """The span is tens of pixels and the nozzle is about two. That asymmetry is the point.

    Unlike the quadrotor, whose motors are a sixteenth of its span and resolvable at close range,
    a jet's nozzle is a fortieth of its span -- so an aircraft's infrared signature is a point
    source on a warm outline, and it is bright because it is hundreds of kelvin hot, not because
    it is large.
    """
    span_px = 1e3 * LIGHT_JET.span_m / 250.0 / BOSON_IFOV_MRAD
    nozzle_px = 1e3 * LIGHT_JET.nozzle_diameter_m / 250.0 / BOSON_IFOV_MRAD
    assert span_px > 60.0
    assert 2.0 < nozzle_px < 6.0
    assert span_px / nozzle_px == pytest.approx(LIGHT_JET.span_m / LIGHT_JET.nozzle_diameter_m)


# --- the track --------------------------------------------------------------------------------


def test_closest_approach_is_where_and_when_the_track_says() -> None:
    track = LOW_PASS
    assert track.cpa_range_m() == pytest.approx(math.hypot(track.altitude_m, track.offset_m))
    assert track.sample(track.cpa_time_s).range_m == pytest.approx(track.cpa_range_m(), rel=1e-12)
    # ...and it really is the minimum, not merely a labelled instant.
    ranges = [track.sample(float(t)).range_m for t in np.linspace(0.0, 10.0, 201)]
    assert min(ranges) == pytest.approx(track.cpa_range_m(), rel=1e-6)


def test_range_is_symmetric_about_closest_approach() -> None:
    track = LOW_PASS
    for delta in (1.0, 2.5, 5.0):
        before = track.sample(track.cpa_time_s - delta).range_m
        after = track.sample(track.cpa_time_s + delta).range_m
        assert before == pytest.approx(after, rel=1e-12)


def test_aspect_sweeps_nose_to_tail_through_the_beam() -> None:
    """0 deg is nose-on and 180 tail-on; the beam is exactly 90 at closest approach.

    The 90 degrees at CPA is not a convention, it is geometry: at closest approach the line of
    sight is perpendicular to the track by definition of closest approach.
    """
    track = LOW_PASS
    aspects = [track.sample(float(t)).aspect_deg for t in np.linspace(0.0, 10.0, 101)]
    assert all(b > a for a, b in zip(aspects, aspects[1:], strict=False)), "monotone, nose to tail"
    assert track.sample(track.cpa_time_s).aspect_deg == pytest.approx(90.0, abs=1e-9)
    assert aspects[0] < 40.0 and aspects[-1] > 140.0


def test_the_nozzle_face_expectation_flips_at_the_beam() -> None:
    """The hot aft face specifically: the cylindrical wall shows from the beam either way."""
    track = LOW_PASS
    assert not track.sample(track.cpa_time_s - 1.0).shows_nozzle_face()
    assert track.sample(track.cpa_time_s + 1.0).shows_nozzle_face()


def test_a_track_that_misses_the_camera_is_refused() -> None:
    for bad in ({"speed_m_s": 0.0}, {"altitude_m": 0.0}, {"offset_m": -10.0}):
        with pytest.raises(ValueError):
            PassTrack(**bad)  # type: ignore[arg-type]


# --- the mount --------------------------------------------------------------------------------


def test_the_mount_aims_where_it_is_told_without_rolling() -> None:
    """The camera's -Z lands on the target and its up stays in the vertical plane.

    No roll is what keeps the sky gradient horizontal across a pass in which the pedestal slews
    through 50 degrees of azimuth. The minimal rotation between two vectors would be shorter and
    would tilt the horizon, which no two-axis pedestal does.
    """
    for t in (0.0, 2.5, 5.0, 7.5, 10.0):
        sample = LOW_PASS.sample(t)
        quat = look_at_quaternion(sample.position_m)
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0, rel=1e-12)
        rotation = _matrix_from_quaternion(quat)
        forward = -rotation[:, 2]  # a USD camera looks along its own -Z
        assert np.allclose(forward, sample.position_m / sample.range_m, atol=1e-9)
        right = rotation[:, 0]
        assert abs(float(right[1])) < 1e-9, "a level pedestal's right axis stays horizontal"


def test_the_aircraft_attitude_puts_its_nose_along_the_track() -> None:
    rotation = _matrix_from_quaternion(aircraft_attitude_quaternion())
    assert np.allclose(-rotation[:, 2], (1.0, 0.0, 0.0), atol=1e-12)  # body -Z is the nose
    assert np.allclose(rotation[:, 1], (0.0, 1.0, 0.0), atol=1e-12)  # wings level


def test_minimal_rotation_handles_the_degenerate_pairs() -> None:
    same = minimal_rotation((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
    assert np.allclose(same, (1.0, 0.0, 0.0, 0.0))
    opposite = minimal_rotation((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    assert float(np.linalg.norm(opposite)) == pytest.approx(1.0)
    assert float(opposite[0]) == pytest.approx(0.0, abs=1e-12), "180 degrees has zero real part"


def _matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --- the scene --------------------------------------------------------------------------------


def test_the_pass_scene_defines_every_node_the_aircraft_needs() -> None:
    scene = Scene.from_file(PASS_YAML)
    assert set(LIGHT_JET.thermal_nodes()) <= set(scene.targets)


def test_the_skin_node_is_the_recovery_temperature_of_the_shared_weather() -> None:
    """Derived, not authored: the config names an airspeed and the law does the rest (ADR 0075)."""
    config = load_scene_config(PASS_YAML)
    spec = next(t for t in config.scene.targets if t.name == "skin")
    assert spec.solver == "ram_skin" and spec.speed_m_s is not None
    scene = Scene.from_file(PASS_YAML)
    for t_rel in (0.0, 5.0, 10.0):
        temps = scene.advance_targets(t_rel, 0.0)
        air = scene.weather_at(t_rel).t_air_k
        expected = float(recovery_temperature_k(air, mach_number(spec.speed_m_s, air)))
        assert temps["skin"] == pytest.approx(expected, abs=2e-3)
        assert temps["skin"] > air + 7.0, "a 150 m/s skin is well above ambient"


def test_the_thermal_ordering_holds_across_the_pass() -> None:
    """nozzle > nacelle > skin > air, at every instant. The aircraft's signature, in one line."""
    scene = Scene.from_file(PASS_YAML)
    for t_rel in np.linspace(0.0, 20.0, 41):
        temps = scene.advance_targets(float(t_rel), 0.0)
        air = scene.weather_at(float(t_rel)).t_air_k
        assert temps["nozzle"] > temps["nacelle"] > temps["skin"] > air


def test_ram_skin_needs_an_airspeed_and_airframe_refuses_one() -> None:
    """The two skin models are not interchangeable and the config will not let them blur.

    ``airframe`` has no airspeed because it *assumes* a slow one; letting it accept one would
    invite a config that names 200 m/s and silently ignores it.
    """
    with pytest.raises(ValueError, match="ram_skin needs speed_m_s"):
        TargetSpec(name="skin", solver="ram_skin")
    with pytest.raises(ValueError, match="airframe has no airspeed"):
        TargetSpec(name="skin", solver="airframe", speed_m_s=180.0)
    with pytest.raises(ValueError, match="only ram_skin takes an airspeed"):
        TargetSpec(name="x", solver="newton", t0_k=300.0, tau_s=900.0, speed_m_s=180.0)
    with pytest.raises(ValueError, match="only airframe takes offset_k"):
        TargetSpec(name="skin", solver="ram_skin", speed_m_s=180.0, offset_k=3.0)
