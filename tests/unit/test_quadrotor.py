"""Engine-free tests for the parametric quadrotor and the aerial solvers behind it (ADR 0074).

Two things are worth holding here and neither is about how the aircraft looks.

The first is **scale**. Angular size is the load-bearing quantity in this whole project -- it
decides whether a target is resolved, whether the sub-pixel handover fires, and what contrast a
test may assert -- and a parametric airframe earns its keep precisely by having dimensions that
are *stated* rather than measured off an imported bounding box. So the layout arithmetic is
checked against the span it claims.

The second is **the thermal ordering**. A drone's infrared signature is not one temperature, it is
motors above speed controllers above the pack above an airframe sitting at air temperature. If
that ordering ever inverted, every frame would still look like a plausible thermal image of a
drone, and every detector trained on it would learn the wrong thing.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.scene import SceneConfig, TargetSpec, load_scene_config
from irsim.scene import Scene
from irsim_isaac.quad_flight import PITCH_AT_FULL_THROTTLE_DEG, tracking_pose
from irsim_isaac.quadrotor import HEAVY_LIFT, QuadrotorSpec

REPO = pathlib.Path(__file__).resolve().parents[2]
FLIGHT_YAML = REPO / "configs" / "scenes" / "quad_flight_clear_noon.yaml"

BOSON_IFOV_MRAD = 0.8571


# --- the airframe ---------------------------------------------------------------------------


def test_the_motors_sit_on_the_span_the_spec_claims() -> None:
    """Every motor is exactly span/2 from the hub along the diagonal, and they are square.

    The span is what a frame is sold by and what every angular-size calculation starts from, so it
    has to be the distance the geometry actually has rather than a label attached to it.
    """
    spec = HEAVY_LIFT
    motors = [p for p in spec.parts() if p.name.startswith("motor_")]
    assert len(motors) == 4
    for part in motors:
        x, _, z = part.centre_m
        assert math.hypot(x, z) == pytest.approx(0.5 * spec.span_m, rel=1e-12)
    offset = spec.motor_offset_m()
    corners = sorted((round(p.centre_m[0], 9), round(p.centre_m[2], 9)) for p in motors)
    expected = sorted(
        (round(x * offset, 9), round(z * offset, 9))
        for x, z in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    )
    assert corners == expected, "all four quadrants, one motor each"


def test_the_arms_reach_from_the_hub_to_the_motors() -> None:
    """An arm is the full diagonal long and centred halfway out, so it meets both ends.

    Centring it at half the *arm length* instead of half the motor offset -- the obvious slip --
    leaves a gap at the hub and overshoots the motor by the same amount, and looks fine.
    """
    spec = HEAVY_LIFT
    for part in (p for p in spec.parts() if p.name.startswith("arm_")):
        x, _, z = part.centre_m
        centre_distance = math.hypot(x, z)
        half_length = 0.5 * part.size_m[0]
        assert centre_distance == pytest.approx(half_length, rel=1e-12)
        assert centre_distance + half_length == pytest.approx(spec.arm_length_m(), rel=1e-12)


def test_the_speed_controllers_sit_between_the_hub_and_the_motors() -> None:
    spec = HEAVY_LIFT
    for part in (p for p in spec.parts() if p.name.startswith("esc_")):
        x, _, z = part.centre_m
        assert 0.0 < math.hypot(x, z) < 0.5 * spec.span_m


def test_every_part_has_a_material_and_a_thermal_node() -> None:
    """A surface with no material has no emissivity and one with no node has no temperature.

    Both are the silent failures this project exists to avoid, so the airframe is not allowed to
    contain a part that has neither -- and the node set is what a scene must then define.
    """
    parts = HEAVY_LIFT.parts()
    assert parts, "an airframe with no parts is not an airframe"
    for part in parts:
        assert part.material, part.name
        assert part.thermal_node, part.name
    assert HEAVY_LIFT.thermal_nodes() == ("airframe", "battery", "esc", "motor")


def test_the_motor_bells_are_high_emissivity_by_default() -> None:
    """Anodised, not bare. Bare aluminium is eps = 0.09 and would render a 70 C motor as cold.

    The library holds both materials precisely so that confusion is catchable, and the default
    here has to be the physically usual one rather than the one that flatters the demo.
    """
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.table import MaterialTable

    table = MaterialTable.from_library(MaterialLibrary.load(), "lwir")
    index = {name: i for i, name in enumerate(table.names)}
    emissivity = np.asarray(table.emissivity)
    assert emissivity[index[HEAVY_LIFT.motor_material]] > 0.8
    assert emissivity[index["bare_aluminium"]] < 0.2


def test_the_heavy_lift_quad_is_resolved_at_the_demo_range() -> None:
    """105 px of span and 6 px of motor at 20 m: the four hot spots are four hot spots.

    Stated as a test because it is the claim the whole stage rests on -- 'you can see the motors
    separately' -- and it is a property of the optics and the airframe together, not of either.
    """
    spec = HEAVY_LIFT
    span_px = 1e3 * spec.span_m / 20.0 / BOSON_IFOV_MRAD
    assert span_px > 90.0
    motor = next(p for p in spec.parts() if p.name == "motor_0")
    assert motor.pixels_across(20.0, BOSON_IFOV_MRAD) > 4.0
    # ...and at the anti-UAV ranges of the other stage it is emphatically not resolved, which is
    # why that stage exists separately and hands sub-pixel targets to the analytic path.
    assert motor.pixels_across(500.0, BOSON_IFOV_MRAD) < 1.0


def test_a_nonsense_airframe_is_refused() -> None:
    for bad in (
        {"span_m": 0.0},
        {"esc_fraction": 0.0},
        {"esc_fraction": 1.0},
        {"motor_diameter_m": -0.1},
    ):
        with pytest.raises(ValueError):
            QuadrotorSpec(**bad)  # type: ignore[arg-type]


# --- the flight -----------------------------------------------------------------------------


def test_attitude_follows_the_throttle_that_heats_the_motors() -> None:
    """A multirotor accelerates by tilting its disc, so pitch and motor heat share one input.

    Without this the picture and the physics are free to disagree -- an aircraft hanging level
    while its motors read full power, which looks fine and is the sort of thing nobody notices.
    """
    _, level = tracking_pose(0.0, 0.0, range_m=20.0, camera_tilt_deg=15.0)
    _, hard = tracking_pose(0.0, 1.0, range_m=20.0, camera_tilt_deg=15.0)
    assert level[0] == pytest.approx(0.0)
    assert hard[0] == pytest.approx(-PITCH_AT_FULL_THROTTLE_DEG)


def test_the_tracker_holds_the_target_near_boresight_but_not_on_it() -> None:
    """The aircraft stays within a degree or so of the camera axis, and keeps moving.

    A target pinned to the exact centre for three hundred frames reads as a compositing error;
    one that wanders out of a 33 degree field is not being tracked.
    """
    offsets = []
    for t in np.linspace(0.0, 1800.0, 60):
        translate, _ = tracking_pose(float(t), 0.5, range_m=20.0, camera_tilt_deg=15.0)
        x, y, z = translate
        assert math.sqrt(x * x + y * y + z * z) == pytest.approx(20.0, rel=1e-9)
        elevation = math.degrees(math.asin(y / 20.0))
        azimuth = math.degrees(math.atan2(x, -z))
        offsets.append((azimuth, elevation - 15.0))
    assert max(abs(a) for a, _ in offsets) < 2.0
    assert max(abs(e) for _, e in offsets) < 2.0
    assert max(abs(a) for a, _ in offsets) > 0.3, "a perfectly still target is not a tracked one"


# --- the scene ------------------------------------------------------------------------------


def test_the_flight_scene_defines_every_node_the_airframe_needs() -> None:
    """The stage and the scene have to agree, or a rendered part has no temperature at all."""
    scene = Scene.from_file(FLIGHT_YAML)
    assert set(HEAVY_LIFT.thermal_nodes()) <= set(scene.targets)


def test_the_thermal_ordering_holds_at_every_point_of_the_flight() -> None:
    """motor > esc > battery > airframe, at every sampled instant of the mission.

    This is the signature itself. The three heat sources share one throttle history and differ
    only in ΔT_max (45 / 30 / 15 K, ADR 0072), so the ordering is a property of the model rather
    than of the profile -- which is exactly why it is cheap to assert and expensive to lose.
    """
    scene = Scene.from_file(FLIGHT_YAML)
    seen_hot = False
    for t in np.linspace(0.0, 1795.0, 120):
        temps = scene.advance_targets(float(t), 0.0)
        air = scene.weather_at(float(t)).t_air_k
        assert temps["motor"] >= temps["esc"] >= temps["battery"] >= temps["airframe"] - 1e-9
        assert temps["airframe"] == pytest.approx(air, abs=1e-6)
        seen_hot = seen_hot or temps["motor"] > air + 40.0
    assert seen_hot, "the mission should take the motors more than 40 K above ambient somewhere"


def test_the_motor_follows_the_square_of_the_throttle() -> None:
    """T - T_air = 45 K u^2, derived by the scene rather than typed into it (ADR 0072, §6.6).

    The point of the ``heat_source`` solver is that a config author writes *what the pilot did*
    and the temperature follows. If the schedule were authored directly this test would be
    comparing a table against itself.
    """
    from irsim.thermal.aerial import MOTOR

    config = load_scene_config(FLIGHT_YAML)
    spec = next(t for t in config.scene.targets if t.name == "motor")
    scene = Scene.from_file(FLIGHT_YAML)
    for t_rel, throttle in zip(spec.throttle_s or [], spec.throttle or [], strict=True):
        temps = scene.advance_targets(float(t_rel), 0.0)
        air = scene.weather_at(float(t_rel)).t_air_k
        assert temps["motor"] - air == pytest.approx(MOTOR.delta_t_k(throttle), abs=2e-3)


def test_a_heat_source_scene_refuses_an_authored_schedule() -> None:
    """Authoring temperatures next to a throttle would leave two answers and no way to pick."""
    with pytest.raises(ValueError, match="derives its schedule"):
        TargetSpec(
            name="motor",
            solver="heat_source",
            source="motor",
            throttle_s=[0.0, 10.0],
            throttle=[0.0, 1.0],
            schedule_s=[0.0, 10.0],
            schedule_k=[300.0, 320.0],
        )


def test_an_unknown_heat_source_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="source must be one of"):
        TargetSpec(
            name="x",
            solver="heat_source",
            source="turbofan",
            throttle_s=[0.0, 1.0],
            throttle=[0.0, 1.0],
        )


def test_a_throttle_outside_zero_to_one_is_refused() -> None:
    for bad in ([-0.1, 0.5], [0.5, 1.4]):
        with pytest.raises(ValueError, match="throttle must lie"):
            TargetSpec(
                name="x",
                solver="heat_source",
                source="motor",
                throttle_s=[0.0, 1.0],
                throttle=bad,
            )


def test_the_other_solvers_still_refuse_aerial_fields() -> None:
    """A newton node with a throttle profile is a config that means two different things."""
    with pytest.raises(ValueError, match="takes no throttle profile"):
        TargetSpec(
            name="x",
            solver="newton",
            t0_k=300.0,
            tau_s=900.0,
            throttle_s=[0.0, 1.0],
            throttle=[0.0, 1.0],
        )


def test_the_airframe_node_starts_at_the_scene_time_not_the_weather_file_start() -> None:
    """An airframe schedule spans the whole weather file; its state must begin at the scene.

    Without the explicit advance its first reading is the air temperature hours earlier -- here
    the difference between a 26 C midday and an 18 C dawn, and entirely plausible either way.
    """
    scene = Scene.from_file(FLIGHT_YAML)
    assert scene.targets["airframe"].temperature() == pytest.approx(
        scene.weather_at(0.0).t_air_k, abs=1e-6
    )


def test_the_flight_scene_is_the_current_schema() -> None:
    from irsim.config.scene import SCENE_SCHEMA_VERSION

    assert isinstance(load_scene_config(FLIGHT_YAML), SceneConfig)
    assert load_scene_config(FLIGHT_YAML).schema_version == SCENE_SCHEMA_VERSION
