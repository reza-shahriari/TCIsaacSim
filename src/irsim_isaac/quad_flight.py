"""The quadrotor flight stage: one resolved multirotor against sky, tracked (ADR 0074).

A second stage beside :mod:`irsim_isaac.aerial_demo`, and the two answer different questions.
The aerial demo asks *can the camera detect this at all* -- six targets at 120 m to 2.5 km, most
of them a pixel or less, which is the anti-UAV range problem. This one asks *what does a drone
look like when you can see it*: a single heavy-lift quadrotor at 20 m, 105 px across its span,
with each motor 6 px of its own, so the four hot bells, the warm speed controllers and the warm
pack are separate objects in the image rather than one blob at one temperature.

**The camera tracks the target.** A drone on a 30-minute mission covers kilometres; nothing that
flies realistically stays in a 33 degree field. So the mount follows it, which is what an anti-UAV
tracker or a gimballed sensor does, and the residual motion in frame is tracking error and
attitude rather than translation. Attitude is *driven by the flight profile*: the aircraft pitches
into the direction it is accelerating, so the same throttle history that heats the motors also
tilts the airframe, and the picture and the physics cannot disagree about what the aircraft is
doing.

docs/physics-model.md §6.6, §15 T3; ADR 0072 (the heat sources), ADR 0073 (the dome),
ADR 0074 (this stage and its time base)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from irsim_isaac.quadrotor import HEAVY_LIFT, QuadrotorSpec, author_quadrotor
from irsim_isaac.stage import DOME_HEIGHT, author_environment, bind_visible_look
from irsim_isaac.visible_sky import DomeSpec

__all__ = [
    "QuadFlightStage",
    "build_quad_flight",
    "tracking_pose",
]

#: Peak nose-down pitch at full throttle, degrees. A multirotor accelerates by tilting its rotor
#: disc, so pitch is the visible face of the same throttle the thermal model is reading. ESTIMATED
#: -- it sets how much of the rotor plane the camera sees and nothing radiometric.
PITCH_AT_FULL_THROTTLE_DEG = 28.0

#: Amplitude and period of the residual tracking error, in degrees of field angle and seconds.
#: A tracker holds a target near boresight, not on it; a target pinned to the exact centre for
#: 300 frames looks like a compositing error rather than a measurement.
TRACK_JITTER_DEG = 0.9
TRACK_PERIOD_S = 420.0


@dataclass
class QuadFlightStage:
    """The built stage and everything a caller needs to fly it."""

    camera_path: str
    quad_path: str
    camera_tilt_deg: float
    range_m: float
    spec: QuadrotorSpec
    prim_to_target: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)

    def thermal_nodes(self) -> tuple[str, ...]:
        return self.spec.thermal_nodes()

    def pixels_across(self, ifov_mrad: float) -> dict[str, float]:
        """Native pixels across each part at this range -- the honest scale of the picture."""
        return {p.name: p.pixels_across(self.range_m, ifov_mrad) for p in self.spec.parts()}


def tracking_pose(
    t_rel_s: float, throttle: float, *, range_m: float, camera_tilt_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Where the aircraft is and how it is oriented at one instant: (translate, rotate XYZ).

    The translation is the boresight point at ``range_m``, nudged by a slow Lissajous of amplitude
    :data:`TRACK_JITTER_DEG` so the aircraft breathes around the centre of the field instead of
    being nailed to it. The rotation yaws slowly -- a tracked aircraft turns relative to the
    sensor -- and pitches nose-down in proportion to ``throttle``, which is the same number
    ADR 0072's law is heating the motors with.
    """
    az = TRACK_JITTER_DEG * math.sin(2.0 * math.pi * t_rel_s / TRACK_PERIOD_S)
    el = 0.6 * TRACK_JITTER_DEG * math.sin(4.0 * math.pi * t_rel_s / TRACK_PERIOD_S + 1.1)
    total_el = math.radians(camera_tilt_deg + el)
    total_az = math.radians(az)
    translate = (
        range_m * math.cos(total_el) * math.sin(total_az),
        range_m * math.sin(total_el),
        -range_m * math.cos(total_el) * math.cos(total_az),
    )
    yaw = 35.0 + 40.0 * math.sin(2.0 * math.pi * t_rel_s / (3.0 * TRACK_PERIOD_S))
    pitch = -PITCH_AT_FULL_THROTTLE_DEG * float(throttle)
    roll = 6.0 * math.sin(2.0 * math.pi * t_rel_s / (1.7 * TRACK_PERIOD_S))
    return translate, (pitch, yaw, roll)


def build_quad_flight(
    *,
    camera_path: str = "/World/IrCamera",
    quad_path: str = "/World/Targets/quad",
    camera_tilt_deg: float = 15.0,
    range_m: float = 20.0,
    spec: QuadrotorSpec = HEAVY_LIFT,
    dome: DomeSpec | None = None,
    dome_texture_path: str | os.PathLike[str] | None = None,
    dome_height: int = DOME_HEIGHT,
) -> QuadFlightStage:
    """Author the stage: the environment dome, a tilted camera, and one quadrotor on boresight.

    No ground plane and no sky geometry, exactly as in the aerial demo: the infrared background is
    computed from the sky model at each ray's own elevation (ADR 0060). The aircraft is the only
    geometry, so every pixel that is not the aircraft is sky.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")

    errors: dict[str, str] = {}
    try:
        author_environment(stage, dome, dome_texture_path, dome_height)
    except Exception as exc:  # noqa: BLE001 - a dark companion frame must not stop the IR render
        errors["environment"] = f"{type(exc).__name__}: {exc}"

    prim_to_target = author_quadrotor(stage, quad_path, spec, look_binder=bind_visible_look)

    # The aircraft's own transform, which the caller rewrites every frame; the parts sit in the
    # body frame underneath it, so one op flies the whole airframe.
    root = UsdGeom.Xformable(stage.GetPrimAtPath(quad_path))
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -range_m))
    root.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddRotateXOp().Set(float(camera_tilt_deg))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    return QuadFlightStage(
        camera_path=camera_path,
        quad_path=quad_path,
        camera_tilt_deg=float(camera_tilt_deg),
        range_m=float(range_m),
        spec=spec,
        prim_to_target=prim_to_target,
        errors=errors,
    )
