"""A tracked aircraft flypast: range and aspect both sweep, and the nozzle appears (ADR 0075).

The aircraft stage's variable is **aspect**, not throttle, and that is the whole reason it exists
beside :mod:`irsim_isaac.quad_flight`. A jet at steady cruise power has a constant nozzle
temperature and a skin pinned to its recovery temperature, so nothing about the *target* changes
during a flypast. What changes is the geometry: the aircraft comes in nose-first with both nozzles
hidden behind their own nacelles, sweeps through the beam, and departs showing them. Its measured
signature therefore varies by a large factor around the clock, which is why rear-aspect detection
ranges are the ones quoted, and it is a property no static frame can show.

Range sweeps too, from the track alone -- about 2:1 over a ten-second pass -- so the same stage
also exercises the inverse-square fall-off and the atmospheric path against a target of known size
and known temperature, with no extra scene machinery.

**The mount tracks the target, by actually slewing.** A 150 m/s aircraft crosses a 33 degree field
in under two seconds, so a fixed camera would show it for a handful of frames. Tracking is also
what removes the motion-blur objection: a stabilised line of sight holds the *target* still on the
focal plane and smears the featureless sky behind it instead.

The aircraft is placed at its **true world position** and the camera is re-aimed at it every
frame, rather than the cheaper trick of rotating the scene onto a fixed boresight. The difference
is the background: over this pass the target's true elevation sweeps from about 16 degrees to 37
and back, and the sky is tens of kelvin colder at the top of that. A fixed boresight would hold one
sky behind a target whose elevation was changing -- a perfectly plausible frame describing a camera
that is not the one being simulated. Re-aiming needs
:meth:`~irsim_isaac.pipeline.ir_camera.IrCamera.refresh_pose`, because the pose behind every
pixel's ray direction is cached at ``open()``.

**This one is real time, not a time-lapse.** ADR 0074 filmed the quadrotor as a time-lapse because
the process was thermal and slow. Here the process is geometric and fast: a ten-second pass filmed
at 30 Hz and played at 30 fps, with no speed-up to declare.

docs/physics-model.md §6.6 (by extension), ADR 0075
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim_isaac.aircraft import LIGHT_JET, AircraftSpec, author_aircraft
from irsim_isaac.stage import DOME_HEIGHT, author_environment, bind_visible_look
from irsim_isaac.visible_sky import DomeSpec

__all__ = [
    "LOW_PASS",
    "PassTrack",
    "PassSample",
    "AircraftPassStage",
    "minimal_rotation",
    "look_at_quaternion",
    "aircraft_attitude_quaternion",
    "build_aircraft_pass",
]


@dataclass(frozen=True)
class PassTrack:
    """A straight, level pass in front of a camera at the origin.

    The aircraft flies along +X at ``speed_m_s``, level at ``altitude_m``, crossing ``offset_m``
    ahead of the camera. Closest approach is therefore ``hypot(altitude, offset)`` and happens at
    ``cpa_time_s``. Three numbers with physical meanings rather than a waypoint list, because
    every quantity the stage reports -- range, aspect, angular size -- has to be derivable in
    closed form for a test to state what it should be.
    """

    speed_m_s: float = 150.0
    altitude_m: float = 150.0
    offset_m: float = 200.0
    cpa_time_s: float = 5.0

    def __post_init__(self) -> None:
        if self.speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive")
        if self.altitude_m <= 0.0 or self.offset_m <= 0.0:
            raise ValueError("the track must pass above and ahead of the camera")

    def cpa_range_m(self) -> float:
        return math.hypot(self.altitude_m, self.offset_m)

    def position_m(self, t_s: float) -> NDArray[np.float64]:
        """World position with the camera at the origin: +X right, +Y up, -Z forward."""
        along = self.speed_m_s * (float(t_s) - self.cpa_time_s)
        return np.array([along, self.altitude_m, -self.offset_m], dtype=np.float64)

    def sample(self, t_s: float) -> PassSample:
        position = self.position_m(t_s)
        range_m = float(np.linalg.norm(position))
        # Aspect: 0 deg is nose-on (the camera sees the front), 180 deg is tail-on. The heading is
        # +X, and the vector from aircraft to camera is -position.
        heading = np.array([1.0, 0.0, 0.0])
        cos_aspect = float(np.dot(heading, -position / range_m))
        return PassSample(
            t_s=float(t_s),
            position_m=position,
            range_m=range_m,
            aspect_deg=math.degrees(math.acos(max(-1.0, min(1.0, cos_aspect)))),
        )


@dataclass(frozen=True)
class PassSample:
    """Where the aircraft is at one instant and which way it is showing."""

    t_s: float
    position_m: NDArray[np.float64]
    range_m: float
    aspect_deg: float

    def shows_nozzle_face(self) -> bool:
        """True past the beam, where the nozzle's **hot aft face** turns toward the camera.

        Not the same as "any of the nozzle is visible": from the beam its cylindrical wall is in
        plain view while the aft face is edge-on, and from ahead a sliver of wall still shows past
        the nacelle at an oblique angle. What flips at 90 degrees is the face, and the face is
        where the area and the temperature are.

        A geometric expectation for a *test* to check the render against, never something the
        render consults -- the renderer decides occlusion by tracing the actual geometry, and if
        the two disagreed it would mean the airframe is not shaped the way this claims.
        """
        return self.aspect_deg > 90.0

    def pixels_across(self, size_m: float, ifov_mrad: float) -> float:
        return 1e3 * size_m / self.range_m / ifov_mrad


def minimal_rotation(source: Any, target: Any) -> NDArray[np.float64]:
    """Quaternion (w, x, y, z) of the shortest rotation taking unit ``source`` onto unit ``target``.

    Shortest rather than any rotation, because the leftover freedom is a roll about the line of
    sight and a tracker that rolled would spin the aircraft in frame for no physical reason.
    """
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis is a valid 180 degree turn; take a stable one.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(a, b)
    quat = np.array([1.0 + dot, *axis])
    return np.asarray(quat / np.linalg.norm(quat))


def _quaternion_from_matrix(m: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotation matrix -> (w, x, y, z), by the branch that avoids cancellation."""
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
    quat = np.zeros(4)
    quat[0] = (m[k, j] - m[j, k]) / s
    quat[1 + i] = 0.25 * s
    quat[1 + j] = (m[j, i] + m[i, j]) / s
    quat[1 + k] = (m[k, i] + m[i, k]) / s
    return quat


def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def look_at_quaternion(direction: Any, world_up: Any = (0.0, 1.0, 0.0)) -> NDArray[np.float64]:
    """Quaternion (w, x, y, z) aiming a USD camera's -Z axis along ``direction``, wings level.

    An azimuth/elevation mount construction rather than the minimal rotation between the two
    vectors: the minimal rotation is shorter but rolls the horizon as it slews, and a real
    two-axis pedestal does not. Keeping the camera's up vector in the vertical plane is what makes
    the sky gradient stay horizontal across the whole pass.
    """
    forward = np.asarray(direction, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    up = np.asarray(world_up, dtype=np.float64)
    right = np.cross(forward, up)
    norm = float(np.linalg.norm(right))
    # Straight up or straight down leaves azimuth undefined; any consistent choice will do.
    right = np.array([1.0, 0.0, 0.0]) if norm < 1e-9 else right / norm
    camera_up = np.cross(right, forward)
    # A USD camera looks along its own -Z, so the third column is -forward.
    return _quaternion_from_matrix(np.column_stack([right, camera_up, -forward]))


def aircraft_attitude_quaternion() -> NDArray[np.float64]:
    """The aircraft's own attitude: nose along +X, wings level, in world axes.

    Constant, because the track is straight and level. It exists as a function so the body-axis
    convention -- forward is the body's **-Z**, as for every other airframe here -- is written
    down once rather than assumed at the call site.
    """
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    return _quaternion_from_matrix(np.column_stack([right, up, -forward]))


#: The default pass: 150 m/s at 150 m, crossing 200 m ahead, closest approach 250 m at t = 5 s.
#: Over ten seconds that sweeps the range 790 -> 250 -> 790 m and the aspect 19 -> 90 -> 162
#: degrees, taking the nozzles from almost fully occluded to fully exposed.
#:
#: Closer than a cruising aircraft would be, deliberately: at 250 m the jet is 75 px across its
#: span and its nozzle is 3.7 px, where a 500 m pass leaves the nozzle at 1.9 px and the whole
#: aircraft a dim mark. The phenomenon is the same at either range -- this one lets you see it.
#: `--altitude-m`, `--offset-m` and `--speed-m-s` move it back out.
LOW_PASS = PassTrack()


@dataclass
class AircraftPassStage:
    """The built stage and the track it flies."""

    camera_path: str
    aircraft_path: str
    camera_tilt_deg: float
    track: PassTrack
    spec: AircraftSpec
    prim_to_target: dict[str, str]
    errors: dict[str, str] = field(default_factory=dict)

    def thermal_nodes(self) -> tuple[str, ...]:
        return self.spec.thermal_nodes()


def build_aircraft_pass(
    *,
    camera_path: str = "/World/IrCamera",
    aircraft_path: str = "/World/Targets/aircraft",
    camera_tilt_deg: float = 25.0,
    track: PassTrack = LOW_PASS,
    spec: AircraftSpec = LIGHT_JET,
    dome: DomeSpec | None = None,
    dome_texture_path: Any = None,
    dome_height: int = DOME_HEIGHT,
) -> AircraftPassStage:
    """Author the stage: environment dome, a tilted camera, and one aircraft on the boresight.

    No ground plane and no sky geometry, as in every stage here: the infrared background is
    computed from the sky model at each ray's own elevation (ADR 0060), so every pixel that is not
    the aircraft is sky.
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

    prim_to_target = author_aircraft(stage, aircraft_path, spec, look_binder=bind_visible_look)

    # The aircraft flies its true track; the caller rewrites these two ops every frame, and the
    # attitude is constant because the pass is straight and level.
    root = UsdGeom.Xformable(stage.GetPrimAtPath(aircraft_path))
    root.AddTranslateOp().Set(Gf.Vec3d(*track.position_m(track.cpa_time_s)))
    attitude = aircraft_attitude_quaternion()
    root.AddOrientOp().Set(
        Gf.Quatf(float(attitude[0]), Gf.Vec3f(*(float(v) for v in attitude[1:])))
    )

    # The camera is a two-axis pedestal at the origin; the caller re-aims it each frame and calls
    # `IrCamera.refresh_pose`, without which the rays would keep coming from the opening aim.
    camera = UsdGeom.Camera.Define(stage, camera_path)
    aim = look_at_quaternion(track.position_m(track.cpa_time_s))
    camera.AddOrientOp().Set(Gf.Quatf(float(aim[0]), Gf.Vec3f(*(float(v) for v in aim[1:]))))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e5))

    return AircraftPassStage(
        camera_path=camera_path,
        aircraft_path=aircraft_path,
        camera_tilt_deg=float(camera_tilt_deg),
        track=track,
        spec=spec,
        prim_to_target=prim_to_target,
        errors=errors,
    )
