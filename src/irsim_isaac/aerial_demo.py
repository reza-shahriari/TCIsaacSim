"""The phase-1 demo stage: small targets against sky, at range, for ``IrCamera`` to render.

docs/physics-model.md §15 T3; roadmap M10.19 (the stage half); ADR 0003 (aerial first),
ADR 0044/MS.2 (the sky profile), ADR 0060 (the thermal bridge), ADR 0014 (ids are the transport).

**There is no sky dome, and no ground plane.** That looks like an omission and is the design. A
ray that hits no geometry takes its apparent temperature from the sky model at its own elevation,
or ``T_ground`` below the horizon (:mod:`irsim_isaac.pipeline.aerial_bridge`), so the background
is computed rather than rendered. Modelling it as an emissive dome instead would push the sky
through a colour AOV -- every one of which is float16 on this build, quantising to ~100 mK against
a 50 mK NETD (ADR 0014) -- and would be worst exactly at the horizon, where the elevation gradient
is steepest and where the targets of interest are. Geometry in this stage is therefore *only* the
things that have a surface: the targets.

The camera is tilted **up**, so the horizon sits near the bottom of the frame and most of the
image is the sky gradient a real anti-UAV camera spends its time looking at. Targets are placed in
camera space at a chosen field angle and range, then rotated into world space by the same tilt, so
"put a drone 8 degrees left of boresight at 500 m" is what the code says.

Sizes are deliberately the real ones -- a 0.35 m quadrotor at 500 m subtends 0.7 mrad, under one
Boson pixel at 0.86 mrad -- so the demo shows what the range problem actually looks like instead
of a comfortable blob. Below about one pixel the renderer's own sampling stops being the right
model and MS.6's analytic point-target injection takes over; the stage marks which targets are in
that regime (:attr:`DemoTarget.subpixel_at`) rather than leaving it to be discovered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DemoTarget",
    "AerialDemoScene",
    "DEMO_TARGETS",
    "build_aerial_demo",
    "camera_space_position",
    "analytic_targets",
    "describe",
]

#: The stage's targets. ``material`` is written as a ``thermal:material`` override so the mapping
#: is exact and does not depend on how anyone names a shader (ADR 0047 precedence rule 1), and
#: ``thermal_node`` names the solver in the scene config that owns its temperature.
#:
#: Ranges span the regime phase 1 cares about: a resolved target, one near the pixel limit, and
#: one well below it. The bird is there because a bird is the false positive every drone detector
#: has to survive, and in LWIR it is warm, small and roughly the same size as a small quadrotor.
DEMO_TARGETS: tuple[tuple[str, float, float, float, float, str, str], ...] = (
    # name,          range_m, size_m, az_deg, el_deg, material,                   thermal node
    ("drone_near", 120.0, 0.35, -6.0, 3.0, "painted_composite", "airframe"),
    ("drone_mid", 500.0, 0.35, 0.0, 6.0, "carbon_fibre", "airframe"),
    ("drone_far", 1500.0, 0.35, 6.5, 4.0, "painted_composite", "airframe"),
    ("aircraft", 2500.0, 12.0, -9.0, 9.0, "aircraft_aluminium_painted", "airframe"),
    ("bird", 250.0, 0.30, 3.0, 1.0, "propeller_rubber", "airframe"),
    ("motor_pod", 120.0, 0.08, -5.4, 3.0, "propeller_rubber", "engine"),
)


@dataclass(frozen=True)
class DemoTarget:
    """One target: where it is, how big, what it is made of, and which solver heats it."""

    name: str
    prim_path: str
    range_m: float
    size_m: float
    azimuth_deg: float
    elevation_deg: float
    material: str
    thermal_node: str

    def angular_size_mrad(self) -> float:
        return 1e3 * self.size_m / self.range_m

    def pixels_across(self, ifov_mrad: float) -> float:
        """How many native pixels the target spans at a given instantaneous field of view."""
        return self.angular_size_mrad() / ifov_mrad

    def subpixel_at(self, ifov_mrad: float) -> bool:
        """True when the renderer is no longer the right model for this target (MS.6)."""
        return self.pixels_across(ifov_mrad) < 1.0


@dataclass
class AerialDemoScene:
    """The built stage: what was authored, and everything a caller needs to drive it."""

    camera_path: str
    camera_tilt_deg: float
    up_axis: str
    targets: dict[str, DemoTarget]
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def prim_to_target(self) -> dict[str, str]:
        """The map ``AerialThermalBridge`` wants: prim path -> thermal node name."""
        return {t.prim_path: t.thermal_node for t in self.targets.values()}

    def boresight_elevation_deg(self) -> float:
        return self.camera_tilt_deg

    def elevation_of(self, name: str) -> float:
        """Absolute elevation above the horizon of one target, degrees."""
        return self.camera_tilt_deg + self.targets[name].elevation_deg


def camera_space_position(
    range_m: float, azimuth_deg: float, elevation_deg: float
) -> tuple[float, float, float]:
    """A point at ``range_m`` from a USD camera, ``azimuth`` right and ``elevation`` up of −Z.

    Right-handed and in the camera's own frame: +X right, +Y up, −Z forward. The caller rotates
    into world space by the camera's tilt; keeping the two steps apart is what lets a target be
    specified as a field angle rather than as three world coordinates nobody can check by eye.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        range_m * math.cos(el) * math.sin(az),
        range_m * math.sin(el),
        -range_m * math.cos(el) * math.cos(az),
    )


def _rotate_x(point: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    """Rotate about +X by ``degrees``; a positive angle tilts the camera's −Z view upward."""
    a = math.radians(degrees)
    x, y, z = point
    return (x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))


def build_aerial_demo(
    *,
    camera_path: str = "/World/IrCamera",
    camera_tilt_deg: float = 8.0,
    targets: tuple[tuple[str, float, float, float, float, str, str], ...] = DEMO_TARGETS,
) -> AerialDemoScene:
    """Author the stage: a tilted camera and the target prims. No sky dome, no ground plane.

    Each target is a front-parallel quad facing the camera. A quad rather than a mesh of a real
    airframe because what this stage exists to exercise is range, size, material and the sky
    behind them; a detailed model would add silhouette structure the radiometry does not yet
    distinguish and would make the expected contrast impossible to state in a test.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Targets", "Xform")
    # A light for the *visible* render only. The infrared path never reads a colour AOV, so this
    # changes nothing about the IR frame; without it the companion RGB capture is a black image.
    UsdLux.DistantLight.Define(stage, "/World/SunForRgb").CreateIntensityAttr(2000.0)

    errors: dict[str, str] = {}
    built: dict[str, DemoTarget] = {}
    for name, range_m, size_m, az, el, material, node in targets:
        path = f"/World/Targets/{name}"
        centre = _rotate_x(camera_space_position(range_m, az, el), camera_tilt_deg)
        half = 0.5 * size_m
        mesh = UsdGeom.Mesh.Define(stage, path)
        # Face the camera: the quad lies in the plane perpendicular to the line of sight only
        # approximately (it is axis-aligned), which at these field angles is within a degree.
        cx, cy, cz = centre
        mesh.GetPointsAttr().Set(
            [
                Gf.Vec3f(cx - half, cy - half, cz),
                Gf.Vec3f(cx + half, cy - half, cz),
                Gf.Vec3f(cx + half, cy + half, cz),
                Gf.Vec3f(cx - half, cy + half, cz),
            ]
        )
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.GetDoubleSidedAttr().Set(True)
        mesh.GetSubdivisionSchemeAttr().Set("none")
        try:
            attr = mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String)
            attr.Set(material)
        except Exception as exc:  # noqa: BLE001 - a failed override is data, not a crash
            errors[f"{name}:override"] = f"{type(exc).__name__}: {exc}"

        built[name] = DemoTarget(
            name=name,
            prim_path=path,
            range_m=float(range_m),
            size_m=float(size_m),
            azimuth_deg=float(az),
            elevation_deg=float(el),
            material=material,
            thermal_node=node,
        )

    camera = UsdGeom.Camera.Define(stage, camera_path)
    # The camera sits at the origin and is tilted up; `IrCamera` writes its optics, not its pose.
    camera.AddRotateXOp().Set(float(camera_tilt_deg))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 1.0e5))

    return AerialDemoScene(
        camera_path=camera_path,
        camera_tilt_deg=float(camera_tilt_deg),
        up_axis="Y",
        targets=built,
        errors=errors,
    )


def describe(scene: AerialDemoScene, ifov_mrad: float) -> list[dict[str, Any]]:
    """A row per target for a report: range, angular size, pixels across, and the sub-pixel flag."""
    return [
        {
            "name": t.name,
            "range_m": t.range_m,
            "size_m": t.size_m,
            "elevation_deg": round(scene.elevation_of(t.name), 2),
            "mrad": round(t.angular_size_mrad(), 4),
            "pixels": round(t.pixels_across(ifov_mrad), 3),
            "subpixel": t.subpixel_at(ifov_mrad),
            "material": t.material,
        }
        for t in scene.targets.values()
    ]


def analytic_targets(scene: AerialDemoScene, ifov_mrad: float, *, hide: bool = True) -> list[Any]:
    """Split the stage at one native pixel: hide the sub-pixel prims and return their MS.6 specs.

    This is the whole of the handover. Below one pixel the renderer samples geometry and gets a
    phase-dependent fraction of the flux (ADR 0071); the analytic path computes the pixel-averaged
    excess exactly. What makes it safe is that the two are **mutually exclusive** -- a prim handed
    to the analytic path is made invisible here, in the same call that produces its spec, so the
    two cannot drift apart in a caller that remembers one and forgets the other.

    ``hide=False`` returns the specs without touching the stage. That is for measuring the
    double-count this design prevents, not for production: with both paths live the target is
    rendered *and* injected and reads too bright by a phase-dependent amount.
    """
    import omni.usd
    from pxr import UsdGeom

    from irsim_isaac.pipeline.ir_camera import AnalyticTarget

    stage = omni.usd.get_context().get_stage()
    out: list[Any] = []
    for target in scene.targets.values():
        if not target.subpixel_at(ifov_mrad):
            continue
        centre = _rotate_x(
            camera_space_position(target.range_m, target.azimuth_deg, target.elevation_deg),
            scene.camera_tilt_deg,
        )
        if hide:
            prim = stage.GetPrimAtPath(target.prim_path)
            if prim and prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        out.append(
            AnalyticTarget(
                name=target.name,
                world_position=centre,
                # The projected area of a front-parallel square; the analytic path needs the area
                # the camera sees, not the object's total surface.
                area_m2=target.size_m**2,
                material=target.material,
                thermal_node=target.thermal_node,
            )
        )
    return out
