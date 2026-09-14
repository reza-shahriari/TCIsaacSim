"""Rigid parts and the one place that turns them into USD prims (ADR 0074, ADR 0075).

Every parametric airframe in this repository -- the quadrotor, the fixed-wing aircraft -- is a list
of :class:`Part`, and :func:`author_parts` is the only code that writes one to a stage. That
matters less for the geometry than for the two things attached to it: the ``thermal:material``
override that gives a surface an emissivity, and the prim-path-to-thermal-node map that gives it a
temperature. A second authoring path would be a second chance to get one of those half-right, and
a surface with no material or no temperature is precisely the silent failure this project exists
to avoid.

The body frame is the stage frame of a level aircraft: **+X right, +Y up, -Z forward**, the same
convention as :func:`irsim_isaac.aerial_demo.camera_space_position`. Whatever flies the airframe
puts one transform on the root and the parts come along.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["Kind", "Part", "author_parts"]

Kind = Literal["box", "cylinder"]


@dataclass(frozen=True)
class Part:
    """One rigid part: where it sits in the body frame, how big, what it is, what heats it.

    ``size_m`` is the full extent of a box. For a cylinder it is the bounding extent too, with the
    two cross-axis entries the diameter and the ``axis`` entry the length -- so a fuselage 1.9 m
    across and 16 m long on the Z axis is ``(1.9, 1.9, 16.0)`` with ``axis="Z"``. Writing it as an
    extent rather than as (radius, length) is what lets :meth:`pixels_across` and every bounding
    check treat boxes and cylinders alike.

    ``rotate_xyz_deg`` is applied about the part's own centre, before the root transform.
    """

    name: str
    kind: Kind
    centre_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    material: str
    thermal_node: str
    axis: Literal["X", "Y", "Z"] = "Y"
    rotate_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a part needs a name")
        if any(v <= 0.0 for v in self.size_m):
            raise ValueError(f"{self.name}: every extent must be positive, got {self.size_m}")
        if not self.material or not self.thermal_node:
            raise ValueError(
                f"{self.name}: a part needs both a material and a thermal node -- a surface with "
                "no emissivity, or none with no temperature, is not renderable physics"
            )

    def largest_dimension_m(self) -> float:
        return max(self.size_m)

    def pixels_across(self, range_m: float, ifov_mrad: float) -> float:
        """How many native pixels the part's largest dimension spans at a range."""
        return 1e3 * self.largest_dimension_m() / range_m / ifov_mrad


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def author_parts(
    stage: Any,
    root_path: str,
    parts: tuple[Part, ...],
    *,
    look_binder: Any = None,
) -> dict[str, str]:
    """Write the parts under ``root_path``; returns prim path -> thermal node.

    The returned map is exactly what
    :class:`~irsim_isaac.pipeline.aerial_bridge.AerialThermalBridge` wants, so a caller never
    writes the correspondence out by hand. ``look_binder(stage, prim, material)`` is the optional
    visible-band hook (:func:`irsim_isaac.stage.bind_visible_look`).
    """
    from pxr import Gf, Sdf, UsdGeom

    UsdGeom.Xform.Define(stage, root_path)
    prim_to_node: dict[str, str] = {}
    for part in parts:
        path = f"{root_path}/{part.name}"
        extent = part.size_m
        if part.kind == "cylinder":
            along = _AXIS_INDEX[part.axis]
            # The two cross-axis extents are the diameter; they must agree or the caller has
            # asked for an ellipse, which `UsdGeom.Cylinder` cannot be.
            cross = [v for i, v in enumerate(extent) if i != along]
            if abs(cross[0] - cross[1]) > 1e-9:
                raise ValueError(
                    f"{part.name}: a cylinder's two cross-axis extents are its diameter and must "
                    f"match, got {extent} about {part.axis}"
                )
            gprim = UsdGeom.Cylinder.Define(stage, path)
            gprim.CreateAxisAttr(getattr(UsdGeom.Tokens, part.axis.lower()))
            gprim.CreateRadiusAttr(0.5 * cross[0])
            gprim.CreateHeightAttr(extent[along])
        else:
            # A unit cube scaled per axis: `UsdGeom.Cube` has one `size`, so the extents have to
            # come from a scale op rather than from the attribute.
            gprim = UsdGeom.Cube.Define(stage, path)
            gprim.CreateSizeAttr(1.0)
        gprim.CreateExtentAttr(
            [
                Gf.Vec3f(*(-0.5 * v for v in extent)),
                Gf.Vec3f(*(0.5 * v for v in extent)),
            ]
        )

        xform = UsdGeom.Xformable(gprim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*part.centre_m))
        if any(part.rotate_xyz_deg):
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*part.rotate_xyz_deg))
        if part.kind == "box":
            xform.AddScaleOp().Set(Gf.Vec3f(*extent))

        prim = gprim.GetPrim()
        prim.CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(part.material)
        if look_binder is not None:
            look_binder(stage, gprim, part.material)
        prim_to_node[path] = part.thermal_node
    return prim_to_node
