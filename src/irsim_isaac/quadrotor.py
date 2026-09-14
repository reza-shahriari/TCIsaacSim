"""A parametric heavy-lift quadrotor: the parts a thermal camera actually separates (ADR 0074).

Built from primitives in code rather than imported as a mesh, and the reason is thermal rather
than aesthetic. What an infrared sensor keys on in a multirotor is not its silhouette -- at any
range where a drone is a threat it is a handful of pixels -- it is that the **motors are hot, the
speed controllers are warm, the battery is warm, and the airframe is at air temperature**. That is
four thermal nodes on one object, and ADR 0072 already models three of them
(:mod:`irsim.thermal.aerial`: ΔT = ΔT_max u^2 above the shared weather's T_air).

Getting that from a downloaded mesh means hand-assigning a material and a thermal node to whatever
sub-prims an artist happened to make, and re-doing it for the next mesh. Getting it from a
parameterised layout means every part is separate by construction, its dimensions are *stated*
rather than measured off a bounding box, and the angular size a test asserts is the angular size
the geometry has. There is no licence to track either.

The layout is engine-free and so are its sizes: :meth:`QuadrotorSpec.parts` is pure arithmetic and
is unit-tested without Isaac Sim. Only :func:`author_quadrotor` touches USD.

**Propellers are deliberately absent.** A 28-inch prop at flight rpm sweeps its whole disc many
times within a 60 Hz integration period, so what a thermal camera records is a faint, low-contrast
annulus, not a solid blade -- and a static disc of the right diameter would be 33 px across at
20 m and would hide the motors underneath it, which are the entire subject. Modelling the smear
properly needs the motion path (M10.1b); until then, nothing is more honest than nothing.

docs/physics-model.md §6.6, §16.2; ADR 0072 (the heat sources), ADR 0074 (this airframe)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from irsim_isaac.airframe import Part, author_parts

__all__ = [
    "Part",
    "QuadrotorSpec",
    "HEAVY_LIFT",
    "author_quadrotor",
]


@dataclass(frozen=True)
class QuadrotorSpec:
    """A quadrotor by its real dimensions. Defaults are a heavy-lift class (see :data:`HEAVY_LIFT`).

    ``span_m`` is motor axis to motor axis across the diagonal, which is how multirotor frames are
    quoted. Motors sit on the diagonals at 45 degrees from forward, so each is ``span/(2 sqrt 2)``
    out along both body axes.
    """

    span_m: float = 1.8
    body_m: tuple[float, float, float] = (0.34, 0.14, 0.34)
    arm_thickness_m: float = 0.05
    motor_diameter_m: float = 0.11
    motor_height_m: float = 0.07
    esc_m: tuple[float, float, float] = (0.09, 0.03, 0.06)
    esc_fraction: float = 0.45  # where along the arm the speed controller sits
    battery_m: tuple[float, float, float] = (0.22, 0.07, 0.15)
    body_material: str = "carbon_fibre"
    # Anodised or painted motor bells, eps ~ 0.9 in LWIR. **Bare** aluminium is eps = 0.09 in this
    # library, and a 336 K motor behind it would read barely above the reflected sky -- the effect
    # the aircraft_aluminium_painted / bare_aluminium pair exists to catch. Real bells are
    # anodised, so the high-emissivity material is the right default, not the flattering one.
    motor_material: str = "aircraft_aluminium_painted"
    esc_material: str = "painted_composite"
    battery_material: str = "painted_composite"

    def __post_init__(self) -> None:
        if self.span_m <= 0.0:
            raise ValueError("span_m must be positive")
        if not 0.0 < self.esc_fraction < 1.0:
            raise ValueError("esc_fraction must lie strictly between the hub and the motor")
        if self.motor_diameter_m <= 0.0 or self.motor_height_m <= 0.0:
            raise ValueError("motor dimensions must be positive")

    def motor_offset_m(self) -> float:
        """Distance of each motor from the hub along one body axis."""
        return self.span_m / (2.0 * math.sqrt(2.0))

    def arm_length_m(self) -> float:
        """Hub centre to motor centre, along the diagonal."""
        return 0.5 * self.span_m

    def parts(self) -> tuple[Part, ...]:
        """Every part, in the body frame. Pure arithmetic -- no engine, no stage, no units but SI.

        The four arms are boxes laid along the diagonals, so each is authored at 45 degrees; the
        caller applies that rotation (:func:`author_quadrotor`). Arms, hub and canopy share the
        ``airframe`` node because forced convection at flight speed pins an unpowered carbon skin
        to the air it is flying through (ADR 0072); everything with a current through it does not.
        """
        offset = self.motor_offset_m()
        arm = self.arm_length_m()
        out: list[Part] = [
            Part("body", "box", (0.0, 0.0, 0.0), self.body_m, self.body_material, "airframe"),
            Part(
                "battery",
                "box",
                (0.0, -0.5 * (self.body_m[1] + self.battery_m[1]), 0.0),
                self.battery_m,
                self.battery_material,
                "battery",
            ),
        ]
        for index, (sx, sz) in enumerate(((1, -1), (-1, -1), (-1, 1), (1, 1))):
            # The arm as an un-rotated box along +X, of the full diagonal length; the authoring
            # step yaws it into place, so the layout arithmetic stays one-dimensional.
            out.append(
                Part(
                    f"arm_{index}",
                    "box",
                    # Midway between the hub and the motor, on both body axes: the box is the full
                    # diagonal long, so its centre sits at half the motor's offset, not half the
                    # arm's length. Getting that wrong leaves a gap at the hub and overshoots the
                    # motor by the same amount, which looks plausible from every angle.
                    (0.5 * offset * sx, 0.0, 0.5 * offset * sz),
                    (arm, self.arm_thickness_m, self.arm_thickness_m),
                    self.body_material,
                    "airframe",
                    # Yawed onto its diagonal. The layout put the centre there; this turns the box
                    # so it points at the motor instead of along +X.
                    rotate_xyz_deg=(0.0, math.degrees(math.atan2(-sz, sx)), 0.0),
                )
            )
            out.append(
                Part(
                    f"motor_{index}",
                    "cylinder",
                    (
                        offset * sx,
                        0.5 * (self.body_m[1] + self.motor_height_m),
                        offset * sz,
                    ),
                    (self.motor_diameter_m, self.motor_height_m, self.motor_diameter_m),
                    self.motor_material,
                    "motor",
                )
            )
            out.append(
                Part(
                    f"esc_{index}",
                    "box",
                    (
                        offset * sx * self.esc_fraction,
                        0.5 * (self.body_m[1] + self.esc_m[1]),
                        offset * sz * self.esc_fraction,
                    ),
                    self.esc_m,
                    self.esc_material,
                    "esc",
                )
            )
        return tuple(out)

    def thermal_nodes(self) -> tuple[str, ...]:
        """The solver names a scene has to define for this airframe to have a temperature."""
        return tuple(sorted({p.thermal_node for p in self.parts()}))


#: A heavy-lift multirotor: 1.8 m diagonal, 110 mm motor bells. At 20 m through a Boson's
#: 0.857 mrad pixel that is 105 px across the span and 6.4 px across each motor -- resolved enough
#: that the four hot spots are unmistakably four hot spots, which is the point of the demo.
HEAVY_LIFT = QuadrotorSpec()


def author_quadrotor(
    stage: Any,
    root_path: str,
    spec: QuadrotorSpec = HEAVY_LIFT,
    *,
    look_binder: Any = None,
) -> dict[str, str]:
    """Author the airframe under ``root_path``; returns prim path -> thermal node."""
    return author_parts(stage, root_path, spec.parts(), look_binder=look_binder)
