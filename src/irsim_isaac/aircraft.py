"""A parametric rear-engined business jet: a skin, two nacelles, two hot nozzles (ADR 0075).

The counterpart to :mod:`irsim_isaac.quadrotor`, and the thermal story is the opposite shape. A
multirotor's heat is spread over four motors that are a sixteenth of its span, so at close range
you resolve them and watch them warm. A jet's heat is concentrated in an exhaust nozzle a
*fortieth* of its span but hundreds of kelvin above ambient, and the skin around it is warmed
almost uniformly by aerodynamic heating. So the jet's signature is a near-point source on a warm
outline, and the thing that changes it is not throttle -- it is **aspect**. The nozzle is hidden
behind its own nacelle from the front and fully exposed from the rear, which is why an aircraft's
measured infrared signature varies by a large factor around the clock and why rear-aspect
detection ranges are the ones quoted.

Rear-fuselage engines rather than under-wing pods, deliberately: it puts both nozzles on the
centreline where one aspect angle governs both, so the occlusion is a clean function of viewing
angle instead of a wing-shadowing problem. It is also the usual layout for the size of aircraft
this represents.

Sizes are a light business jet: 16 m span, 18 m long, 1.1 m nacelles, 0.8 m nozzles. Through a
Boson's 0.857 mrad pixel at 500 m that is 37 px of span and **1.9 px of nozzle** -- which is the
honest answer and the interesting one. You do not resolve the hot part; you see a point source
that dominates the frame anyway because it is 400 K above its background.

**What is not modelled.** No exhaust plume: the gas behind the nozzle radiates in CO2 and H2O
bands rather than as a grey surface (§6.6 notes this for automotive exhaust) and it is not a
surface the renderer can carry. No leading-edge stagnation hot spot -- the recovery temperature
here is the flat-plate one, applied to the whole skin, where a real leading edge runs a few kelvin
hotter still. No engine intake, which from the front is a cavity that reads warmer than the skin.

docs/physics-model.md §6.6 (by extension; the spec has no aircraft section), ADR 0075
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from irsim_isaac.airframe import Part, author_parts

__all__ = ["AircraftSpec", "LIGHT_JET", "author_aircraft"]


@dataclass(frozen=True)
class AircraftSpec:
    """A fixed-wing aircraft by its real dimensions. Body frame +X right, +Y up, **-Z forward**.

    ``nozzle_material`` is the one real compromise in this airframe: a turbofan nozzle is oxidised
    Inconel or titanium and the library has no high-temperature alloy, so the painted-aluminium
    entry stands in for it at eps = 0.90 LWIR. That is close to a hot oxidised metal and very far
    from the eps = 0.09 of the *bare* aluminium a careless mapping would reach for, which would
    render an 800 K nozzle as barely warm.
    """

    span_m: float = 16.0
    length_m: float = 18.0
    fuselage_diameter_m: float = 1.9
    wing_chord_m: float = 2.2
    wing_thickness_m: float = 0.26
    tailplane_span_m: float = 6.0
    tailplane_chord_m: float = 1.3
    fin_height_m: float = 3.0
    fin_chord_m: float = 1.8
    nacelle_diameter_m: float = 1.1
    nacelle_length_m: float = 3.2
    nacelle_offset_m: float = 1.65
    nozzle_diameter_m: float = 0.8
    nozzle_length_m: float = 0.5
    skin_material: str = "aircraft_aluminium_painted"
    nacelle_material: str = "aircraft_aluminium_painted"
    nozzle_material: str = "aircraft_aluminium_painted"

    def __post_init__(self) -> None:
        for name in ("span_m", "length_m", "fuselage_diameter_m", "nozzle_diameter_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.nozzle_diameter_m > self.nacelle_diameter_m:
            raise ValueError(
                "the nozzle cannot be wider than the nacelle it exhausts through -- if it were, "
                "it would never be occluded and the aspect dependence this stage measures would "
                "disappear"
            )

    def parts(self) -> tuple[Part, ...]:
        """Every part in the body frame. Pure arithmetic: no engine, no stage, SI throughout.

        The fuselage runs along Z with the nose at -Z. Nacelles sit either side of the rear
        fuselage and each nozzle caps its nacelle at the **+Z** (aft) end, so from ahead the
        nacelle body is between the camera and the nozzle.
        """
        half_len = 0.5 * self.length_m
        tail_z = half_len - self.fin_chord_m
        nacelle_z = half_len - 0.5 * self.nacelle_length_m - 1.2
        nozzle_z = nacelle_z + 0.5 * (self.nacelle_length_m + self.nozzle_length_m)
        parts: list[Part] = [
            Part(
                "fuselage",
                "cylinder",
                (0.0, 0.0, 0.0),
                (self.fuselage_diameter_m, self.fuselage_diameter_m, self.length_m),
                self.skin_material,
                "skin",
                axis="Z",
            ),
            Part(
                "wing",
                "box",
                (0.0, -0.3, 0.6),
                (self.span_m, self.wing_thickness_m, self.wing_chord_m),
                self.skin_material,
                "skin",
            ),
            Part(
                "tailplane",
                "box",
                (0.0, 0.5 * self.fin_height_m, tail_z + 0.5 * self.tailplane_chord_m),
                (self.tailplane_span_m, 0.18, self.tailplane_chord_m),
                self.skin_material,
                "skin",
            ),
            Part(
                "fin",
                "box",
                (0.0, 0.25 * self.fin_height_m, tail_z),
                (0.22, self.fin_height_m, self.fin_chord_m),
                self.skin_material,
                "skin",
            ),
        ]
        for index, side in ((0, -1.0), (1, 1.0)):
            offset = side * self.nacelle_offset_m
            parts.append(
                Part(
                    f"nacelle_{index}",
                    "cylinder",
                    (offset, 0.35, nacelle_z),
                    (self.nacelle_diameter_m, self.nacelle_diameter_m, self.nacelle_length_m),
                    self.nacelle_material,
                    "nacelle",
                    axis="Z",
                )
            )
            parts.append(
                Part(
                    f"nozzle_{index}",
                    "cylinder",
                    (offset, 0.35, nozzle_z),
                    (self.nozzle_diameter_m, self.nozzle_diameter_m, self.nozzle_length_m),
                    self.nozzle_material,
                    "nozzle",
                    axis="Z",
                )
            )
        return tuple(parts)

    def thermal_nodes(self) -> tuple[str, ...]:
        """The solver names a scene has to define for this aircraft to have a temperature."""
        return tuple(sorted({p.thermal_node for p in self.parts()}))


#: A light business jet. At 500 m through a Boson pixel: 37 px of span, 1.9 px of nozzle.
LIGHT_JET = AircraftSpec()


def author_aircraft(
    stage: Any,
    root_path: str,
    spec: AircraftSpec = LIGHT_JET,
    *,
    look_binder: Any = None,
) -> dict[str, str]:
    """Author the aircraft under ``root_path``; returns prim path -> thermal node."""
    return author_parts(stage, root_path, spec.parts(), look_binder=look_binder)
