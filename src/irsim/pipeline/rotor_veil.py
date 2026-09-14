"""Stage 2c — rotor discs composited onto the k× radiance plane (ADR 0081).

A spinning rotor is not geometry (`irsim.optics.rotor`): what the detector records is the time
average of an intermittent opaque occluder, a veil covering each pixel for the fraction `alpha` of
the integration window that blade material stood in the way. This module is where that veil meets
the frame.

**It composites, it does not inject an excess.** MS.6's point targets
(`irsim.pipeline.point_target`) add `phi tau (L_t - L_beyond)` because a sub-pixel target occults a
*sky column* whose radiance the plane does not carry separately. A veil is different: the plane
already holds the correctly attenuated background at every pixel -- sky beyond the disc for some,
the aircraft's own arm or motor bell for others, each having travelled its own path -- so the right
operation is the blend itself,

    L = L_plane + alpha (L_blade,at-sensor - L_plane)

with the blade's own radiance carried through the atmosphere at the **disc's** range. That is exact
for both cases at once and needs no decision about what is behind, which the excess form would
have: applied with a `sky_beyond` term over a pixel where the disc veils the airframe, it would
subtract a sky column that is not there.

**The blade's path is stage 2's, reused rather than re-derived.** `blade_radiance_at_sensor` calls
the same `apply_layered_gbuffer` / `apply_atmosphere` the plane went through, on a one-element
array, so the veil and the pixels under it cannot disagree about the atmosphere.

**Windowed.** A disc is a small ellipse on a large grid, and `coverage_map` allocates per call; a
4x supersampled Boson frame is 2560 x 2048, so a full-frame map per rotor would be 40 MB of
float64 each and four of them per frame. The coverage is built on the ellipse's bounding box only.

docs/physics-model.md §13.4 stage 2, §8.3; ADR 0081
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.optics.rotor import DiscEllipse, RotorDisc, coverage_map, veil_radiance
from irsim.pipeline.atmosphere import apply_layered_gbuffer
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = [
    "RotorVeil",
    "blade_radiance_at_sensor",
    "veil_window",
    "inject_rotor_veils",
]


@dataclass(frozen=True)
class RotorVeil:
    """One rotor, projected and ready to composite.

    ``ellipse`` is in **k× supersampled pixels** -- build it with
    ``Intrinsics.from_sensor(spec, supersample=k)`` so the grid the coverage is drawn on is the
    grid the plane is on. ``blade_radiance`` is the band radiance the blade shows at the disc,
    already ``eps L_B(T) + (1 - eps) L_env`` for the blade material; the atmosphere between the
    disc and the sensor is applied here.

    ``occluded`` is a k×-grid mask, ``True`` where something opaque stands in **front** of the disc
    plane. Without it the veil paints a blade over the motor bell it is bolted to.
    """

    disc: RotorDisc
    ellipse: DiscEllipse
    swept_rad: float
    blade_radiance: float
    range_m: float
    phase_rad: float = 0.0
    occluded: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.range_m <= 0.0:
            raise ValueError("range_m must be positive")
        if self.blade_radiance < 0.0:
            raise ValueError("blade_radiance must be non-negative")
        if self.swept_rad < 0.0:
            raise ValueError("swept_rad must be non-negative")


def blade_radiance_at_sensor(
    veil: RotorVeil,
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity = "lb",
) -> float:
    """The blade's band radiance as it arrives, through the same stage-2 path the plane took."""
    l_blade = np.array([[veil.blade_radiance]], dtype=np.float64)
    distance = np.array([[veil.range_m]], dtype=np.float64)
    if atmosphere is None:
        return float(veil.blade_radiance)
    if isinstance(atmosphere, LayeredAtmosphere):
        out = apply_layered_gbuffer(atmosphere, band, t_s, l_blade, distance, quantity)
        return float(np.asarray(out)[0, 0])
    state = atmosphere.state(t_s)
    l_air = float(lut.lookup(np.float64(state.t_air_k), quantity)[()])
    out = apply_atmosphere(l_blade, distance, state.gamma_per_m[band], l_air)
    return float(np.asarray(out)[0, 0])


def veil_window(ellipse: DiscEllipse, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Bounding box ``(y0, y1, x0, x1)`` of the ellipse clipped to the grid, or ``None`` if off it.

    The box is built from the semi-major axis in both directions rather than from the rotated
    extent: it is at most 41 % larger in area and costs one subtraction, where getting the rotated
    bound wrong costs a clipped disc that nothing would flag.
    """
    height, width = int(shape[0]), int(shape[1])
    reach = math.ceil(ellipse.semi_major_px) + 1
    cx, cy = ellipse.centre_px
    x0 = max(0, int(math.floor(cx)) - reach)
    x1 = min(width, int(math.ceil(cx)) + reach)
    y0 = max(0, int(math.floor(cy)) - reach)
    y1 = min(height, int(math.ceil(cy)) + reach)
    if x0 >= x1 or y0 >= y1:
        return None
    return (y0, y1, x0, x1)


def inject_rotor_veils(
    radiance_ss: NDArray[np.floating],
    veils: Sequence[RotorVeil],
    atmosphere: Atmosphere | LayeredAtmosphere | None,
    band: str,
    t_s: float,
    lut: BandLUT,
    quantity: Quantity = "lb",
) -> NDArray[np.floating]:
    """Composite every veil onto the post-stage-2 k× radiance plane, in place of nothing.

    Returns the plane unchanged (same object) when there are no veils, so a scene without rotors
    pays nothing and its goldens are untouched.
    """
    if not veils:
        return radiance_ss
    out: Any = np.array(radiance_ss, dtype=np.float64, copy=True)
    for veil in veils:
        window = veil_window(veil.ellipse, out.shape[:2])
        if window is None or veil.ellipse.semi_minor_px <= 0.0:
            continue
        y0, y1, x0, x1 = window
        alpha = coverage_map(
            veil.disc,
            (y1 - y0, x1 - x0),
            (veil.ellipse.centre_px[0] - x0, veil.ellipse.centre_px[1] - y0),
            veil.ellipse.semi_major_px,
            veil.ellipse.semi_minor_px,
            veil.swept_rad,
            rotation_deg=veil.ellipse.rotation_deg,
            phase_rad=veil.phase_rad,
        )
        if not alpha.any():
            continue
        l_blade = blade_radiance_at_sensor(veil, atmosphere, band, t_s, lut, quantity)
        mask = None if veil.occluded is None else np.asarray(veil.occluded)[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = veil_radiance(out[y0:y1, x0:x1], alpha, l_blade, occluded=mask)
    return np.asarray(out, dtype=radiance_ss.dtype)
