"""Per-cell shadow: which parts of a surface the sun actually reaches (PT.1).

docs/physics-model.md §6.1 (where ``q_solar`` enters the balance), §5.2; ADR 0087 (the field).

ADR 0087's headline case is **a wall half in sun, spanning 10-20 K across one prim** -- the defect
that point-wise temperature exists to fix. Until this module, nothing could produce it. A patch gave
every cell its own temperature, but `SceneSurfaceForcing` carried one ``shaded`` bool per *surface*,
so the only spatial variation a field could express came from ``q_internal_w_m2`` -- an engine bay
under a bonnet. That is why both point-wise scenes shipped so far are **pre-dawn**: the machinery
could not have rendered a sunlit one differently from a flat surface.

**Shadow gates the direct beam only.** A shaded cell still sees diffuse sky through its own view
factor, and zeroing all solar in shade is the usual shortcut that renders shaded surfaces far too
cold. :func:`~irsim.thermal.solar.solar_loading` already has that split; this module supplies the
per-cell ``shadow`` argument it has always accepted and nothing has ever varied.

**Rectangles, hard-edged, single-bounce.** An occluder is a rectangle -- the same primitive ADR 0088
uses for a radiator, because a slab, a parapet, a wing or a container reads as one. The test is an
exact ray-rectangle intersection, so a cell is lit or not lit with nothing in between: no penumbra
(the sun's 0.53 deg disc would soften the edge over ~1 cm per metre of standoff), and no light
bouncing off the occluder back onto the shaded part. Both make the shaded side slightly warmer than
this model says, and both are recorded rather than hidden -- the step across the terminator is the
quantity of interest and it is a lower bound here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.surface_field import PlanarPatch

__all__ = ["ShadowRectangle", "cell_shadow", "patch_solar_loading"]


@dataclass(frozen=True)
class ShadowRectangle:
    """A rectangular opaque occluder, in the same frame as the patch it shades."""

    centre_m: NDArray[np.float64]
    u_axis: NDArray[np.float64]
    v_axis: NDArray[np.float64]
    half_u_m: float
    half_v_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre_m", np.asarray(self.centre_m, dtype=np.float64).reshape(3))
        for name in ("u_axis", "v_axis"):
            v = np.asarray(getattr(self, name), dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(v))
            if norm <= 0.0:
                raise ValueError(f"{name} has zero length")
            object.__setattr__(self, name, v / norm)
        if abs(float(np.dot(self.u_axis, self.v_axis))) > 1e-9:
            raise ValueError("u_axis and v_axis must be perpendicular")
        if self.half_u_m <= 0.0 or self.half_v_m <= 0.0:
            raise ValueError("half extents must be positive")

    @property
    def normal(self) -> NDArray[np.float64]:
        return np.asarray(np.cross(self.u_axis, self.v_axis))


def cell_shadow(
    patch: PlanarPatch,
    sun_direction: Any,
    occluders: Sequence[ShadowRectangle] = (),
) -> NDArray[np.float64]:
    """``(n_cells,)`` direct-beam visibility: 1.0 where the sun reaches the cell, 0.0 where not.

    ``sun_direction`` points **toward** the sun, in the patch's own frame -- the same convention
    :func:`~irsim.thermal.solar.sun_direction` uses for ENU. Converting a scene's sun vector into
    the patch's frame is the caller's job and is deliberately not guessed at here: a patch declares
    a ``frame`` name, not a transform.

    A cell facing away from the sun is returned as lit. That is not a bug and it matters: the
    ``max(0, n·s)`` in `solar_loading` is what zeroes the beam for a back-facing surface, and
    duplicating it here would make a self-shadowing test look like an occlusion result.
    """
    s = np.asarray(sun_direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(s))
    if norm <= 0.0:
        raise ValueError("sun_direction has zero length")
    s = s / norm

    centres = patch.cell_centres()
    lit = np.ones(centres.shape[0], dtype=np.float64)
    for rect in occluders:
        denominator = float(np.dot(s, rect.normal))
        if abs(denominator) < 1e-12:
            # The beam runs along the occluder's plane: it can graze but not be stopped by it.
            continue
        # Distance along the ray from each cell to the occluder's plane.
        t = ((rect.centre_m - centres) @ rect.normal) / denominator
        ahead = t > 1e-9  # behind the cell is behind the sun's side; it cannot shade
        hit = centres + t[:, None] * s
        offset = hit - rect.centre_m
        inside = (np.abs(offset @ rect.u_axis) <= rect.half_u_m) & (
            np.abs(offset @ rect.v_axis) <= rect.half_v_m
        )
        lit[ahead & inside] = 0.0
    return lit


def patch_solar_loading(
    patch: PlanarPatch,
    sun_direction: Any,
    dni_w_m2: float,
    dhi_w_m2: float,
    sky_view_factor: Any,
    occluders: Sequence[ShadowRectangle] = (),
) -> NDArray[np.float64]:
    """``(n_cells,)`` Q_sol for a patch, with the direct beam gated per cell (§6.1).

    The whole point of the module in one call: the same `solar_loading` every surface has always
    used, handed a shadow array that varies across the surface instead of one bool for all of it.
    """
    from irsim.thermal.solar import solar_loading

    shade = cell_shadow(patch, sun_direction, occluders)
    normals = np.broadcast_to(patch.normal, (patch.n_cells, 3))
    return np.asarray(
        solar_loading(normals, sun_direction, dni_w_m2, dhi_w_m2, sky_view_factor, shade)
    )
