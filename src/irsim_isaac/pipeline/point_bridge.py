"""Per-pixel temperature from a solved surface field, keyed by the position AOV.

docs/physics-model.md §13.1, §13.3; roadmap MP.3; ADR 0087 (the field and why the position AOV is
the parameterisation), ADR 0014 (what this build actually transports per pixel).

`AerialThermalBridge.temperature_plane` gathers `table[instance_id]`: one temperature per prim,
which is the defect ADR 0087 exists to fix. This module is the other half of that fix on the render
path, and it is deliberately **additive** -- it takes a finished plane and overwrites only the
pixels belonging to prims that have a field bound to them. A prim with no binding keeps the
per-instance value it always had, so every existing scene renders bit-identically.

The lookup is the one ADR 0014 leaves available. Temperature cannot cross a colour AOV (all fp16),
but `Camera3dPositionSD` carries a **float32 position good to 3.4 mm**, which M2.4 established is
**camera** space on this build -- so :func:`world_positions` applies the same rotation
`ray_directions` does, from the same `camera_to_world`, rather than deriving a second one that
could disagree with it by twice the camera's tilt without raising anything.

**A pixel that lands on a bound prim but outside all of its patches raises.** It is the symptom of
a patch authored smaller than the geometry it is meant to cover, and the alternative -- taking the
prim's fallback value, or the nearest cell -- produces a frame in which part of a bonnet is a field
and part of it is a flat patch, with a seam that looks like physics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.surface_field import PlanarThermalField
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths

__all__ = ["SurfaceBinding", "PointwiseTemperature", "world_positions"]


def world_positions(
    position: Any,
    *,
    frame: str,
    camera_position: Any = None,
    camera_to_world: Any = None,
) -> NDArray[np.float64]:
    """``(H, W, 3)`` world-space surface positions from the position AOV.

    The world-space counterpart of :func:`~irsim_isaac.pipeline.gbuffer_isaac.ray_directions`,
    which needs only the *direction*: here the magnitude matters, because it is what selects a cell.
    ``frame="camera"`` (what M2.4 measured on this build) rotates by ``camera_to_world`` and adds
    the camera's own position; ``frame="world"`` passes through.
    """
    pos = np.asarray(position, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"position must be (H, W, 3), got {pos.shape}")
    if pos.dtype == np.float16:  # pragma: no cover - the reader never produces one
        raise TypeError("a float16 position plane cannot resolve a thermal cell (CLAUDE.md #2)")
    if frame == "world":
        return pos
    if frame != "camera":
        raise ValueError(f"unknown position frame {frame!r}")
    if camera_position is None:
        raise ValueError("frame='camera' needs camera_position to reach world space")
    out = pos
    if camera_to_world is not None:
        rot = np.asarray(camera_to_world, dtype=np.float64)
        if rot.shape == (4, 4):
            rot = rot[:3, :3]
        if rot.shape != (3, 3):
            raise ValueError(f"camera_to_world must be 3x3 or 4x4, got {rot.shape}")
        out = out @ rot.T
    return np.asarray(out + np.asarray(camera_position, dtype=np.float64).reshape(3))


@dataclass(frozen=True)
class SurfaceBinding:
    """One prim's temperature field. Several may share a prim -- a bonnet and a roof, say.

    ``field.patch.frame`` is a name the caller sets and this module **checks**: a patch authored in
    a prim's local frame cannot be sampled with world positions, and the two are indistinguishable
    from the array shapes alone.
    """

    prim_path: str
    field: PlanarThermalField

    def __post_init__(self) -> None:
        if not self.prim_path:
            raise ValueError("a binding needs a prim path")


class PointwiseTemperature:
    """Overwrites the patch-backed prims of a per-instance temperature plane with their fields."""

    def __init__(self, bindings: Sequence[SurfaceBinding]) -> None:
        self.bindings = tuple(bindings)
        for binding in self.bindings:
            if binding.field.patch.frame != "world":
                raise ValueError(
                    f"{binding.prim_path}: patch frame is {binding.field.patch.frame!r}; this "
                    "bridge samples with world positions, so its patches must be authored in "
                    "world space"
                )
        self._by_path: dict[str, list[PlanarThermalField]] = {}
        for binding in self.bindings:
            self._by_path.setdefault(binding.prim_path, []).append(binding.field)

    @property
    def prim_paths(self) -> tuple[str, ...]:
        return tuple(self._by_path)

    def advance_to(self, t_s: float) -> None:
        """Push every bound field to ``t_s``. The only method that changes anything."""
        for fields in self._by_path.values():
            for field in fields:
                field.advance_to(t_s)

    def apply(
        self,
        plane: Any,
        instance_ids: Any,
        id_to_labels: Mapping[Any, Any] | None,
        positions_world: Any,
        t_s: float,
        *,
        strict: bool = True,
    ) -> NDArray[np.float32]:
        """Return ``plane`` with every bound prim's pixels replaced by its field's own values.

        Untouched when no bound prim is on screen, which is what makes attaching this to an
        existing scene a no-op until a patch is authored for something in it.
        """
        out = np.array(plane, dtype=np.float32, copy=True)
        ids = np.asarray(instance_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"instance_ids must be an integer plane, got {ids.dtype}")
        if ids.shape != out.shape:
            raise ValueError(f"instance_ids {ids.shape} does not match the plane {out.shape}")
        points = np.asarray(positions_world, dtype=np.float64)
        if points.shape != (*out.shape, 3):
            raise ValueError(f"positions_world {points.shape} does not match the plane {out.shape}")

        paths = labels_to_paths(id_to_labels)
        for ident, path in paths.items():
            if ident == BACKGROUND_INSTANCE_ID:
                continue
            fields = self._by_path.get(path)
            if not fields:
                continue
            mask = ids == ident
            if not mask.any():
                continue
            selected = points[mask]
            filled = np.zeros(selected.shape[0], dtype=bool)
            values = np.zeros(selected.shape[0], dtype=np.float32)
            for field in fields:
                pending = ~filled
                if not pending.any():
                    break
                sampled = field.sample_at(t_s, selected[pending])
                hit = np.isfinite(sampled)
                idx = np.flatnonzero(pending)[hit]
                values[idx] = sampled[hit]
                filled[idx] = True
            if not filled.all() and strict:
                raise ValueError(
                    f"{int((~filled).sum())} pixels of {path!r} fall outside every patch bound to "
                    "it. A patch authored smaller than its geometry renders part of a surface as "
                    "a field and part as a flat value, with a seam that looks like physics; widen "
                    "the patch or its thickness_m rather than letting them take a fallback."
                )
            if not filled.all():
                # Non-strict: keep the per-instance value, which is what the pixel had before.
                values[~filled] = out[mask][~filled]
            out[mask] = values
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"PointwiseTemperature({len(self.bindings)} bindings over {len(self._by_path)} prims)"
        )
