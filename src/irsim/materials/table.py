"""Scalar-per-band material table: material id → in-band emissivity ε₀.

The minimal table the radiance kernel needs (docs/physics-model.md §13.5 ``matEps``): one grey
emissivity per material id for the current band, derived from spectral data by
:func:`irsim.radiometry.band_average.band_average` (ADR 0010) or authored directly for tests.
Id **0 is the UNMAPPED sentinel** (an asset the material resolver could not map); a kernel that
meets it must fail, not silently render a default. The angular model, reflectance and
transmittance columns arrive with the material milestone (M7.18 / M7.10).

docs/physics-model.md §4.1, §12.3, §13.5
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["UNMAPPED_MATERIAL_ID", "MaterialTable"]

UNMAPPED_MATERIAL_ID = 0


@dataclass(frozen=True)
class MaterialTable:
    """Dense id → ε₀ table for one band. Ids are small non-negative integers."""

    emissivity: NDArray[np.float32]  # index = material id; NaN where undefined
    band_id: str = ""

    @classmethod
    def from_mapping(cls, eps_by_id: Mapping[int, float], band_id: str = "") -> MaterialTable:
        if not eps_by_id:
            raise ValueError("material table needs at least one material")
        if UNMAPPED_MATERIAL_ID in eps_by_id:
            raise ValueError(
                "material id 0 is the UNMAPPED sentinel and cannot carry an emissivity"
            )
        max_id = max(eps_by_id)
        if min(eps_by_id) < 0 or max_id > 65535:
            raise ValueError("material ids must lie in 1..65535")
        table = np.full(max_id + 1, np.nan, dtype=np.float32)
        for mid, eps in eps_by_id.items():
            if not 0.0 < eps <= 1.0:
                raise ValueError(f"material {mid}: emissivity {eps} must lie in (0, 1]")
            table[mid] = np.float32(eps)
        return cls(emissivity=table, band_id=band_id)

    @classmethod
    def constant(cls, eps: float, ids: tuple[int, ...] = (1,), band_id: str = "") -> MaterialTable:
        return cls.from_mapping(dict.fromkeys(ids, eps), band_id)

    def emissivity_for(
        self, material_id: NDArray[np.integer], sky_mask: NDArray[np.bool_] | None = None
    ) -> NDArray[np.float32]:
        """Per-pixel ε₀; raises on the UNMAPPED sentinel or an id the table does not define.

        Pixels under ``sky_mask`` are blackbody-equivalent (ε₀ = 1: the G-buffer carries the
        *apparent* sky temperature there, see irsim.config.gbuffer) and their ids are not
        checked, so the renderer's background id 0 is not mistaken for an unmapped asset.
        """
        ids = np.asarray(material_id)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"material_id must be an integer plane, got {ids.dtype}")
        if sky_mask is not None:
            sky = np.asarray(sky_mask)
            if sky.dtype != np.bool_ or sky.shape != ids.shape:
                raise ValueError("sky_mask must be a bool plane with the material_id shape")
            eps = np.ones(ids.shape, dtype=np.float32)
            if np.any(~sky):
                eps[~sky] = self.emissivity_for(ids[~sky])
            return eps
        if np.any(ids == UNMAPPED_MATERIAL_ID):
            raise ValueError(
                "G-buffer contains material id 0 (UNMAPPED): an asset has no material mapping; "
                "fix the resolver rather than rendering a default emissivity"
            )
        if np.any(ids < 0) or np.any(ids >= self.emissivity.size):
            raise ValueError(f"material id outside the table (0..{self.emissivity.size - 1})")
        eps = self.emissivity[ids]
        if np.any(np.isnan(eps)):
            bad = sorted(set(np.unique(ids[np.isnan(eps)]).tolist()))
            raise ValueError(f"material ids {bad} have no emissivity in this table")
        return np.asarray(eps, dtype=np.float32)
