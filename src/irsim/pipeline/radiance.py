"""Stage 1 — band radiance from the G-buffer: L = ε₀ · L_B(T), emission only.

The first stage of docs/physics-model.md §13.4 in its §16.4-step-3 form: constant per-material
emissivity, no reflected term, no atmosphere. Those arrive as additive terms in later steps
(M7.13 reflection, M8 atmosphere); this stage stays the ε₀ L_B(T) core that every later form
reduces to when L_env = 0 and τ_atm = 1.

**No π, no aperture factor, no cos⁴ here.** Radiance is a property of the scene; the optics stage
turns it into power on a pixel. Works at whatever grid it is given (native or supersampled):
it is per-pixel, so supersampling is the caller's choice of G-buffer resolution (ADR 0014 --
ids never blend, so anti-aliasing is by box-filtering *radiance* downstream, never ids).

docs/physics-model.md §5.1, §13.5, §16.4 step 3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes, require_fp32_or_better
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = ["band_radiance", "band_radiance_stage", "BandRadianceStage"]


def band_radiance(
    temperature_k: NDArray[np.floating],
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    lut: BandLUT,
    quantity: Quantity = "lb",
    sky_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """ε₀[material] · L_B(T) as float32, same shape as the inputs (``lb_q`` for photon FPAs).

    Under ``sky_mask`` the temperature is the *apparent* sky temperature, so ε₀ = 1 there
    (irsim.config.gbuffer) and the material id is ignored.
    """
    t = require_fp32_or_better(np.asarray(temperature_k), "temperature_k")
    ids = np.asarray(material_id)
    if ids.shape != t.shape:
        raise ValueError(f"material_id shape {ids.shape} != temperature shape {t.shape}")
    eps = materials.emissivity_for(ids, sky_mask)
    lb = lut.lookup(t, quantity)
    return np.asarray(eps * lb, dtype=np.float32)


def band_radiance_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-1 entry point on the plane dict: adds ``radiance`` (float32, W m⁻² sr⁻¹)."""
    del state
    out = band_radiance(
        planes["temperature_k"],
        planes["material_id"],
        config.materials,
        config.lut,
        sky_mask=planes.get("sky_mask"),
    )
    return {"radiance": out}


class BandRadianceStage:
    name = "band_radiance"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return band_radiance_stage(planes, config, state)
