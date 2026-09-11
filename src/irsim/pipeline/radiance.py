"""Stage 1 — band radiance from the G-buffer: L = ε₀ L_B(T) + (1 − ε₀) L_env.

The first stage of docs/physics-model.md §13.4: constant per-material emissivity, the reflected
environment term when an ``l_env`` plane is given (M7.13: V_s L_sky,eff + (1 − V_s) L_ground,
built by irsim.pipeline.environment from the SkyModel), emission only otherwise. (1 − ε₀)
stands for ρ + τ: a transmitting material passes the environment behind it (ADR 0046).

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
from irsim.pipeline.environment import environment_radiance
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = ["band_radiance", "band_radiance_stage", "BandRadianceStage"]


def band_radiance(
    temperature_k: NDArray[np.floating],
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    lut: BandLUT,
    quantity: Quantity = "lb",
    sky_mask: NDArray[np.bool_] | None = None,
    l_env: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """ε₀[material] L_B(T) + (1 − ε₀) L_env as float32 (``lb_q`` for photon FPAs).

    Under ``sky_mask`` the temperature is the *apparent* sky temperature, so ε₀ = 1 there
    (irsim.config.gbuffer) and the material id is ignored. Without ``l_env`` the stage is
    emission only (the M3 form).
    """
    t = require_fp32_or_better(np.asarray(temperature_k), "temperature_k")
    ids = np.asarray(material_id)
    if ids.shape != t.shape:
        raise ValueError(f"material_id shape {ids.shape} != temperature shape {t.shape}")
    eps = materials.emissivity_for(ids, sky_mask)
    lb = lut.lookup(t, quantity)
    if l_env is None:
        return np.asarray(eps * lb, dtype=np.float32)
    env = require_fp32_or_better(np.asarray(l_env), "l_env")
    if env.shape != t.shape:
        raise ValueError(f"l_env shape {env.shape} != temperature shape {t.shape}")
    out = eps.astype(np.float64) * lb + (1.0 - eps.astype(np.float64)) * env.astype(np.float64)
    return np.asarray(out, dtype=np.float32)


def band_radiance_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-1 entry point on the plane dict: adds ``radiance`` (float32, W m⁻² sr⁻¹)."""
    l_env = None
    if config.sky is not None:
        l_env = environment_radiance(
            config.sky,
            config.lut,
            state.t_s,
            np.asarray(planes["sky_view_factor"]),
            config.quantity,
        )
    out = band_radiance(
        planes["temperature_k"],
        planes["material_id"],
        config.materials,
        config.lut,
        config.quantity,
        sky_mask=planes.get("sky_mask"),
        l_env=l_env,
    )
    return {"radiance": out}


class BandRadianceStage:
    name = "band_radiance"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return band_radiance_stage(planes, config, state)
