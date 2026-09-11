"""Stage 2 — atmosphere on the G-buffer: L' = τ(d) L + (1 − τ(d)) L_B(T_air), sky pixels untouched.

Per-pixel Beer–Lambert (irsim.atmosphere.beer_lambert) with the scalars the ``Atmosphere``
(M8.5) supplies for the frame time: one γ_B and one L_B(T_air) per band, both evaluated with the
pipeline's own LUT so the isothermal invariance holds bit-for-bit against stage 1. Runs on the
k× supersampled grid before the PSF (the atmosphere is a property of each ray, the blur of the
optics). Sky-pixel policy (ADR 0050): pixels under ``sky_mask`` carry the apparent sky
temperature, which already includes the atmosphere to space, and are returned untouched --
bit-identical -- in both the Beer–Lambert and the constant-τ (L1) paths.

docs/physics-model.md §13.4 stage 2, §7.1, §3.3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.beer_lambert import apply_atmosphere, apply_tau_override
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = [
    "apply_atmosphere_gbuffer",
    "apply_layered_gbuffer",
    "atmosphere_stage",
    "AtmosphereStage",
]


def apply_atmosphere_gbuffer(
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    gamma_per_m: float,
    l_air: float,
    sky_mask: NDArray[np.bool_] | None = None,
    tau_override: float | None = None,
) -> NDArray[np.floating]:
    """Per-pixel M8.1 on the radiance plane; dtype preserved (float16 refused)."""
    l_in = np.asarray(radiance)
    d = np.asarray(distance_m)
    if d.shape != l_in.shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {l_in.shape}")
    if tau_override is None:
        out = apply_atmosphere(l_in, d, gamma_per_m, l_air)
    else:
        out = apply_tau_override(l_in, tau_override, l_air)
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != l_in.shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")
        out = np.where(sky, l_in, out).astype(l_in.dtype, copy=False)
    return out


def apply_layered_gbuffer(
    atmosphere: LayeredAtmosphere,
    band: str,
    t_s: float,
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    quantity: str,
    sky_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.floating]:
    """Per-term horizontal form of MS.1 on the radiance plane (per-pixel slant paths: MS.8)."""
    l_in = np.asarray(radiance)
    if l_in.dtype == np.float16:
        raise TypeError("radiance is float16 (non-negotiable #2)")
    d = np.asarray(distance_m)
    if d.shape != l_in.shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {l_in.shape}")
    out = atmosphere.apply(band, t_s, l_in, d, 0.0, quantity)  # type: ignore[arg-type]
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != l_in.shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")
        out = np.where(sky, l_in, out).astype(l_in.dtype, copy=False)
    return out


def atmosphere_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-2 entry point on the plane dict: replaces ``radiance``; identity without an
    Atmosphere."""
    radiance = np.asarray(planes["radiance"])
    if config.atmosphere is None:
        return {"radiance": radiance}
    if isinstance(config.atmosphere, LayeredAtmosphere):
        return {
            "radiance": apply_layered_gbuffer(
                config.atmosphere,
                config.sensor.sensor.band.band_id,
                state.t_s,
                radiance,
                np.asarray(planes["distance_m"]),
                config.quantity,
                sky_mask=planes.get("sky_mask"),
            )
        }
    atm_state = config.atmosphere.state(state.t_s)
    band = config.sensor.sensor.band.band_id
    l_air = float(config.lut.lookup(np.float64(atm_state.t_air_k), config.quantity)[()])
    out = apply_atmosphere_gbuffer(
        radiance,
        np.asarray(planes["distance_m"]),
        atm_state.gamma_per_m[band],
        l_air,
        sky_mask=planes.get("sky_mask"),
        tau_override=config.tau_override,
    )
    return {"radiance": out}


class AtmosphereStage:
    name = "atmosphere"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return atmosphere_stage(planes, config, state)
