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

from irsim.materials.surface import surface_radiance
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes, require_fp32_or_better
from irsim.pipeline.environment import environment_radiance
from irsim.pipeline.illumination import Illumination, illumination_from_planes
from irsim.radiometry.lut import BandLUT, Quantity

__all__ = [
    "band_radiance",
    "band_radiance_stage",
    "stage_illumination",
    "BandRadianceStage",
]


def band_radiance(
    temperature_k: NDArray[np.floating],
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    lut: BandLUT,
    quantity: Quantity = "lb",
    sky_mask: NDArray[np.bool_] | None = None,
    l_env: NDArray[np.floating] | None = None,
    l_behind: NDArray[np.floating] | None = None,
    illumination: Illumination | None = None,
    normal_dot_view: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    """ε₀ L_B(T) + ρ L_env + τ L_behind as float32 (``lb_q`` for photon FPAs).

    Under ``sky_mask`` the temperature is the *apparent* sky temperature, so ε₀ = 1 there
    (irsim.config.gbuffer) and the material id is ignored. Without ``l_env`` the stage is
    emission only (the M3 form). ``l_behind`` is what a second ray through a semi-transparent
    material returns (M7.15); without it L_behind = L_env (ADR 0046), which collapses the three
    terms back to the ε L_B + (1 − ε) L_env of M7.13 exactly, so an opaque scene is unaffected.

    ``illumination`` is the M11.2 bundle and is the general form of ``l_env``: the environment
    plus whatever solar and night sources the band's regime lets through, already gated and
    carrying its own units tag, which is checked here (ADR 0063). It is mutually exclusive with
    ``l_env`` -- two ways to say the same thing is how the two drift apart.

    ``normal_dot_view`` switches stage 1 to §4.2's directional ε(θ) (M7.14). It is used **only**
    when the material table was packed with an angle LUT (M7.10), so every scene and every golden
    written before this is bit-identical: the directional path is something a caller opts into by
    packing the table for it, not something that appears because a G-buffer happens to carry a
    plane it has always carried.

    **ε L_B(T) is evaluated whatever the regime.** A reflective band is a statement about a 300 K
    scene, not about the band: a 500 K exhaust glows in SWIR at night with no illumination at all.
    """
    if illumination is not None:
        if l_env is not None:
            raise ValueError(
                "pass either l_env or illumination, not both: the bundle already carries the "
                "environment term as l_env (ADR 0063)"
            )
        illumination.require_quantity(quantity, "band_radiance")
        l_env = illumination.total_incident()
    t = require_fp32_or_better(np.asarray(temperature_k), "temperature_k")
    ids = np.asarray(material_id)
    if ids.shape != t.shape:
        raise ValueError(f"material_id shape {ids.shape} != temperature shape {t.shape}")
    lb = lut.lookup(t, quantity)
    if l_env is None:
        if l_behind is not None:
            raise ValueError(
                "l_behind was given without l_env: a second ray needs an environment model, "
                "because the surface still reflects (rho L_env) as well as transmits. Configure "
                "PipelineConfig.sky (M7.13) or drop the radiance_behind plane -- silently "
                "ignoring it would make a transparent material render as opaque."
            )
        return np.asarray(materials.emissivity_for(ids, sky_mask) * lb, dtype=np.float32)
    env = require_fp32_or_better(np.asarray(l_env), "l_env")
    # A 0-d environment is a *uniform* one -- isotropic airglow over the whole frame, a fixed
    # overcast -- and broadcasting it is unambiguous. Anything else must match the grid exactly:
    # general broadcasting would let a (H, 1) column through as if it were a full plane, which is
    # the misalignment this check exists to catch.
    if env.ndim != 0 and env.shape != t.shape:
        raise ValueError(f"l_env shape {env.shape} != temperature shape {t.shape}")
    behind = None
    if l_behind is not None:
        behind = require_fp32_or_better(np.asarray(l_behind), "l_behind")
        if behind.shape != t.shape:
            raise ValueError(f"l_behind shape {behind.shape} != temperature shape {t.shape}")
    if normal_dot_view is not None and materials.angle_lut is not None:
        eps, rho, tau = materials.directional_properties_for(ids, normal_dot_view, sky_mask)
    else:
        eps, rho, tau = materials.properties_for(ids, sky_mask)
    out = surface_radiance(eps, rho, tau, lb, env, behind)
    return np.asarray(out, dtype=np.float32)


def stage_illumination(
    planes: Planes, config: PipelineConfig, state: PipelineState
) -> Illumination:
    """The M11.2 bundle for this frame: environment + the regime-gated solar and night planes.

    One function so that ``run_frame`` and ``band_radiance_stage`` cannot end up illuminating
    the scene differently -- they did diverge once already, which is how the stage and the frame
    path came to hold two copies of the environment lookup.
    """
    l_env = None
    if config.sky is not None:
        l_env = environment_radiance(
            config.sky,
            config.lut,
            state.t_s,
            np.asarray(planes["sky_view_factor"]),
            config.quantity,
        )
    return illumination_from_planes(
        config.sensor.sensor.band.regime, config.quantity, planes, l_env=l_env
    )


def band_radiance_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-1 entry point on the plane dict: adds ``radiance`` (float32, W m⁻² sr⁻¹)."""
    out = band_radiance(
        planes["temperature_k"],
        planes["material_id"],
        config.materials,
        config.lut,
        config.quantity,
        sky_mask=planes.get("sky_mask"),
        l_behind=planes.get("radiance_behind"),
        illumination=stage_illumination(planes, config, state),
        normal_dot_view=planes.get("normal_dot_view"),
    )
    return {"radiance": out}


class BandRadianceStage:
    name = "band_radiance"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return band_radiance_stage(planes, config, state)
