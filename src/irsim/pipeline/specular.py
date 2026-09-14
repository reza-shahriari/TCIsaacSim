"""Stage-1 specular reflection: the sky the lobe actually sees, and the sun in it (§4.3, §5.4).

:mod:`irsim.materials.lobe` gives the kernel; this module gives it something to reflect. The
incident field a surface sees is the sky above its horizon and the ground below it, and the
kernel's directions run over both -- which matters most for exactly the surfaces the lobe is for.
A near-vertical windshield or a ship's flank reflects **ground** over half its lobe, and a model
that sampled only the sky would render it far too cold.

The sun is in the same field, and arrives through the same kernel (ADR 0067), so a glint is not a
second mechanism bolted on: it is what the kernel returns when the incident field contains a small
very bright patch. What the code does add is the **cap** -- the reflected radiance of a source can
never exceed the source's own radiance.

docs/physics-model.md §4.3, §5.4, §5.3, §13.7; ADR 0067
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sky import SkyModel
from irsim.materials.lobe import (
    ReflectionKernel,
    glint_radiance,
    hemisphere_kernel,
    reflected_radiance,
)
from irsim.pipeline.environment import ground_temperature_k

__all__ = [
    "incident_field",
    "specular_reflected_radiance",
    "specular_glint_radiance",
    "kernel_for",
]


def kernel_for(view_dir: Any, normal: Any, roughness: float, **kwargs: Any) -> ReflectionKernel:
    """The blended kernel for one surface; a thin re-export so callers need one import."""
    return hemisphere_kernel(view_dir, normal, roughness, **kwargs)


def incident_field(
    sky: SkyModel, t_s: float, kernel: ReflectionKernel, up: Any = (0.0, 0.0, 1.0)
) -> NDArray[np.float64]:
    """L_incident along each kernel direction: the sky above the horizon, the ground below it.

    ``up`` is the **world** up axis, not the surface normal: what counts as sky is set by the
    horizon, not by which way the surface is facing. Passing the normal here is the mistake this
    argument exists to make explicit -- it would give a vertical panel a sky in every direction.
    """
    up_v = np.asarray(up, dtype=np.float64)
    up_v = up_v / np.linalg.norm(up_v)
    sin_el = np.clip(np.sum(kernel.directions * up_v, axis=-1), -1.0, 1.0)
    above = sin_el > 0.0
    # The sky model is defined on [0, 90] and refuses anything else, so the below-horizon
    # directions are evaluated at the horizon and then discarded -- not clamped into the answer.
    elevation = np.arcsin(np.where(above, sin_el, 0.0))
    l_sky = np.asarray(sky.radiance(t_s, elevation), dtype=np.float64)
    l_ground = float(sky.lut.lookup(np.float64(ground_temperature_k(sky, t_s)), sky.quantity)[()])
    return np.asarray(np.where(above, l_sky, l_ground))


def specular_reflected_radiance(
    reflectance: Any,
    sky: SkyModel,
    t_s: float,
    view_dir: Any,
    normal: Any,
    roughness: float,
    up: Any = (0.0, 0.0, 1.0),
    **kernel_kwargs: Any,
) -> float:
    """ρ · ⟨L_sky/ground⟩ over the lobe -- the M7.13 environment term with a direction."""
    kernel = kernel_for(view_dir, normal, roughness, **kernel_kwargs)
    return float(reflected_radiance(reflectance, kernel, incident_field(sky, t_s, kernel, up=up)))


def specular_glint_radiance(
    reflectance: Any,
    view_dir: Any,
    normal: Any,
    sun_dir: Any,
    roughness: float,
    band_irradiance: float,
    tau_sun: float = 1.0,
    **kernel_kwargs: Any,
) -> float:
    """The sun through the same kernel, normalised on the same quadrature as the sky term."""
    kernel = kernel_for(view_dir, normal, roughness, **kernel_kwargs)
    return glint_radiance(
        reflectance,
        view_dir,
        normal,
        sun_dir,
        roughness,
        band_irradiance,
        tau_sun=tau_sun,
        kernel_normalisation=kernel.specular_normalisation,
    )
