"""The reflection lobe: one microfacet kernel for reflected sky and solar glint (§4.3, §5.4).

§4.3's headline is that an infrared reflection lobe is **narrower and stronger** than the same
material's visible one -- roughness that scatters 0.5 µm light diffusely can be optically smooth
at 10 µm -- so painted bodywork and glass show near-mirror sky reflections in LWIR while asphalt
stays Lambertian. That is why ``roughness_per_band`` is per band and why "do not reuse visible
roughness values" is in the spec in bold.

**The specular/diffuse split is explicit, because a microfacet lobe does not become Lambertian
and it was worth measuring rather than assuming (ADR 0067).** The tidy version of this module
would carry one GGX kernel and let roughness → 1 produce the diffuse answer. It does not: at
α = 1 GGX is uniform in the half-vector, and the resulting kernel sits a **total-variation
distance of 0.30** from the cosine hemisphere, reading a sky 4.7 units warmer out of 280 -- 1.7 %,
and not converging as α grows further. So:

    K(ω_i) = w_s · K_GGX(ω_i) + (1 − w_s) · cos θ_i / π ,    ∫ K dω_i = 1
    L_reflected = ρ · ∫ K(ω_i) L_incident(ω_i) dω_i ,        w_s = (1 − roughness)²

Energy is conserved **by construction** -- the reflected coefficient is exactly ρ for every
roughness, to machine precision -- and both endpoints are exact rather than fitted: roughness 0 is
a mirror reading L_sky(mirror direction), roughness 1 is the V_s blend of M7.13 to 1e-6.

The (1 − r)² shape is a modelling choice, and the check on it is §4.3's own prose: glass 0.03 →
**94 %** specular, painted bodywork 0.12 → **77 %**, asphalt 0.70 → **9 %**. That is exactly
"painted bodywork and glass show near-mirror sky reflections in LWIR ... asphalt and concrete stay
near-Lambertian even in LWIR", reproduced from the authored roughness values without being fitted
to them.

**The sun is not a special case, it is a small bright patch of the incident field.** Its
contribution is K(ŝ)·Ω_sun·L_sun with L_sun = E_B τ_sun /(Ω_sun cos θ_s), so it reduces to §5.4's
``L = R E_B τ / Ω_sun`` in the mirror limit, and is **capped there**: K(ŝ)Ω_sun ≤ 1 means you
cannot see the sun brighter than the sun, however smooth the surface is. A model that let the NDF
peak run free would put 1/(πα²) into a pixel and produce a glint brighter than its source.

docs/physics-model.md §4.3, §5.4, §12.3, §13.7; ADR 0067
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SOLID_ANGLE_SUN_SR",
    "MIN_ALPHA",
    "roughness_to_alpha",
    "specular_weight",
    "ggx_ndf",
    "mirror_direction",
    "kernel_value",
    "ReflectionKernel",
    "hemisphere_kernel",
    "reflected_radiance",
    "glint_radiance",
]

#: Solid angle of the solar disc at 1 AU: π (R_sun / d)² with R_sun/d = 4.6524e-3 rad (0.2665°).
SOLID_ANGLE_SUN_SR = math.pi * (4.6524e-3) ** 2

#: A floor on the GGX width. At α below this the lobe is narrower than the solar disc and the
#: numerical kernel cannot resolve it; the physics there is "perfect mirror", which the glint cap
#: and the α → 0 limit both already give exactly.
MIN_ALPHA = 1e-4


def roughness_to_alpha(roughness: Any) -> NDArray[np.float64]:
    """α = roughness² — the Disney/Burley convention (ADR 0067).

    The squaring is what makes the authored numbers behave the way the material library's comments
    intend: glass at 0.03 becomes α = 9e-4, a near-mirror, and asphalt at 0.80 becomes α = 0.64,
    essentially Lambertian. Using roughness directly as α would put glass at 0.03 — visibly
    blurred — and compress every interesting material into the top of the range.
    """
    r = np.asarray(roughness, dtype=np.float64)
    if np.any((r < 0.0) | (r > 1.0)):
        raise ValueError("roughness must lie in [0, 1]")
    return np.asarray(np.maximum(r * r, MIN_ALPHA))


def specular_weight(roughness: Any) -> NDArray[np.float64]:
    """w_s = (1 − roughness)²: the share of ρ that stays directional (ADR 0067).

    1 at roughness 0 and 0 at roughness 1, so both endpoints of the blended kernel are exact. The
    square rather than a straight line because it is what reproduces §4.3's qualitative claims
    from the library's own authored values -- glass 94 %, paint 77 %, asphalt 9 % -- where a linear
    weight would leave asphalt 30 % specular and visibly mirror-like in LWIR, which §4.3 says it
    is not.
    """
    r = np.asarray(roughness, dtype=np.float64)
    if np.any((r < 0.0) | (r > 1.0)):
        raise ValueError("roughness must lie in [0, 1]")
    return np.asarray((1.0 - r) ** 2)


def ggx_ndf(cos_theta_h: Any, alpha: Any) -> NDArray[np.float64]:
    """GGX / Trowbridge–Reitz microfacet distribution D(θ_h), normalised so ∫D cos θ_h dω_h = 1.

    GGX rather than Beckmann (ADR 0067): its power-law tails match measured rough surfaces, and a
    thermal reflection's *tail* is what puts a faint warm smear around a hot reflection rather
    than a hard-edged disc.
    """
    c = np.clip(np.asarray(cos_theta_h, dtype=np.float64), 0.0, 1.0)
    a = np.asarray(alpha, dtype=np.float64)
    a2 = a * a
    denom = c * c * (a2 - 1.0) + 1.0
    return np.asarray(a2 / (math.pi * denom * denom))


def _unit(v: Any) -> NDArray[np.float64]:
    x = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("direction vector has zero length")
    return np.asarray(x / n)


def mirror_direction(view_dir: Any, normal: Any) -> NDArray[np.float64]:
    """The specular direction for a ray **leaving** along ``view_dir``: 2(n·v)n − v.

    ``view_dir`` points from the surface toward the camera and ``normal`` away from the surface,
    both unit. The result points from the surface toward where the reflected energy comes from.
    """
    v = _unit(view_dir)
    n = _unit(normal)
    cos_o = np.sum(v * n, axis=-1, keepdims=True)
    if np.any(cos_o <= 0.0):
        raise ValueError("view_dir is below the surface: n·v must be positive")
    return np.asarray(2.0 * cos_o * n - v)


def kernel_value(view_dir: Any, normal: Any, incident_dir: Any, alpha: Any) -> NDArray[np.float64]:
    """K(ω_i) in sr⁻¹: the microfacet reflection kernel, before normalisation.

    K = D(θ_h) cos θ_h / (4 (ω_i·h)); the cos θ_h / (4 ω_i·h) is the half-vector Jacobian
    dω_h/dω_i, which is what turns the NDF's own normalisation ∫D cos θ_h dω_h = 1 into a kernel
    that integrates to 1 over incident directions. Shadowing/masking G is taken as 1 (ADR 0067)
    and the residual is renormalised away by :func:`hemisphere_kernel`.
    """
    v = _unit(view_dir)
    n = _unit(normal)
    i = _unit(incident_dir)
    h = i + v
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    h = np.where(norm > 0.0, h / np.maximum(norm, 1e-300), n)
    cos_h = np.clip(np.sum(h * n, axis=-1), 0.0, 1.0)
    i_dot_h = np.maximum(np.sum(i * h, axis=-1), 1e-12)
    return np.asarray(ggx_ndf(cos_h, alpha) * cos_h / (4.0 * i_dot_h))


def _basis(n: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(n[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = _unit(np.cross(n, helper))
    return t1, np.cross(n, t1)


def _cone_nodes(
    axis: NDArray[np.float64], cos_min: float, n_theta: int, n_phi: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gauss–Legendre in cos θ over [cos_min, 1] × midpoint φ, about ``axis``."""
    t1, t2 = _basis(axis)
    nodes, gl = np.polynomial.legendre.leggauss(n_theta)
    cos_t = 0.5 * (nodes + 1.0) * (1.0 - cos_min) + cos_min
    d_cos = 0.5 * gl * (1.0 - cos_min)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = (np.arange(n_phi) + 0.5) * (2.0 * math.pi / n_phi)
    dirs = (
        cos_t[:, None, None] * axis[None, None, :]
        + (sin_t[:, None] * np.cos(phi)[None, :])[:, :, None] * t1[None, None, :]
        + (sin_t[:, None] * np.sin(phi)[None, :])[:, :, None] * t2[None, None, :]
    ).reshape(-1, 3)
    solid = (d_cos[:, None] * (2.0 * math.pi / n_phi) * np.ones(n_phi)[None, :]).reshape(-1)
    return dirs, solid


#: Fraction of the GGX half-angle distribution the specular cone is sized to contain. GGX has
#: power-law tails, so "all of it" is the whole hemisphere; this is where the cone is cut and the
#: remainder renormalised away.
SPECULAR_CONE_FRACTION = 0.9999


@dataclass(frozen=True)
class ReflectionKernel:
    """Incident directions and weights summing to exactly 1, plus the specular bookkeeping."""

    directions: NDArray[np.float64]
    weights: NDArray[np.float64]
    specular_weight: float
    alpha: float
    #: ∫K_GGX dω over the cone, on this quadrature. The glint divides by it so that a point
    #: source and the quadrature agree about how much of the lobe they are talking about.
    specular_normalisation: float

    def average(self, incident_radiance: Any) -> NDArray[np.float64]:
        values = np.asarray(incident_radiance, dtype=np.float64)
        if values.shape[-1] != self.weights.shape[0]:
            raise ValueError(
                f"incident_radiance last axis {values.shape[-1]} != kernel size "
                f"{self.weights.shape[0]}"
            )
        return np.asarray(np.sum(values * self.weights, axis=-1))


def hemisphere_kernel(
    view_dir: Any,
    normal: Any,
    roughness: float,
    n_theta: int = 40,
    n_phi: int = 64,
) -> ReflectionKernel:
    """The blended reflection kernel: a cone about the mirror direction plus a cosine hemisphere.

    The specular part is quadratured **about the mirror direction**, in a cone sized to hold
    99.99 % of the GGX half-angle distribution, rather than on a grid tied to the normal. That is
    not an optimisation: a normal-aligned grid at α = 1e-4 cannot see a lobe 1° wide at all, and
    a glass windshield came out 0.09 units short of the exact mirror answer before this changed.
    The diffuse part keeps the normal-aligned cosine quadrature, where it is exact.

    Directions whose incident elevation is below the surface get zero weight, and the remainder is
    renormalised -- the same G = 1 simplification as :func:`kernel_value`, applied consistently.

    This is the engine-free oracle. It costs ~2·n_theta·n_phi direction evaluations per surface and
    is not what a renderer would run -- a cubemap probe or an importance-sampled estimator is --
    but it is what those are *checked against*.
    """
    v = _unit(view_dir)
    n = _unit(normal)
    alpha = float(roughness_to_alpha(roughness))
    w_s = float(specular_weight(roughness))
    mirror = mirror_direction(v, n)

    # cone half-angle: tan(theta_h) = alpha * sqrt(F/(1-F)) holds F of the NDF; the reflected
    # direction moves about twice as far as the half-vector, hence the factor 2.
    tan_h = alpha * math.sqrt(SPECULAR_CONE_FRACTION / (1.0 - SPECULAR_CONE_FRACTION))
    theta_max = min(math.pi, 2.0 * math.atan(tan_h) + 1e-6)
    spec_dirs, spec_solid = _cone_nodes(mirror, math.cos(theta_max), n_theta, n_phi)
    above = np.sum(spec_dirs * n, axis=-1) > 0.0
    spec_raw = np.where(above, kernel_value(v, n, spec_dirs, alpha) * spec_solid, 0.0)
    spec_norm = float(spec_raw.sum())
    if not spec_norm > 0.0:
        raise ValueError("specular kernel integrated to zero")

    diff_dirs, diff_solid = _cone_nodes(n, 0.0, n_theta, n_phi)
    cos_i = np.sum(diff_dirs * n, axis=-1)
    diff_raw = (cos_i / math.pi) * diff_solid

    weights = np.concatenate(
        [w_s * spec_raw / spec_norm, (1.0 - w_s) * diff_raw / float(diff_raw.sum())]
    )
    return ReflectionKernel(
        directions=np.concatenate([spec_dirs, diff_dirs]),
        weights=weights,
        specular_weight=w_s,
        alpha=alpha,
        specular_normalisation=spec_norm,
    )


def reflected_radiance(
    reflectance: Any, kernel: ReflectionKernel, incident_radiance: Any
) -> NDArray[np.float64]:
    """ρ · Σ w_i L_i. Exactly ρ·L for a uniform field, at any roughness, to machine precision."""
    rho = np.asarray(reflectance, dtype=np.float64)
    if np.any((rho < 0.0) | (rho > 1.0)):
        raise ValueError("reflectance must lie in [0, 1]")
    return np.asarray(rho * kernel.average(incident_radiance))


def glint_radiance(
    reflectance: Any,
    view_dir: Any,
    normal: Any,
    sun_dir: Any,
    roughness: float,
    band_irradiance: float,
    tau_sun: float = 1.0,
    solid_angle_sun_sr: float = SOLID_ANGLE_SUN_SR,
    kernel_normalisation: float = 1.0,
) -> float:
    """ρ w_s · min(K(ŝ)·Ω_sun, 1) · E_B τ_sun / (Ω_sun cos θ_s) — §5.4's glint, capped at the sun.

    The cap is the physics, not a guard: the reflected radiance of a source cannot exceed the
    source's own radiance, and a microfacet NDF evaluated at its peak will happily claim
    1/(πα²) sr⁻¹, which for glass (α = 9e-4) is 4e5 — enough to render a glint four orders of
    magnitude brighter than the sun it came from.

    ``kernel_normalisation`` is the un-normalised kernel's hemisphere integral, from
    :func:`hemisphere_kernel`; pass 1.0 to use the analytic kernel directly.
    """
    n = _unit(normal)
    s = _unit(sun_dir)
    cos_s = float(np.sum(s * n))
    if cos_s <= 0.0 or band_irradiance <= 0.0 or tau_sun <= 0.0:
        return 0.0
    if not 0.0 < solid_angle_sun_sr < 2.0 * math.pi:
        raise ValueError("solid_angle_sun_sr must be a small positive solid angle")
    alpha = float(roughness_to_alpha(roughness))
    k = float(kernel_value(view_dir, n, s, alpha)) / max(kernel_normalisation, 1e-300)
    captured = min(k * solid_angle_sun_sr, 1.0)
    l_sun = band_irradiance * tau_sun / (solid_angle_sun_sr * cos_s)
    return float(np.asarray(reflectance) * float(specular_weight(roughness)) * captured * l_sun)
