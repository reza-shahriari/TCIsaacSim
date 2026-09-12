"""Aerial-target contrast against the sky background, and the elevation where it vanishes.

A target in the sky is seen against the sky, and the sky is not a constant: it runs from roughly
air temperature at the horizon to tens of kelvin colder at zenith (MS.1, ADR 0071). So the
contrast of a drone flying at a fixed temperature *changes sign somewhere*, and where it does is a
detectability statement that a sky-target simulator has to get right. Along a ray of elevation θ:

    L_target(θ) = τ(R, θ) · [ε L_B(T_t) + (1 − ε) L_env] + L_path(R, θ)      (§5.1, §7.1)
    L_bg(θ)     = L_sky(θ)                                                    (§5.3 a, ADR 0044)
    C(θ)        = L_target(θ) − L_bg(θ)

with L_env = V_s L_sky,eff + (1 − V_s) L_B(T_ground) from the same M7.13 relation the pipeline
uses. Two mechanisms fight: the target's own emission wins at zenith, where the sky is cold, while
near the horizon τ → small, L_path → (1 − τ) L_B(T_air) and the sky itself approaches L_B(T_air),
leaving only the reflected term (1 − ε)(L_env − L_B(T_air)) — negative for a cold-sky reflection.
``zero_contrast_elevation`` finds the crossing.

**Scope (ADR 0072).** L_env is evaluated with the ground-level sky model: a target at altitude sees
slightly less atmosphere above it than a surface does, and its belly sees ground at a range, not
underfoot. Neither is modelled here, so the belly case (V_s → 0) carries an unquantified bias and
the honest configuration is V_s = 1. Elevation is the geometric ray angle, flat-earth as in MS.1.

docs/physics-model.md §5.1, §5.3(a), §7.1, §15; ADR 0072
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from irsim.atmosphere.sky import SkyModel
from irsim.pipeline.environment import environment_radiance

__all__ = [
    "AerialTarget",
    "target_leaving_radiance",
    "apparent_target_radiance",
    "background_radiance",
    "target_contrast",
    "zero_contrast_elevation",
]


@dataclass(frozen=True)
class AerialTarget:
    """A resolved target in the sky: its surface temperature, band emissivity, range and sky view.

    ``sky_view_factor`` is V_s of the face the camera sees — 1 for a face that sees only sky
    (the honest case, see the module note), 0 for a belly over warm ground.
    """

    temperature_k: float
    emissivity: float
    range_m: float
    sky_view_factor: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.temperature_k < 1e4 or not math.isfinite(self.temperature_k):
            raise ValueError("temperature_k must be a positive, finite kelvin value")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must lie in [0, 1]")
        if not self.range_m > 0.0 or not math.isfinite(self.range_m):
            raise ValueError("range_m must be positive and finite")
        if not 0.0 <= self.sky_view_factor <= 1.0:
            raise ValueError("sky_view_factor must lie in [0, 1]")


def target_leaving_radiance(target: AerialTarget, sky: SkyModel, t_s: float) -> float:
    """ε L_B(T_t) + (1 − ε) L_env at the target, before any atmosphere (§5.1, M7.13).

    L_env comes from :func:`irsim.pipeline.environment.environment_radiance`, so this and the
    pipeline cannot disagree about the reflected term.
    """
    lut = sky.lut
    l_b = float(lut.lookup(np.float64(target.temperature_k), sky.quantity)[()])
    l_env = float(
        environment_radiance(
            sky, lut, t_s, np.array([target.sky_view_factor], dtype=np.float64), sky.quantity
        )[0]
    )
    return target.emissivity * l_b + (1.0 - target.emissivity) * l_env


def apparent_target_radiance(
    target: AerialTarget, sky: SkyModel, t_s: float, elevation_rad: float
) -> float:
    """τ(R, θ) L_target + L_path(R, θ): what reaches the aperture from the target (§7.1)."""
    atm = sky.atmosphere
    band = sky.band
    tau = float(np.asarray(atm.transmittance(band, t_s, target.range_m, elevation_rad))[()])
    l_path = float(atm.path_radiance(band, t_s, target.range_m, elevation_rad, sky.quantity))
    return tau * target_leaving_radiance(target, sky, t_s) + l_path


def background_radiance(sky: SkyModel, t_s: float, elevation_rad: float) -> float:
    """L_sky(θ) behind the target — the SkyModel's value, so cloud blending is included."""
    return float(np.asarray(sky.radiance(t_s, elevation_rad))[()])


def target_contrast(target: AerialTarget, sky: SkyModel, t_s: float, elevation_rad: float) -> float:
    """C(θ) = L_target,apparent(θ) − L_sky(θ). Positive means the target is brighter than sky."""
    return apparent_target_radiance(target, sky, t_s, elevation_rad) - background_radiance(
        sky, t_s, elevation_rad
    )


def zero_contrast_elevation(
    target: AerialTarget,
    sky: SkyModel,
    t_s: float,
    bracket_deg: tuple[float, float] = (0.5, 90.0),
    tolerance_deg: float = 1e-4,
) -> float | None:
    """The elevation (degrees) where C(θ) = 0, or ``None`` when C does not change sign.

    ``None`` is the physically ordinary answer for a hot target — it is brighter than the sky at
    every elevation — so callers must handle it rather than treating it as a failure.
    """
    lo, hi = bracket_deg
    if not 0.0 <= lo < hi <= 90.0:
        raise ValueError("bracket_deg must satisfy 0 <= lo < hi <= 90")

    def c_of_deg(deg: float) -> float:
        return target_contrast(target, sky, t_s, math.radians(deg))

    c_lo, c_hi = c_of_deg(lo), c_of_deg(hi)
    if c_lo == 0.0:
        return lo
    if c_hi == 0.0:
        return hi
    if math.copysign(1.0, c_lo) == math.copysign(1.0, c_hi):
        return None
    return float(brentq(c_of_deg, lo, hi, xtol=tolerance_deg))
