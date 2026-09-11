"""Beer–Lambert transmittance and path radiance: the atmosphere reference kernel (§7.1, §2).

    τ(d) = exp(−γ_B d),        L_path = (1 − τ) L_B(T_air),        L' = τ L + (1 − τ) L_B(T_air)

with γ_B the band-averaged extinction coefficient (m⁻¹) from the atmosphere model and L_B(T_air)
the band radiance of the air at the (uniform) air temperature. Everything is in radiance space:
an isothermal scene (L = L_B(T_air)) is invariant at every distance, and the contrast between two
radiances is scaled by exactly τ — neither holds for any Kelvin-space shortcut (non-negotiable
#3). d = ∞ gives τ = 0 (a sky pixel sees only the path). ``apply_tau_override`` is the documented
L1 fallback (a constant τ independent of distance). Inputs float32 or float64, dtype preserved,
float16 refused.

docs/physics-model.md §7.1, §2 (three-term radiance), §13.4 stage 2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["transmittance", "path_radiance", "apply_atmosphere", "apply_tau_override"]


def _floating(x: object, what: str) -> NDArray[np.floating]:
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError(f"{what} is float16 (non-negotiable #2)")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def transmittance(distance_m: object, gamma_per_m: object) -> NDArray[np.floating]:
    """τ = exp(−γ d); d ≥ 0 (∞ allowed → 0), γ ≥ 0. Result dtype follows the distance array."""
    d = _floating(distance_m, "distance_m")
    g = _floating(gamma_per_m, "gamma_per_m")
    if np.any(g < 0.0) or not np.all(np.isfinite(g)):
        raise ValueError("gamma_per_m must be finite and non-negative")
    if np.any(d < 0.0) or np.any(np.isnan(d)):
        raise ValueError("distance_m must be non-negative (inf allowed)")
    with np.errstate(over="ignore", invalid="ignore"):
        tau = np.exp(-(g.astype(np.float64) * d.astype(np.float64)))
    tau = np.where(np.isinf(d) & (g > 0), 0.0, tau)
    tau = np.where(np.isinf(d) & (g == 0), 1.0, tau)
    return np.asarray(tau, dtype=d.dtype if np.issubdtype(d.dtype, np.floating) else np.float64)


def path_radiance(tau: object, l_air_band: object) -> NDArray[np.floating]:
    """L_path = (1 − τ) · L_B(T_air) (§7.1)."""
    t = _floating(tau, "tau")
    if np.any(t < 0.0) or np.any(t > 1.0):
        raise ValueError("tau must lie in [0, 1]")
    l_air = _floating(l_air_band, "l_air_band")
    return np.asarray((1.0 - t.astype(np.float64)) * l_air.astype(np.float64), dtype=t.dtype)


def apply_atmosphere(
    l_band: object, distance_m: object, gamma_per_m: object, l_air_band: object
) -> NDArray[np.floating]:
    """L' = τ L + (1 − τ) L_B(T_air): attenuation plus path radiance, in radiance space (§2)."""
    lb = _floating(l_band, "l_band")
    tau = transmittance(distance_m, gamma_per_m).astype(np.float64)
    l_air = _floating(l_air_band, "l_air_band").astype(np.float64)
    out = tau * lb.astype(np.float64) + (1.0 - tau) * l_air
    return np.asarray(out, dtype=lb.dtype)


def apply_tau_override(l_band: object, tau: float, l_air_band: object) -> NDArray[np.floating]:
    """L1 fallback: a constant, distance-independent τ (documented as such; no physics claim)."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    lb = _floating(l_band, "l_band")
    l_air = _floating(l_air_band, "l_air_band").astype(np.float64)
    return np.asarray(tau * lb.astype(np.float64) + (1.0 - tau) * l_air, dtype=lb.dtype)
