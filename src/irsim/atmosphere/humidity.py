"""Saturation vapour pressure, absolute humidity and the molecular extinction γ(w) (§7.3).

Magnus form with Bolton (1980) coefficients: e_s(T_C) = 6.112 exp(17.67 T_C / (T_C + 243.5)) hPa,
accurate to 0.1 % over −30…+35 °C. Absolute humidity (water-vapour density, g m⁻³):

    w = e / (R_v T) = 216.7 · RH · e_s(T) / T_K,        216.7 = 10⁵ / R_v   (R_v = 461.5 J kg⁻¹ K⁻¹)

The spec calls w "precipitable water"; it is the absolute humidity (spec issue S5), and the
factor is derived from R_v here rather than typed in. RH is a **fraction**: 80 raises. The
molecular part of the band extinction is the engineering form γ_mol,B = γ₀,B + β_B w.

docs/physics-model.md §7.3
"""

from __future__ import annotations

import math

from irsim.radiometry.constants import R_V_WATER

__all__ = [
    "saturation_vapour_pressure_hpa",
    "vapour_pressure_hpa",
    "absolute_humidity_g_m3",
    "gamma_molecular",
    "ABS_HUMIDITY_FACTOR",
]

MAGNUS_A_HPA = 6.112
MAGNUS_B = 17.67
MAGNUS_C_C = 243.5
# g m^-3 per (hPa / K): 100 Pa/hPa * 1000 g/kg / R_v
ABS_HUMIDITY_FACTOR = 100.0 * 1000.0 / R_V_WATER  # 216.68
T_AIR_MIN_K = 150.0


def saturation_vapour_pressure_hpa(t_c: float) -> float:
    """Magnus/Bolton 1980: 6.112 hPa at 0 °C, 23.39 at 20 °C, 42.47 at 30 °C."""
    if t_c < -100.0 or t_c > 100.0:
        raise ValueError(f"t_c = {t_c} is not a Celsius air temperature")
    return MAGNUS_A_HPA * math.exp(MAGNUS_B * t_c / (t_c + MAGNUS_C_C))


def _check_rh(rh_fraction: float) -> None:
    if not 0.0 <= rh_fraction <= 1.0:
        raise ValueError(
            f"relative humidity {rh_fraction} is not a fraction in [0, 1] (80 means percent)"
        )


def _check_t_k(t_k: float) -> None:
    if t_k < T_AIR_MIN_K:
        raise ValueError(f"t_k = {t_k} K is below {T_AIR_MIN_K} K -- kelvin, not celsius")


def vapour_pressure_hpa(t_k: float, rh_fraction: float) -> float:
    """e = RH · e_s(T) in hPa (the input the broadband sky emissivity of M6.5 uses)."""
    _check_t_k(t_k)
    _check_rh(rh_fraction)
    return rh_fraction * saturation_vapour_pressure_hpa(t_k - 273.15)


def absolute_humidity_g_m3(t_k: float, rh_fraction: float) -> float:
    """w = 216.7 · RH · e_s / T_K in g m⁻³ (24.3 at 30 °C / 80 %)."""
    return ABS_HUMIDITY_FACTOR * vapour_pressure_hpa(t_k, rh_fraction) / t_k


def gamma_molecular(w_g_m3: float, gamma0_per_m: float, beta_per_m_per_g_m3: float) -> float:
    """γ_mol = γ₀ + β w (m⁻¹) -- the §7.3 engineering form; γ₀, β fitted per band (M8.3)."""
    if w_g_m3 < 0.0 or gamma0_per_m < 0.0 or beta_per_m_per_g_m3 < 0.0:
        raise ValueError("w, gamma0 and beta must be non-negative")
    return gamma0_per_m + beta_per_m_per_g_m3 * w_g_m3
