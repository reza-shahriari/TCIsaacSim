"""Atmospheric propagation -- Beer-Lambert transmission + path radiance.

Implements docs/thermal_camera_model.md SS3.
"""
from __future__ import annotations
import numpy as np


def transmittance(gamma_per_m: float, range_m: np.ndarray) -> np.ndarray:
    """tau(R) = exp(-gamma * R). SS3."""
    return np.exp(-gamma_per_m * np.asarray(range_m, dtype=np.float64))


def path_radiance(gamma_per_m: float, range_m: np.ndarray, l_bb_t_atm: float) -> np.ndarray:
    """(1 - tau(R)) * L_bb(T_atm) -- the atmosphere's own self-emission ("airlight"). SS3."""
    return (1 - transmittance(gamma_per_m, range_m)) * l_bb_t_atm


def apply_atmosphere(source_radiance: np.ndarray, gamma_per_m: float, range_m: np.ndarray,
                      l_bb_t_atm: float) -> np.ndarray:
    """tau(R)*L_source + (1-tau(R))*L_bb(T_atm). SS3.

    range_m == 0        -> output == source_radiance   (tau=1)
    range_m -> large     -> output -> l_bb_t_atm         (tau=0)
    """
    tau = transmittance(gamma_per_m, range_m)
    return tau * source_radiance + (1 - tau) * l_bb_t_atm
