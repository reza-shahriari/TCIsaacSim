"""Atmospheric propagation -- Beer-Lambert transmission + path radiance.

Implements docs/thermal_camera_model.md SS3.
"""
from __future__ import annotations
import numpy as np


def transmittance(gamma_per_m: float, range_m: np.ndarray) -> np.ndarray:
    """tau(R) = exp(-gamma * R). SS3."""
    raise NotImplementedError("TODO: implement Beer-Lambert transmittance, SS3")


def path_radiance(gamma_per_m: float, range_m: np.ndarray, l_bb_t_atm: float) -> np.ndarray:
    """(1 - tau(R)) * L_bb(T_atm) -- the atmosphere's own self-emission ("airlight"). SS3."""
    raise NotImplementedError("TODO: implement path radiance, SS3")


def apply_atmosphere(source_radiance: np.ndarray, gamma_per_m: float, range_m: np.ndarray,
                      l_bb_t_atm: float) -> np.ndarray:
    """tau(R)*L_source + (1-tau(R))*L_bb(T_atm). SS3.

    Sanity checks this should satisfy once implemented (see docs/isaacsim_checklist.md 1.6):
    range_m == 0        -> output == source_radiance
    range_m -> large     -> output -> l_bb_t_atm
    """
    raise NotImplementedError("TODO: combine transmittance + path_radiance, SS3")
