"""Optics -- radiance to focal-plane irradiance.

Implements docs/thermal_camera_model.md SS4.
"""
from __future__ import annotations
import numpy as np


def focal_plane_irradiance(radiance: np.ndarray, f_number: float, tau_optics: float,
                            field_angle_rad: np.ndarray | float = 0.0) -> np.ndarray:
    """E = L * tau_optics * pi / (4*N^2) * cos^4(theta_field). SS4.

    Sanity check (docs/isaacsim_checklist.md 1.7): doubling f_number should
    quarter the irradiance, all else equal.
    """
    raise NotImplementedError("TODO: implement the camera radiometry equation, SS4")
