"""Optics -- radiance to focal-plane irradiance.

Implements docs/thermal_camera_model.md SS4.
"""
from __future__ import annotations
import numpy as np


def focal_plane_irradiance(radiance: np.ndarray, f_number: float, tau_optics: float,
                            field_angle_rad: np.ndarray | float = 0.0) -> np.ndarray:
    """E = L * tau_optics * pi / (4*N^2) * cos^4(theta_field). SS4."""
    cos4 = np.cos(np.asarray(field_angle_rad, dtype=np.float64)) ** 4
    return radiance * tau_optics * (np.pi / (4 * f_number**2)) * cos4
