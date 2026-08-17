"""Detector response -- irradiance to raw signal.

Implements docs/thermal_camera_model.md SS5. Two detector families: uncooled
microbolometer (SS5a) and cooled photon detector (SS5b).
"""
from __future__ import annotations
import numpy as np

from .constants import PLANCK_H, SPEED_OF_LIGHT

_trapz = getattr(np, "trapezoid", None) or np.trapz


def linear_detector_signal(irradiance: np.ndarray, k_resp: float, offset: float = 0.0) -> np.ndarray:
    """Signal = k_resp * E + offset -- the collapsed-responsivity shortcut. SS5."""
    return k_resp * irradiance + offset


def bolometer_temperature_rise(power_abs_w: float, thermal_capacitance: float,
                                thermal_conductance: float, dt_s: float,
                                prev_delta_t_k: float = 0.0) -> float:
    """One Euler step of C_th*d(dT)/dt = P_abs - G_th*dT. SS5a."""
    d_delta_t = (power_abs_w - thermal_conductance * prev_delta_t_k) / thermal_capacitance * dt_s
    return prev_delta_t_k + d_delta_t


def bolometer_signal(delta_t_px_k: float, tcr_per_k: float, r0_ohm: float) -> float:
    """dR = TCR * dT_px * R0 -- resistance change from pixel temperature rise. SS5a."""
    return tcr_per_k * delta_t_px_k * r0_ohm


def photon_detector_electrons(irradiance_spectrum: np.ndarray, wavelength_m: np.ndarray,
                               pixel_area_m2: float, integration_time_s: float,
                               quantum_efficiency: float) -> float:
    """N_e = (A*t_int*eta/(h*c)) * integral(E(lambda)*lambda dlambda). SS5b."""
    integral = _trapz(np.asarray(irradiance_spectrum) * np.asarray(wavelength_m), wavelength_m)
    return (pixel_area_m2 * integration_time_s * quantum_efficiency
            / (PLANCK_H * SPEED_OF_LIGHT)) * integral
