"""Detector response -- irradiance to raw signal.

Implements docs/thermal_camera_model.md SS5. Two detector families: uncooled
microbolometer (SS5a) and cooled photon detector (SS5b). Start with
linear_detector_signal (the simplification SS5 explicitly allows) before
attempting the full bolometer thermal ODE.
"""
from __future__ import annotations
import numpy as np


def linear_detector_signal(irradiance: np.ndarray, k_resp: float, offset: float = 0.0) -> np.ndarray:
    """Signal = k_resp * E + offset -- the collapsed-responsivity shortcut. SS5."""
    raise NotImplementedError("TODO: implement the linear detector shortcut, SS5")


def bolometer_temperature_rise(power_abs_w: float, thermal_capacitance: float,
                                thermal_conductance: float, dt_s: float,
                                prev_delta_t_k: float = 0.0) -> float:
    """One Euler step of C_th*d(dT)/dt = P_abs - G_th*dT. SS5a."""
    raise NotImplementedError("TODO: implement the bolometer thermal ODE step, SS5a")


def bolometer_signal(delta_t_px_k: float, tcr_per_k: float, r0_ohm: float) -> float:
    """dR/R0 ~= TCR * dT_px -- resistance change from pixel temperature rise. SS5a."""
    raise NotImplementedError("TODO: implement bolometer resistance response, SS5a")


def photon_detector_electrons(irradiance_spectrum: np.ndarray, wavelength_m: np.ndarray,
                               pixel_area_m2: float, integration_time_s: float,
                               quantum_efficiency: float) -> float:
    """Photon-counting detector response (cooled MWIR). SS5b."""
    raise NotImplementedError("TODO: implement photon detector response, SS5b")
