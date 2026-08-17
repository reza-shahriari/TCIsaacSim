"""Temporal noise + fixed-pattern noise.

Implements docs/thermal_camera_model.md SS7-SS8.
"""
from __future__ import annotations
import numpy as np


def netd_noise_sigma(netd_k: float, d_signal_d_t: float) -> float:
    """sigma_noise = NETD * dSignal/dT. SS8."""
    raise NotImplementedError("TODO: implement NETD-based noise sigma, SS8")


def add_temporal_noise(image: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Independent per-pixel, per-frame Gaussian noise. SS8."""
    raise NotImplementedError("TODO: add temporal noise, SS8")


def add_fixed_pattern_noise(image: np.ndarray, gain_sigma: float, offset_sigma: float,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-pixel gain/offset non-uniformity, constant across frames. SS7."""
    raise NotImplementedError("TODO: add fixed-pattern noise, SS7")
