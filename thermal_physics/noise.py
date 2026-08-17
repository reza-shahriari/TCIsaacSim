"""Temporal noise + fixed-pattern noise.

Implements docs/thermal_camera_model.md SS7-SS8.
"""
from __future__ import annotations
import numpy as np


def netd_noise_sigma(netd_k: float, d_signal_d_t: float) -> float:
    """sigma_noise = NETD * dSignal/dT. SS8."""
    return netd_k * d_signal_d_t


def add_temporal_noise(image: np.ndarray, sigma: float,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Independent per-pixel, per-frame Gaussian noise. SS8."""
    rng = rng or np.random.default_rng()
    image = np.asarray(image, dtype=np.float64)
    return image + rng.normal(0.0, sigma, size=image.shape)


def add_fixed_pattern_noise(image: np.ndarray, gain_sigma: float, offset_sigma: float,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-pixel gain/offset non-uniformity, constant across frames. SS7."""
    rng = rng or np.random.default_rng()
    image = np.asarray(image, dtype=np.float64)
    gain = rng.normal(1.0, gain_sigma, size=image.shape) if gain_sigma > 0 else 1.0
    offset = rng.normal(0.0, offset_sigma, size=image.shape) if offset_sigma > 0 else 0.0
    return image * gain + offset
