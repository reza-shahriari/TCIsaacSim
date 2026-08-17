"""Automatic Gain Control -- dynamic range compression for display.

Implements docs/thermal_camera_model.md SS10.
"""
from __future__ import annotations
import numpy as np


def linear_agc(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Percentile-clipped linear scaling to 8-bit. SS10."""
    raise NotImplementedError("TODO: implement linear/percentile AGC, SS10")


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """CDF-based histogram equalization to 8-bit. SS10."""
    raise NotImplementedError("TODO: implement histogram equalization, SS10")
