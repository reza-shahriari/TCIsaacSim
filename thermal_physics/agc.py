"""Automatic Gain Control -- dynamic range compression for display.

Implements docs/thermal_camera_model.md SS10.
"""
from __future__ import annotations
import numpy as np


def linear_agc(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Percentile-clipped linear scaling to 8-bit. SS10."""
    image = np.asarray(image, dtype=np.float64)
    lo = np.percentile(image, low_percentile)
    hi = np.percentile(image, high_percentile)
    if hi <= lo:
        hi = lo + 1e-9
    normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.round(normalized * 255).astype(np.uint8)


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """CDF-based histogram equalization to 8-bit. SS10."""
    image = np.asarray(image, dtype=np.float64)
    hist, bin_edges = np.histogram(image.flatten(), bins=256)
    cdf = hist.cumsum().astype(np.float64)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    mapped = np.interp(image.flatten(), bin_edges[:-1], cdf)
    return (mapped.reshape(image.shape) * 255).astype(np.uint8)
