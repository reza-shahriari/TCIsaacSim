"""ADC quantization. Implements docs/thermal_camera_model.md SS9."""
from __future__ import annotations
import numpy as np


def quantize(signal: np.ndarray, bits: int, v_min: float, v_max: float) -> np.ndarray:
    """Map signal in [v_min, v_max] to an integer in [0, 2^bits - 1], clamped
    (not wrapped) for values outside that range. SS9."""
    max_level = 2**bits - 1
    span = v_max - v_min
    normalized = (np.asarray(signal, dtype=np.float64) - v_min) / (span if span != 0 else 1e-12)
    clipped = np.clip(normalized, 0.0, 1.0)
    return np.round(clipped * max_level).astype(np.int64)
