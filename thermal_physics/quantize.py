"""ADC quantization. Implements docs/thermal_camera_model.md SS9."""
from __future__ import annotations
import numpy as np


def quantize(signal: np.ndarray, bits: int, v_min: float, v_max: float) -> np.ndarray:
    """Map signal in [v_min, v_max] to an integer in [0, 2^bits - 1]. SS9."""
    raise NotImplementedError("TODO: implement quantization, SS9")
