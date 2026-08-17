"""Orchestrates SS1-SS11 into one per-frame render. Implements docs/thermal_camera_model.md SS12.

This is the function Phase 2's Isaac Sim writer prototype (and, later, the
SPG kernel) both exist to feed real per-pixel arrays into. Get this fully
working and tested against synthetic inputs (docs/isaacsim_checklist.md
Phase 1) before wiring it to anything Isaac-Sim-shaped.
"""
from __future__ import annotations
import numpy as np


def render_frame(temperature_k: np.ndarray, emissivity: np.ndarray, range_m: np.ndarray,
                  params: dict) -> np.ndarray:
    """Full per-pixel chain: emission -> atmosphere -> optics -> detector ->
    blur -> noise -> quantize -> AGC -> colormap. Returns an (H, W, 3) uint8 image.

    Args:
        temperature_k, emissivity, range_m: (H, W) arrays -- the point-wise
            scene fields this whole project exists to support (see SS0).
        params: sensor/scene parameters -- f_number, tau_optics, netd_k,
            atmosphere gamma, bit depth, palette name, etc. Consider a
            dataclass once the shape stabilizes.
    """
    raise NotImplementedError("TODO: wire SS1-SS11 together, SS12")
