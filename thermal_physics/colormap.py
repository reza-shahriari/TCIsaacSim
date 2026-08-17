"""Signal to color -- palette lookup tables.

Implements docs/thermal_camera_model.md SS11. Consider an open colormap
(matplotlib inferno/magma/plasma, or Google's Turbo) rather than trying to
reproduce a proprietary vendor palette -- see SS11's note on this.
"""
from __future__ import annotations
import numpy as np

PALETTES: dict[str, object] = {
    # TODO: "white_hot", "black_hot", and at least one pseudocolor palette. SS11.
}


def apply_palette(image_8bit: np.ndarray, palette: str = "white_hot") -> np.ndarray:
    """RGB(u,v) = Palette[DN_8(u,v)]. SS11."""
    raise NotImplementedError("TODO: implement palette lookup, SS11")
