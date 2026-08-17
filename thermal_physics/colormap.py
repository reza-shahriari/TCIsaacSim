"""Signal to color -- palette lookup tables.

Implements docs/thermal_camera_model.md SS11. "ironbow" here is an original,
generic black -> purple -> red -> orange -> yellow -> white progression in
the spirit of a hot-body colormap (similar construction to matplotlib's
"hot") -- not a reproduction of any proprietary vendor palette's actual
control-point values.
"""
from __future__ import annotations
import numpy as np

PALETTES: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "ironbow": [
        (0.00, (0.00, 0.00, 0.00)),
        (0.25, (0.25, 0.00, 0.35)),
        (0.50, (0.75, 0.10, 0.00)),
        (0.75, (1.00, 0.65, 0.00)),
        (1.00, (1.00, 1.00, 0.85)),
    ],
}


def apply_palette(image_8bit: np.ndarray, palette: str = "white_hot") -> np.ndarray:
    """RGB(u,v) = Palette[DN_8(u,v)]. SS11. Returns an (H, W, 3) uint8 image."""
    image_8bit = np.asarray(image_8bit)
    if palette == "white_hot":
        return np.stack([image_8bit] * 3, axis=-1).astype(np.uint8)
    if palette == "black_hot":
        inv = (255 - image_8bit).astype(np.uint8)
        return np.stack([inv] * 3, axis=-1).astype(np.uint8)
    if palette in PALETTES:
        return _interpolate_palette(image_8bit, PALETTES[palette])
    raise ValueError(f"unknown palette {palette!r}; known: white_hot, black_hot, {list(PALETTES)}")


def _interpolate_palette(image_8bit: np.ndarray,
                          control_points: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    positions = np.array([p[0] for p in control_points])
    colors = np.array([p[1] for p in control_points])  # (N, 3) in [0,1]
    t = image_8bit.astype(np.float64) / 255.0
    channels = [np.interp(t, positions, colors[:, c]) for c in range(3)]
    rgb = np.stack(channels, axis=-1) * 255.0
    return rgb.astype(np.uint8)
