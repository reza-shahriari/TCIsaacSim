"""Polarity, palette look-up tables and 8-bit quantisation (§11.4, §13.4, ADR 0030).

DN8 = round(255 · clip(y, 0, 1)); white-hot indexes the table with DN8, black-hot with
255 − DN8. Output is always RGBA8 (H, W, 4) with alpha 255: the Isaac viewport displays RGBA
unorm textures, and a fixed layout avoids a per-consumer channel switch.

Palette provenance: gray is the identity. Ironbow, lava, rainbow and arctic are **public
approximations** of the FLIR palettes of the same names, built from a few RGB control points
with linear interpolation (the control-point sets are the ones circulating in open-source
thermal viewers; they are not FLIR's tables). Perception models are sensitive to palette (§11.4),
so the point is reproducibility -- training and deployment use the same table -- not fidelity to
a particular vendor's curve. Ironbow and lava have non-decreasing Rec.601 luma along the index;
rainbow has 256 distinct entries.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PALETTES",
    "PaletteName",
    "Polarity",
    "palette_table",
    "quantise_display",
    "to_display8",
]

PaletteName = Literal["gray", "ironbow", "rainbow", "lava", "arctic"]
Polarity = Literal["white_hot", "black_hot"]

# (index in 0..255, (R, G, B)) control points; linear interpolation between them.
_CONTROL_POINTS: dict[str, list[tuple[int, tuple[int, int, int]]]] = {
    "ironbow": [
        (0, (0, 0, 0)),
        (36, (32, 0, 64)),
        (80, (128, 0, 128)),
        (120, (192, 0, 128)),
        (160, (255, 64, 0)),
        (200, (255, 160, 0)),
        (232, (255, 224, 64)),
        (255, (255, 255, 255)),
    ],
    "lava": [
        (0, (0, 0, 0)),
        (48, (64, 0, 0)),
        (112, (192, 0, 0)),
        (176, (255, 128, 0)),
        (224, (255, 224, 32)),
        (255, (255, 255, 255)),
    ],
    "arctic": [
        (0, (0, 0, 64)),
        (64, (0, 96, 192)),
        (128, (160, 224, 255)),
        (176, (255, 255, 255)),
        (216, (255, 224, 96)),
        (255, (255, 96, 0)),
    ],
}


def _interpolate(points: list[tuple[int, tuple[int, int, int]]]) -> NDArray[np.uint8]:
    idx = np.array([p[0] for p in points], dtype=np.float64)
    rgb = np.array([p[1] for p in points], dtype=np.float64)
    x = np.arange(256, dtype=np.float64)
    table = np.stack([np.interp(x, idx, rgb[:, c]) for c in range(3)], axis=1)
    return np.asarray(np.rint(table), dtype=np.uint8)


def _rainbow() -> NDArray[np.uint8]:
    """Hue sweep blue (240°) → red (0°) at full saturation and value: 256 distinct entries."""
    h = np.linspace(240.0, 0.0, 256) / 60.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    v, p, q, t = 1.0, 0.0, 1.0 - f, f
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.asarray(np.rint(np.stack([r, g, b], axis=1) * 255.0), dtype=np.uint8)


def _build() -> dict[str, NDArray[np.uint8]]:
    gray = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    return {
        "gray": gray,
        "ironbow": _interpolate(_CONTROL_POINTS["ironbow"]),
        "lava": _interpolate(_CONTROL_POINTS["lava"]),
        "arctic": _interpolate(_CONTROL_POINTS["arctic"]),
        "rainbow": _rainbow(),
    }


PALETTES: dict[str, NDArray[np.uint8]] = _build()


def palette_table(name: str) -> NDArray[np.uint8]:
    """The 256×3 uint8 table for ``name``."""
    if name not in PALETTES:
        raise ValueError(f"unknown palette {name!r}; known: {sorted(PALETTES)}")
    return PALETTES[name]


def quantise_display(y: object) -> NDArray[np.uint8]:
    """DN8 = round(255 · clip(y, 0, 1)) as uint8; never wraps."""
    arr = np.asarray(y)
    if arr.dtype == np.float16:
        raise TypeError("display input is float16 (non-negotiable #2)")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError("display input must be the float image in [0, 1]")
    return np.asarray(np.rint(np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0), dtype=np.uint8)


def to_display8(
    y: object, polarity: Polarity = "white_hot", palette: str = "gray"
) -> NDArray[np.uint8]:
    """[0, 1] image → RGBA8 (H, W, 4) through polarity and palette (alpha = 255)."""
    dn8 = quantise_display(y)
    if polarity == "black_hot":
        dn8 = (255 - dn8.astype(np.int16)).astype(np.uint8)
    elif polarity != "white_hot":
        raise ValueError(f"unknown polarity {polarity!r}")
    rgb = palette_table(palette)[dn8]
    alpha = np.full((*dn8.shape, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)
