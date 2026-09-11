"""Minimal PNG writer (8-bit gray, RGB, RGBA) with only the standard library.

The physics core may not import PIL / imageio / cv2 (layering guard, CLAUDE.md #1), yet a
human needs to look at the first image. PNG is zlib + a CRC, so this is thirty lines of stdlib.
Display images are uint8 only; anything radiometric goes to .npy / EXR, never to a PNG.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray

__all__ = ["write_png", "png_bytes"]

_COLOR_TYPES = {1: 0, 3: 2, 4: 6}  # channels -> PNG colour type (gray, RGB, RGBA)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def png_bytes(image: NDArray[np.uint8]) -> bytes:
    """Encode a uint8 (H, W), (H, W, 3) or (H, W, 4) array as PNG bytes."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(f"PNG output is uint8 display data, got {arr.dtype}")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in _COLOR_TYPES:
        raise ValueError("image must be (H, W), (H, W, 3) or (H, W, 4)")
    h, w, c = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(h))  # filter type 0 per row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _COLOR_TYPES[c], 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: str | os.PathLike[str], image: NDArray[np.uint8]) -> None:
    with open(path, "wb") as fh:
        fh.write(png_bytes(image))
