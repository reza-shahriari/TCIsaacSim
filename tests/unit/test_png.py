"""The stdlib PNG writer (M5.7): valid signature and chunks, and a decodable payload."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from irsim.io.png import png_bytes


def test_png_structure_and_payload_round_trip() -> None:
    img = np.arange(4 * 3 * 4, dtype=np.uint8).reshape(3, 4, 4)
    data = png_bytes(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, ctype) == (4, 3, 8, 6)
    idat_len = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    raw = zlib.decompress(data[41 : 41 + idat_len])
    rows = np.frombuffer(raw, np.uint8).reshape(3, 1 + 4 * 4)
    assert np.all(rows[:, 0] == 0) and np.array_equal(rows[:, 1:].reshape(3, 4, 4), img)
    assert data[-12:-8] == struct.pack(">I", 0) and data[-8:-4] == b"IEND"
    gray = png_bytes(np.zeros((2, 2), np.uint8))
    assert struct.unpack(">B", gray[25:26])[0] == 0
    with pytest.raises(TypeError):
        png_bytes(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        png_bytes(np.zeros((2, 2, 2), np.uint8))
