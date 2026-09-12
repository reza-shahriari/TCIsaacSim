"""Minimal OpenEXR writer: single-part, scanline, **float32**, no compression. Stdlib only.

docs/physics-model.md §12.2 ``outputs``; roadmap M10.10a; CLAUDE.md non-negotiable #2 (the disk
rule) and #1 (the core imports no imaging stack).

``.npy`` already stores float32 exactly and is what the test suite reads back, so why EXR at all:
a radiance or apparent-temperature frame has to be openable by the people who look at imagery --
in a DCC, a viewer, an annotation tool -- and none of them read ``.npy``. The alternative in
practice is a 16-bit PNG, which is precisely the silent failure non-negotiable #2 exists to
prevent: at 300 K over a 200--400 K range, 16-bit unorm quantises to 3 mK and **half** float to
0.25 K, five times coarser than a 50 mK NETD, while the picture still looks perfectly good.

So this writer only ever emits ``FLOAT`` (32-bit) samples. :data:`HALF_REFUSED` explains why a
half-float request raises rather than being honoured -- EXR's default and most common pixel type
is exactly the one that would destroy the sensor's sensitivity.

No compression, deliberately: EXR's compressors (ZIP, PIZ, DWA) range from lossless-but-fiddly to
lossy, the files here are small, and an uncompressed scanline file is about eighty lines of
struct packing that can be checked against the specification byte for byte. If file size ever
matters, ZIPS is the next step and it is still lossless.

Format reference: the OpenEXR file layout -- magic ``0x01312f76``, version 2, a null-terminated
attribute list, a scanline offset table, then one block per scanline. Everything is little-endian.
"""

from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MAGIC",
    "PIXEL_TYPE_FLOAT",
    "HALF_REFUSED",
    "exr_bytes",
    "write_exr",
    "read_exr",
]

MAGIC = 0x01312F76
_VERSION = 2
PIXEL_TYPE_FLOAT = 2  # 0 = UINT, 1 = HALF, 2 = FLOAT
_NO_COMPRESSION = 0
_INCREASING_Y = 0

HALF_REFUSED = (
    "half-float EXR is refused: at 300 K its spacing is 0.25 K, five times coarser than a 50 mK "
    "NETD, so a half frame looks correct and has no sensitivity left in it (CLAUDE.md #2). "
    "Write float32 -- that is the only pixel type this writer emits."
)


def _attr(name: str, kind: str, payload: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\0"
        + kind.encode("ascii")
        + b"\0"
        + struct.pack("<i", len(payload))
        + payload
    )


def _channel_list(names: tuple[str, ...]) -> bytes:
    # Channels must be stored in alphabetical order; a reader matches them by name, so an
    # out-of-order list silently pairs the wrong data with the wrong channel.
    body = b""
    for name in sorted(names):
        body += (
            name.encode("ascii")
            + b"\0"
            + struct.pack("<i", PIXEL_TYPE_FLOAT)
            + struct.pack("<B", 0)  # pLinear
            + b"\0\0\0"  # reserved
            + struct.pack("<ii", 1, 1)  # xSampling, ySampling
        )
    return body + b"\0"


def exr_bytes(image: Any, *, channels: tuple[str, ...] | None = None, half: bool = False) -> bytes:
    """Encode a float32 ``(H, W)`` or ``(H, W, C)`` array as an uncompressed scanline EXR.

    ``channels`` names the channels (default ``("Y",)`` for one, ``("R", "G", "B")`` for three).
    ``half=True`` raises: see :data:`HALF_REFUSED`.
    """
    if half:
        raise ValueError(HALF_REFUSED)
    arr = np.asarray(image)
    if arr.dtype == np.float16:
        raise TypeError(f"refusing to write a float16 array to EXR. {HALF_REFUSED}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"EXR output is floating-point data, got {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype="<f4")
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"image must be (H, W) or (H, W, C), got {np.asarray(image).shape}")
    height, width, count = arr.shape
    if height == 0 or width == 0:
        raise ValueError("image must not be empty")
    if channels is None:
        channels = ("Y",) if count == 1 else ("R", "G", "B", "A")[:count]
    if len(channels) != count:
        raise ValueError(f"{len(channels)} channel names for {count} channels")

    header = struct.pack("<II", MAGIC, _VERSION)
    header += _attr("channels", "chlist", _channel_list(channels))
    header += _attr("compression", "compression", struct.pack("<B", _NO_COMPRESSION))
    header += _attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, width - 1, height - 1))
    header += _attr("lineOrder", "lineOrder", struct.pack("<B", _INCREASING_Y))
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"

    order = [channels.index(name) for name in sorted(channels)]
    row_bytes = width * 4 * count
    block = 8 + row_bytes  # y (int32) + data size (int32) + the samples
    first = len(header) + 8 * height
    offsets = b"".join(struct.pack("<Q", first + y * block) for y in range(height))

    body = bytearray()
    for y in range(height):
        body += struct.pack("<ii", y, row_bytes)
        for index in order:
            body += arr[y, :, index].tobytes()
    return header + offsets + bytes(body)


def write_exr(
    path: str | os.PathLike[str],
    image: Any,
    *,
    channels: tuple[str, ...] | None = None,
    half: bool = False,
) -> None:
    with open(path, "wb") as fh:
        fh.write(exr_bytes(image, channels=channels, half=half))


def read_exr(path: str | os.PathLike[str]) -> NDArray[np.float32]:
    """Read back a file written by :func:`write_exr`; returns ``(H, W)`` or ``(H, W, C)`` float32.

    Deliberately narrow -- it reads this writer's own dialect (uncompressed, FLOAT, increasing y)
    and raises on anything else rather than pretending to be an EXR library.

    **Multi-channel files come back in alphabetical channel order**, because that is the order EXR
    stores them in and the file records names, not positions. Writing ``("R", "G", "B")`` and
    reading back therefore yields B, G, R. Single-channel planes -- which is what every
    radiometric output here is -- are unaffected.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MAGIC:
        raise ValueError(f"not an EXR file (magic {magic:#x})")
    if version & 0xFF != _VERSION:
        raise ValueError(f"unsupported EXR version {version & 0xFF}")

    pos = 8
    names: list[str] = []
    window: tuple[int, int, int, int] | None = None
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode("ascii")
        pos = end + 1
        end = data.index(b"\0", pos)
        kind = data[pos:end].decode("ascii")
        pos = end + 1
        (size,) = struct.unpack_from("<i", data, pos)
        pos += 4
        payload = data[pos : pos + size]
        pos += size
        if name == "channels":
            at = 0
            while payload[at] != 0:
                stop = payload.index(b"\0", at)
                names.append(payload[at:stop].decode("ascii"))
                (pixel_type,) = struct.unpack_from("<i", payload, stop + 1)
                if pixel_type != PIXEL_TYPE_FLOAT:
                    raise ValueError(f"channel {names[-1]!r} is not FLOAT (type {pixel_type})")
                at = stop + 1 + 16
        elif name == "compression" and payload[0] != _NO_COMPRESSION:
            raise ValueError(f"compressed EXR (method {payload[0]}) is not supported here")
        elif name == "dataWindow":
            window = struct.unpack_from("<iiii", payload, 0)
        del kind
    pos += 1

    if window is None or not names:
        raise ValueError("EXR header has no dataWindow or no channels")
    x_min, y_min, x_max, y_max = window
    width, height = x_max - x_min + 1, y_max - y_min + 1
    pos += 8 * height
    out = np.zeros((height, width, len(names)), dtype=np.float32)
    for _ in range(height):
        y, _size = struct.unpack_from("<ii", data, pos)
        pos += 8
        for index in range(len(names)):
            row = np.frombuffer(data, dtype="<f4", count=width, offset=pos)
            out[y - y_min, :, index] = row
            pos += width * 4
    return out[:, :, 0] if len(names) == 1 else out
