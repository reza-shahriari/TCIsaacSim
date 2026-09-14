"""Putting a synthetic clip through a real video codec, to find out what it costs.

roadmap ME.2b; ADR 0003, ADR 0023.

The indexed sets are stored as lossily-coded 8-bit video (`data/validation/datasets.yaml`), and
the analysers in :mod:`irsim.validation.codec` state a floor under everything measured on them.
This module is how that floor is *calibrated*: take a cube whose noise is known exactly, send it
through the same kind of encoder, and measure what comes back. Nothing here analyses anything --
it shells out to ffmpeg and returns an array -- which is why it lives in ``irsim_eval`` and the
statistics live in the physics core.

**Full range, stated explicitly.** ``gray`` and ``yuv420p`` are both YUV, but H.264 defaults to
the 16-235 studio range, and letting ffmpeg decide costs +-1 code on a round trip that should be
exact. With ``scale=in_range=full:out_range=full`` and ``-color_range pc`` on both ends, a CRF 0
round trip is **bit-exact** (``tests/unit/test_codec_floor.py`` asserts it), which is what makes a
lossy measurement attributable to the encoder rather than to the harness.

Measured with this module on a 320x256x60 cube of Boson-ratio noise at sigma_TVH = 1.5 DN
(ADR 0023 addendum): CRF 18 leaves 5 % of the temporal noise, CRF 23 leaves none at all, a
2000 kbit/s CBR stream leaves 75 % and 800 kbit/s leaves 57 %. The Halmstad set's own bitrate is
recorded as UNVERIFIED in the index, so these are the shape of the effect and not a correction to
apply to it.
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np
from numpy.typing import NDArray

from irsim_eval.video import ffmpeg_available

__all__ = ["h264_round_trip"]

_FULL_RANGE = ("-vf", "scale=in_range=full:out_range=full", "-color_range", "pc")


def h264_round_trip(
    cube: object,
    *,
    fps: float = 30.0,
    crf: int | None = None,
    bitrate_kbps: float | None = None,
    preset: str = "medium",
) -> NDArray[np.uint8]:
    """Encode a uint8 ``(T, V, H)`` cube with libx264 and decode it back, losses included.

    Exactly one of ``crf`` (constant quality; 0 is lossless) and ``bitrate_kbps`` (constant
    bitrate, the way a recorder is usually configured) must be given. Both dimensions must be
    even: 4:2:0 needs them, and padding here would change the array the caller measures.
    """
    if (crf is None) == (bitrate_kbps is None):
        raise ValueError("give exactly one of crf and bitrate_kbps")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not on PATH; install it to measure a codec floor")
    frames = np.asarray(cube)
    if frames.dtype != np.uint8:
        raise TypeError(f"the codec takes stored 8-bit codes, got {frames.dtype}")
    if frames.ndim != 3:
        raise ValueError(f"cube must be (T, V, H), got shape {frames.shape}")
    count, height, width = (int(n) for n in frames.shape)
    if height % 2 or width % 2:
        raise ValueError(f"4:2:0 needs even dimensions, got {height}x{width}")

    if crf is not None:
        quality = ["-crf", str(int(crf))]
    else:
        assert bitrate_kbps is not None  # guarded above; narrows the type for mypy
        quality = ["-b:v", f"{float(bitrate_kbps):g}k"]
    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        raw, coded = work / "in.raw", work / "out.mp4"
        raw.write_bytes(np.ascontiguousarray(frames).tobytes())
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{width}x{height}", "-r", f"{float(fps):g}", "-i", str(raw),
                "-c:v", "libx264", "-preset", preset, *quality, *_FULL_RANGE,
                "-pix_fmt", "yuv420p", str(coded),
            ],
            check=True,
        )  # fmt: skip
        decoded = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(coded),
                "-color_range", "pc", "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            check=True,
            capture_output=True,
        ).stdout  # fmt: skip

    out = np.frombuffer(decoded, dtype=np.uint8)
    if out.size != count * height * width:
        raise RuntimeError(
            f"decoded {out.size} bytes, expected {count * height * width} "
            f"({count} frames of {height}x{width}) -- the encoder dropped or added frames"
        )
    return np.asarray(out.reshape(count, height, width))
