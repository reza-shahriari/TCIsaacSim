"""Decode a published video clip to the 8-bit frames every Tier 4 statistic is measured on.

roadmap ME.5; ADR 0003, ADR 0068.

Lives in ``irsim_eval`` and not in ``irsim`` because it shells out to ffmpeg, which CLAUDE.md keeps
out of the physics core.

**Two facts about this decode bound every number derived from it, and both are recorded rather than
smoothed over.**

*The luma plane is the image.* These clips are ``yuv420p``: the chroma planes are subsampled 2x2 and
carry nothing, because the source was monochrome. Decoding to RGB and averaging would mix the
decoder's own YUV->RGB matrix into the data, so the luma plane is taken directly with
``-pix_fmt gray``.

*The range flag is missing.* ``color_range`` is ``unknown`` on every clip in the reference set, so
nothing in the file says whether code 16 means black (limited/TV range) or code 0 does (full/PC).
The two readings differ by a gain of 255/219 = 1.16 and an offset of 16 codes. ffmpeg's default for
an unflagged h.264 stream is limited range, and that is what :func:`decode_gray8` returns -- but
scale-free statistics are the only ones this ambiguity leaves intact, which is exactly why ADR 0068
made them the deliverable. :func:`range_ambiguity_codes` quantifies it for the report.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FFMPEG",
    "FFPROBE",
    "ClipInfo",
    "ffmpeg_available",
    "probe_clip",
    "decode_gray8",
    "range_ambiguity_codes",
]

FFMPEG = os.environ.get("IRSIM_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("IRSIM_FFPROBE", "ffprobe")


class ClipInfo(dict[str, object]):
    """What ffprobe says about one clip. A dict so it lands in the report's JSON unchanged."""

    @property
    def shape(self) -> tuple[int, int]:
        return int(str(self["height"])), int(str(self["width"]))

    @property
    def fps(self) -> float:
        return float(str(self["frame_rate_hz"]))


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def probe_clip(path: str | os.PathLike[str]) -> ClipInfo:
    """Width, height, frame rate, codec and colour range, straight from the container.

    The frame rate is read from the file and **not** from the index: the reference set's camera
    core runs at 60 Hz and its clips are stored at 30, and a one-pole temporal fit run at the wrong
    rate returns a time constant that is wrong by the same factor while looking entirely plausible.
    """
    out = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,codec_name,pix_fmt,color_range,nb_frames,r_frame_rate,profile",
            "-show_entries", "format=duration,bit_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    raw = json.loads(out.stdout)
    stream = raw["streams"][0]
    fmt = raw.get("format", {})
    num, _, den = str(stream["r_frame_rate"]).partition("/")
    return ClipInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec=stream.get("codec_name"),
        profile=stream.get("profile"),
        pix_fmt=stream.get("pix_fmt"),
        colour_range=stream.get("color_range", "unknown"),
        frame_rate_hz=float(num) / float(den or 1),
        frames=int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        duration_s=float(fmt["duration"]) if fmt.get("duration") else None,
        bit_rate_kbps=float(fmt["bit_rate"]) / 1e3 if fmt.get("bit_rate") else None,
    )


def decode_gray8(path: str | os.PathLike[str], max_frames: int | None = None) -> NDArray[np.uint8]:
    """``(T, H, W)`` uint8 luma. Raises if the stream is empty or truncated mid-frame."""
    info = probe_clip(path)
    height, width = info.shape
    command = [FFMPEG, "-v", "error", "-i", str(path)]
    if max_frames is not None:
        command += ["-frames:v", str(int(max_frames))]
    command += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    result = subprocess.run(command, capture_output=True, check=True)
    stride = height * width
    if not result.stdout:
        raise ValueError(f"{pathlib.Path(path).name}: decoded to no frames")
    if len(result.stdout) % stride:
        raise ValueError(
            f"{pathlib.Path(path).name}: {len(result.stdout)} bytes is not a whole number of "
            f"{height}x{width} frames"
        )
    flat = np.frombuffer(result.stdout, dtype=np.uint8)
    return np.asarray(flat.reshape(-1, height, width).copy(), dtype=np.uint8)


def range_ambiguity_codes(frames: NDArray[np.uint8]) -> float:
    """How many DN8 codes the missing ``color_range`` flag is worth, on these frames.

    Reading an unflagged stream as full range rather than limited scales it by 219/255 and shifts
    it by 16, so the disagreement grows with the level: it is ``(255/219 - 1)·mean + 16·(…)``
    evaluated on the frames themselves rather than assumed. Reported next to every absolute-level
    statistic so a reader can see which ones the ambiguity swallows.
    """
    values = np.asarray(frames, dtype=np.float64)
    limited = values
    full = np.clip((values - 16.0) * (255.0 / 219.0), 0.0, 255.0)
    return float(np.mean(np.abs(limited - full)))
