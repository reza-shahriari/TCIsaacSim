"""Dataset writers (M10.10a): what reaches disk must still be the physics that left the pipeline.

The failure this module guards is not a crash. It is a frame that writes, opens, looks right, and
has had its sensitivity quantised away -- a 16-bit PNG of apparent temperature, or a half-float
EXR. So the tests are about **exactness in physical units** and about the refusals, not about
whether a file appears.

docs/physics-model.md §12.2 outputs; CLAUDE.md non-negotiable #2.
"""

from __future__ import annotations

import json
import pathlib
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from irsim.io import PLANE_UNITS, read_float_plane, write_frame, write_png
from irsim.io.exr import MAGIC, PIXEL_TYPE_FLOAT, exr_bytes, read_exr, write_exr

#: The §12.2 radiometric span. A 16-bit integer container over it quantises to 3 mK; half-float at
#: 300 K to 250 mK. Both are compared against a 50 mK NETD below.
RANGE_K = (233.15, 473.15)
NETD_MK = 50.0


@dataclass
class FakeOutputs:
    """Just the four §12.2 outputs -- the writer takes anything shaped like ``Outputs``."""

    radiance: Any = None
    apparent_t: Any = None
    dn16: Any = None
    display8: Any = None
    isp_hash: str | None = None


def temperature_ramp(h: int = 16, w: int = 32) -> np.ndarray:
    """A float32 ramp across the radiometric range, with sub-millikelvin structure on it."""
    base = np.linspace(RANGE_K[0], RANGE_K[1], h * w, dtype=np.float64).reshape(h, w)
    fine = 1e-4 * np.arange(h * w, dtype=np.float64).reshape(h, w) % 1e-3
    return np.asarray(base + fine, dtype=np.float32)


# --- the disk rule --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["npy", "exr"])
def test_a_temperature_ramp_round_trips_to_under_a_millikelvin(
    tmp_path: pathlib.Path, fmt: str
) -> None:
    """Write, read, compare in kelvin: both float32 containers must be bit-exact, so 0 mK.

    Stated as a millikelvin budget rather than "equal" because that is the quantity that matters:
    the same ramp through a 16-bit PNG would come back 3 mK out and through half-float 250 mK out,
    the second being five times the NETD the whole sensor is anchored to.
    """
    ramp = temperature_ramp()
    out = FakeOutputs(apparent_t=ramp)
    record = write_frame(tmp_path, out, float_format=fmt)  # type: ignore[arg-type]
    back = read_float_plane(record.files["apparent_t"])
    assert back.dtype == np.float32
    error_mk = float(np.abs(back.astype(np.float64) - ramp.astype(np.float64)).max()) * 1e3
    assert error_mk == 0.0
    assert error_mk < 0.001 * NETD_MK


def test_the_quantisation_a_16_bit_png_would_have_cost_is_not_small(tmp_path: pathlib.Path) -> None:
    """The reason for the rule, stated as a number rather than an opinion.

    This does not test our code -- it tests the premise the code is built on, so that a future
    reader who wonders why radiance may not go to a PNG has the arithmetic in front of them.
    """
    span = RANGE_K[1] - RANGE_K[0]
    png16_mk = span / (2**16 - 1) * 1e3
    half_at_300k_mk = float(np.spacing(np.float16(300.0))) * 1e3
    assert png16_mk == pytest.approx(3.66, abs=0.01)
    assert half_at_300k_mk == pytest.approx(250.0, abs=1.0)
    # CLAUDE.md #2 says "five times coarser than a 50 mK NETD". It is exactly five times.
    assert half_at_300k_mk == pytest.approx(5 * NETD_MK, rel=1e-9)


def test_a_float16_plane_is_refused_at_the_disk_boundary(tmp_path: pathlib.Path) -> None:
    """float16 must not reach disk by any route: the writer, and the EXR encoder underneath it."""
    half = temperature_ramp().astype(np.float16)
    with pytest.raises(TypeError, match="float16"):
        write_frame(tmp_path, FakeOutputs(apparent_t=half))
    with pytest.raises(TypeError, match="float16"):
        write_exr(tmp_path / "x.exr", half)


def test_a_half_float_exr_is_refused_even_when_asked_for(tmp_path: pathlib.Path) -> None:
    """EXR's most common pixel type is the one that destroys the sensor, so asking still fails."""
    with pytest.raises(ValueError, match="half-float EXR is refused"):
        write_exr(tmp_path / "x.exr", temperature_ramp(), half=True)


def test_radiometric_planes_never_land_in_an_integer_container(tmp_path: pathlib.Path) -> None:
    """radiance and apparent_t go to .npy/.exr; only dn16 and display8 are PNGs."""
    out = FakeOutputs(
        radiance=np.full((4, 4), 12.5, np.float32),
        apparent_t=temperature_ramp(4, 4),
        dn16=np.arange(16, dtype=np.uint16).reshape(4, 4),
        display8=np.zeros((4, 4, 4), np.uint8),
    )
    record = write_frame(tmp_path, out)
    assert record.files["radiance"].suffix == ".npy"
    assert record.files["apparent_t"].suffix == ".npy"
    assert record.files["dn16"].suffix == ".png"
    assert record.files["display8"].suffix == ".png"


# --- the containers -------------------------------------------------------------------------


def test_dn16_survives_the_png_exactly(tmp_path: pathlib.Path) -> None:
    """An ADC code is an integer, so the one PNG that carries physics loses nothing.

    Decoded here from the raw chunks rather than with an image library -- the core may not import
    one (CLAUDE.md #1), and decoding the file we wrote against the PNG specification is a stronger
    check than round-tripping through our own encoder twice.
    """
    codes = np.array([[0, 1, 255, 256], [4095, 30000, 65534, 65535]], dtype=np.uint16)
    path = tmp_path / "dn.png"
    write_png(path, codes)
    data = path.read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    (length,) = struct.unpack(">I", data[8:12])
    assert data[12:16] == b"IHDR"
    width, height, depth, colour = struct.unpack(">IIBB", data[16 : 16 + 10])
    assert (width, height, depth, colour) == (4, 2, 16, 0)  # 16-bit grayscale

    pos = 8 + 12 + length
    idat = b""
    while pos < len(data):
        (size,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        if tag == b"IDAT":
            idat += data[pos + 8 : pos + 8 + size]
        pos += 12 + size
    raw = zlib.decompress(idat)
    rows = [raw[r * (1 + width * 2) : (r + 1) * (1 + width * 2)] for r in range(height)]
    assert all(row[0] == 0 for row in rows), "filter type 0 on every row"
    decoded = np.stack([np.frombuffer(row[1:], dtype=">u2") for row in rows])
    assert np.array_equal(decoded, codes)


def test_a_uint16_png_must_be_single_channel(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="single-channel"):
        write_png(tmp_path / "x.png", np.zeros((2, 2, 3), np.uint16))


def test_exr_header_matches_the_specification(tmp_path: pathlib.Path) -> None:
    """Magic, version, FLOAT pixel type and the little-endian sample bytes, checked by hand.

    The reader below is ours, so a round trip alone would pass even with the byte order wrong in
    both directions. This pins the parts a real EXR reader depends on.
    """
    image = np.array([[1.5, -2.25], [0.0, 1e6]], dtype=np.float32)
    blob = exr_bytes(image)
    magic, version = struct.unpack_from("<II", blob, 0)
    assert magic == MAGIC
    assert version & 0xFF == 2
    assert b"channels\x00chlist\x00" in blob
    assert struct.pack("<i", PIXEL_TYPE_FLOAT) in blob
    # Every sample appears verbatim, little-endian, in row order.
    assert np.ascontiguousarray(image[0], dtype="<f4").tobytes() in blob
    assert np.ascontiguousarray(image[1], dtype="<f4").tobytes() in blob


def test_exr_round_trips_a_two_dimensional_plane(tmp_path: pathlib.Path) -> None:
    ramp = temperature_ramp(9, 7)  # deliberately not square and not a power of two
    path = tmp_path / "t.exr"
    write_exr(path, ramp)
    back = read_exr(path)
    assert back.shape == ramp.shape
    assert np.array_equal(back, ramp)


def test_exr_channels_come_back_in_alphabetical_order(tmp_path: pathlib.Path) -> None:
    """EXR stores channels by name, so RGB written comes back B, G, R -- pinned, not surprising."""
    rgb = np.stack(
        [np.full((3, 3), v, np.float32) for v in (1.0, 2.0, 3.0)], axis=-1
    )  # R=1, G=2, B=3
    path = tmp_path / "rgb.exr"
    write_exr(path, rgb, channels=("R", "G", "B"))
    back = read_exr(path)
    assert [float(back[0, 0, i]) for i in range(3)] == [3.0, 2.0, 1.0]


# --- the sidecar ----------------------------------------------------------------------------


def test_the_sidecar_says_what_every_plane_is(tmp_path: pathlib.Path) -> None:
    """Units, dtypes, shapes and hashes: without them a directory of arrays is not a dataset."""
    out = FakeOutputs(
        radiance=np.full((4, 4), 12.5, np.float32),
        apparent_t=temperature_ramp(4, 4),
        dn16=np.zeros((4, 4), np.uint16),
        display8=np.zeros((4, 4, 4), np.uint8),
        isp_hash="ispdeadbeef",
    )
    record = write_frame(
        tmp_path,
        out,
        frame_index=7,
        t_s=3600.0 + 1.5,
        start_utc=datetime(2024, 6, 21, 4, 0, tzinfo=timezone.utc),
        config_hash="cfg123",
        band_hash="band456",
        quantity="lb",
    )
    meta = json.loads(record.sidecar.read_text())
    assert meta["frame_index"] == 7
    assert meta["config_hash"] == "cfg123"
    assert meta["band_hash"] == "band456"
    assert meta["isp_hash"] == "ispdeadbeef"
    assert meta["quantity"] == "lb"
    assert meta["utc"] == "2024-06-21T05:00:01.500000+00:00"
    assert meta["planes"]["apparent_t"]["unit"] == PLANE_UNITS["apparent_t"] == "K"
    assert meta["planes"]["apparent_t"]["dtype"] == "float32"
    assert meta["planes"]["dn16"]["shape"] == [4, 4]
    assert meta["planes"]["display8"]["shape"] == [4, 4, 4]


def test_a_frame_index_sorts_lexicographically(tmp_path: pathlib.Path) -> None:
    """frame_000009 before frame_000010: a bare 9/10 puts a sequence out of order in every tool."""
    blank = FakeOutputs(dn16=np.zeros((2, 2), np.uint16))
    names = [write_frame(tmp_path, blank, frame_index=i).metadata["name"] for i in (9, 10, 100)]
    assert names == sorted(names)


def test_an_output_that_is_off_is_absent_rather_than_zero(tmp_path: pathlib.Path) -> None:
    """A disabled output must leave no file: zeros are indistinguishable from a real dark frame."""
    record = write_frame(tmp_path, FakeOutputs(apparent_t=temperature_ramp(4, 4)))
    assert set(record.files) == {"apparent_t"}
    assert not list(tmp_path.glob("*dn16*"))
    assert not list(tmp_path.glob("*display8*"))
    assert json.loads(record.sidecar.read_text())["planes"].keys() == {"apparent_t"}


def test_missing_time_is_recorded_as_null_not_invented(tmp_path: pathlib.Path) -> None:
    record = write_frame(tmp_path, FakeOutputs(dn16=np.zeros((2, 2), np.uint16)))
    meta = json.loads(record.sidecar.read_text())
    assert meta["utc"] is None and meta["t_s"] is None and meta["config_hash"] is None
