"""M10.10b: the ROS 2 message arithmetic, checked without a ROS graph.

ROS 2 is a system-level install, not a pip dependency this repository can pull in, and it is not
available here. So everything that can be checked without it is: the intrinsics, the encodings, the
stamp, and which outputs get published at all. The part that needs rclpy is one function, and it
names what is missing rather than failing with an import traceback.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim_isaac.ros2_bridge import (
    TOPIC_ENCODINGS,
    camera_info,
    frame_messages,
    image_message,
    publish_frame,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


@dataclass
class _Outputs:
    apparent_t: np.ndarray | None = None
    radiance: np.ndarray | None = None
    dn16: np.ndarray | None = None
    display8: np.ndarray | None = None


def test_fx_is_the_focal_length_over_the_pitch() -> None:
    """The one number a `CameraInfo` exists to carry, derived from the sensor's own blocks so a
    camera and its intrinsics cannot disagree."""
    sensor = load_sensor_config(BOSON)
    spec = sensor.sensor
    info = camera_info(sensor, t_s=1.5)
    expected = spec.optics.focal_length_mm * 1000.0 / spec.fpa.pitch_um
    assert info.fx == pytest.approx(expected, rel=1e-3)
    assert info.fy == pytest.approx(info.fx), "square pixels"
    assert (info.width, info.height) == (spec.fpa.width, spec.fpa.height)


def test_the_principal_point_uses_the_pixel_edge_convention() -> None:
    """``width/2``, not ``(width - 1)/2``. The half-pixel difference is invisible in a preview and
    is a systematic half-pixel bias in anything that triangulates."""
    sensor = load_sensor_config(BOSON)
    info = camera_info(sensor, t_s=0.0)
    assert info.cx == pytest.approx(info.width / 2.0)
    assert info.cy == pytest.approx(info.height / 2.0)
    assert info.k[1] == 0.0 and info.k[3] == 0.0 and info.k[8] == 1.0


def test_the_distortion_travels_exactly_as_the_config_wrote_it() -> None:
    sensor = load_sensor_config(BOSON)
    info = camera_info(sensor, t_s=0.0)
    optics = sensor.sensor.optics
    assert info.distortion_model == optics.distortion.model
    assert info.d == tuple(float(c) for c in optics.distortion.coeffs)


def test_the_stamp_is_integer_nanoseconds_of_sim_time() -> None:
    """A ROS stamp is (sec, nanosec); a float second at hour scale has already lost sub-microsecond
    resolution, so the conversion happens once, here, in integers."""
    sensor = load_sensor_config(BOSON)
    assert camera_info(sensor, t_s=0.0).stamp_ns == 0
    assert camera_info(sensor, t_s=1.5).stamp_ns == 1_500_000_000
    assert camera_info(sensor, t_s=3600.000_000_5).stamp_ns == 3_600_000_000_500


def test_apparent_temperature_goes_out_in_float32_and_never_narrower() -> None:
    """Non-negotiable #2 on the wire: at 300 K a float16 grid is 0.25 K coarse, five times a 50 mK
    NETD, so a subscriber would be reading a camera five times worse than the one that sent it."""
    plane = np.full((4, 4), 300.0, np.float32)
    message = image_message("apparent_t", plane, t_s=2.0)
    assert message.encoding == "32FC1" and message.data.dtype == np.float32
    assert message.step == 4 * 4  # one float32 channel
    assert message.stamp_ns == 2_000_000_000
    # float64 in is narrowed to the wire form; float16 is refused rather than widened, because a
    # float16 plane has already lost the precision and widening it would hide that.
    assert image_message("apparent_t", plane.astype(np.float64), t_s=0.0).data.dtype == np.float32
    with pytest.raises(TypeError, match="32FC1"):
        image_message("apparent_t", plane.astype(np.float16), t_s=0.0)


def test_each_output_takes_the_encoding_that_does_not_lose_it() -> None:
    assert TOPIC_ENCODINGS["dn16"][1] == "mono16"
    assert TOPIC_ENCODINGS["display8"][1] == "rgba8"
    raw = image_message("dn16", np.zeros((3, 5), np.uint16), t_s=0.0)
    assert raw.step == 5 * 2 and raw.topic.endswith("/image_raw")
    colour = image_message("display8", np.zeros((3, 5, 4), np.uint8), t_s=0.0)
    assert colour.step == 5 * 4 and colour.topic.endswith("/image_color")
    with pytest.raises(TypeError, match="mono16"):
        image_message("dn16", np.zeros((3, 5), np.float32), t_s=0.0)
    with pytest.raises(KeyError, match="unknown output"):
        image_message("flux", np.zeros((3, 5), np.float32), t_s=0.0)


def test_an_output_the_config_switched_off_is_skipped_not_zeroed() -> None:
    """A subscriber receiving a black `apparent_t` topic has no way to tell it from a scene at
    absolute zero."""
    sensor = load_sensor_config(BOSON)
    outputs = _Outputs(dn16=np.zeros((2, 2), np.uint16), display8=np.zeros((2, 2, 4), np.uint8))
    messages, info = frame_messages(outputs, sensor, t_s=0.25)
    topics = {m.topic for m in messages}
    assert topics == {"/ir/image_raw", "/ir/image_color"}
    assert all(m.stamp_ns == 250_000_000 for m in messages)
    assert info.stamp_ns == 250_000_000 and info.frame_id == "ir_camera"


def test_publishing_without_ros_says_what_is_missing() -> None:
    """The one call that needs a ROS graph names the dependency instead of raising ImportError
    from somewhere inside a library nobody asked for."""
    sensor = load_sensor_config(BOSON)
    messages, info = frame_messages(_Outputs(dn16=np.zeros((2, 2), np.uint16)), sensor, t_s=0.0)
    with pytest.raises(RuntimeError, match="system-level install"):
        publish_frame(object(), messages, info)
