"""M10.23/M10.24: the manual display span, and why it has to know which band it is in.

Both of §11.3's AGC modes rescale from the current frame's own histogram, so a target whose
temperature is the subject looks the same in every frame -- the gain follows the target and cancels
exactly the change being filmed. A real operator switches to a manual span for that, and so do these
renders.

The span therefore has to be in a quantity the band actually produces. §12.1 switches
``apparent_temperature`` off for SWIR and NIR, because inverting L_B does not give a scene
temperature when the signal is reflected sunlight -- so a kelvin span has no plane to apply to
there, and the failure mode this module exists to prevent is rendering *the other plane* instead
and producing a plausible picture of the wrong quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from irsim_isaac.display_span import (
    DisplaySpan,
    span_from_apparent_t,
    span_from_dn16,
    span_from_nodes,
)


@dataclass
class _Outputs:
    apparent_t: np.ndarray | None = None
    dn16: np.ndarray | None = None


def test_a_span_must_increase() -> None:
    with pytest.raises(ValueError, match="must increase"):
        DisplaySpan(kind="apparent_t", low=300.0, high=300.0)
    with pytest.raises(ValueError, match="must increase"):
        DisplaySpan(kind="dn16", low=500.0, high=100.0)


def test_each_kind_reads_its_own_plane_and_says_so() -> None:
    kelvin = DisplaySpan(kind="apparent_t", low=291.15, high=340.15)
    counts = DisplaySpan(kind="dn16", low=200.0, high=800.0)
    assert kelvin.is_temperature and not counts.is_temperature
    assert kelvin.caption == "span 18-67C"
    assert counts.caption == "span DN 200-800"


def test_asking_for_kelvin_on_a_band_that_has_none_raises(  # noqa: D103
) -> None:
    """The bug M10.23 fixed, pinned. A SWIR render has no `apparent_t`, and quietly spanning
    `dn16` instead would produce a plausible picture of a quantity nobody asked for."""
    kelvin = DisplaySpan(kind="apparent_t", low=290.0, high=340.0)
    reflective = _Outputs(apparent_t=None, dn16=np.full((4, 4), 500, np.uint16))
    with pytest.raises(ValueError, match="apparent_t"):
        kelvin.plane(reflective)
    with pytest.raises(ValueError, match="outputs.apparent_temperature"):
        kelvin.scale(reflective)
    # ...and the converse: a camera with `dn_16` switched off cannot be spanned on the ADC.
    counts = DisplaySpan(kind="dn16", low=0.0, high=1000.0)
    with pytest.raises(ValueError, match="dn16"):
        counts.plane(_Outputs(apparent_t=np.full((4, 4), 300.0, np.float32), dn16=None))


def test_scale_puts_the_span_ends_at_zero_and_one() -> None:
    span = DisplaySpan(kind="dn16", low=200.0, high=700.0)
    out = _Outputs(dn16=np.array([[200, 450, 700]], dtype=np.uint16))
    assert span.scale(out).ravel() == pytest.approx([0.0, 0.5, 1.0])
    kelvin = DisplaySpan(kind="apparent_t", low=290.0, high=390.0)
    frame = _Outputs(apparent_t=np.array([[290.0, 340.0, 390.0]], dtype=np.float32))
    assert kelvin.scale(frame).ravel() == pytest.approx([0.0, 0.5, 1.0])


def test_the_printed_key_is_clamped_to_the_display_axis() -> None:
    """The overlay prints `node -> display code`; a node outside the span must read 0 or 255 and
    never a negative code or one past the end."""
    span = DisplaySpan(kind="apparent_t", low=290.0, high=340.0)
    assert span.code_for(290.0) == 0
    assert span.code_for(340.0) == 255
    assert span.code_for(315.0) == pytest.approx(128, abs=1)
    assert span.code_for(100.0) == 0
    assert span.code_for(1000.0) == 255


def test_a_node_span_comes_from_the_targets_own_history() -> None:
    samples = [{"motor": 300.0, "skin": 295.0}, {"motor": 340.0, "skin": 296.0}]
    span = span_from_nodes(samples)
    assert span.is_temperature
    assert span.low < 295.0 and span.high > 340.0, "padded around the nodes"


@pytest.mark.parametrize(
    ("builder", "kind"), [(span_from_dn16, "dn16"), (span_from_apparent_t, "apparent_t")]
)
def test_a_percentile_span_clips_the_tails_and_keeps_the_body(builder, kind) -> None:  # type: ignore[no-untyped-def]
    """Percentiles, not min/max: one hot specular pixel would otherwise own the whole span and
    render everything else black."""
    frame = np.full((100, 100), 500.0)
    frame[0, 0] = 1e6  # a glint
    frame[0, 1] = -1e6  # and a dead pixel
    span = builder(frame)
    assert span.kind == kind
    assert 400.0 < span.low < 600.0 and 400.0 < span.high < 600.0
    assert span.high > span.low


@pytest.mark.parametrize("builder", [span_from_dn16, span_from_apparent_t])
def test_a_flat_frame_gives_a_usable_span_rather_than_raising(builder) -> None:  # type: ignore[no-untyped-def]
    """A uniform picture is a real thing to render -- an overcast sky, a calibration target -- and
    it should come out flat, not take the renderer down."""
    span = builder(np.full((16, 16), 42.0))
    assert span.high > span.low
    with pytest.raises(ValueError, match="no pixels"):
        builder(np.zeros((0, 0)))


def test_a_percentile_span_brackets_a_gradient() -> None:
    """The maritime case: the sea and sky span a range and the span must contain most of it."""
    frame = np.linspace(280.0, 300.0, 10_000).reshape(100, 100)
    span = span_from_apparent_t(frame)
    inside = float(((frame >= span.low) & (frame <= span.high)).mean())
    assert inside > 0.97, inside
