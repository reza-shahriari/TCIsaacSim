"""Temporal PSD shape on a flat region (ME.2b, ADR 0023).

A flat patch of sky is the only place in a thermal clip where the temporal spectrum is the
sensor's and nothing else, and its *shape* is the fingerprint of any first-order filter in the
chain: the bolometer membrane, an in-camera temporal noise filter, or a codec's inter-frame
prediction. Whether the shape is flat decides whether a per-pixel temporal sigma read off the clip
is a NETD-like number at all, so the analyser has to separate white from filtered and then say
which filter.

The generator is the project's own :class:`~irsim.detector.lowpass.BolometerLowPass` -- an
independent implementation of the same one-pole recursion, written for the physics and not for
this test. The sampled IIR's time constant in frames is exactly tau/dt, which is why the analyser
can report seconds when it is told the frame interval.

(In the simulator the membrane filters flux *before* the noise is added, ADR 0052, so the
simulator's own frames are not low-passed this way. The filter here stands for the in-camera
filtering that public clips do show, which is what ME.2b has to detect.)
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.detector.lowpass import BolometerLowPass
from irsim.validation import temporal_shape

FRAMES = 256
SHAPE = (64, 64)
DT_S = 1.0 / 60.0


def _white(seed: int = 1, frames: int = FRAMES) -> np.ndarray:
    return np.random.default_rng(seed).normal(120.0, 2.0, size=(frames, *SHAPE))


def _filtered(tau_s: float, seed: int = 1) -> np.ndarray:
    """White frames through the project's one-pole filter, as an in-camera filter would."""
    iir = BolometerLowPass(tau_s=tau_s)
    return np.stack([iir.step(frame, DT_S) for frame in _white(seed)])


def test_white_noise_reads_as_white() -> None:
    """The null. It is not free: keeping the one-sided Nyquist bin, which carries half the weight
    of its neighbours, would make every clip read 2 % low-pass -- five times this null sigma."""
    shape = temporal_shape(_white(), dt_s=DT_S)
    assert shape.white_consistent
    assert abs(shape.z) < 3.0
    assert shape.low_over_high == pytest.approx(1.0, abs=4.0 * shape.null_sigma)
    assert shape.null_sigma < 0.01
    assert shape.tau_frames < 0.25, "the fit floor on 256 frames, not a real time constant"
    assert 0.0 not in shape.frequencies and 0.5 not in shape.frequencies


def test_a_ten_millisecond_membrane_is_recovered_in_seconds() -> None:
    """tau_th = 10 ms at 60 Hz is 0.6 frames, and the spectrum falls 2.15x from DC to Nyquist."""
    shape = temporal_shape(_filtered(0.010), dt_s=DT_S)
    assert not shape.white_consistent
    assert shape.tau_frames == pytest.approx(0.6, rel=0.05)
    assert shape.tau_s == pytest.approx(0.010, rel=0.05)
    assert shape.low_over_high == pytest.approx(2.07, rel=0.10)


def test_a_slower_filter_gives_a_longer_time_constant() -> None:
    """Four times the time constant, four times the answer: the fit is not reading one number."""
    fast = temporal_shape(_filtered(0.010), dt_s=DT_S)
    slow = temporal_shape(_filtered(0.040), dt_s=DT_S)
    assert slow.tau_s == pytest.approx(0.040, rel=0.05)
    assert slow.tau_frames == pytest.approx(4.0 * fast.tau_frames, rel=0.10)
    assert slow.low_over_high > 5.0 * fast.low_over_high


def test_a_slow_drift_is_reported_rather_than_absorbed_into_tau() -> None:
    """Drift and a filter both put power at low frequency; only one of them is a time constant.

    Excluding the bins below ``f_fit_min`` buys a lot but not everything -- a linear ramp's
    spectrum falls as 1/f^2 and still reaches into the fit band. Measured on a 256-frame clip of
    the 10 ms filter above: a ramp of one noise sigma across the clip costs 1.4 % on tau, and one
    of five sigma costs 28 %. The defence is that ``drift_fraction`` moves first and much further
    (0.14 -> 0.23 -> 0.75), so the report says the clip drifted instead of quietly claiming a
    longer membrane. A clip with no drift at all still sits near 0.14 here, because the filter's
    own low-frequency emphasis is real power (white noise would sit at f_fit_min/0.5 = 0.1).
    """
    cube = _filtered(0.010)
    quiet = temporal_shape(cube, dt_s=DT_S)
    assert quiet.drift_fraction == pytest.approx(0.14, abs=0.03)

    noise_sigma = 2.0
    gentle = temporal_shape(
        cube + np.linspace(0.0, noise_sigma, cube.shape[0])[:, None, None], dt_s=DT_S
    )
    assert gentle.tau_s == pytest.approx(0.010, rel=0.05)
    assert gentle.drift_fraction > quiet.drift_fraction + 0.05

    heavy = temporal_shape(
        cube + np.linspace(0.0, 5.0 * noise_sigma, cube.shape[0])[:, None, None], dt_s=DT_S
    )
    assert heavy.drift_fraction > 0.7, "a drifting clip has to announce itself"
    assert heavy.tau_s == pytest.approx(0.0128, rel=0.10), "and this is what it costs"


def test_a_clip_too_short_to_have_a_shape_says_so() -> None:
    with pytest.raises(ValueError, match="too few bins"):
        temporal_shape(_white(frames=12))
    with pytest.raises(ValueError, match="at least two samples"):
        temporal_shape(_white(frames=1))
