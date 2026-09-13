"""Is the camera holding still? The gate in front of every per-pixel temporal statistic.

roadmap ME.1b; §10.2 (the 3-D noise model), §15 T4.

ME.2b wants a per-pixel temporal standard deviation on flat sky, because that is the closest thing
to a laboratory blackbody these public clips offer. The measurement is only about the *sensor* if
the scene stayed on the same pixels: pan the camera by a pixel a frame and the same statistic
measures how fast the sky gradient moves through the detector, which on a clear sky is a much
larger number and looks exactly as plausible. Nothing in the frames announces which happened.

So a clip is classified before it is measured, and the classification turns on a distinction that
a simple motion magnitude gets wrong:

* **jitter** -- a hand-held or wind-shaken mount wobbling by a fraction of a pixel, zero-mean, not
  going anywhere. Scene content stays inside its own detector pixel, so per-pixel temporal
  statistics still describe the sensor. This is **static**.
* **a pan or a drift** -- small per-frame shifts that all point the same way and accumulate. After
  a hundred frames the scene has crossed many pixels. This is **moving**, however small each step
  was.

The discriminator is therefore *cumulative displacement from the first frame*, not per-frame shift:
jitter's cumulative displacement stays bounded while a pan's grows without limit. That choice is
what :func:`classify_clip` exists to make, and the two synthetic controls in the tests -- a
translated clip and a jittered one with the same per-frame step size -- are what keep it honest.

Shifts come from phase correlation (an FFT cross-power spectrum), which is robust to the brightness
changes an AGC introduces between frames because it uses only the phase. Sub-pixel accuracy comes
from Foroosh's ratio around the correlation peak rather than the usual parabolic fit: a phase-only
peak is a Dirichlet kernel that splits linearly between its two nearest samples, and a parabola
fitted to that shape under-reads by about 30 % at a third of a pixel -- small enough to pass for
noise, biased enough to drag a drifting clip under the threshold. The ratio is good to roughly a
tenth of a pixel on real texture, well inside the one-pixel decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "STATIC_DISPLACEMENT_PX",
    "StaticVerdict",
    "estimate_shift",
    "cumulative_shifts",
    "classify_clip",
]

#: Cumulative displacement, in native pixels, below which a clip counts as static. One pixel is
#: the physically meaningful line: below it scene content has not left the detector pixel it
#: started in, so a per-pixel temporal statistic is still about that pixel.
STATIC_DISPLACEMENT_PX = 1.0

_EPS = 1e-12


@dataclass(frozen=True)
class StaticVerdict:
    """The classification, with the numbers that produced it."""

    is_static: bool
    max_displacement_px: float
    rms_step_px: float
    net_displacement_px: float
    reason: str

    @property
    def drift_ratio(self) -> float:
        """Net displacement over the total path walked.

        Near 1 the steps all point the same way (a pan); near 0 they cancel (jitter). Reported
        rather than used as a threshold, because it is the number that explains a verdict to a
        human -- two clips can share a cumulative displacement and be entirely different things.
        """
        total = self.rms_step_px
        return 0.0 if total <= _EPS else min(self.net_displacement_px / (total + _EPS), 1e6)


def _window(shape: tuple[int, int]) -> NDArray[np.float64]:
    """A separable Hann window: the FFT assumes periodicity and a frame is not periodic.

    Without it the frame edges act like a step, which puts a cross-shaped artefact through the
    correlation and biases the peak toward zero shift -- that is, it biases every clip toward
    "static", which is exactly the wrong way for this decision to fail.
    """
    rows = np.hanning(shape[0])
    cols = np.hanning(shape[1])
    return np.asarray(np.outer(rows, cols), dtype=np.float64)


def _dirichlet_offset(low: float, peak: float, high: float) -> float:
    """Sub-pixel peak offset in (-1, 1) from three samples, for a **phase** correlation.

    A parabolic fit is the usual recipe and is wrong here. It is the right estimator for an
    ordinary cross-correlation, whose peak is smooth; a phase-only correlation of a shifted image
    has a Dirichlet-kernel peak instead, which splits linearly between the two samples either side
    of the true position. For a shift of ``n + t`` the samples are proportional to ``1 - t`` and
    ``t``, so the offset is the ratio ``t / ((1 - t) + t)`` -- Foroosh's estimator. Fitting a
    parabola to that shape under-reads systematically, by about 30 % at a third of a pixel, which
    is small enough to look like noise and biased enough to drag a drifting clip under a
    displacement threshold.
    """
    lo, hi, pk = max(low, 0.0), max(high, 0.0), max(peak, 0.0)
    if hi >= lo:
        denominator = pk + hi
        return 0.0 if denominator < _EPS else float(min(hi / denominator, 1.0 - _EPS))
    denominator = pk + lo
    return 0.0 if denominator < _EPS else -float(min(lo / denominator, 1.0 - _EPS))


def estimate_shift(reference: Any, moved: Any) -> tuple[float, float]:
    """``(dx, dy)`` in pixels that ``moved`` is displaced from ``reference``, by phase correlation.

    Positive ``dx`` means the content moved to the right (towards larger columns) and positive
    ``dy`` downwards, matching the image convention used everywhere else in the project.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(moved, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"frames must be two matching 2-D arrays, got {a.shape} and {b.shape}")
    height, width = a.shape
    if height < 4 or width < 4:
        raise ValueError("frames must be at least 4x4 for a correlation peak to mean anything")

    window = _window((height, width))
    fa = np.fft.rfft2((a - a.mean()) * window)
    fb = np.fft.rfft2((b - b.mean()) * window)
    cross = fa.conj() * fb
    magnitude = np.abs(cross)
    # Phase only: the amplitude carries the scene's contrast, which an AGC changes between frames.
    correlation = np.fft.irfft2(np.divide(cross, magnitude + _EPS), s=(height, width))

    peak = int(np.argmax(correlation))
    row, col = divmod(peak, width)
    d_row = _dirichlet_offset(
        float(correlation[(row - 1) % height, col]),
        float(correlation[row, col]),
        float(correlation[(row + 1) % height, col]),
    )
    d_col = _dirichlet_offset(
        float(correlation[row, (col - 1) % width]),
        float(correlation[row, col]),
        float(correlation[row, (col + 1) % width]),
    )
    # The peak wraps: a shift of -1 appears at index height - 1.
    shift_y = row - height if row > height // 2 else row
    shift_x = col - width if col > width // 2 else col
    return (float(shift_x + d_col), float(shift_y + d_row))


def cumulative_shifts(images: Any) -> NDArray[np.float64]:
    """``(N, 2)`` displacement of each frame from the **first**, in pixels.

    Measured against frame zero rather than accumulated frame to frame: summing per-frame
    estimates would also sum their errors, so a long jittery clip would random-walk its way past
    the threshold and be called moving for a reason that is purely estimator noise.
    """
    stack = np.asarray(images)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"need at least two frames as (N, H, W), got {stack.shape}")
    first = stack[0]
    out = np.zeros((stack.shape[0], 2), dtype=np.float64)
    for i in range(1, stack.shape[0]):
        out[i] = estimate_shift(first, stack[i])
    return out


def classify_clip(
    images: Any, *, max_displacement_px: float = STATIC_DISPLACEMENT_PX
) -> StaticVerdict:
    """Decide whether a clip is static enough for a per-pixel temporal statistic.

    Returns the verdict with the numbers behind it, so a report can say *why* a clip was excluded
    rather than only that it was.
    """
    shifts = cumulative_shifts(images)
    distances = np.hypot(shifts[:, 0], shifts[:, 1])
    max_displacement = float(distances.max())
    net = float(np.hypot(shifts[-1, 0], shifts[-1, 1]))
    steps = np.diff(shifts, axis=0)
    rms_step = float(np.sqrt((steps**2).sum(axis=1).mean())) if len(steps) else 0.0

    is_static = max_displacement < max_displacement_px
    if is_static:
        reason = (
            f"scene stayed within {max_displacement:.2f} px of its starting position "
            f"(limit {max_displacement_px:.2f}), so per-pixel temporal statistics describe "
            "the sensor"
        )
    else:
        reason = (
            f"scene moved {max_displacement:.2f} px from its starting position "
            f"(limit {max_displacement_px:.2f}); a per-pixel temporal statistic here would "
            "measure the scene crossing the detector, not the detector"
        )
    return StaticVerdict(
        is_static=is_static,
        max_displacement_px=max_displacement,
        rms_step_px=rms_step,
        net_displacement_px=net,
        reason=reason,
    )
