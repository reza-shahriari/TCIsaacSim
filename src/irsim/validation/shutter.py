"""Reading a camera's shutter off its output: freezes, their interval, and what grows between them.

roadmap ME.3a; docs/physics-model.md §11.2, §10.3; §15 Tier 4. ADR 0068.

A flat-field correction is the most conspicuous thing a thermal core does to its own video. The
shutter closes, the image **holds** for half a second to a second, and when it opens the fixed
pattern has been replaced (§11.2). Both halves of that are measurable from frames alone, which
makes them two of the few sensor behaviours a public 8-bit clip can still support -- unlike a
radiometric claim, a held frame is a held frame whatever the recorder did to the codes.

**A freeze is a run of still frames; an FFC is a freeze followed by a new fixed pattern.** The
second half is what separates a shutter from a dropped-frame stall in the recording, and the two
look identical if you only count repeated frames. :func:`find_freezes` reports both: every run of
still frames with its length, and ``pattern_change``, the step in the time-averaged frame across
the run measured in units of what the temporal noise alone would produce. A shutter event lands
far above 1; a stalled recorder lands at 1. Neither is thrown away, because a stall is worth
knowing about too -- it is a reason to distrust every temporal statistic from that clip.

**The interval needs a long clip and is a distribution, not a number.** The Boson's schedule is
180 s, so a 10-second Halmstad clip can show a freeze *length* and can never show an interval;
:func:`freeze_intervals_s` refuses a clip shorter than :data:`MIN_INTERVAL_CLIP_S` rather than
returning one interval from two events that happened to be close. And because a real core also
fires on FPA temperature change, the intervals between events are a distribution whose *upper*
edge is the configured schedule -- a mean interval below the schedule is the expected observation,
not evidence against it.

**What grows between events is the residual, and it grows from zero.** After a shutter the
correction is recalibrated, so the part of the fixed pattern the shutter owns is zero and then
relaxes back toward its stationary level (§11.2, §10.3). For an Ornstein-Uhlenbeck pattern of
correlation time tau started at zero, the *variance* recovers as ``1 - exp(-2t/tau)`` -- the factor
of two is in the variance, not the amplitude, and fitting the variance with ``exp(-t/tau)`` reports
twice the true correlation time. :func:`fit_pattern_growth` fits

    sigma^2(t) = floor + amplitude * (1 - exp(-2 t / tau))

where ``floor`` absorbs the white noise and any pattern the shutter does *not* recalibrate.

The column and row means are the right projection to watch it in: §10.3's stripe noise lives
there, they average the white noise down by the width of the array, and they are what a NUC
residual moves. Scene structure lives there too, so run this on the flat windows
:mod:`irsim.validation.flat` returns, not on a whole frame with a horizon in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Freeze",
    "LineMeans",
    "MIN_INTERVAL_CLIP_S",
    "PatternGrowth",
    "find_freezes",
    "fit_pattern_growth",
    "freeze_intervals_s",
    "line_means",
    "pattern_energy",
]

#: Shortest clip an FFC *interval* may be reported from. The Boson's schedule is 180 s, so a clip
#: below this can only ever catch one event and an interval measured on it is an artefact.
MIN_INTERVAL_CLIP_S = 180.0


@dataclass(frozen=True)
class Freeze:
    """One run of held frames: where it starts, how long it lasts, and what it left behind."""

    first_frame: int
    frames: int
    duration_s: float
    pattern_change: float

    @property
    def looks_like_ffc(self) -> bool:
        """A pattern step several times what the temporal noise alone could produce."""
        return bool(np.isfinite(self.pattern_change) and self.pattern_change >= 3.0)


def _as_cube(cube: object) -> NDArray[np.float64]:
    x = np.asarray(cube)
    if x.dtype == np.float16:
        raise TypeError("cube is float16; promote to float32 or better before analysis")
    if x.ndim != 3:
        raise ValueError(f"cube must be (T, V, H), got shape {x.shape}")
    if x.shape[0] < 3:
        raise ValueError("a shutter statistic needs at least three frames")
    return x.astype(np.float64)


def _pattern_change(cube: NDArray[np.float64], first: int, frames: int, context: int) -> float:
    """Step in the time-averaged frame across a freeze, in units of the temporal noise's own step.

    1.0 means "exactly what averaging a few noisy frames either side would give anyway"; a shutter
    that redrew the fixed pattern lands far above it.
    """
    before = cube[max(0, first - context) : first]
    after = cube[first + frames : first + frames + context]
    if before.shape[0] < 2 or after.shape[0] < 2:
        return float("nan")
    step = after.mean(axis=0) - before.mean(axis=0)
    variance = (
        before.var(axis=0, ddof=1) / before.shape[0] + after.var(axis=0, ddof=1) / after.shape[0]
    )
    expected = float(np.sqrt(variance.mean()))
    if not expected > 0.0:
        return float("inf") if float(step.std()) > 0.0 else float("nan")
    return float(step.std() / expected)


def find_freezes(
    cube: object,
    *,
    fps: float,
    min_freeze_s: float = 0.3,
    still_ratio: float = 0.1,
    context: int = 8,
) -> list[Freeze]:
    """Every run of near-identical frames lasting at least ``min_freeze_s``.

    A gap between two frames counts as still when its mean absolute difference is below
    ``still_ratio`` of the clip's median gap -- scale-free, so it works on a recorder-stretched
    clip and on a raw one alike. A camera that holds a frame gives gaps of exactly zero; a codec
    leaves them small but non-zero, which is why the test is a ratio and not equality.

    ``frames`` is the number of held frames, which is the ISP's own freeze length: the controller
    emits its held copy for N frames, so N+1 consecutive frames are equal and N gaps are still.
    """
    frames_cube = _as_cube(cube)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    if not 0.0 < still_ratio < 1.0:
        raise ValueError(f"still_ratio must be in (0, 1), got {still_ratio}")

    gaps = np.abs(np.diff(frames_cube, axis=0)).mean(axis=(1, 2))
    reference = float(np.median(gaps))
    if not reference > 0.0:
        raise ValueError(
            "the clip does not change from frame to frame at all; there is no moving reference "
            "to call a freeze still against"
        )
    still = gaps < still_ratio * reference
    minimum = max(1, int(round(float(min_freeze_s) * float(fps))))

    out: list[Freeze] = []
    start = None
    for index, value in enumerate([*still.tolist(), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length >= minimum:
                first = start + 1  # the first *repeated* frame, i.e. the firing frame
                out.append(
                    Freeze(
                        first_frame=first,
                        frames=length,
                        duration_s=length / float(fps),
                        pattern_change=_pattern_change(frames_cube, first, length, context),
                    )
                )
            start = None
    return out


def freeze_intervals_s(
    freezes: list[Freeze], *, fps: float, n_frames: int, min_clip_s: float = MIN_INTERVAL_CLIP_S
) -> NDArray[np.float64]:
    """Seconds between consecutive freeze events, for a clip long enough to mean something.

    Raises on a clip shorter than ``min_clip_s``: two events inside ten seconds are a burst, not a
    schedule, and reporting their spacing as an interval is how a 0.4 s number ends up in a table
    next to a camera whose schedule is 180 s.
    """
    duration = float(n_frames) / float(fps)
    if duration < float(min_clip_s):
        raise ValueError(
            f"{duration:.1f} s of video cannot show an FFC interval (need {min_clip_s:.0f} s; the "
            "Boson's schedule alone is 180 s). Freeze *length* is measurable on any clip."
        )
    starts = np.array([f.first_frame for f in freezes], dtype=np.float64)
    if starts.size < 2:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(np.diff(np.sort(starts)) / float(fps), dtype=np.float64)


@dataclass(frozen=True)
class LineMeans:
    """Column and row means per frame -- where §10.3's stripe noise and a NUC residual live."""

    column: NDArray[np.float64]  # (T, H)
    row: NDArray[np.float64]  # (T, V)


def line_means(cube: object) -> LineMeans:
    """Per-frame column and row means of a cube, the projection the growth curve is measured in."""
    x = _as_cube(cube)
    return LineMeans(
        column=np.asarray(x.mean(axis=1), dtype=np.float64),
        row=np.asarray(x.mean(axis=2), dtype=np.float64),
    )


def pattern_energy(series: object) -> NDArray[np.float64]:
    """Spatial standard deviation of a ``(T, N)`` line-mean series, one number per frame."""
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"series must be (T, N), got shape {x.shape}")
    if x.shape[1] < 2:
        raise ValueError("a spatial standard deviation needs at least two lines")
    return np.asarray(x.std(axis=1, ddof=1), dtype=np.float64)


@dataclass(frozen=True)
class PatternGrowth:
    """The §11.2 between-shutter growth curve and the correlation time fitted to it."""

    tau_s: float
    sigma_inf: float
    floor: float
    n_segments: int
    residual: float

    @property
    def grows(self) -> bool:
        """Whether there is a growing component at all, or only the stationary floor."""
        return self.sigma_inf > 0.0 and np.isfinite(self.tau_s)


def _model(elapsed: NDArray[np.float64], tau_s: float) -> NDArray[np.float64]:
    return np.asarray(-np.expm1(-2.0 * elapsed / tau_s), dtype=np.float64)


def fit_pattern_growth(
    energy: object,
    *,
    fps: float,
    resets: object,
    tau_grid: object | None = None,
) -> PatternGrowth:
    """Fit ``sigma^2(t) = floor + amplitude (1 - e^{-2t/tau})`` to the energy since each reset.

    ``resets`` are the frame indices the shutter fired on (``Freeze.first_frame``). Every frame is
    placed at its elapsed time since the reset before it -- frames before the first reset are
    dropped, because their history is unknown -- and one curve is fitted to all the segments
    together, which is what makes a 3-minute schedule measurable from a clip that only contains a
    couple of events.

    The factor of two is in the variance and is the whole difficulty of the fit: the amplitude
    recovers as ``1 - e^{-t/tau}`` and the variance as ``1 - e^{-2t/tau}``, so fitting the wrong
    one of the pair reports 2 tau or tau/2 and looks perfectly reasonable.
    """
    values = np.asarray(energy, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"energy must be one value per frame, got shape {values.shape}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    marks = np.sort(np.asarray(resets, dtype=np.int64).ravel())
    if marks.size < 1:
        raise ValueError("the growth curve is measured since a shutter event; none were given")

    index = np.arange(values.size)
    previous = np.searchsorted(marks, index, side="right") - 1
    keep = previous >= 0
    if not np.any(keep):
        raise ValueError("no frames follow a reset; the clip ends before the first shutter event")
    elapsed = (index[keep] - marks[previous[keep]]).astype(np.float64) / float(fps)
    variance = values[keep] ** 2

    grid = (
        np.geomspace(0.25 / float(fps), max(elapsed.max(), 1.0 / float(fps)) * 4.0, 300)
        if tau_grid is None
        else np.asarray(tau_grid, dtype=np.float64)
    )
    best = (np.inf, float(grid[0]), 0.0, 0.0)
    for tau in grid:
        basis = np.column_stack([np.ones_like(elapsed), _model(elapsed, float(tau))])
        solution, *_ = np.linalg.lstsq(basis, variance, rcond=None)
        residual = float(np.sum((basis @ solution - variance) ** 2))
        if residual < best[0] and solution[1] >= 0.0:
            best = (residual, float(tau), float(solution[0]), float(solution[1]))
    residual, tau_s, floor, amplitude = best
    return PatternGrowth(
        tau_s=tau_s,
        sigma_inf=float(np.sqrt(max(amplitude, 0.0))),
        floor=float(np.sqrt(max(floor, 0.0))),
        n_segments=int(np.unique(previous[keep]).size),
        residual=float(np.sqrt(residual / elapsed.size)),
    )
