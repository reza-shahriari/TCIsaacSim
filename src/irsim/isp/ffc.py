"""Flat-field correction: the schedule, the freeze, and what ΔT_FPA the residual actually sees.

docs/physics-model.md §11.2, §12.2 ``nuc``, §15 Tier 3. ADR 0057.

§11.2 asks for three things, and is explicit about why the third matters most:

    the FFC event itself: a shutter closes, the image freezes for 0.5–1 s, then the pattern resets.
    That freeze is a real behavioural artefact that a perception stack must survive. Simulating it
    is worth more than another decimal place of radiometry.

This module is that event. It owns the schedule, the freeze, and — the part that is easy to put in
the wrong place — **the ΔT_FPA the NUC residual is evaluated at**. The three §12.2 modes differ
almost entirely in that one quantity:

* ``shuttered`` — ΔT = T_FPA − T_cal, growing since the last shutter event and reset to zero by it.
  The image is held for ``round(ffc_freeze_ms · fps)`` frames while the shutter is closed.
* ``shutterless`` — no shutter, so nothing resets. A scene-based correction [R32, R33] estimates
  the pattern continuously from scene motion, and the residual settles at the level where the
  estimator's convergence balances the drift. That is a first-order lag on ΔT, with time constant
  ``nuc.shutterless_tau_s``: ``dΔT_eff/dt = dΔT/dt − ΔT_eff/τ``, so under a constant drift rate r
  it approaches ``r·τ`` instead of growing without bound. No freeze, ever.
* ``ideal`` — ΔT_eff ≡ 0. Perfect correction, no residual, no freeze. The control case.

**Why the freeze holds rather than blanks.** A closed shutter means the FPA is looking at the
shutter blade, not the scene; a camera that emitted those frames would show a flat field, and a
perception stack would see the scene vanish and reappear. Real cores hold the last good frame
instead, which is what makes the artefact subtle: for 0.7 s the image is *stale* rather than
obviously wrong, and anything tracking through it sees motion stop dead and then jump. Holding is
the behaviour worth simulating.

**The rounding rule.** ``freeze_frames = round(freeze_s · fps)``, so 60 Hz × 0.7 s is exactly 42
frames and 9 Hz × 0.7 s is 6 (6.3 rounded), not 7. Round rather than ceil: a freeze is a physical
duration being sampled by the frame clock, and at 9 Hz the last partial frame is more likely to
complete than not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import NucSpec

__all__ = ["Resettable", "FfcEvent", "FfcController"]


@runtime_checkable
class Resettable(Protocol):
    """Anything the shutter recalibrates: the NUC residual, the drift, the T_cal holder."""

    def reset(self, frame_index: int) -> None: ...


@dataclass(frozen=True)
class FfcEvent:
    """What happened to one frame."""

    fired: bool  # the shutter closed on this frame and the resets ran
    frozen: bool  # this frame's output is the held one, not the live one
    frames_held: int  # how many frames have been held so far in this freeze (0 when live)


@dataclass
class FfcController:
    """The §11.2 flat-field schedule for one camera (ADR 0057).

    ``fps`` is the frame rate the schedule is sampled at; it comes from ``fpa.frame_rate_hz`` and is
    taken explicitly so a bench can run the controller at a rate the config does not name.
    """

    nuc: NucSpec
    fps: float
    _held: NDArray[np.floating] | None = field(default=None, init=False, repr=False)
    _frames_held: int = field(default=0, init=False, repr=False)
    _delta_t_eff: float = field(default=0.0, init=False, repr=False)
    _previous_raw: float = field(default=0.0, init=False, repr=False)
    _t_cal_k: float | None = field(default=None, init=False, repr=False)
    _resettables: list[Resettable] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError(f"fps must be positive and finite, got {self.fps}")
        if self.nuc.mode == "shutterless" and not self.nuc.shutterless_tau_s > 0.0:
            raise ValueError("shutterless mode needs a positive nuc.shutterless_tau_s")

    # -- the schedule ---------------------------------------------------------------------
    @property
    def shutters(self) -> bool:
        """Only a shuttered core closes a shutter; the other two modes never freeze."""
        return self.nuc.mode == "shuttered"

    @property
    def interval_frames(self) -> int:
        """Frames between shutter events. Below one frame is a schedule nobody can run."""
        n = int(round(float(self.nuc.ffc_interval_s) * float(self.fps)))
        if n < 1:
            raise ValueError(
                f"ffc_interval_s {self.nuc.ffc_interval_s} at {self.fps} Hz is under one frame"
            )
        return n

    @property
    def freeze_frames(self) -> int:
        """round(freeze_s · fps): 42 at 60 Hz / 700 ms, 6 at 9 Hz (6.3, not 7)."""
        if not self.shutters:
            return 0
        return int(round(float(self.nuc.ffc_freeze_ms) * 1e-3 * float(self.fps)))

    def fires_on(self, frame_index: int) -> bool:
        """True when the shutter closes on this frame: every ``interval_frames``, never at 0.

        Frame 0 is excluded because the camera was calibrated when it started; firing there would
        put a freeze in the first frames of every sequence, including every golden fixture.
        """
        if not self.shutters or frame_index <= 0:
            return False
        return frame_index % self.interval_frames == 0

    def register(self, *resettables: Resettable) -> None:
        """Attach the objects the shutter recalibrates (M9.6's residual, M9.4's drift, T_cal)."""
        for obj in resettables:
            if not isinstance(obj, Resettable):
                raise TypeError(f"{obj!r} has no reset(frame_index) hook")
            self._resettables.append(obj)

    # -- the ΔT the residual sees ---------------------------------------------------------
    def update_delta_t(self, t_fpa_k: float, dt_s: float) -> float:
        """Advance and return ΔT_eff, the temperature offset M9.6's residual is evaluated at.

        Which of the three forms applies is the substance of the mode, not a detail of it:
        ``ideal`` returns 0, ``shuttered`` returns the raw offset from the last calibration, and
        ``shutterless`` returns the first-order lag that a scene-based correction leaves behind.
        """
        if self.nuc.mode == "ideal":
            self._delta_t_eff = 0.0
            return 0.0
        if self._t_cal_k is None:
            self._t_cal_k = float(t_fpa_k)
        raw = float(t_fpa_k) - float(self._t_cal_k)
        if self.shutters:
            self._delta_t_eff = raw
            return raw

        # shutterless: dΔT_eff/dt = dΔT/dt − ΔT_eff/τ, integrated exactly over the step for a
        # drift rate held constant across it. Under constant r this tends to r·τ, never to
        # infinity -- which is the whole difference between a scene-based correction and none.
        tau = float(self.nuc.shutterless_tau_s)
        rate = (raw - self._previous_raw) / float(dt_s) if dt_s > 0.0 else 0.0
        decay = math.exp(-float(dt_s) / tau)
        self._delta_t_eff = self._delta_t_eff * decay + rate * tau * (1.0 - decay)
        self._previous_raw = raw
        return self._delta_t_eff

    @property
    def delta_t_eff_k(self) -> float:
        return self._delta_t_eff

    @property
    def t_cal_k(self) -> float | None:
        """The FPA temperature the current correction was calibrated at."""
        return self._t_cal_k

    # -- the frame ------------------------------------------------------------------------
    def process(
        self, frame: NDArray[np.floating], frame_index: int, t_fpa_k: float | None = None
    ) -> tuple[NDArray[np.floating], FfcEvent]:
        """Return the frame the camera emits, and what happened to it.

        On a firing frame the shutter closes: every registered object is reset, ``t_cal`` becomes
        the current ``t_fpa_k``, ΔT_eff returns to zero, and the previously held frame is emitted.
        The freeze then continues for ``freeze_frames`` frames in total.
        """
        fired = self.fires_on(frame_index)
        if fired:
            for obj in self._resettables:
                obj.reset(frame_index)
            if t_fpa_k is not None:
                self._t_cal_k = float(t_fpa_k)
            self._delta_t_eff = 0.0
            self._previous_raw = 0.0
            self._frames_held = 0

        freezing = self.shutters and (fired or 0 < self._frames_held < self.freeze_frames)
        if freezing and self.freeze_frames > 0:
            if self._held is None:
                # Nothing to hold yet (an FFC on the very first emitted frame): pass it through
                # rather than inventing a frame the camera never saw.
                self._held = np.array(frame, copy=True)
            self._frames_held += 1
            held = np.array(self._held, copy=True)
            return held, FfcEvent(fired=fired, frozen=True, frames_held=self._frames_held)

        self._frames_held = 0
        self._held = np.array(frame, copy=True)
        return np.array(frame, copy=True), FfcEvent(fired=fired, frozen=False, frames_held=0)
