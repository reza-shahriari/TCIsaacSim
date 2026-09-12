"""Microbolometer thermal time constant: the per-pixel exponential IIR across frames (M9.1).

docs/physics-model.md §9.2. The membrane is a first-order thermal system,

    C_th dΔT/dt = α_abs Φ(t) − G_th ΔT,     τ_th = C_th / G_th   (8–12 ms for VOx),

so for a signal held constant across an integration period the sampled response is the exact
solution of that ODE over one frame interval:

    S_n = S_{n−1} + (S_ideal_n − S_{n−1}) (1 − e^{−Δt/τ_th}).

This is exact rather than an approximation of the ODE: zero-order hold on Φ over the frame makes
the exponential update the closed-form solution, which is why it agrees with an RK4 integration of
the membrane equation to integration error alone (tests/unit/test_bolometer_lowpass.py).

**Where it goes (ADR 0052).** The filter acts on the *ideal* signal, in the detector's physical
signal space, **before** noise is added. The membrane low-passes the incident flux; it does not
low-pass the ROIC's read noise or the FPA's fixed-pattern terms, which are injected downstream of
the thermal integration. Filtering after the noise stage would correlate σ_TVH across frames and
quietly destroy the NETD anchor (M4.6) by shrinking the per-frame temporal variance.

**Who owns the state (ADR 0052).** The per-pixel state is float32 (CLAUDE.md #2 — it carries
signal) and lives in ``PipelineState.buffers``, not in the detector object, so that a detector
instance stays immutable and shareable and every cross-frame buffer has one owner. M9.8 wires it.

**On "0.6 frames" (spec issue S8).** §9.2's "smears over roughly 0.6 frames" is τ_th/Δt (10 ms at
60 Hz), not the extent of the smear. One frame of the IIR already reaches 1 − e^{−1.667} = 81 %.
The physically meaningful extent is the trailing exponential decay length of a moving edge, which
is ``trailing_decay_length_px`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BolometerLowPass",
    "alpha_for",
    "responsivity_rolloff",
    "trailing_decay_length_px",
]


def alpha_for(dt_s: float, tau_s: float) -> float:
    """The IIR blend weight α = 1 − e^{−Δt/τ_th} (§9.2).

    α → 1 as Δt/τ → ∞ (a slow camera sees the settled membrane) and α → Δt/τ as Δt/τ → 0.
    """
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"frame interval must be finite and positive, got {dt_s}")
    if not np.isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError(f"thermal time constant must be finite and positive, got {tau_s}")
    return float(-np.expm1(-dt_s / tau_s))


def responsivity_rolloff(
    frequency_hz: NDArray[np.floating] | float, tau_s: float
) -> NDArray[np.float64] | float:
    """R(f)/R_0 = 1 / sqrt(1 + (2π f τ_th)²) — the membrane's continuous frequency response (§9.2).

    This is the response of the underlying ODE, the quantity a datasheet or a bench measurement
    reports. It is *not* the transfer function of the sampled IIR above (which is periodic in f
    with the frame rate); the two agree for f well below Nyquist. Use this for the temporal MTF
    and for reasoning about τ_th, and the IIR for propagating frames.
    """
    if not np.isfinite(tau_s) or tau_s <= 0.0:
        raise ValueError(f"thermal time constant must be finite and positive, got {tau_s}")
    # not named `f`: that is reserved for the f-number repo-wide (tests/unit/test_aperture_guard.py)
    freq = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(freq < 0.0):
        raise ValueError("frequency must be non-negative")
    out = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * freq * tau_s) ** 2)
    return out if np.ndim(frequency_hz) else float(out)


def trailing_decay_length_px(speed_px_per_frame: float, dt_s: float, tau_s: float) -> float:
    """Exponential decay length, in pixels, of the tail behind an edge moving at a constant speed.

    A feature crossing the array at v pixels per frame leaves a trail whose amplitude falls by
    e^{−1} every v τ_th/Δt pixels: each pixel is vacated for Δt/v seconds of membrane decay. At
    60 Hz with τ_th = 10 ms an edge at 4 px/frame trails with a 6.7 px decay length — this is the
    number to quote, not §9.2's "0.6 frames" (spec issue S8).
    """
    if not np.isfinite(speed_px_per_frame) or speed_px_per_frame <= 0.0:
        raise ValueError("speed must be finite and positive")
    return float(speed_px_per_frame) * tau_s / dt_s


@dataclass
class BolometerLowPass:
    """Stateful per-pixel first-order filter for the membrane's thermal lag (§9.2, ADR 0052).

    The state is float32 and starts *settled*: the first ``step`` adopts its input rather than
    ramping from zero, so a sequence does not open with a spurious frame-long transient that no
    real camera shows (a core that has been staring at the scene is already in equilibrium with
    it). Call ``reset`` to model a genuine cold start.

    Not thread-safe and deliberately mutable — one instance per camera. M9.8 keeps the array in
    ``PipelineState.buffers`` so the pipeline owns cross-frame state uniformly.
    """

    tau_s: float
    state: NDArray[np.float32] | None = field(default=None)

    def __post_init__(self) -> None:
        if not np.isfinite(self.tau_s) or self.tau_s <= 0.0:
            raise ValueError(f"thermal time constant must be finite and positive, got {self.tau_s}")

    def reset(self) -> None:
        """Forget the settled state; the next ``step`` re-adopts its input."""
        self.state = None

    def step(self, s_ideal: NDArray[np.floating], dt_s: float) -> NDArray[np.float32]:
        """Advance one frame and return the filtered signal (float32, a fresh array).

        ``s_ideal`` is the noise-free signal the detector would produce with no thermal lag, in
        whatever physical signal space the caller uses (the filter is linear, so the space only
        has to be consistent between frames).
        """
        plane = _checked(s_ideal)
        alpha = alpha_for(dt_s, self.tau_s)
        if self.state is None or self.state.shape != plane.shape:
            if self.state is not None:
                raise ValueError(
                    f"frame shape {plane.shape} does not match the filter state "
                    f"{self.state.shape}; call reset() when the format changes"
                )
            self.state = plane.copy()
            fresh: NDArray[np.float32] = self.state.copy()
            return fresh
        # float32 throughout: the state carries signal (CLAUDE.md #2)
        self.state = (self.state + (plane - self.state) * np.float32(alpha)).astype(
            np.float32, copy=False
        )
        out: NDArray[np.float32] = self.state.copy()
        return out


def _checked(plane: NDArray[Any]) -> NDArray[np.float32]:
    if plane.dtype == np.float16:
        raise TypeError("bolometer signal is float16 (CLAUDE.md non-negotiable #2); use float32")
    if not np.issubdtype(plane.dtype, np.floating):
        raise TypeError(f"bolometer signal must be a float plane, got {plane.dtype}")
    return np.asarray(plane, dtype=np.float32)
