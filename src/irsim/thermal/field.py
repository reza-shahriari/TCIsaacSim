"""The object the engine glue reads: a fixed-tick solve behind a render-time query (§6.4, §1).

§6.4: "Decouple this from the render loop entirely — run it on a fixed thermal tick (say 1 Hz) and
interpolate." That is one sentence and two separate requirements, and the second is the one that
gets broken:

* the solve advances on **its own clock**, so a render at 240 fps and a render at 1 fps see the
  same temperatures;
* a query **never mutates** anything. A renderer asks for a temperature many times per tick — per
  band, per AOV, per pass — and a query that advanced the solver would make the answer depend on
  how many times it was asked, which is not a bug that shows up as an error message.

So :class:`ThermalField` keeps two bracketing ticks and interpolates between them, and
``temperature_at`` is const. Ticks are produced on demand up to the requested time and then kept;
asking for an earlier time than the ticks already cover raises rather than re-integrating, because
a thermal history is not reversible and silently re-running it would give a different answer.

**float32 is the boundary and only the boundary** (CLAUDE.md #2). The solve is float64 throughout;
``temperature_at`` narrows once, on the way out, because that is what the G-buffer carries. Over
48 h the difference between carrying float32 internally and narrowing at the end is measurable,
which is why the narrowing is in one place with a test on it.

docs/physics-model.md §6.4, §1; ADR 0036, ADR 0037
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver

__all__ = ["DEFAULT_TICK_S", "ThermalField"]

#: §6.4's "say 1 Hz". Fast enough that the linear interpolation between ticks is far below a
#: millikelvin for every material in the library, and slow enough that a 48 h spin-up is cheap.
DEFAULT_TICK_S = 1.0


@dataclass(frozen=True)
class _Tick:
    t_s: float
    temperatures_k: NDArray[np.float64]


class ThermalField:
    """A facet solve on a fixed tick, queried at arbitrary render times.

    ``forcing_at`` is the scene's own weather-driven forcing; the field never looks at a clock of
    its own. Construction takes the state at ``t0_s`` -- normally M6.10's spin-up result -- so the
    field begins where the scene begins and the first frame is not a transient.
    """

    def __init__(
        self,
        properties: FacetProperties,
        forcing_at: Callable[[float], FacetForcing],
        t0_s: float,
        initial_k: NDArray[np.float64],
        tick_s: float = DEFAULT_TICK_S,
    ) -> None:
        if tick_s <= 0.0:
            raise ValueError("tick_s must be positive")
        self.properties = properties
        self.forcing_at = forcing_at
        self.tick_s = float(tick_s)
        self.t0_s = float(t0_s)
        self._solver = FacetSolver(properties, initial_k)
        self._ticks: list[_Tick] = [_Tick(float(t0_s), self._solver.temperatures_k)]

    # -- the solve ---------------------------------------------------------------------------

    @property
    def n_ticks(self) -> int:
        return len(self._ticks)

    @property
    def latest_t_s(self) -> float:
        return self._ticks[-1].t_s

    def _extend_to(self, t_s: float) -> None:
        while self._ticks[-1].t_s < t_s - 1e-12:
            start = self._ticks[-1].t_s
            self._solver.advance(self.forcing_at(start), self.tick_s)
            self._ticks.append(_Tick(start + self.tick_s, self._solver.temperatures_k))

    def advance_to(self, t_s: float) -> None:
        """Produce ticks up to ``t_s``. The **only** method that changes anything."""
        if t_s < self.t0_s - 1e-12:
            raise ValueError(
                f"cannot advance to {t_s} s, before the field's start {self.t0_s} s: a thermal "
                "history is not reversible, and re-running it would give a different answer"
            )
        self._extend_to(t_s)

    # -- the query ---------------------------------------------------------------------------

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        """Per-facet temperature at an arbitrary render time, float32, **without mutating**.

        Linear between the two bracketing ticks. Linear rather than anything cleverer because the
        tick is 1 s and the fastest surface in the library has a time constant of minutes: the
        interpolation error is far below a millikelvin, and a higher-order scheme would need more
        stored ticks to buy nothing.
        """
        if t_s < self.t0_s - 1e-12:
            raise ValueError(f"t_s = {t_s} is before the field's start {self.t0_s}")
        if t_s > self.latest_t_s + 1e-12:
            raise ValueError(
                f"t_s = {t_s} is past the last tick {self.latest_t_s}: call advance_to first. A "
                "query must not advance the solve -- a renderer asks many times per tick, and an "
                "answer that depended on how often it was asked would not look like an error"
            )
        index = min(int((t_s - self.t0_s) / self.tick_s), len(self._ticks) - 1)
        lower = self._ticks[index]
        if index + 1 >= len(self._ticks):
            return np.asarray(lower.temperatures_k, dtype=np.float32)
        upper = self._ticks[index + 1]
        span = upper.t_s - lower.t_s
        weight = 0.0 if span <= 0.0 else (t_s - lower.t_s) / span
        blended = lower.temperatures_k + weight * (upper.temperatures_k - lower.temperatures_k)
        return np.asarray(blended, dtype=np.float32)

    def state_hash(self) -> str:
        """SHA-256 over every tick produced so far: what a query must not change."""
        h = hashlib.sha256()
        for tick in self._ticks:
            h.update(np.float64(tick.t_s).tobytes())
            h.update(np.ascontiguousarray(tick.temperatures_k, dtype=np.float64).tobytes())
        return h.hexdigest()
