"""Housing temperature source for the optics self-emission term (M9.3).

docs/physics-model.md §8.2 ("optics self-emission"), §12.2 ``optics``. ADR 0015, ADR 0016.

``irsim.optics.self_emission`` needs a band radiance ``L_B(T_housing)`` every frame, and §8.2
leaves open where that temperature comes from. This module is the answer: one small node with the
three modes §12.2's ``housing_temp_mode`` already names.

* ``fixed`` — a bench number. The housing is held at ``housing_temp_k`` whatever the weather does.
  This is the mode the radiometric calibration of ADR 0021 assumes, and the mode in which a camera
  shows no self-emission drift at all.
* ``ambient`` — the housing *is* the air. No lag, no dissipation. The right model for a core whose
  thermal time constant is far below the frame interval, and a useful degenerate case for isolating
  whether an artefact comes from the lag or from the drift.
* ``coupled`` — the lumped node, ``dT/dt = (T_air + ΔT_self − T)/τ``, with τ = ``housing_tau_s``
  and ΔT_self = ``housing_self_heating_k``. This is the physical origin of shutterless offset
  drift on the *optics* side: a housing that warms 87 mK of apparent temperature per kelvin
  (ADR 0016) while the NUC table was calibrated at another housing temperature.

**Why this delegates to ``NewtonCoolingSolver`` rather than integrating its own node.** The M6.6
solver already implements the exact exponential update, ``T(t+dt) = T∞ + (T(t) − T∞) e^{−dt/τ}``,
which is unconditionally stable and free of overshoot at any step size — and the housing is
precisely the linear, radiation-free, solar-free case that solver was scoped for (it is inside the
camera body, not exchanging with the sky). Writing a second integrator here would give the
project two lumped-node implementations to keep in agreement for no physical gain. The FPA node
(M9.2) is the deliberate exception: it uses RK2 because ADR 0053 keeps it independently
parameterised, and the two nodes are not the same node.

**Ambient comes from one place (CLAUDE.md #6).** The node takes the scene's shared
``WeatherSeries`` or an explicit provider, never both, and exposes whichever it holds as
``.weather`` — which is all ``Scene.__post_init__`` needs to refuse a housing model built on a
different weather object than the atmosphere and the thermal solvers see. A camera whose lens
warms on a summer afternoon while the atmosphere runs a winter night is exactly the failure
non-negotiable #6 exists to prevent, and it would be invisible in the image: the self-emission
term is a smooth pedestal, not a recognisable artefact.

Temperatures are float64 here and narrow to float32 only at the buffer boundary (CLAUDE.md #2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from irsim.thermal.solvers import NewtonCoolingSolver

if TYPE_CHECKING:
    from irsim.config.sensor import OpticsSpec
    from irsim.thermal.weather import WeatherSeries

__all__ = ["HousingTempMode", "HousingTemperature"]

HousingTempMode = Literal["fixed", "ambient", "coupled"]

# Time equality on the weather axis. Seconds-scale times with float64 carry far more precision
# than this; the slack exists so that re-reading at_s the frame already advanced to is a read
# rather than a rejected rewind.
_T_EPS_S = 1e-9


@dataclass
class HousingTemperature:
    """T_housing(t) for the §8.2 self-emission term; see the module docstring for the modes.

    ``weather`` and ``ambient_provider`` are mutually exclusive; exactly one is required outside
    ``fixed`` mode. ``t0_k`` seeds a ``coupled`` node, defaulting to its own steady state at
    ``t0_s`` — a camera that has been switched on long enough to settle, which is the state a
    frame sequence should normally start from (call :meth:`settle` to return to it).
    """

    mode: HousingTempMode
    fixed_temp_k: float | None = None
    tau_s: float | None = None
    self_heating_k: float = 0.0
    weather: WeatherSeries | None = None
    ambient_provider: Callable[[float], float] | None = None
    t0_s: float = 0.0
    t0_k: float | None = None
    _solver: NewtonCoolingSolver | None = field(default=None, init=False, repr=False)
    _temperature_k: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "ambient", "coupled"):
            raise ValueError(f"unknown housing_temp_mode {self.mode!r}")
        if self.weather is not None and self.ambient_provider is not None:
            raise ValueError(
                "the housing node takes the shared WeatherSeries or an ambient provider, not "
                "both (CLAUDE.md non-negotiable #6: one weather object feeds every consumer)"
            )
        self.t0_s = float(self.t0_s)
        self.self_heating_k = float(self.self_heating_k)

        if self.mode == "fixed":
            if self.fixed_temp_k is None or not float(self.fixed_temp_k) > 0.0:
                raise ValueError("housing_temp_mode 'fixed' requires a positive housing_temp_k")
            self._temperature_k = float(self.fixed_temp_k)
            return

        if self.weather is None and self.ambient_provider is None:
            raise ValueError(
                f"housing_temp_mode {self.mode!r} needs an ambient source: pass the scene's "
                "shared WeatherSeries, or an explicit ambient_provider"
            )
        if self.mode == "ambient":
            self._temperature_k = self.ambient_k(self.t0_s)
            return

        if self.tau_s is None or not np.isfinite(self.tau_s) or float(self.tau_s) <= 0.0:
            raise ValueError("housing_temp_mode 'coupled' requires a positive housing_tau_s")
        if self.self_heating_k < 0.0:
            raise ValueError("housing_self_heating_k must be non-negative")
        self.tau_s = float(self.tau_s)
        start_k = self.steady_state_k(self.t0_s) if self.t0_k is None else float(self.t0_k)
        if not start_k > 0.0 or not np.isfinite(start_k):
            raise ValueError("the housing node needs a positive finite starting temperature")
        self._solver = NewtonCoolingSolver(start_k, self.tau_s, self._target_k, t0_s=self.t0_s)
        self._temperature_k = start_k

    # -- ambient --------------------------------------------------------------------------
    def ambient_k(self, t_s: float) -> float:
        """Air temperature at ``t_s`` from whichever single source was configured (kelvin)."""
        if self.ambient_provider is not None:
            return float(self.ambient_provider(float(t_s)))
        if self.weather is not None:
            return float(self.weather.at(float(t_s)).t_air_k)
        raise ValueError("this is 'fixed' mode and has no ambient source; read fixed_temp_k")

    def _target_k(self, t_s: float) -> float:
        """T∞ = T_air(t) + ΔT_self, the temperature the coupled node relaxes toward."""
        return self.ambient_k(t_s) + self.self_heating_k

    def steady_state_k(self, t_s: float) -> float:
        """The temperature this node settles to under the weather held at ``t_s``."""
        if self.mode == "fixed":
            assert self.fixed_temp_k is not None
            return float(self.fixed_temp_k)
        if self.mode == "ambient":
            return self.ambient_k(t_s)
        return self._target_k(t_s)

    # -- state ----------------------------------------------------------------------------
    @property
    def temperature_k(self) -> float:
        """The node's current temperature, without advancing it."""
        return self._temperature_k

    @property
    def time_s(self) -> float:
        """The time the node has been advanced to, on the weather's axis."""
        return self.t0_s if self._solver is None else self._solver.state.t_s

    def settle(self, t_s: float) -> float:
        """Put the node at its steady state for the weather at ``t_s`` (the spin-up entry point).

        A coupled node started cold takes ~5τ to forget it; for a camera that has been running,
        starting settled is both cheaper and more honest than simulating the warm-up.
        """
        t_s = float(t_s)
        self._temperature_k = self.steady_state_k(t_s)
        if self.mode == "coupled":
            assert self.tau_s is not None
            self._solver = NewtonCoolingSolver(
                self._temperature_k, self.tau_s, self._target_k, t0_s=t_s
            )
        else:
            self.t0_s = t_s
        return self._temperature_k

    def at(self, t_s: float) -> float:
        """T_housing at ``t_s`` (kelvin); advances a ``coupled`` node to that time.

        ``fixed`` and ``ambient`` are algebraic and may be read at any time in any order.
        ``coupled`` carries state, so time must not run backwards: re-reading the current time is
        a read, but asking for an earlier one raises rather than silently returning a temperature
        from a history that was never integrated.
        """
        t_s = float(t_s)
        if not np.isfinite(t_s):
            raise ValueError("t_s must be finite")
        if self.mode == "fixed":
            assert self.fixed_temp_k is not None
            self._temperature_k = float(self.fixed_temp_k)
            return self._temperature_k
        if self.mode == "ambient":
            self._temperature_k = self.ambient_k(t_s)
            return self._temperature_k

        assert self._solver is not None
        dt = t_s - self._solver.state.t_s
        if dt < -_T_EPS_S:
            raise ValueError(
                f"the coupled housing node is at t = {self._solver.state.t_s} s and cannot step "
                f"back to {t_s} s; call settle() to restart it"
            )
        if dt > _T_EPS_S:
            self._temperature_k = self._solver.advance(self._solver.state.t_s, dt)
        return self._temperature_k

    def step(self, t_s: float, dt_s: float) -> float:
        """Advance from ``t_s`` over ``dt_s``; the tick form of :meth:`at`."""
        if not np.isfinite(dt_s) or float(dt_s) < 0.0:
            raise ValueError(f"dt_s must be finite and non-negative, got {dt_s}")
        return self.at(float(t_s) + float(dt_s))

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_optics(
        cls,
        optics: OpticsSpec,
        weather: WeatherSeries | None = None,
        ambient_provider: Callable[[float], float] | None = None,
        t0_s: float = 0.0,
        t0_k: float | None = None,
    ) -> HousingTemperature:
        """Build the node the sensor YAML describes (§12.2 ``optics``).

        The mode and its parameters are read from the config so that a camera's housing behaviour
        lives with the rest of that camera, not in calling code.
        """
        return cls(
            mode=optics.housing_temp_mode,
            fixed_temp_k=optics.housing_temp_k,
            tau_s=optics.housing_tau_s,
            self_heating_k=optics.housing_self_heating_k,
            weather=weather,
            ambient_provider=ambient_provider,
            t0_s=t0_s,
            t0_k=t0_k,
        )

    def band_radiance(self, t_s: float, lut: Any, quantity: str = "lb") -> float:
        """``L_B(T_housing(t))`` from a band LUT — what §8.2's Φ_self actually consumes.

        Kept here so the temperature never has to leave this object as a bare float that some
        caller then inverts with a different LUT than the rest of the frame used.
        """
        return float(np.asarray(lut.lookup(self.at(t_s), quantity))[()])
