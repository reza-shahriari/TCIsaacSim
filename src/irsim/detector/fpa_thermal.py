"""FPA temperature node and the gain/offset(T_FPA) polynomials (M9.2).

docs/physics-model.md §9.2 ("FPA temperature coupling"), §6.4, §11.2. ADR 0053.

A TEC-less core's focal plane sits above ambient by its own dissipation and lags ambient by its own
time constant. Bolometer responsivity and offset are functions of that temperature, so the FPA node
is the *physical origin of shutterless drift*: model it and NUC behaviour falls out of the
correction being calibrated at one T_FPA and applied at another, instead of having to be faked.

The node is the same lumped form as the optics housing (ADR 0016),

    C dT/dt = P − h (T − T_amb),   equivalently   dT/dt = (T_amb + ΔT_self − T) / τ,

with τ = C/h and the steady rise ΔT_self = P/h. The config authors τ and ΔT_self rather than C, h
and P separately: only those two combinations are observable from the outside, and authoring three
numbers to determine two invites them to disagree.

**Ambient comes from one place (CLAUDE.md #6).** The node takes either the scene's shared
``WeatherSeries`` or an explicit provider, never both — a camera whose FPA warms on a different
day from the one the atmosphere and the thermal solver see is exactly the failure that
non-negotiable forbids.

**Where the split with NUC sits (ADR 0053).** ``gain_of_t``/``offset_of_t`` here are the *raw,
uncorrected* response of the detector, and they are large: parts in 10³ per kelvin, an image that
would be unusable without correction. What the NUC residual (M9.6) models is the small part that
*survives* correction, parameterised separately by ``nuc.residual_gain_ppm_per_k`` and
``nuc.residual_offset_mk_per_k``. The two must not be confused or the drift is counted twice.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # a runtime import would close the thermal <-> atmosphere cycle through here
    from irsim.thermal.weather import WeatherSeries

__all__ = [
    "FpaTempMode",
    "FpaThermalModel",
    "gain_of_t",
    "offset_of_t",
]

FpaTempMode = Literal["fixed", "ambient", "coupled"]


def gain_of_t(t_fpa_k: float, t_cal_k: float, coeffs_per_k: Sequence[float]) -> float:
    """Responsivity gain g(T_FPA) = 1 + Σ c_i (T_FPA − T_cal)^i, i starting at 1 (§9.2).

    Normalised at the calibration temperature *by construction*: the coefficient list carries no
    constant term, so ``g(T_cal) == 1`` exactly and cannot be authored wrong. Empty coefficients
    mean a temperature-independent detector.
    """
    dt = float(t_fpa_k) - float(t_cal_k)
    return 1.0 + sum(float(c) * dt ** (i + 1) for i, c in enumerate(coeffs_per_k))


def offset_of_t(t_fpa_k: float, t_cal_k: float, coeffs_dn_per_k: Sequence[float]) -> float:
    """Additive offset o(T_FPA) = Σ d_i (T_FPA − T_cal)^i in DN, i starting at 1 (§9.2, §11.2).

    ``o(T_cal) == 0`` by construction, for the same reason as ``gain_of_t``.
    """
    dt = float(t_fpa_k) - float(t_cal_k)
    return sum(float(d) * dt ** (i + 1) for i, d in enumerate(coeffs_dn_per_k))


@dataclass
class FpaThermalModel:
    """The FPA's own temperature over time (§9.2, ADR 0053).

    ``mode`` selects the physics, mirroring the housing node's three modes (ADR 0016):

    * ``fixed`` — TEC-pinned. ``T_fpa`` is ``fixed_temp_k`` regardless of weather; this is what a
      stabilised core does and why such cores show no ambient-driven drift.
    * ``ambient`` — tracks air temperature with no lag or self-heating (a useful degenerate case,
      and the right model for a core whose τ is far below the frame interval).
    * ``coupled`` — the lumped node above, integrated by RK2 on the thermal tick.

    ``weather`` and ``ambient_provider`` are mutually exclusive and exactly one is required outside
    ``fixed`` mode (CLAUDE.md #6).
    """

    mode: FpaTempMode
    tau_s: float | None = None
    self_heating_k: float = 0.0
    fixed_temp_k: float | None = None
    weather: WeatherSeries | None = None
    ambient_provider: Callable[[float], float] | None = None
    temperature_k: float | None = field(default=None)

    def __post_init__(self) -> None:
        if self.mode == "fixed":
            if self.fixed_temp_k is None or self.fixed_temp_k <= 0.0:
                raise ValueError("fpa_temp_mode 'fixed' needs a positive fpa_temp_k")
            self.temperature_k = float(self.fixed_temp_k)
            return
        if self.weather is not None and self.ambient_provider is not None:
            raise ValueError(
                "the FPA node takes the shared WeatherSeries or an ambient provider, not both "
                "(CLAUDE.md non-negotiable #6: one weather object feeds every consumer)"
            )
        if self.weather is None and self.ambient_provider is None:
            raise ValueError(
                f"fpa_temp_mode {self.mode!r} needs an ambient source: pass the scene's shared "
                "WeatherSeries, or an explicit ambient_provider"
            )
        if self.mode == "coupled":
            if self.tau_s is None or not np.isfinite(self.tau_s) or self.tau_s <= 0.0:
                raise ValueError("fpa_temp_mode 'coupled' needs a positive fpa_tau_s")
            if self.self_heating_k < 0.0:
                raise ValueError("fpa_self_heating_k must be non-negative")

    def ambient_k(self, t_s: float) -> float:
        """Air temperature at ``t_s`` from whichever single source was configured."""
        if self.ambient_provider is not None:
            return float(self.ambient_provider(t_s))
        if self.weather is not None:
            return float(self.weather.at(t_s).t_air_k)
        raise ValueError("no ambient source (this is 'fixed' mode; read fixed_temp_k)")

    def steady_state_k(self, t_s: float) -> float:
        """T_amb + P/h, the temperature the node relaxes to under constant weather."""
        if self.mode == "fixed":
            assert self.fixed_temp_k is not None
            return float(self.fixed_temp_k)
        if self.mode == "ambient":
            return self.ambient_k(t_s)
        return self.ambient_k(t_s) + float(self.self_heating_k)

    def settle(self, t_s: float) -> float:
        """Put the node at its steady state for the weather at ``t_s`` (the spin-up entry point)."""
        self.temperature_k = self.steady_state_k(t_s)
        return self.temperature_k

    def step(self, t_s: float, dt_s: float) -> float:
        """Advance the node to ``t_s`` over ``dt_s`` and return the new T_FPA (kelvin).

        ``fixed`` and ``ambient`` are algebraic. ``coupled`` uses RK2 (Heun) on
        ``dT/dt = (T_target − T)/τ`` with ``T_target = T_amb(t) + ΔT_self`` evaluated at the start
        and end of the step, so a moving ambient is integrated rather than sampled once.
        """
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"thermal tick must be finite and positive, got {dt_s}")
        if self.mode == "fixed":
            assert self.fixed_temp_k is not None
            self.temperature_k = float(self.fixed_temp_k)
            return self.temperature_k
        if self.mode == "ambient":
            self.temperature_k = self.ambient_k(t_s)
            return self.temperature_k

        assert self.tau_s is not None
        if self.temperature_k is None:
            return self.settle(t_s - dt_s)
        t0 = self.temperature_k
        target_0 = self.ambient_k(t_s - dt_s) + float(self.self_heating_k)
        target_1 = self.ambient_k(t_s) + float(self.self_heating_k)
        k1 = (target_0 - t0) / self.tau_s
        k2 = (target_1 - (t0 + dt_s * k1)) / self.tau_s
        self.temperature_k = t0 + 0.5 * dt_s * (k1 + k2)
        return self.temperature_k

    def gain(self, t_cal_k: float, coeffs_per_k: Sequence[float]) -> float:
        """``gain_of_t`` at the node's current temperature."""
        return gain_of_t(self._current(), t_cal_k, coeffs_per_k)

    def offset(self, t_cal_k: float, coeffs_dn_per_k: Sequence[float]) -> float:
        """``offset_of_t`` at the node's current temperature."""
        return offset_of_t(self._current(), t_cal_k, coeffs_dn_per_k)

    def _current(self) -> float:
        if self.temperature_k is None:
            raise ValueError("the FPA node has no temperature yet; call settle() or step() first")
        return float(self.temperature_k)
