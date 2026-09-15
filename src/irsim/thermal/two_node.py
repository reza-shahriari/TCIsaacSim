"""§6.4's lumped two-node solver: a fast surface on a slow substrate.

    C₁ dT₁/dt = net_flux(T₁) − (T₁ − T₂)/R₁₂
    C₂ dT₂/dt = (T₁ − T₂)/R₁₂ − (T₂ − T_deep)/R₂d

with R₁₂ = δ₁/(2k₁) + δ₂/(2k₂), the centre-to-centre conduction resistance of two adjacent slabs.
One node gives a surface that tracks the air and forgets yesterday; two give it somewhere to put
the day's heat, which is what makes a road still warm at midnight and a thin panel not.

**The stability bound in §6.4 is stated correctly and evaluated wrongly, by a factor of 400.**
§6.4 writes Δt < 2C₁/(h + 4εσT³ + 1/R₁₂) and then says that "for thin painted metal
(C₁ ~ 5 kJ m⁻² K⁻¹) lands around 60–200 s". It does not. For the §16.2 car-paint row the
substrate term 1/R₁₂ is about **37 500 W m⁻² K⁻¹**, three orders above h + 4εσT³ ≈ 33, and the
bound is **0.235 s**. The quoted range is what the *single-node* bound gives at high wind, i.e. the
formula with the term that dominates it removed. Recorded as spec issue S39; this module computes
the bound from its own equations and refuses a step that violates it, so the discrepancy cannot be
absorbed silently.

The consequence is a design decision rather than a nuisance: a thin panel is not integrated
explicitly at a scene tick at all. It is a **single node with a resistive back boundary**, whose
bound is the comfortable 2C/(h + 4εσT³), or it is stepped at its own rate inside one tick. ADR
0036 records both the parameters and that choice.

docs/physics-model.md §6.3, §6.4; ADR 0036, spec issue S39
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.balance import SurfaceForcing, ThermalProperties, net_flux

__all__ = [
    "GUARD_T_MAX_K",
    "NodeLayer",
    "TwoNodeProperties",
    "TwoNodeState",
    "LumpedTwoNodeSolver",
    "contact_resistance",
]

#: Worst-case temperature for the radiative linearisation in the stability guard. A guard has to
#: be valid over the whole run, and 4εσT³ grows with T, so it is evaluated where it is largest.
#: 400 K is above any surface these scenes reach and below anything the LUT refuses.
GUARD_T_MAX_K = 400.0


@dataclass(frozen=True)
class NodeLayer:
    """One slab: how much heat it holds and how well it conducts across itself."""

    thickness_m: float
    conductivity_w_mk: float
    density_kg_m3: float
    specific_heat_j_kgk: float

    def __post_init__(self) -> None:
        for name, value in (
            ("thickness_m", self.thickness_m),
            ("conductivity_w_mk", self.conductivity_w_mk),
            ("density_kg_m3", self.density_kg_m3),
            ("specific_heat_j_kgk", self.specific_heat_j_kgk),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def heat_capacity_j_m2_k(self) -> float:
        return self.density_kg_m3 * self.specific_heat_j_kgk * self.thickness_m

    @property
    def half_resistance_m2k_w(self) -> float:
        """δ/(2k): centre-to-face conduction resistance."""
        return self.thickness_m / (2.0 * self.conductivity_w_mk)

    @classmethod
    def from_material(cls, material: Any, thickness_m: float | None = None) -> NodeLayer:
        thermal = material.spec.thermal
        return cls(
            thickness_m=thermal.thickness_m if thickness_m is None else thickness_m,
            conductivity_w_mk=thermal.conductivity_w_mk,
            density_kg_m3=thermal.density_kg_m3,
            specific_heat_j_kgk=thermal.specific_heat_j_kgk,
        )


def contact_resistance(surface: NodeLayer, substrate: NodeLayer) -> float:
    """R₁₂ = δ₁/(2k₁) + δ₂/(2k₂), exactly as §6.4 writes it."""
    return surface.half_resistance_m2k_w + substrate.half_resistance_m2k_w


@dataclass(frozen=True)
class TwoNodeProperties:
    """The two layers, the optical properties of the top one, and the deep boundary.

    ``back_resistance_m2k_w`` is R₂d and ``deep_temperature_k`` is T_deep; §6.4 names both and
    defines neither (ADR 0036). The default is an **adiabatic back** -- infinite resistance -- so
    a caller who has not thought about what is behind the substrate gets a closed system rather
    than a silent leak to a temperature nobody chose.
    """

    surface: NodeLayer
    substrate: NodeLayer
    optical: ThermalProperties
    back_resistance_m2k_w: float = math.inf
    deep_temperature_k: float | None = None

    def __post_init__(self) -> None:
        if self.back_resistance_m2k_w <= 0.0:
            raise ValueError("back resistance must be positive (use inf for an adiabatic back)")
        if math.isfinite(self.back_resistance_m2k_w) and self.deep_temperature_k is None:
            raise ValueError(
                "a finite back resistance needs a deep_temperature_k: R₂d without T_deep is a "
                "path to an unspecified reservoir (ADR 0036)"
            )
        if self.deep_temperature_k is not None and self.deep_temperature_k <= 0.0:
            raise ValueError("deep_temperature_k must be positive (kelvin)")
        if self.optical.heat_capacity_j_m2_k <= 0.0:
            raise ValueError("optical properties carry a non-positive capacity")

    @property
    def r12_m2k_w(self) -> float:
        return contact_resistance(self.surface, self.substrate)

    def stability_limit_s(self, h_max_w_m2_k: float, t_max_k: float = GUARD_T_MAX_K) -> float:
        """§6.4's bound, Δt < 2C₁/(h + 4εσT³ + 1/R₁₂), at the worst case of both arguments."""
        if h_max_w_m2_k < 0.0:
            raise ValueError("h_max must be non-negative")
        conductance = (
            h_max_w_m2_k
            + 4.0 * self.optical.emissivity * SIGMA_SB * t_max_k**3
            + 1.0 / self.r12_m2k_w
        )
        return 2.0 * self.surface.heat_capacity_j_m2_k / conductance


@dataclass(frozen=True)
class TwoNodeState:
    surface_k: float
    substrate_k: float

    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.surface_k, self.substrate_k], dtype=np.float64)


class LumpedTwoNodeSolver:
    """RK2 on the two-node system, with the §6.4 step bound enforced at construction.

    The guard is in the **constructor**, not in ``advance``: a caller that has chosen a tick has
    chosen it once, and discovering on frame 4000 that the tick was unstable is discovering it
    after the scene has already been rendered.
    """

    def __init__(
        self,
        properties: TwoNodeProperties,
        dt_s: float,
        h_max_w_m2_k: float,
        initial: TwoNodeState,
        t_max_k: float = GUARD_T_MAX_K,
    ) -> None:
        limit = properties.stability_limit_s(h_max_w_m2_k, t_max_k)
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if dt_s > limit:
            raise ValueError(
                f"dt = {dt_s:g} s exceeds the §6.4 explicit bound {limit:g} s for this pair of "
                f"layers (R₁₂ = {properties.r12_m2k_w:.3e} m²K/W contributes "
                f"{1.0 / properties.r12_m2k_w:.1f} W/m²/K of the "
                f"{2.0 * properties.surface.heat_capacity_j_m2_k / limit:.1f} total). A thin "
                f"panel wants a single node with a resistive back boundary, not this solver at a "
                f"scene tick (ADR 0036, spec issue S39)."
            )
        self.properties = properties
        self.dt_s = float(dt_s)
        self._state = np.array([initial.surface_k, initial.substrate_k], dtype=np.float64)

    @property
    def state(self) -> TwoNodeState:
        return TwoNodeState(float(self._state[0]), float(self._state[1]))

    def derivative(
        self, temperatures: NDArray[np.float64], forcing: SurfaceForcing
    ) -> NDArray[np.float64]:
        t1, t2 = temperatures[0], temperatures[1]
        props = self.properties
        conduction = (t1 - t2) / props.r12_m2k_w
        d1 = (float(net_flux(t1, props.optical, forcing)) - conduction) / (
            props.surface.heat_capacity_j_m2_k
        )
        deep = 0.0
        if math.isfinite(props.back_resistance_m2k_w):
            assert props.deep_temperature_k is not None
            deep = (t2 - props.deep_temperature_k) / props.back_resistance_m2k_w
        d2 = (conduction - deep) / props.substrate.heat_capacity_j_m2_k
        return np.array([d1, d2], dtype=np.float64)

    def advance(self, forcing: SurfaceForcing, dt_s: float | None = None) -> TwoNodeState:
        """One midpoint step under a constant forcing; returns the new state."""
        dt = self.dt_s if dt_s is None else float(dt_s)
        half = self._state + 0.5 * dt * self.derivative(self._state, forcing)
        self._state = self._state + dt * self.derivative(half, forcing)
        return self.state

    def equilibrium(self, forcing: SurfaceForcing) -> TwoNodeState:
        """The steady state, found on the **surface** node alone.

        At equilibrium every flux through the stack is the same, so T₂ is determined by T₁ and the
        two-node root reduces to a one-dimensional problem -- which keeps the bisection of M6.7
        usable and its uniqueness argument intact.
        """
        props = self.properties
        adiabatic = not math.isfinite(props.back_resistance_m2k_w)

        def substrate_for(t1: float) -> float:
            if adiabatic:
                return t1
            assert props.deep_temperature_k is not None
            ratio = props.back_resistance_m2k_w / (props.r12_m2k_w + props.back_resistance_m2k_w)
            return props.deep_temperature_k + ratio * (t1 - props.deep_temperature_k)

        def residual(t1: float) -> float:
            conduction = (t1 - substrate_for(t1)) / props.r12_m2k_w
            return float(net_flux(t1, props.optical, forcing)) - conduction

        lo, hi = 150.0, 900.0
        if residual(lo) < 0.0 or residual(hi) > 0.0:
            raise ValueError("two-node equilibrium is outside the 150-900 K bracket")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        t1 = 0.5 * (lo + hi)
        return TwoNodeState(t1, substrate_for(t1))
