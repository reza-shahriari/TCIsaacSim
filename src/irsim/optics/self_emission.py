"""Optics self-emission: the warm lens and housing the detector also sees.

In LWIR the optics glow. For a single-lens uncooled core (docs/physics-model.md §8.2):

    Φ_self ≈ A_d · Ω_eff · (1 − τ_opt) · L_B(T_housing),     Ω_eff = π / (4F² + 1)

i.e. the lens is a grey body of emissivity 1 − τ_opt (ρ_lens = 0, ADR 0016) at the housing
temperature, filling the same cone as the scene. Because T_housing drifts, this term is the physical
origin of shutterless drift and the reason for periodic flat-field correction (§11.2).

The general N-element form for a window/lens/mirror stack, each element attenuated by everything
downstream of it (§8.2, [R19]):

    L_self = Σ_i ε_i L_B(T_i) Π_{j>i} τ_j,      ε_i = 1 − τ_i − ρ_i   (Kirchhoff, non-negotiable #4)

Everything here is in radiance/power space -- never kelvin (non-negotiable #3). The caller supplies
band radiances L_B(T) from the band LUT or the closed-form top-hat, so the module is band-agnostic.

docs/physics-model.md §8.2, §2 (Φ_self), §11.2
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from irsim.optics.aperture import aperture_factor

__all__ = [
    "KIRCHHOFF_TOL",
    "OpticalElement",
    "self_emission_power",
    "stack_transmittance",
    "stack_self_radiance",
]

KIRCHHOFF_TOL = 1e-6


def self_emission_power(
    active_area_m2: float, f_number: float, tau_opt: float, lb_housing: float
) -> float:
    """Φ_self = A_d Ω_eff (1 − τ_opt) L_B(T_housing), single-lens form (§8.2). Units follow L_B."""
    if not 0.0 < tau_opt <= 1.0:
        raise ValueError(f"tau_opt must be in (0, 1], got {tau_opt}")
    if not active_area_m2 > 0.0:
        raise ValueError("active_area_m2 must be positive")
    if lb_housing < 0.0:
        raise ValueError("lb_housing is a band radiance and cannot be negative")
    return active_area_m2 * aperture_factor(f_number) * (1.0 - tau_opt) * lb_housing


@dataclass(frozen=True)
class OpticalElement:
    """One element of the optical train: transmittance τ, reflectance ρ, temperature T.

    Emissivity is derived, ε = 1 − τ − ρ, never authored (non-negotiable #4); τ + ρ > 1 raises.
    """

    tau: float
    temperature_k: float
    rho: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0 or not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"{self.name or 'element'}: tau and rho must lie in [0, 1]")
        if self.tau + self.rho > 1.0 + KIRCHHOFF_TOL:
            raise ValueError(
                f"{self.name or 'element'}: tau + rho = {self.tau + self.rho:.6f} > 1 violates "
                "Kirchhoff closure (epsilon = 1 - tau - rho would be negative)"
            )
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")

    @property
    def emissivity(self) -> float:
        return max(0.0, 1.0 - self.tau - self.rho)


def stack_transmittance(elements: Sequence[OpticalElement]) -> float:
    """Π τ_i over the whole train."""
    out = 1.0
    for e in elements:
        out *= e.tau
    return out


def stack_self_radiance(
    elements: Sequence[OpticalElement], lb_of_t: Callable[[float], float]
) -> float:
    """L_self = Σ_i ε_i L_B(T_i) Π_{j>i} τ_j, elements ordered scene → detector (§8.2).

    ``lb_of_t`` maps a temperature to band radiance (LUT lookup or closed form). Multiply by
    A_d Ω_eff for the power on a pixel; with one element of τ = τ_opt, ρ = 0 this reduces exactly to
    :func:`self_emission_power` / (A_d Ω_eff).
    """
    total = 0.0
    for i, e in enumerate(elements):
        downstream = 1.0
        for later in elements[i + 1 :]:
            downstream *= later.tau
        total += e.emissivity * lb_of_t(e.temperature_k) * downstream
    return total
