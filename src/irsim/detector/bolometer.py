"""Ideal microbolometer static transfer: pixel power → membrane ΔT → signal → DN.

Physics (docs/physics-model.md §9.2): absorbed power α_abs Φ heats a membrane of conductance G_th,
giving a steady-state temperature rise ΔT = α_abs Φ / G_th and a fractional resistance change
β ΔT (β = TCR). The static responsivity is R₀ = α_abs β I_bias R / G_th [V/W]. Everything is
**linear in absorbed power** -- a bolometer is a power detector, so it takes the energy-form band
radiance L_B, never the photon form (a plausibility guard refuses photon-scale inputs).

DN representation (ADR 0019): the volts-per-DN chain of a real ROIC is unknown for any datasheet
camera, so the transfer carries one **effective gain in DN per watt** and an offset in watts, sized
so that a chosen scene-temperature range spans the ADC. Responsivity R₀ is computed and reported
for the record but does not feed the DN. NETD is anchored later (M4), not derived from R₀.

Dynamic behaviour (the τ_th IIR across frames, §9.2) is M9.1; this module is the static transfer.

docs/physics-model.md §9.2, §8.1, §2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.detector.params import BolometerParams
from irsim.detector.quantise import dn_max_for_bits, quantise

__all__ = [
    "PHOTON_SCALE_GUARD",
    "absorbed_power_w",
    "membrane_delta_t_k",
    "static_responsivity_v_per_w",
    "BolometerTransfer",
]

# In-band power on one LWIR pixel is ~1e-8 W; a photon *rate* on the same pixel is ~1e12 /s.
# Anything above this is not watts and would be a form mix-up (§9.2 energy form).
PHOTON_SCALE_GUARD = 1e-3

FloatArray = NDArray[np.floating]


def _power(phi_w: object) -> NDArray[np.float64]:
    p = np.asarray(phi_w)
    if p.dtype == np.float16:
        raise TypeError("pixel power is float16; use float32 or better")
    p = p.astype(np.float64)
    if np.any(p < 0.0):
        raise ValueError("pixel power cannot be negative")
    if np.any(p > PHOTON_SCALE_GUARD):
        raise ValueError(
            f"pixel power {float(p.max()):.3e} exceeds {PHOTON_SCALE_GUARD} W: this looks like a "
            "photon rate, not watts -- the bolometer takes energy-form band radiance (§9.2)"
        )
    return p


def absorbed_power_w(phi_w: object, params: BolometerParams) -> NDArray[np.float64]:
    """α_abs · Φ: the power the membrane actually absorbs (§9.2)."""
    return params.absorptance * _power(phi_w)


def membrane_delta_t_k(phi_w: object, params: BolometerParams) -> NDArray[np.float64]:
    """Steady-state membrane temperature rise α_abs Φ / G_th (§9.2)."""
    return absorbed_power_w(phi_w, params) / params.g_th_w_per_k


def static_responsivity_v_per_w(params: BolometerParams) -> float:
    """R₀ = α_abs β I_bias R / G_th [V/W]; negative for VOx (β < 0). Reported, not used for DN."""
    return (
        params.absorptance
        * params.tcr_per_k
        * params.bias_current_a
        * params.resistance_ohm
        / params.g_th_w_per_k
    )


@dataclass(frozen=True)
class BolometerTransfer:
    """DN = quantise(gain_dn_per_w · (Φ − offset_w)): ideal linear static transfer (ADR 0019)."""

    gain_dn_per_w: float
    offset_w: float
    bit_depth: int

    def __post_init__(self) -> None:
        if not self.gain_dn_per_w > 0.0 or self.offset_w < 0.0:
            raise ValueError("gain must be positive and offset non-negative")
        dn_max_for_bits(self.bit_depth)

    @classmethod
    def from_power_range(
        cls, phi_min_w: float, phi_max_w: float, bit_depth: int
    ) -> BolometerTransfer:
        """Size the gain so that [Φ_min, Φ_max] maps onto [0, 2^bits − 1]."""
        if not 0.0 <= phi_min_w < phi_max_w:
            raise ValueError("need 0 <= phi_min < phi_max")
        top = dn_max_for_bits(bit_depth)
        return cls(
            gain_dn_per_w=top / (phi_max_w - phi_min_w), offset_w=phi_min_w, bit_depth=bit_depth
        )

    def signal_dn(self, phi_w: object) -> NDArray[np.float32]:
        """The un-quantised, un-clipped signal in DN units (float32) -- noise is added here."""
        s = self.gain_dn_per_w * (_power(phi_w) - self.offset_w)
        return np.asarray(s, dtype=np.float32)

    def dn(self, phi_w: object) -> NDArray[np.uint16]:
        return quantise(self.signal_dn(phi_w), self.bit_depth)

    def power_from_signal_w(self, signal_dn: object) -> NDArray[np.float64]:
        """Inverse of :meth:`signal_dn` for the radiometric branch (M3.10)."""
        s = np.asarray(signal_dn)
        if s.dtype == np.float16:
            raise TypeError("signal is float16")
        return s.astype(np.float64) / self.gain_dn_per_w + self.offset_w
