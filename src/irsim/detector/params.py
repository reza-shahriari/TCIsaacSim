"""Detector (FPA) parameters as the detector models consume them.

``FpaParams`` mirrors the validated ``fpa:`` block (:mod:`irsim.config.sensor`) plus the §9
constants §12.2 omits, as a frozen typed object with the derived quantities every detector model
needs: active area A_d, frame period, DN range. Discriminated on ``type`` so a bolometer model can
never be handed photon parameters by accident.

Two conventions are fixed here and asserted downstream (ADR 0009):

* **R(λ) is peak-normalised** (a shape). The photon detector's quantum efficiency η is
  ``quantum_efficiency``, a separate scalar, applied **exactly once** in
  :func:`irsim.detector.photon.photoelectrons` -- never folded into R(λ), never applied by the LUT.
* **Bolometers have no η.** Their absorptance α_abs and thermal conductance G_th set the
  membrane response; responsivity is anchored by NETD (§9.4), not predicted (Appendix A #7).

docs/physics-model.md §12.2, §9.1, §9.2, §2 (A_d)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from irsim.config.sensor import BolometerFpa, PhotonFpa, SensorConfig

__all__ = ["FpaParams", "BolometerParams", "PhotonParams", "fpa_params_from_config"]


@dataclass(frozen=True)
class _Common:
    width: int
    height: int
    pitch_um: float
    fill_factor: float
    frame_rate_hz: float
    bit_depth: int
    fpa_temp_k: float | None
    fpa_tau_s: float | None
    fpa_self_heating_k: float

    @property
    def active_area_m2(self) -> float:
        """A_d = (pitch · 1e-6)² · fill_factor (§2, §9.1)."""
        return (self.pitch_um * 1e-6) ** 2 * self.fill_factor

    @property
    def active_width_um(self) -> float:
        return math.sqrt(self.fill_factor) * self.pitch_um

    @property
    def frame_dt_s(self) -> float:
        return 1.0 / self.frame_rate_hz

    @property
    def dn_max(self) -> int:
        return int(2**self.bit_depth - 1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class BolometerParams(_Common):
    """§9.2 microbolometer: thermal time constant, TCR β, absorptance α_abs, G_th, bias, R."""

    type: Literal["bolometer"]
    thermal_time_constant_s: float
    tcr_per_k: float
    absorptance: float
    g_th_w_per_k: float
    bias_current_a: float
    resistance_ohm: float

    @property
    def c_th_j_per_k(self) -> float:
        """Membrane heat capacity C_th = τ_th · G_th (§9.2)."""
        return self.thermal_time_constant_s * self.g_th_w_per_k


@dataclass(frozen=True)
class PhotonParams(_Common):
    """§9.1 photon detector: η, well capacity, integration time, read noise, dark current."""

    type: Literal["photon"]
    quantum_efficiency: float
    well_capacity_e: float
    integration_time_s: float
    dark_current_model: str
    read_noise_e: float | None
    dark_current_i_ref_a: float | None
    dark_current_t_ref_k: float | None
    dark_current_band_gap_ev: float | None


FpaParams = BolometerParams | PhotonParams


def fpa_params_from_config(config: SensorConfig) -> FpaParams:
    """Build the typed detector parameters from a validated sensor config."""
    fpa = config.sensor.fpa
    if isinstance(fpa, BolometerFpa):
        return BolometerParams(
            width=fpa.width,
            height=fpa.height,
            pitch_um=fpa.pitch_um,
            fill_factor=fpa.fill_factor,
            frame_rate_hz=fpa.frame_rate_hz,
            bit_depth=fpa.bit_depth,
            fpa_temp_k=fpa.fpa_temp_k,
            fpa_tau_s=fpa.fpa_tau_s,
            fpa_self_heating_k=fpa.fpa_self_heating_k,
            type="bolometer",
            thermal_time_constant_s=fpa.thermal_time_constant_ms * 1e-3,
            tcr_per_k=fpa.tcr_per_k,
            absorptance=fpa.absorptance,
            g_th_w_per_k=fpa.g_th_w_per_k,
            bias_current_a=fpa.bias_current_a,
            resistance_ohm=fpa.resistance_ohm,
        )
    if isinstance(fpa, PhotonFpa):
        dc = fpa.dark_current
        return PhotonParams(
            width=fpa.width,
            height=fpa.height,
            pitch_um=fpa.pitch_um,
            fill_factor=fpa.fill_factor,
            frame_rate_hz=fpa.frame_rate_hz,
            bit_depth=fpa.bit_depth,
            fpa_temp_k=fpa.fpa_temp_k,
            fpa_tau_s=fpa.fpa_tau_s,
            fpa_self_heating_k=fpa.fpa_self_heating_k,
            type="photon",
            quantum_efficiency=fpa.quantum_efficiency,
            well_capacity_e=fpa.well_capacity_e,
            integration_time_s=fpa.integration_time_ms * 1e-3,
            dark_current_model=fpa.dark_current_model,
            read_noise_e=fpa.read_noise_e,
            dark_current_i_ref_a=dc.i_ref_a_per_pixel if dc else None,
            dark_current_t_ref_k=dc.t_ref_k if dc else None,
            dark_current_band_gap_ev=dc.band_gap_ev if dc else None,
        )
    raise TypeError(f"unknown FPA spec {type(fpa).__name__}")  # pragma: no cover
