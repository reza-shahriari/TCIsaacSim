"""NETD predictor for both detector classes (docs/physics-model.md §9.4 photon-count form).

    NETD(T) = σ_total(T) / (∂S/∂T)(T)

The signal S and its derivative are in the detector's *physical* pre-quantisation unit:

* **bolometer**: S = Φ, the in-band pixel power in W. ∂S/∂T = A_d · π τ/(4F²+1) · (∂L_B/∂T);
  noise is scene-independent Gaussian (Johnson, ROIC, 1/f lumped) -- so NETD ∝ 1/(∂L_B/∂T) and
  falls with scene temperature exactly as the derivative ratio (the [R9] 39 → 23 mK anchor);
* **photon**: S = N_e, photoelectrons per integration. ∂S/∂T = η A_d t_int · π τ/(4F²+1) ·
  (∂L_{q,B}/∂T); σ² = N_e(T) + N_dark + N_background + σ_read² (Poisson terms plus Gaussian read).

The aperture factor enters only through :mod:`irsim.optics` (non-negotiable #5); an f-number can
be passed explicitly so that a datasheet NETD measured at one F can be anchored and the camera run
at another (ADR 0025). The DN gain never enters: NETD is gain-invariant (ADR 0019). The §10.1 1/f
term is represented by the drift model (M9.4), not here (ADR 0054).

docs/physics-model.md §9.4, §9.3, §3.4, §10.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from irsim.config.sensor import SensorSpec
from irsim.detector.figures_of_merit import (
    enbw_first_order_hz,
    johnson_noise_v,
    temperature_fluctuation_nep_w,
)
from irsim.detector.params import BolometerParams, PhotonParams, fpa_params_from_config
from irsim.optics.aperture import aperture_factor
from irsim.radiometry.lut import BandLUT

__all__ = [
    "NoiseBudget",
    "signal_derivative_per_k",
    "shot_variance",
    "sigma_total",
    "predict_netd_k",
    "bolometer_floors",
]


@dataclass(frozen=True)
class NoiseBudget:
    """Scene-independent noise of one camera in its signal unit, plus the Poisson bookkeeping.

    ``sigma_gaussian``: the lumped Gaussian σ (W for bolometers: Johnson + ROIC + 1/f; e⁻ for
    photon FPAs: read noise). ``dark_electrons`` / ``background_electrons``: mean Poisson counts
    per integration (photon FPAs). ``floors``: first-principles figures reported for the record.
    """

    kind: Literal["bolometer", "photon"]
    sigma_gaussian: float
    dark_electrons: float = 0.0
    background_electrons: float = 0.0
    floors: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.sigma_gaussian < 0.0
            or self.dark_electrons < 0.0
            or self.background_electrons < 0.0
        ):
            raise ValueError("noise terms cannot be negative")


def _f_number(sensor: SensorSpec, f_number: float | None) -> float:
    return sensor.optics.f_number if f_number is None else f_number


def signal_derivative_per_k(
    temperature_k: float, sensor: SensorSpec, lut: BandLUT, f_number: float | None = None
) -> float:
    """∂S/∂T at a blackbody temperature: W/K (bolometer) or e⁻/K (photon)."""
    params = fpa_params_from_config_spec(sensor)
    f = _f_number(sensor, f_number)
    geometry = params.active_area_m2 * aperture_factor(f) * sensor.optics.transmittance
    if isinstance(params, BolometerParams):
        return geometry * float(lut.lookup(temperature_k, "dlb_dt")[()])
    return (
        params.quantum_efficiency
        * params.integration_time_s
        * geometry
        * float(lut.lookup(temperature_k, "dlb_q_dt")[()])
    )


def mean_signal(
    temperature_k: float, sensor: SensorSpec, lut: BandLUT, f_number: float | None = None
) -> float:
    """S(T) for a blackbody: pixel power (W) or photoelectrons (e⁻)."""
    params = fpa_params_from_config_spec(sensor)
    f = _f_number(sensor, f_number)
    geometry = params.active_area_m2 * aperture_factor(f) * sensor.optics.transmittance
    if isinstance(params, BolometerParams):
        return geometry * float(lut.lookup(temperature_k, "lb")[()])
    return (
        params.quantum_efficiency
        * params.integration_time_s
        * geometry
        * float(lut.lookup(temperature_k, "lb_q")[()])
    )


def shot_variance(
    temperature_k: float,
    sensor: SensorSpec,
    lut: BandLUT,
    budget: NoiseBudget,
    f_number: float | None = None,
) -> float:
    """Poisson variance N_e(T) + N_dark + N_bg (photon); 0 for a bolometer."""
    if budget.kind == "bolometer":
        return 0.0
    return (
        mean_signal(temperature_k, sensor, lut, f_number)
        + budget.dark_electrons
        + budget.background_electrons
    )


def sigma_total(
    temperature_k: float,
    sensor: SensorSpec,
    lut: BandLUT,
    budget: NoiseBudget,
    f_number: float | None = None,
) -> float:
    """σ_total(T) = √(σ_shot²(T) + σ_gaussian²) in the signal unit (§9.4)."""
    return math.sqrt(
        shot_variance(temperature_k, sensor, lut, budget, f_number) + budget.sigma_gaussian**2
    )


def predict_netd_k(
    temperature_k: float,
    sensor: SensorSpec,
    lut: BandLUT,
    budget: NoiseBudget,
    f_number: float | None = None,
) -> float:
    """NETD(T) = σ_total(T) / (∂S/∂T)(T) in kelvin (§9.4 photon-count form)."""
    return sigma_total(temperature_k, sensor, lut, budget, f_number) / signal_derivative_per_k(
        temperature_k, sensor, lut, f_number
    )


def bolometer_floors(sensor: SensorSpec, lut: BandLUT, t_ref_k: float = 300.0) -> dict[str, float]:
    """First-principles bolometer noise floors at T_ref, reported for the record (§10.1):
    ENBW = 1/(4τ_th); temperature-fluctuation and Johnson NEPs and their NETD equivalents."""
    params = fpa_params_from_config_spec(sensor)
    if not isinstance(params, BolometerParams):
        raise TypeError("bolometer_floors needs a bolometer FPA")
    enbw = enbw_first_order_hz(params.thermal_time_constant_s)
    t_fpa = params.fpa_temp_k if params.fpa_temp_k is not None else t_ref_k
    nep_tf = temperature_fluctuation_nep_w(t_fpa, params.g_th_w_per_k, enbw)
    r0 = abs(
        params.absorptance
        * params.tcr_per_k
        * params.bias_current_a
        * params.resistance_ohm
        / params.g_th_w_per_k
    )
    nep_j = johnson_noise_v(t_fpa, params.resistance_ohm, enbw) / r0
    d_absorbed = params.absorptance * signal_derivative_per_k(t_ref_k, sensor, lut)
    return {
        "enbw_hz": enbw,
        "nep_temperature_fluctuation_w": nep_tf,
        "nep_johnson_w": nep_j,
        "netd_temperature_fluctuation_k": nep_tf / d_absorbed,
        "netd_johnson_k": nep_j / d_absorbed,
        "responsivity_v_per_w": r0,
    }


def fpa_params_from_config_spec(sensor: SensorSpec) -> BolometerParams | PhotonParams:
    """FpaParams from a SensorSpec (the fpa block alone is enough)."""
    from irsim.config.sensor import SensorConfig

    return fpa_params_from_config(SensorConfig(sensor=sensor))
