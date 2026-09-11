"""NETD anchoring: scale the Gaussian noise so the predicted NETD at 300 K equals the datasheet.

docs/physics-model.md §9.4 steps 1-4 and Appendix A #7: NETD is a calibration handle, not a noise
generator. The physical chain (transfer, aperture, band derivative, Poisson shot) sets the
*structure*; the datasheet number sets the *magnitude*, by solving for the one scene-independent
Gaussian σ (ADR 0025):

    σ_gaussian² = (NETD_target · ∂S/∂T(T_ref, F_ref))² − σ_shot²(T_ref)

Poisson shot terms are never rescaled: if shot noise alone already exceeds the target the config
is physically unattainable and this raises. F_ref is ``noise.netd_ref_f_number`` (the f-number the
datasheet NETD was measured at) or the configured f-number; the configured F enters the running
camera only through the transfer, so a faster lens *improves* the predicted NETD and a slower one
degrades it by the aperture-factor ratio, e.g. NETD(F/1.4)/NETD(F/1.0) = (4·1.96+1)/(4+1) = 1.768,
not 1.96 (§9.4 datasheet form vs +1 form, ADR 0024).

docs/physics-model.md §9.4, §12.2 noise.netd_mk_at_300k, §16.1 NETD grades, Appendix A #7
"""

from __future__ import annotations

import math
from typing import Literal

from irsim.config.sensor import SensorSpec
from irsim.detector.netd import (
    NoiseBudget,
    bolometer_floors,
    fpa_params_from_config_spec,
    shot_variance,
    signal_derivative_per_k,
)
from irsim.detector.params import BolometerParams, PhotonParams
from irsim.radiometry.lut import BandLUT

__all__ = ["anchor_noise", "T_REF_ANCHOR_K"]

T_REF_ANCHOR_K = 300.0


def anchor_noise(
    sensor: SensorSpec,
    lut: BandLUT,
    target_netd_k: float | None = None,
    t_ref_k: float = T_REF_ANCHOR_K,
    dark_electrons: float = 0.0,
    background_electrons: float = 0.0,
) -> NoiseBudget:
    """Solve for the scene-independent Gaussian σ (W or e⁻) that reproduces the target NETD."""
    target = sensor.noise.netd_mk_at_300k * 1e-3 if target_netd_k is None else target_netd_k
    if target <= 0.0:
        raise ValueError("target NETD must be positive")
    f_ref = sensor.noise.netd_ref_f_number or sensor.optics.f_number
    params = fpa_params_from_config_spec(sensor)
    kind: Literal["bolometer", "photon"] = (
        "bolometer" if isinstance(params, BolometerParams) else "photon"
    )
    d_signal = signal_derivative_per_k(t_ref_k, sensor, lut, f_number=f_ref)
    provisional = NoiseBudget(
        kind=kind,
        sigma_gaussian=0.0,
        dark_electrons=dark_electrons,
        background_electrons=background_electrons,
    )
    var_shot = shot_variance(t_ref_k, sensor, lut, provisional, f_number=f_ref)
    var_needed = (target * d_signal) ** 2
    if var_needed <= var_shot:
        shot_netd = math.sqrt(var_shot) / d_signal
        raise ValueError(
            f"NETD target {target * 1e3:.1f} mK is unattainable: Poisson shot noise alone gives "
            f"{shot_netd * 1e3:.1f} mK at {t_ref_k:.0f} K and F/{f_ref:g}; shot terms are never "
            "rescaled (ADR 0025) -- change eta, t_int, F or the target"
        )
    sigma_gaussian = math.sqrt(var_needed - var_shot)
    floors = bolometer_floors(sensor, lut, t_ref_k) if isinstance(params, BolometerParams) else {}
    if isinstance(params, PhotonParams) and params.read_noise_e is not None:
        floors["read_noise_e_configured"] = params.read_noise_e
    return NoiseBudget(
        kind=kind,
        sigma_gaussian=sigma_gaussian,
        dark_electrons=dark_electrons,
        background_electrons=background_electrons,
        floors=floors,
    )
