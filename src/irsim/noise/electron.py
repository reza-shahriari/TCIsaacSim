"""Electron-space noise for photon FPAs, and the budget that replaces the NETD anchor (§10.1, §9.4).

ADR 0025 made NETD the *calibration handle*: the physical chain sets the structure of the noise
and the datasheet number sets its magnitude, by solving for one scene-independent Gaussian σ. That
is the right rule for a **bolometer**, whose datasheet quotes NETD and nothing else usable.

It is the wrong rule for a photon FPA, and M11.1's SWIR camera is where it breaks in the open.
NETD is defined against a 300 K blackbody (§9.4), and a 300 K blackbody puts **1.24
photoelectrons per pixel per 16 ms frame** into 0.9-1.7 µm against 120 e⁻ of read noise. Evaluated
honestly, that camera's NETD is **976 K** — correct physics and a useless anchor. Anchoring to it
would scale the Gaussian term by a factor of 2e4 and produce a camera whose noise is invented.

What an InGaAs or InSb datasheet *does* quote is the thing the noise is actually made of: quantum
efficiency, well capacity, integration time, read noise in electrons, dark current. So for a
photon FPA with those authored, this module builds the budget **from them**, and uses NETD only as
a cross-check — reversing which of the two is derived (ADR 0025 addendum).

Three rules carried over unchanged, because they were never about which handle is primary:

* **Poisson terms are never rescaled.** Shot noise is a property of the photon arrival, not a
  tuning knob. Variance equals mean, and the test checks that on the generator itself.
* **Noise is added in electron space, never in kelvin** (non-negotiable #3).
* **An infeasible configuration raises** rather than quietly producing a camera that cannot
  exist — but only where NETD means something: the check is skipped, with its reason named, for a
  reflective band where the §9.4 definition is vacuous.

docs/physics-model.md §10.1, §9.1, §9.4, §12.1; ADR 0025 (and its M11.6 addendum)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.detector.dark_current import dark_current_a, dark_electrons
from irsim.detector.netd import NoiseBudget, fpa_params_from_config_spec, predict_netd_k
from irsim.detector.params import PhotonParams
from irsim.radiometry.lut import BandLUT

__all__ = [
    "electron_noise",
    "dark_electrons_for",
    "electron_budget",
    "InfeasibleNoiseConfigError",
    "shot_limited_netd_k",
]


class InfeasibleNoiseConfigError(ValueError):
    """A camera whose own shot noise already exceeds the NETD its datasheet claims."""


def electron_noise(
    n_e: Any,
    i_dark_a: float,
    t_int_s: float,
    sigma_read_e: float,
    rng: np.random.Generator,
    background_e: float = 0.0,
) -> NDArray[np.float64]:
    """Poisson(N_signal + N_dark + N_bg) + Normal(0, σ_read), in electrons.

    The Poisson draw covers **every** arriving-carrier term together, which is not a shortcut:
    the sum of independent Poissons is Poisson in their sum, so one draw is exactly right and two
    draws added would be too. Read noise is Gaussian and additive because it is a readout voltage,
    not a count.

    Clipped at zero. A negative electron count is not a physical measurement, and letting one
    through turns into a negative DN that the ISP's histogram then has to have an opinion about.
    """
    mean = np.asarray(n_e, dtype=np.float64)
    if np.any(mean < 0.0):
        raise ValueError("signal electrons cannot be negative")
    if sigma_read_e < 0.0:
        raise ValueError("read noise cannot be negative")
    if background_e < 0.0:
        raise ValueError("background electrons cannot be negative")
    dark = dark_electrons(i_dark_a, t_int_s) if i_dark_a > 0.0 else 0.0
    total_mean = mean + dark + background_e
    counts = rng.poisson(total_mean).astype(np.float64)
    if sigma_read_e > 0.0:
        counts = counts + rng.normal(0.0, sigma_read_e, size=counts.shape)
    return np.asarray(np.maximum(counts, 0.0))


def dark_electrons_for(params: PhotonParams, t_fpa_k: float | None = None) -> float:
    """Mean dark electrons per integration from the config's Arrhenius block; 0 without one."""
    if params.dark_current_i_ref_a is None:
        return 0.0
    assert params.dark_current_t_ref_k is not None
    assert params.dark_current_band_gap_ev is not None
    temperature = t_fpa_k if t_fpa_k is not None else params.fpa_temp_k
    if temperature is None:
        temperature = params.dark_current_t_ref_k
    return dark_electrons(
        dark_current_a(
            float(temperature),
            params.dark_current_i_ref_a,
            params.dark_current_t_ref_k,
            params.dark_current_band_gap_ev,
        ),
        params.integration_time_s,
    )


def electron_budget(
    sensor: SensorSpec,
    lut: BandLUT,
    t_fpa_k: float | None = None,
    background_electrons: float = 0.0,
    t_ref_k: float = 300.0,
    check_netd: bool = True,
) -> NoiseBudget:
    """Build a photon FPA's noise budget from its **electron** datasheet numbers.

    σ_gaussian is the configured ``read_noise_e`` -- used, not solved for. The dark and background
    counts are Poisson terms alongside the signal. Nothing is scaled to hit a NETD.

    ``check_netd`` raises :class:`InfeasibleNoiseConfig` when the resulting budget cannot reach the
    datasheet NETD, i.e. when the camera as specified is worse than it claims. The check is
    **skipped for a reflective band**, where §9.4's 300 K definition is vacuous (a 300 K scene
    emits essentially nothing there) and the configured field is a placeholder by construction.
    """
    params = fpa_params_from_config_spec(sensor)
    if not isinstance(params, PhotonParams):
        raise TypeError(
            "electron_budget is for photon FPAs; a bolometer's only usable datasheet handle is "
            "NETD, so it keeps the ADR 0025 anchor (irsim.detector.anchor.anchor_noise)"
        )
    if params.read_noise_e is None:
        raise ValueError(
            "photon FPA has no read_noise_e: either author it (the number every InGaAs/InSb "
            "datasheet quotes) or use the NETD anchor deliberately"
        )
    dark = dark_electrons_for(params, t_fpa_k)
    budget = NoiseBudget(
        kind="photon",
        sigma_gaussian=float(params.read_noise_e),
        dark_electrons=dark,
        background_electrons=float(background_electrons),
        floors={
            "read_noise_e_configured": float(params.read_noise_e),
            "dark_electrons_per_integration": dark,
            "background_electrons_per_integration": float(background_electrons),
        },
    )
    if check_netd and sensor.band.regime != "reflective":
        predicted = predict_netd_k(t_ref_k, sensor, lut, budget)
        target = sensor.noise.netd_mk_at_300k * 1e-3
        if predicted > target:
            raise InfeasibleNoiseConfigError(
                f"the electron budget gives NETD {predicted * 1e3:.1f} mK at {t_ref_k:.0f} K, "
                f"worse than the datasheet's {target * 1e3:.1f} mK. Poisson terms are never "
                f"rescaled (ADR 0025), so this camera cannot reach its own claim -- raise eta, "
                f"t_int or the aperture, lower read_noise_e, or correct the datasheet figure."
            )
    return budget


def shot_limited_netd_k(sensor: SensorSpec, lut: BandLUT, t_ref_k: float = 300.0) -> float:
    """The best NETD this camera could have: shot noise alone, nothing else.

    A floor in the strict sense -- nothing in §10 can go below it, because the only term left is
    the photon arrival statistics themselves.
    """
    return predict_netd_k(t_ref_k, sensor, lut, NoiseBudget(kind="photon", sigma_gaussian=0.0))
