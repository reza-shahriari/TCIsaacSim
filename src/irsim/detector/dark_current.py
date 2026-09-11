"""Arrhenius dark current for photon detectors (docs/physics-model.md §9.1).

    i_dark(T) = i_ref · (T / T_ref)^{3/2} · exp(−E_g / 2k_B · (1/T − 1/T_ref))

so that a diffusion-limited detector's dark current grows exponentially with FPA temperature --
why MWIR InSb is cryocooled to 77-150 K. Band gaps live in :mod:`irsim.radiometry.constants`.
Dark electrons per integration are i_dark · t_int / q and are a Poisson term in the noise budget.
"""

from __future__ import annotations

import math

from irsim.radiometry.constants import (
    BAND_GAP_INGAAS_EV,
    BAND_GAP_INSB_EV,
    EV_PER_K,
    Q_E,
)

__all__ = ["dark_current_a", "dark_electrons", "BAND_GAPS_EV", "FPA_TEMPERATURE_MIN_K"]

BAND_GAPS_EV = {"insb": BAND_GAP_INSB_EV, "ingaas": BAND_GAP_INGAAS_EV}
# No infrared FPA modelled here runs below 30 K; a value this small is Celsius by mistake.
FPA_TEMPERATURE_MIN_K = 30.0


def dark_current_a(t_fpa_k: float, i_ref_a: float, t_ref_k: float, band_gap_ev: float) -> float:
    """Dark current at T_FPA from the reference (i_ref at T_ref) and the band gap, in amperes."""
    for name, t in (("t_fpa_k", t_fpa_k), ("t_ref_k", t_ref_k)):
        if t < FPA_TEMPERATURE_MIN_K:
            raise ValueError(
                f"{name} = {t} K is below {FPA_TEMPERATURE_MIN_K} K -- kelvin, not celsius"
            )
    if i_ref_a < 0.0 or band_gap_ev <= 0.0:
        raise ValueError("i_ref must be non-negative and the band gap positive")
    ratio = (t_fpa_k / t_ref_k) ** 1.5
    arrhenius = math.exp(-band_gap_ev / (2.0 * EV_PER_K) * (1.0 / t_fpa_k - 1.0 / t_ref_k))
    return float(i_ref_a * ratio * arrhenius)


def dark_electrons(i_dark_a: float, integration_time_s: float) -> float:
    """Mean dark electrons per integration: i_dark · t_int / q."""
    if i_dark_a < 0.0 or integration_time_s <= 0.0:
        raise ValueError("dark current must be non-negative and t_int positive")
    return i_dark_a * integration_time_s / Q_E
