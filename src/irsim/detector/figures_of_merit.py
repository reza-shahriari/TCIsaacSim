"""Datasheet figures of merit and the noise bandwidth of a bolometer (docs/physics-model.md §9.3).

Responsivity R [V/W], NEP = V_n/R [W], D* = √(A_d Δf)/NEP [cm·Hz^½/W], and the two NETD forms
of §9.4 as *documented conversions*:

* the **photon-count / +1 form** NETD = σ / (∂S/∂T), which the noise chain uses (ADR 0024);
* the **datasheet / paraxial form** NETD = 4F² NEP / (A_d M'), which is 4F²/(4F² + 1) = 0.8 of
  the +1 form at F/1.0 (M' = π ∂L_B/∂T, the exitance derivative) -- provided only for comparing
  against published numbers.

Bolometer noise bandwidth: the equivalent noise bandwidth of the first-order membrane response is
ENBW = 1/(4τ_th) (§9.2, §10.1) for white noise at the input; the sampled per-frame IIR reaches that
value only when Δt ≪ τ_th, and at 60 Hz with τ_th = 10 ms it is ~20 % lower -- both are provided.
The physical bolometer floors: Johnson V_n = √(4 k_B T R Δf) and temperature-fluctuation
NEP = √(4 k_B T² G_th Δf) (§10.1).

docs/physics-model.md §9.3, §9.4, §10.1, §8.1
"""

from __future__ import annotations

import math

from irsim.optics.aperture import aperture_factor
from irsim.radiometry.constants import K_BOLTZMANN

__all__ = [
    "nep_w",
    "d_star_cm_hz_w",
    "netd_photon_count_form_k",
    "netd_datasheet_form_k",
    "datasheet_over_photon_count_ratio",
    "paraxial_aperture_term",
    "enbw_first_order_hz",
    "enbw_sampled_iir_hz",
    "johnson_noise_v",
    "temperature_fluctuation_nep_w",
]

M2_PER_CM2 = 1e-4


def nep_w(noise_voltage_v: float, responsivity_v_per_w: float) -> float:
    """NEP = V_n / |R|: the power giving SNR = 1 (§9.3)."""
    if responsivity_v_per_w == 0.0:
        raise ValueError("responsivity must be non-zero")
    return abs(noise_voltage_v / responsivity_v_per_w)


def d_star_cm_hz_w(active_area_m2: float, bandwidth_hz: float, nep_w_: float) -> float:
    """D* = √(A_d[cm²] · Δf) / NEP in cm·Hz^½·W⁻¹ (§9.3), with the explicit m² → cm² conversion."""
    if active_area_m2 <= 0.0 or bandwidth_hz <= 0.0 or nep_w_ <= 0.0:
        raise ValueError("area, bandwidth and NEP must be positive")
    return math.sqrt(active_area_m2 / M2_PER_CM2 * bandwidth_hz) / nep_w_


def netd_photon_count_form_k(sigma_signal: float, d_signal_dt: float) -> float:
    """NETD = σ / (∂S/∂T) in the signal's own unit (§9.4 photon-count form; the chain's form)."""
    if d_signal_dt <= 0.0:
        raise ValueError("signal derivative must be positive")
    return sigma_signal / d_signal_dt


def paraxial_aperture_term(f_number: float) -> float:
    """4F² expressed through the single aperture-factor definition: π/Ω_eff − 1
    (non-negotiable #5)."""
    return math.pi / aperture_factor(f_number) - 1.0


def netd_datasheet_form_k(
    f_number: float, nep_w_: float, active_area_m2: float, m_prime_w_m2_k: float
) -> float:
    """The paraxial datasheet form 4F² NEP / (A_d M') -- **conversion only**, never used by the
    noise chain (§9.4, §2 convention warning).

    M' is the §9.4 "change in power per unit area radiated by the object within the band", i.e.
    the in-band **exitance** derivative ∂M/∂T = π ∂L_B/∂T (W m⁻² K⁻¹) for a Lambertian source --
    not the radiance derivative. With that reading the form differs from the +1 photon-count form
    by exactly (4F² + 1)/(4F²).
    """
    if active_area_m2 <= 0.0 or m_prime_w_m2_k <= 0.0:
        raise ValueError("area and M' must be positive")
    return paraxial_aperture_term(f_number) * nep_w_ / (active_area_m2 * m_prime_w_m2_k)


def datasheet_over_photon_count_ratio(f_number: float) -> float:
    """4F²/(4F² + 1): the paraxial datasheet form *understates* NETD by this factor (0.8 at
    F/1.0), because π/(4F²) overstates the irradiance. A datasheet NETD quoted in the paraxial
    convention must be multiplied by (4F² + 1)/(4F²) = 1.25 at F/1 to compare with the chain."""
    four_f2 = paraxial_aperture_term(f_number)
    return four_f2 / (four_f2 + 1.0)


def enbw_first_order_hz(tau_s: float) -> float:
    """Equivalent noise bandwidth of a first-order low-pass with time constant τ: 1/(4τ)."""
    if tau_s <= 0.0:
        raise ValueError("tau must be positive")
    return 1.0 / (4.0 * tau_s)


def enbw_sampled_iir_hz(tau_s: float, dt_s: float) -> float:
    """ENBW of the per-frame IIR S_n = S_{n−1} + (x_n − S_{n−1})·a, a = 1 − e^{−Δt/τ}, fed with
    white noise sampled at 1/Δt: (1/2Δt) · a/(2 − a). Tends to 1/(4τ) as Δt → 0."""
    if tau_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("tau and dt must be positive")
    a = 1.0 - math.exp(-dt_s / tau_s)
    return 0.5 / dt_s * a / (2.0 - a)


def johnson_noise_v(temperature_k: float, resistance_ohm: float, bandwidth_hz: float) -> float:
    """Johnson noise voltage √(4 k_B T R Δf) (§10.1)."""
    return math.sqrt(4.0 * K_BOLTZMANN * temperature_k * resistance_ohm * bandwidth_hz)


def temperature_fluctuation_nep_w(
    temperature_k: float, g_th_w_per_k: float, bandwidth_hz: float
) -> float:
    """Temperature-fluctuation noise as an NEP: √(4 k_B T² G_th Δf) (§10.1, the physical floor)."""
    return math.sqrt(4.0 * K_BOLTZMANN * temperature_k**2 * g_th_w_per_k * bandwidth_hz)
