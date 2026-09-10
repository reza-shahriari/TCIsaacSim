"""Blackbody radiance: Planck's law in energy and photon form, plus band integration helpers.

This is the reference implementation. It is written for clarity and correctness, not speed --
fast paths (LUTs, GPU kernels) are validated against it, never the other way round.

docs/physics-model.md §3.1, §3.2, §3.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import (
    C1L,
    C1Q,
    C2,
    EXP_ARG_MAX,
    SIGMA_SB,
    TEMPERATURE_MAX_K,
    TEMPERATURE_MIN_K,
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "spectral_radiance",
    "spectral_photon_radiance",
    "d_spectral_radiance_dT",
    "fractional_exitance",
    "band_radiance_tophat",
]


def _validate(wavelength_um: FloatArray, temperature_k: FloatArray) -> None:
    """Fail loudly on unit mistakes rather than returning plausible nonsense.

    Passing metres instead of micrometres is the single most common error at this
    boundary and it produces a finite, wrong answer that survives all the way to
    the output image.
    """
    if np.any(wavelength_um <= 0):
        raise ValueError("wavelength must be positive")
    if np.any(wavelength_um < WAVELENGTH_MIN_UM) or np.any(wavelength_um > WAVELENGTH_MAX_UM):
        raise ValueError(
            f"wavelength outside [{WAVELENGTH_MIN_UM}, {WAVELENGTH_MAX_UM}] um -- "
            "this function takes MICROMETRES, not metres or nanometres"
        )
    if np.any(temperature_k < TEMPERATURE_MIN_K) or np.any(temperature_k > TEMPERATURE_MAX_K):
        raise ValueError(
            f"temperature outside [{TEMPERATURE_MIN_K}, {TEMPERATURE_MAX_K}] K -- "
            "this function takes KELVIN, not celsius"
        )


def _planck_denominator(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
    """exp(C2 / (lam * T)) - 1, computed without overflow and accurately for small x.

    expm1 matters at the hot end (large lam*T gives small x, where exp(x)-1 loses
    precision); the clip matters at the cold end, where exp overflows to inf.
    """
    x = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    return np.expm1(x)


def spectral_radiance(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
    """Blackbody spectral radiance in W m^-2 sr^-1 um^-1.

    Use this form for thermal detectors (microbolometers), which absorb power.

    Args:
        wavelength_um: wavelength(s) in micrometres.
        temperature_k: temperature(s) in kelvin. Broadcast against wavelength_um.

    docs/physics-model.md §3.1
    """
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    _validate(wavelength_um, temperature_k)
    return C1L / (wavelength_um**5 * _planck_denominator(wavelength_um, temperature_k))


def spectral_photon_radiance(
    wavelength_um: FloatArray, temperature_k: FloatArray
) -> FloatArray:
    """Blackbody spectral photon radiance in photons s^-1 m^-2 sr^-1 um^-1.

    Use this form for photon detectors (MWIR/SWIR/NIR), where quantum efficiency is
    a per-photon quantity. Using the energy form with a QE produces an error that
    varies across the band, so it does not show up as a simple gain offset.

    docs/physics-model.md §3.1
    """
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    _validate(wavelength_um, temperature_k)
    return C1Q / (wavelength_um**4 * _planck_denominator(wavelength_um, temperature_k))


def d_spectral_radiance_dT(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
    """dL/dT in W m^-2 sr^-1 um^-1 K^-1.

    This is what NETD divides by, and why NETD falls as scene temperature rises.
    Any code treating NETD as a scene-independent constant in kelvin is wrong.

    docs/physics-model.md §3.4
    """
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    _validate(wavelength_um, temperature_k)

    x = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    radiance = C1L / (wavelength_um**5 * np.expm1(x))
    # dL/dT = L * (C2 / (lam T^2)) * exp(x) / (exp(x) - 1)
    # written as exp(x)/expm1(x) = 1 + 1/expm1(x) to stay accurate for small x
    return radiance * (C2 / (wavelength_um * temperature_k**2)) * (1.0 + 1.0 / np.expm1(x))


def fractional_exitance(
    wavelength_um: float,
    temperature_k: float,
    rtol: float = 1e-12,
    max_terms: int = 2000,
) -> float:
    """Fraction of total blackbody exitance emitted below `wavelength_um`.

    Closed-form series, summed adaptively until terms stop contributing.

    The term count needed depends strongly on x = C2 / (lam * T): for x > 2 (every
    realistic scene temperature in LWIR and MWIR) 3-8 terms reach double precision,
    but the far Rayleigh-Jeans tail (x << 1) converges slowly. A fixed term count
    silently saturates below 1.0 there, so the loop runs to a convergence criterion
    instead. This matters whenever the function is used to check a numerical band
    integral -- a truncated series looks exactly like an integration error.

    Useful as an independent check on numerical band integration: if quadrature and
    this disagree by more than 0.1% on a top-hat response, one of them has a bug.

    docs/physics-model.md §3.2 (a)
    """
    x = C2 / (wavelength_um * temperature_k)
    if x > EXP_ARG_MAX:
        return 0.0

    total = 0.0
    for n in range(1, max_terms + 1):
        nx = n * x
        if nx > EXP_ARG_MAX:
            break
        term = np.exp(-nx) / n * (x**3 + 3.0 * x**2 / n + 6.0 * x / n**2 + 6.0 / n**3)
        total += term
        if total > 0.0 and term < rtol * total:
            break

    return float(15.0 / np.pi**4 * total)


def band_radiance_tophat(
    lambda_min_um: float, lambda_max_um: float, temperature_k: float
) -> float:
    """Band radiance in W m^-2 sr^-1 for a top-hat spectral response.

    Exact for a rectangular response. Real responses need quadrature against R(lambda);
    this is the cross-check and the fast approximation for cores whose response is
    close to rectangular.

    docs/physics-model.md §3.2 (a)
    """
    if lambda_max_um <= lambda_min_um:
        raise ValueError("lambda_max_um must exceed lambda_min_um")
    f_hi = fractional_exitance(lambda_max_um, temperature_k)
    f_lo = fractional_exitance(lambda_min_um, temperature_k)
    return SIGMA_SB * temperature_k**4 / np.pi * (f_hi - f_lo)
