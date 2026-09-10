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
    SIGMA_Q,
    SIGMA_SB,
    TEMPERATURE_MAX_K,
    TEMPERATURE_MIN_K,
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
    ZETA_3,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "spectral_radiance",
    "spectral_photon_radiance",
    "d_spectral_radiance_dT",
    "d_spectral_photon_radiance_dT",
    "fractional_exitance",
    "fractional_photon_exitance",
    "band_radiance_tophat",
    "band_photon_radiance_tophat",
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
    x: FloatArray = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    return np.asarray(np.expm1(x), dtype=np.float64)


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


def spectral_photon_radiance(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
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


def _thermal_derivative_factor(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
    """d ln(1/(e^x − 1)) / dT with x = C2/(λT): the factor shared by both derivative forms.

    (C2 / (λ T²)) · e^x / (e^x − 1), with e^x/(e^x − 1) written as 1 + 1/expm1(x) so it stays
    accurate for small x (the hot / long-wavelength end) and finite under the overflow clip.
    """
    x: FloatArray = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    factor = (C2 / (wavelength_um * temperature_k**2)) * (1.0 + 1.0 / np.expm1(x))
    return np.asarray(factor, dtype=np.float64)


def d_spectral_radiance_dT(wavelength_um: FloatArray, temperature_k: FloatArray) -> FloatArray:
    """dL/dT in W m^-2 sr^-1 um^-1 K^-1 (energy form, for bolometers).

    This is what NETD divides by, and why NETD falls as scene temperature rises.
    Any code treating NETD as a scene-independent constant in kelvin is wrong.

    docs/physics-model.md §3.4
    """
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    _validate(wavelength_um, temperature_k)
    x: FloatArray = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    radiance = C1L / (wavelength_um**5 * np.expm1(x))
    return np.asarray(
        radiance * _thermal_derivative_factor(wavelength_um, temperature_k), dtype=np.float64
    )


def d_spectral_photon_radiance_dT(
    wavelength_um: FloatArray, temperature_k: FloatArray
) -> FloatArray:
    """dL_q/dT in photons s^-1 m^-2 sr^-1 um^-1 K^-1 (photon form, for photon detectors).

    Photon-detector NETD divides by the *band-integrated photon* derivative (§9.4), not the
    energy one: the two differ by the spectrally varying hc/λ, so using the energy form with a
    QE mis-weights the band. Identity: dL_q/dT · hc/λ == dL/dT, tested to 1e-12.

    docs/physics-model.md §3.4, §9.4
    """
    wavelength_um = np.asarray(wavelength_um, dtype=np.float64)
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    _validate(wavelength_um, temperature_k)
    x: FloatArray = np.clip(C2 / (wavelength_um * temperature_k), None, EXP_ARG_MAX)
    photon_radiance = C1Q / (wavelength_um**4 * np.expm1(x))
    return np.asarray(
        photon_radiance * _thermal_derivative_factor(wavelength_um, temperature_k), dtype=np.float64
    )


# Bernoulli numbers B_2 .. B_14 for the small-x expansion of the Planck integral.
_BERNOULLI_EVEN = (1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730, 7.0 / 6)
_SMALL_X = 0.5


def _planck_integral_above(x: float, rtol: float, max_terms: int) -> float:
    """∫_x^∞ t³ / (e^t − 1) dt: the dimensionless Planck integral over wavenumbers above x,
    i.e. over wavelengths *below* λ = C2 / (x T). Total over all x is π⁴/15.

    Two independent expansions, each used where it converges fast:

    * ``x >= _SMALL_X``: the exponential series Σ_n e^{-nx}(x³/n + 3x²/n² + 6x/n³ + 6/n⁴).
      For x > 2 (every scene temperature in LWIR/MWIR) 3-8 terms reach double precision.
    * ``x < _SMALL_X``: π⁴/15 minus the Taylor series of the complementary integral ∫_0^x,
      x³/3 − x⁴/8 + Σ_k B_2k x^{2k+3} / ((2k+3)(2k)!). The exponential series cannot reach
      ``rtol`` here in any practical number of terms (its tail falls only as 1/N³, so 2000
      terms leave ~4e-11).

    The crossover is tested for continuity to 1e-13 (tests/unit/test_planck.py).
    """
    if x < _SMALL_X:
        below = x**3 / 3.0 - x**4 / 8.0
        factorial = 1.0
        for k, b2k in enumerate(_BERNOULLI_EVEN, start=1):
            factorial *= (2 * k - 1) * (2 * k)  # (2k)!
            below += b2k * x ** (2 * k + 3) / ((2 * k + 3) * factorial)
        return float(np.pi**4 / 15.0 - below)

    if x > EXP_ARG_MAX:
        return 0.0
    total = 0.0
    # Successive terms shrink at least geometrically by e^{-x}, so the remainder after a term is
    # bounded by term * e^{-x} / (1 - e^{-x}); stop when that bound is below rtol * total.
    # (Stopping on the term alone leaves a 1 / (1 - e^{-x}) ≈ 2.5x larger residual at x = 0.5.)
    tail_factor = np.exp(-x) / (-np.expm1(-x))
    for n in range(1, max_terms + 1):
        nx = n * x
        if nx > EXP_ARG_MAX:
            break
        term = np.exp(-nx) / n * (x**3 + 3.0 * x**2 / n + 6.0 * x / n**2 + 6.0 / n**3)
        total += term
        if total > 0.0 and term * tail_factor < rtol * total:
            break
    return total


def fractional_exitance(
    wavelength_um: float,
    temperature_k: float,
    rtol: float = 1e-12,
    max_terms: int = 2000,
) -> float:
    """Fraction of total blackbody exitance emitted below ``wavelength_um``: F(0 → λT).

    F = (15/π⁴) ∫_x^∞ t³/(e^t − 1) dt with x = C2/(λT) (short wavelengths are large x),
    summed in closed form (see
    :func:`_planck_integral_above` for the two expansions and why both are needed).

    Useful as an independent check on numerical band integration: if quadrature and this
    disagree by more than 0.1% on a top-hat response, one of them has a bug. Note that F
    never reaches exactly 1 at a finite wavelength -- at 1000 µm and 300 K the tail beyond
    still carries 1 − F ≈ 5.6e-6 -- so "F(large λ) == 1" is not a valid test.

    Inputs are validated like every other boundary here: micrometres and kelvin only.

    docs/physics-model.md §3.2 (a)
    """
    _validate(
        np.asarray(wavelength_um, dtype=np.float64), np.asarray(temperature_k, dtype=np.float64)
    )
    x = C2 / (wavelength_um * temperature_k)
    return 15.0 / np.pi**4 * _planck_integral_above(x, rtol, max_terms)


def _photon_integral_above(x: float, rtol: float, max_terms: int) -> float:
    """∫_x^∞ t² / (e^t − 1) dt: the photon-form analogue of :func:`_planck_integral_above`.
    Total over all x is 2ζ(3).

    * ``x >= _SMALL_X``: exponential series Σ_n e^{-nx}(x²/n + 2x/n² + 2/n³), geometric tail bound.
    * ``x < _SMALL_X``: 2ζ(3) minus the Taylor series of ∫_0^x, x²/2 − x³/6 +
      Σ_k B_2k x^{2k+2} / ((2k+2)(2k)!).
    """
    if x < _SMALL_X:
        below = x**2 / 2.0 - x**3 / 6.0
        factorial = 1.0
        for k, b2k in enumerate(_BERNOULLI_EVEN, start=1):
            factorial *= (2 * k - 1) * (2 * k)
            below += b2k * x ** (2 * k + 2) / ((2 * k + 2) * factorial)
        return float(2.0 * ZETA_3 - below)

    if x > EXP_ARG_MAX:
        return 0.0
    total = 0.0
    tail_factor = np.exp(-x) / (-np.expm1(-x))
    for n in range(1, max_terms + 1):
        nx = n * x
        if nx > EXP_ARG_MAX:
            break
        term = np.exp(-nx) / n * (x**2 + 2.0 * x / n + 2.0 / n**2)
        total += term
        if total > 0.0 and term * tail_factor < rtol * total:
            break
    return total


def fractional_photon_exitance(
    wavelength_um: float,
    temperature_k: float,
    rtol: float = 1e-12,
    max_terms: int = 2000,
) -> float:
    """Fraction of total blackbody *photon* exitance emitted below ``wavelength_um``.

    F_q = (1 / 2ζ(3)) ∫_x^∞ t²/(e^t − 1) dt with x = C2/(λT). The photon analogue of
    :func:`fractional_exitance`; the only implementation-independent oracle for the photon
    band-radiance table (``Lb_q``) that photon-detector NETD depends on (§9.4). Like F, it never
    reaches exactly 1 at finite wavelength (1 − F_q(1000 µm, 300 K) ≈ 4.8e-4 -- the photon
    spectrum's Rayleigh-Jeans tail is fatter than the energy spectrum's).

    docs/physics-model.md §3.2 (a), §9.4
    """
    _validate(
        np.asarray(wavelength_um, dtype=np.float64), np.asarray(temperature_k, dtype=np.float64)
    )
    x = C2 / (wavelength_um * temperature_k)
    return _photon_integral_above(x, rtol, max_terms) / (2.0 * ZETA_3)


def band_radiance_tophat(lambda_min_um: float, lambda_max_um: float, temperature_k: float) -> float:
    """Band radiance in W m^-2 sr^-1 for a top-hat spectral response.

    Exact for a rectangular response. Real responses need quadrature against R(lambda);
    this is the cross-check and the fast approximation for cores whose response is
    close to rectangular.

    docs/physics-model.md §3.2 (a)
    """
    _validate(
        np.asarray([lambda_min_um, lambda_max_um], dtype=np.float64),
        np.asarray(temperature_k, dtype=np.float64),
    )
    if lambda_max_um <= lambda_min_um:
        raise ValueError("lambda_max_um must exceed lambda_min_um")
    f_hi = fractional_exitance(lambda_max_um, temperature_k)
    f_lo = fractional_exitance(lambda_min_um, temperature_k)
    return SIGMA_SB * temperature_k**4 / np.pi * (f_hi - f_lo)


def band_photon_radiance_tophat(
    lambda_min_um: float, lambda_max_um: float, temperature_k: float
) -> float:
    """Band photon radiance in photons s^-1 m^-2 sr^-1 for a top-hat spectral response.

    σ_q T³ / π · (F_q(λ_max) − F_q(λ_min)); the photon-form counterpart of
    :func:`band_radiance_tophat`.

    docs/physics-model.md §3.2 (a)
    """
    _validate(
        np.asarray([lambda_min_um, lambda_max_um], dtype=np.float64),
        np.asarray(temperature_k, dtype=np.float64),
    )
    if lambda_max_um <= lambda_min_um:
        raise ValueError("lambda_max_um must exceed lambda_min_um")
    f_hi = fractional_photon_exitance(lambda_max_um, temperature_k)
    f_lo = fractional_photon_exitance(lambda_min_um, temperature_k)
    return SIGMA_Q * temperature_k**3 / np.pi * (f_hi - f_lo)
