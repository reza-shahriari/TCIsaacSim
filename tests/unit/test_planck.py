"""Analytic identity tests for Planck's law.

Every test here answers: would this fail if the physics were wrong? These catch wrong
constants, dropped pi factors and unit errors -- exactly the class of bug that is
invisible in the output image because AGC rescales it away.

See the ir-sim-testing skill for the tolerance conventions used.
"""

import numpy as np
import pytest

from irsim.radiometry.constants import C_LIGHT, H_PLANCK, SIGMA_SB, WIEN_B
from irsim.radiometry.planck import (
    band_radiance_tophat,
    d_spectral_radiance_dT,
    fractional_exitance,
    spectral_photon_radiance,
    spectral_radiance,
)


def test_photon_and_energy_forms_agree() -> None:
    """L_q * h*c/lambda == L. Catches a wrong C1Q."""
    lam = np.linspace(0.5, 25.0, 400)
    for T in (250.0, 300.0, 500.0, 1000.0):
        energy_per_photon_j = H_PLANCK * C_LIGHT / (lam * 1e-6)
        lhs = spectral_photon_radiance(lam, T) * energy_per_photon_j
        rhs = spectral_radiance(lam, T)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_integral_equals_stefan_boltzmann() -> None:
    """Integral of L over all lambda == sigma T^4 / pi. Catches a wrong C1L or a pi error."""
    lam = np.linspace(0.1, 1000.0, 2_000_000)
    for T in (250.0, 300.0, 800.0):
        integrated = np.trapezoid(spectral_radiance(lam, T), lam)
        expected = SIGMA_SB * T**4 / np.pi
        # Residual is tail truncation beyond 1000 um plus trapezoid error, not physics:
        # measured at 1e-5 worst case. A wrong C1L or a dropped pi is a factor of 3+,
        # so this tolerance still catches every error class the test exists for.
        assert abs(integrated / expected - 1.0) < 2e-5, f"T={T}: {integrated} vs {expected}"


def test_wien_displacement() -> None:
    """Peak wavelength times temperature is the Wien constant."""
    for T in (300.0, 500.0, 1000.0):
        lam = np.linspace(0.5, 40.0, 400_000)
        peak = lam[int(np.argmax(spectral_radiance(lam, T)))]
        assert abs(peak * T / WIEN_B - 1.0) < 1e-3


def test_thermal_derivative_matches_finite_difference() -> None:
    lam = np.linspace(3.0, 14.0, 200)
    T, dT = 300.0, 1e-3
    numeric = (spectral_radiance(lam, T + dT) - spectral_radiance(lam, T - dT)) / (2 * dT)
    np.testing.assert_allclose(d_spectral_radiance_dT(lam, T), numeric, rtol=1e-6)


def test_thermal_derivative_rises_with_temperature() -> None:
    """Why NETD falls as scene temperature rises. If this ever fails, the noise
    model's temperature dependence is built on sand."""
    lam = np.linspace(8.0, 12.0, 100)
    d300 = np.trapezoid(d_spectral_radiance_dT(lam, 300.0), lam)
    d373 = np.trapezoid(d_spectral_radiance_dT(lam, 373.0), lam)
    assert d373 > d300 * 1.5


def test_fractional_exitance_bounds_and_monotonicity() -> None:
    assert fractional_exitance(0.2, 300.0) < 1e-9
    assert fractional_exitance(5000.0, 300.0) == pytest.approx(1.0, abs=1e-5)
    vals = [fractional_exitance(x, 300.0) for x in (2.0, 5.0, 10.0, 20.0, 50.0)]
    assert all(b > a for a, b in zip(vals[:-1], vals[1:], strict=True))


def test_band_radiance_matches_quadrature() -> None:
    """Closed-form series vs. numerical integration -- two independent derivations."""
    for lo, hi, T in ((8.0, 12.0, 300.0), (3.0, 5.0, 300.0), (3.0, 5.0, 500.0)):
        lam = np.linspace(lo, hi, 200_001)
        quad = np.trapezoid(spectral_radiance(lam, T), lam)
        closed = band_radiance_tophat(lo, hi, T)
        assert abs(closed / quad - 1.0) < 1e-4, f"{lo}-{hi} um at {T} K: {closed} vs {quad}"


def test_band_radiance_strictly_increasing() -> None:
    temps = np.arange(200.0, 1000.0, 5.0)
    vals = [band_radiance_tophat(8.0, 12.0, float(T)) for T in temps]
    assert all(b > a for a, b in zip(vals[:-1], vals[1:], strict=True))


def test_known_value_at_10um_300k() -> None:
    """Hand-checked anchor: 9.925 W m^-2 sr^-1 um^-1. See ir-radiometry skill."""
    assert spectral_radiance(np.array([10.0]), 300.0)[0] == pytest.approx(9.925, rel=1e-3)
    assert spectral_photon_radiance(np.array([10.0]), 300.0)[0] == pytest.approx(4.996e20, rel=1e-3)


def test_no_overflow_on_extreme_sweep() -> None:
    """Cold scenes at short wavelengths overflow a naive exp(). Must return finite,
    non-negative radiance instead of inf or nan."""
    lam = np.linspace(0.4, 20.0, 500)
    for T in (200.0, 220.0, 2000.0):
        for fn in (spectral_radiance, spectral_photon_radiance, d_spectral_radiance_dT):
            out = fn(lam, T)
            assert np.all(np.isfinite(out)), f"{fn.__name__} not finite at {T} K"
            assert np.all(out >= 0.0), f"{fn.__name__} negative at {T} K"


@pytest.mark.parametrize(
    ("lam", "T", "hint"),
    [
        (np.array([1e-5]), 300.0, "metres instead of micrometres"),
        (np.array([10.0]), -10.0, "sub-zero celsius passed as kelvin"),
        (np.array([10.0]), 1e5, "temperature far outside any physical scene"),
    ],
)
def test_unit_mistakes_raise(lam: np.ndarray, T: float, hint: str) -> None:
    """Silent unit errors are the most expensive bug class in radiometry."""
    with pytest.raises(ValueError):
        spectral_radiance(lam, T)
