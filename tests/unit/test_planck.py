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
    d_spectral_photon_radiance_dT,
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


def _simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson on a uniform, odd-length grid (kept local: no scipy in the oracle)."""
    assert len(x) % 2 == 1
    h = x[1] - x[0]
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


@pytest.mark.parametrize("T", [250.0, 300.0, 800.0])
def test_integral_equals_stefan_boltzmann(T: float) -> None:
    """∫L dλ == σT⁴/π to 1e-6 (§15 Tier 1). Catches a wrong C1L, a dropped π, a wrong σ.

    Two independent routes: Simpson quadrature of spectral_radiance on 0.1-200 µm, plus the
    closed-form tail (σT⁴/π)(1 − F(200 µm)) from fractional_exitance for the part of the
    spectrum the grid does not cover. Neither the quadrature nor the series is trusted alone;
    the 1e-6 agreement is the evidence (ADR 0005).
    """
    lam = np.linspace(0.1, 200.0, 19_991)  # odd length, 0.01 µm spacing
    quad = _simpson(spectral_radiance(lam, T), lam)
    total = SIGMA_SB * T**4 / np.pi
    tail = total * (1.0 - fractional_exitance(200.0, T))
    rel = (quad + tail) / total - 1.0
    assert abs(rel) < 1e-6, f"T={T}: relative error {rel:.2e} (tail fraction {tail / total:.2e})"


def test_fractional_exitance_converges_to_unity() -> None:
    """F → 1 as λT → ∞, from below, with the residual matching the Rayleigh-Jeans limit.

    F(1000 µm, 6000 K) has x = 2.4e-3 and 1 − F ≈ (15/π⁴)·x³/3 ≈ 7e-10: a truncated series
    or a wrong 15/π⁴ prefactor puts the residual far outside [0.5, 1.5] × that value.
    """
    from irsim.radiometry.constants import C2

    lam, T = 1000.0, 6000.0
    x = C2 / (lam * T)
    residual = 1.0 - fractional_exitance(lam, T)
    expected = 15.0 / np.pi**4 * (x**3 / 3.0 - x**4 / 8.0)
    assert 0.0 < residual < 1e-9
    assert abs(residual / expected - 1.0) < 1e-6


def test_fractional_exitance_branches_agree_at_crossover() -> None:
    """The small-x Taylor branch and the exponential-series branch are independent expansions
    of the same integral; they must agree where the implementation switches (x = 0.5)."""
    from irsim.radiometry.constants import C2
    from irsim.radiometry.planck import _SMALL_X, _planck_integral_above

    for x in (_SMALL_X * (1 - 1e-9), _SMALL_X, 0.3, 0.45):
        # evaluate the exponential series directly at small x, bypassing the switch
        series = sum(
            np.exp(-n * x) / n * (x**3 + 3 * x**2 / n + 6 * x / n**2 + 6 / n**3)
            for n in range(1, 200_000)
        )
        taylor = _planck_integral_above(x, 1e-15, 5000)
        assert abs(taylor / series - 1.0) < 1e-13, f"x={x}: {taylor} vs {series}"
    # and the function itself stays monotone through a λ sweep that crosses x = 0.5
    T = 300.0
    lams = np.linspace(C2 / (0.6 * T), C2 / (0.4 * T), 201)
    f = np.array([fractional_exitance(float(lam_), T) for lam_ in lams])
    assert np.all(np.diff(f) > 0)


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


def test_photon_derivative_times_photon_energy_is_energy_derivative() -> None:
    """dL_q/dT · hc/λ == dL/dT to 1e-12 (§3.4). Catches a wrong C1Q or a derivative factor
    that differs between the two forms."""
    lam = np.linspace(0.4, 20.0, 600)
    for T in (200.0, 300.0, 500.0, 1000.0, 2000.0):
        energy_per_photon_j = H_PLANCK * C_LIGHT / (lam * 1e-6)
        np.testing.assert_allclose(
            d_spectral_photon_radiance_dT(lam, T) * energy_per_photon_j,
            d_spectral_radiance_dT(lam, T),
            rtol=1e-12,
        )


def test_photon_derivative_matches_finite_difference() -> None:
    lam = np.linspace(0.9, 14.0, 300)
    dT = 1e-3
    for T in (250.0, 300.0, 800.0):
        numeric = (
            spectral_photon_radiance(lam, T + dT) - spectral_photon_radiance(lam, T - dT)
        ) / (2 * dT)
        np.testing.assert_allclose(d_spectral_photon_radiance_dT(lam, T), numeric, rtol=1e-6)


def test_photon_derivative_finite_and_positive_over_sweep() -> None:
    """200-2000 K × 0.4-20 µm: the cold/short corner overflows a naive exp()."""
    lam = np.linspace(0.4, 20.0, 400)
    for T in np.linspace(200.0, 2000.0, 19):
        out = d_spectral_photon_radiance_dT(lam, T)
        assert np.all(np.isfinite(out)) and np.all(out >= 0.0), f"T={T}"


def test_thermal_derivative_rises_with_temperature() -> None:
    """Why NETD falls as scene temperature rises. If this ever fails, the noise
    model's temperature dependence is built on sand."""
    lam = np.linspace(8.0, 12.0, 100)
    d300 = np.trapezoid(d_spectral_radiance_dT(lam, 300.0), lam)
    d373 = np.trapezoid(d_spectral_radiance_dT(lam, 373.0), lam)
    assert d373 > d300 * 1.5


def test_fractional_exitance_bounds_and_monotonicity() -> None:
    assert fractional_exitance(0.2, 300.0) < 1e-9
    assert fractional_exitance(1000.0, 300.0) == pytest.approx(1.0, abs=1e-5)
    vals = [fractional_exitance(x, 300.0) for x in (2.0, 5.0, 10.0, 20.0, 50.0, 1000.0)]
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
        for fn in (
            spectral_radiance,
            spectral_photon_radiance,
            d_spectral_radiance_dT,
            d_spectral_photon_radiance_dT,
        ):
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


def test_band_helpers_reject_metre_valued_wavelengths() -> None:
    """The closed-form helpers share the boundary validation: 8e-6 'µm' is a metre slip."""
    with pytest.raises(ValueError, match="MICROMETRES"):
        band_radiance_tophat(8e-6, 12e-6, 300.0)
    with pytest.raises(ValueError, match="MICROMETRES"):
        fractional_exitance(12e-6, 300.0)
    with pytest.raises(ValueError, match="KELVIN"):
        band_radiance_tophat(8.0, 12.0, 27.0 - 300.0)
