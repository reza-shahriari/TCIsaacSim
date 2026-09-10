"""Aperture factor and focal-plane irradiance (M3.1): identities, an independent cone integral,
the spec's 20 % known answer, and the Boson pixel-power hand value."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from irsim.optics import aperture_factor, cone_half_angle, fpa_irradiance, pixel_power

BOSON_A_D = 1.296e-10  # m^2: (12 um)^2 * 0.90


@pytest.mark.parametrize("f_number", [0.8, 1.0, 1.4, 2.0, 4.0])
def test_aperture_factor_is_pi_sin2_of_cone_half_angle(f_number: float) -> None:
    """π/(4F²+1) == π sin²(atan(1/2F)) to 1e-12: the paraxial π/(4F²) fails this."""
    exact = math.pi * math.sin(cone_half_angle(f_number)) ** 2
    assert abs(aperture_factor(f_number) / exact - 1.0) < 1e-12
    paraxial = math.pi / (4.0 * f_number**2)
    assert paraxial > aperture_factor(f_number)


@pytest.mark.parametrize("f_number", [1.0, 1.4, 2.8])
def test_irradiance_matches_lambertian_cone_integral(f_number: float) -> None:
    """E = ∫₀^θmax 2π L cosθ sinθ dθ (independent numerical derivation) to 1e-10."""
    L = 3.7
    theta_max = cone_half_angle(f_number)
    numeric, _ = quad(
        lambda th: 2 * math.pi * L * math.cos(th) * math.sin(th),
        0.0,
        theta_max,
        epsabs=0,
        epsrel=1e-13,
    )
    assert abs(float(fpa_irradiance(L, f_number, 1.0)) / numeric - 1.0) < 1e-10


def test_known_answers_at_f1() -> None:
    """§2: at F/1.0 the +1 form is 20 % below the paraxial form."""
    assert aperture_factor(1.0) / (math.pi / 4.0) == pytest.approx(0.8, abs=1e-15)
    assert aperture_factor(1.0) == pytest.approx(0.628319, abs=1e-6)


def test_boson_pixel_power_hand_value() -> None:
    """A_d 1.296e-10 m², F/1, τ 0.92, L_B = 1 W m⁻² sr⁻¹ -> 7.492e-11 W (hand computed)."""
    phi = float(pixel_power(1.0, 1.0, 0.92, BOSON_A_D))
    assert phi == pytest.approx(0.92 * math.pi / 5.0 * BOSON_A_D, rel=1e-12)
    assert phi == pytest.approx(7.492e-11, rel=1e-3)


def test_quantity_agnostic_and_linear() -> None:
    lb = np.array([1.0, 2.0, 55.49], dtype=np.float64)
    e = fpa_irradiance(lb, 1.0, 0.92)
    np.testing.assert_allclose(e / lb, 0.92 * math.pi / 5.0, rtol=1e-15)
    lq = lb * 1e20
    np.testing.assert_allclose(fpa_irradiance(lq, 1.0, 0.92) / lq, 0.92 * math.pi / 5.0, rtol=1e-15)


def test_cos4_field_multiplies_in() -> None:
    cos4 = np.array([[1.0, 0.7924]], dtype=np.float32)
    lb = np.full((1, 2), 10.0, dtype=np.float32)
    e = fpa_irradiance(lb, 1.0, 1.0, cos4)
    assert e[0, 1] / e[0, 0] == pytest.approx(0.7924, rel=1e-6)


def test_dtype_guard() -> None:
    with pytest.raises(TypeError, match="float16"):
        fpa_irradiance(np.array([1.0], dtype=np.float16), 1.0, 0.9)
    out = pixel_power(np.array([1.0], dtype=np.float32), 1.0, 0.9, BOSON_A_D)
    assert out.dtype == np.float32
    assert fpa_irradiance(np.array([1.0]), 1.0, 0.9).dtype == np.float64


@pytest.mark.parametrize(
    ("f_number", "tau", "area"), [(0.0, 0.9, 1e-10), (1.0, 1.2, 1e-10), (1.0, 0.9, 0.0)]
)
def test_bad_parameters_raise(f_number: float, tau: float, area: float) -> None:
    with pytest.raises(ValueError):
        pixel_power(1.0, f_number, tau, area)
