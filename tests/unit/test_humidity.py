"""Magnus saturation pressure, absolute humidity and gamma(w) (M8.2)."""

from __future__ import annotations

import pytest

from irsim.atmosphere import (
    absolute_humidity_g_m3,
    gamma_molecular,
    saturation_vapour_pressure_hpa,
    vapour_pressure_hpa,
)
from irsim.atmosphere.humidity import ABS_HUMIDITY_FACTOR
from irsim.radiometry.constants import R_V_WATER


def test_saturation_pressure_known_values() -> None:
    assert saturation_vapour_pressure_hpa(0.0) == pytest.approx(6.112, rel=1e-12)
    assert saturation_vapour_pressure_hpa(20.0) == pytest.approx(23.39, rel=5e-3)
    assert saturation_vapour_pressure_hpa(30.0) == pytest.approx(42.47, rel=5e-3)


def test_absolute_humidity_known_values() -> None:
    assert pytest.approx(1e5 / R_V_WATER, rel=1e-12) == ABS_HUMIDITY_FACTOR
    assert pytest.approx(216.7, rel=1e-3) == ABS_HUMIDITY_FACTOR
    assert absolute_humidity_g_m3(303.15, 0.80) == pytest.approx(24.3, rel=1e-2)
    assert absolute_humidity_g_m3(293.15, 1.00) == pytest.approx(17.3, rel=1e-2)
    assert absolute_humidity_g_m3(268.15, 0.70) == pytest.approx(2.4, rel=2e-2)


def test_guards_and_monotonicity() -> None:
    with pytest.raises(ValueError, match="fraction"):
        absolute_humidity_g_m3(300.0, 80.0)
    with pytest.raises(ValueError, match="kelvin"):
        absolute_humidity_g_m3(25.0, 0.5)
    ws = [absolute_humidity_g_m3(290.0, rh) for rh in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert all(b > a for a, b in zip(ws[:-1], ws[1:], strict=True))
    ts = [absolute_humidity_g_m3(t, 0.5) for t in (270.0, 280.0, 290.0, 300.0)]
    assert all(b > a for a, b in zip(ts[:-1], ts[1:], strict=True))
    assert vapour_pressure_hpa(293.15, 0.5) == pytest.approx(
        0.5 * saturation_vapour_pressure_hpa(20.0)
    )


def test_gamma_molecular_is_linear_in_w() -> None:
    g0, beta = 1.4e-4, 4.4e-5
    assert gamma_molecular(0.0, g0, beta) == g0
    dw = 1e-3
    fd = (gamma_molecular(10.0 + dw, g0, beta) - gamma_molecular(10.0 - dw, g0, beta)) / (2 * dw)
    assert fd == pytest.approx(beta, rel=1e-9)
    with pytest.raises(ValueError):
        gamma_molecular(-1.0, g0, beta)
