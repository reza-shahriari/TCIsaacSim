"""Convection coefficient (M6.3): the §6.2 known answers, the free/forced crossover, scalar
speed composition, and the parked-vs-highway ratio that makes vehicle speed first-order."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.convection import (
    DEFAULT_CONVECTION,
    ConvectionParams,
    convection_coefficient,
    free_forced_crossover_k,
    relative_air_speed,
)


def test_known_answers() -> None:
    assert float(convection_coefficient(0.0, 0.0)) == 5.0
    h28 = float(convection_coefficient(0.0, 28.0))
    assert h28 == pytest.approx(5.0 + 4.0 * 28.0**0.8, rel=1e-9)
    assert h28 == pytest.approx(62.51, abs=0.02)
    assert h28 / float(convection_coefficient(0.0, 0.0)) > 10.0


def test_free_forced_crossover() -> None:
    x = free_forced_crossover_k()
    assert x == pytest.approx((5.0 / 1.5) ** 3, rel=1e-12) and x == pytest.approx(37.0, abs=0.1)
    assert float(convection_coefficient(x * 0.99, 0.0)) == 5.0, "forced wins below the crossover"
    assert float(convection_coefficient(x * 1.01, 0.0)) > 5.0, "free wins above it"
    assert float(convection_coefficient(-x * 1.5, 0.0)) == float(
        convection_coefficient(x * 1.5, 0.0)
    )
    # free convection never matters in a real wind: 100 K hot surface at 5 m/s is forced
    assert float(convection_coefficient(100.0, 5.0)) == pytest.approx(5.0 + 4.0 * 5.0**0.8)


def test_scalar_speed_composition_and_broadcasting() -> None:
    assert float(convection_coefficient(3.0, 2.0, 28.0)) == float(
        convection_coefficient(3.0, 30.0, 0.0)
    )
    assert float(relative_air_speed(-2.0, -28.0)) == 30.0
    dt = np.array([0.0, 10.0, 60.0])
    wind = np.array([0.0, 1.0, 3.0])
    h = convection_coefficient(dt, wind, 0.0)
    expect = [float(convection_coefficient(a, b)) for a, b in zip(dt, wind, strict=True)]
    np.testing.assert_allclose(h, expect, rtol=1e-15)
    assert h.dtype == np.float64


def test_parameter_guards() -> None:
    with pytest.raises(ValueError):
        ConvectionParams(a=0.0)
    from irsim.thermal.convection import forced_convection

    with pytest.raises(ValueError, match="non-negative"):
        forced_convection(-1.0, DEFAULT_CONVECTION)
    p = ConvectionParams(a=5.7, b=3.8, n=1.0, c=1.5)  # McAdams-type linear form
    assert float(convection_coefficient(0.0, 10.0, params=p)) == pytest.approx(43.7)
