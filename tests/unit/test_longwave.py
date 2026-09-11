"""Broadband longwave down (M6.5): the overcast and enclosed limits to 1e-9, the clear dry sky
range, the overcast margin, the "never below the LWIR-window proxy" guard, and monotonicity."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pytest

from irsim.atmosphere.humidity import vapour_pressure_hpa
from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal import WeatherSample
from irsim.thermal.longwave import (
    clear_sky_emissivity,
    longwave_down,
    longwave_down_from_sample,
    sky_emissivity,
)

Formula = Literal["brunt", "idso"]
FORMULAS: tuple[Formula, ...] = ("brunt", "idso")


@pytest.mark.parametrize("formula", FORMULAS)
def test_overcast_and_enclosed_limits(formula: Formula) -> None:
    t_air = 288.0
    q = float(longwave_down(t_air, 8.0, 1.0, 1.0, 300.0, formula))
    assert q == pytest.approx(SIGMA_SB * t_air**4, rel=1e-9)
    assert SIGMA_SB * t_air**4 == pytest.approx(390.1, abs=0.1)
    q0 = float(longwave_down(t_air, 8.0, 0.2, 0.0, 300.0, formula))
    assert q0 == pytest.approx(SIGMA_SB * 300.0**4, rel=1e-12)
    half = float(longwave_down(t_air, 8.0, 0.0, 0.5, 300.0, formula))
    clear = float(longwave_down(t_air, 8.0, 0.0, 1.0, 300.0, formula))
    assert half == pytest.approx(0.5 * clear + 0.5 * SIGMA_SB * 300.0**4, rel=1e-12)


def _window_proxy_hemispheric(t_air: float, delta_t: float = 60.0, q: float = 0.75) -> float:
    """Cosine-weighted hemispheric sigma T_sky^4 of the §5.3(a) LWIR-window T_sky(theta)."""
    theta = np.linspace(0.0, np.pi / 2, 20001)
    t_sky = t_air - delta_t * np.cos(theta) ** q
    w = np.sin(theta) * np.cos(theta)
    return float(np.trapezoid(SIGMA_SB * t_sky**4 * w, theta) / np.trapezoid(w, theta))


@pytest.mark.parametrize("formula", FORMULAS)
def test_clear_dry_sky_range_and_overcast_margin(formula: Formula) -> None:
    t_air, e_dry = 288.0, 5.0  # ~45 % RH at 15 C
    clear = float(longwave_down(t_air, e_dry, 0.0, 1.0, t_air, formula))
    assert 230.0 <= clear <= 320.0, clear
    overcast = float(longwave_down(t_air, e_dry, 1.0, 1.0, t_air, formula))
    assert overcast - clear >= 60.0, (clear, overcast)
    # a zenith window temperature used as a broadband sky would be off by 100+ W/m^2
    assert clear - SIGMA_SB * (t_air - 60.0) ** 4 > 100.0
    assert clear >= _window_proxy_hemispheric(t_air), "broadband never below the window proxy"


@pytest.mark.parametrize("formula", FORMULAS)
def test_monotone_in_cloud_and_vapour_and_bounded(formula: Formula) -> None:
    e = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
    eps = clear_sky_emissivity(e, 293.0, formula)
    assert np.all(np.diff(eps) > 0.0) and np.all(eps <= 1.0) and np.all(eps > 0.5)
    clouds = np.linspace(0.0, 1.0, 11)
    q = longwave_down(288.0, 8.0, clouds, 1.0, 288.0, formula)
    assert np.all(np.diff(q) > 0.0) and q[-1] == pytest.approx(SIGMA_SB * 288.0**4, rel=1e-9)
    assert float(sky_emissivity(8.0, 288.0, 1.0, formula)) == 1.0
    with pytest.raises(ValueError):
        longwave_down(288.0, 8.0, 1.2, 1.0, 288.0, formula)
    with pytest.raises(ValueError):
        longwave_down(288.0, -1.0, 0.0, 1.0, 288.0, formula)
    with pytest.raises(ValueError, match="formula"):
        clear_sky_emissivity(8.0, 288.0, cast(Formula, "swinbank"))


def test_sample_feeds_the_same_vapour_pressure_as_the_atmosphere() -> None:
    s = WeatherSample(293.15, 0.6, 1.0, 0.3, 0.0, 0.0, 23000.0, 0.0)
    q = float(longwave_down_from_sample(s, 0.8, 295.0))
    e = vapour_pressure_hpa(293.15, 0.6)
    assert q == pytest.approx(float(longwave_down(293.15, e, 0.3, 0.8, 295.0)), rel=1e-15)
    assert 250.0 < q < 420.0
