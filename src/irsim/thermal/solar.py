"""Solar geometry (NOAA algorithm) and per-facet solar loading (§6.1, §5.4).

Sun position from the NOAA Solar Calculator equations (Meeus-derived; better than 0.01° over
1950–2050, geometric -- atmospheric refraction is *not* applied: it is < 0.6° and only at the
horizon, where the direct irradiance is already ~0). No new dependency (ADR 0034). Times are
timezone-aware; the core works on Julian day arrays so the thermal tick can vectorise.

Solar loading on a facet with unit normal n, sun unit vector s (both in the local ENU frame:
x east, y north, z up), shadow flag S ∈ {0, 1} (an input -- from the renderer or a script) and
sky-view factor V_s:

    Q_sol = S · max(0, n·s) · DNI  +  V_s · DHI          absorbed = α_sol · Q_sol

DNI/DHI come from the WeatherSeries (surface values: the file's pyranometer already saw the
solar path), so nothing here attenuates them again. The reflected term of §5.4 later reuses
``sun_direction`` for its geometry.

docs/physics-model.md §6.1 (energy balance, [R15] shadow flag S and cosine ψ), §5.4
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SunPosition",
    "julian_day",
    "sun_position",
    "sun_position_utc",
    "sun_direction",
    "solar_noon_utc",
    "solar_loading",
    "absorbed_solar",
]

J2000 = 2451545.0
UNIX_EPOCH_JD = 2440587.5


def julian_day(when: datetime | Any) -> NDArray[np.float64]:
    """Julian day of an aware datetime (or an array of POSIX seconds)."""
    if isinstance(when, datetime):
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        seconds = np.asarray(when.timestamp(), dtype=np.float64)
    else:
        seconds = np.asarray(when, dtype=np.float64)
    return np.asarray(seconds / 86400.0 + UNIX_EPOCH_JD, dtype=np.float64)


@dataclass(frozen=True)
class SunPosition:
    """Degrees. Elevation is geometric (no refraction); azimuth clockwise from north."""

    elevation_deg: NDArray[np.float64]
    azimuth_deg: NDArray[np.float64]
    declination_deg: NDArray[np.float64]
    equation_of_time_min: NDArray[np.float64]
    hour_angle_deg: NDArray[np.float64]


def _noaa_core(jd: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    """Declination (deg), equation of time (min) and geometric mean longitude for Julian days."""
    t = (jd - J2000) / 36525.0
    l0 = np.mod(280.46646 + t * (36000.76983 + 0.0003032 * t), 360.0)
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_rad = np.deg2rad(m)
    c = (
        np.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + np.sin(3 * m_rad) * 0.000289
    )
    true_long = l0 + c
    omega = np.deg2rad(125.04 - 1934.136 * t)
    app_long = true_long - 0.00569 - 0.00478 * np.sin(omega)
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - 0.001813 * t))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * np.cos(omega)
    decl = np.rad2deg(np.arcsin(np.sin(np.deg2rad(eps)) * np.sin(np.deg2rad(app_long))))
    y = np.tan(np.deg2rad(eps) / 2.0) ** 2
    l0_rad = np.deg2rad(l0)
    eot = 4.0 * np.rad2deg(
        y * np.sin(2 * l0_rad)
        - 2 * ecc * np.sin(m_rad)
        + 4 * ecc * y * np.sin(m_rad) * np.cos(2 * l0_rad)
        - 0.5 * y * y * np.sin(4 * l0_rad)
        - 1.25 * ecc * ecc * np.sin(2 * m_rad)
    )
    return decl, eot, l0


def sun_position(lat_deg: float, lon_deg: float, jd: Any) -> SunPosition:
    """NOAA sun position for Julian day(s); longitude east-positive."""
    if not -90.0 <= lat_deg <= 90.0 or not -180.0 <= lon_deg <= 180.0:
        raise ValueError("latitude in [-90, 90], longitude in [-180, 180] (east positive)")
    jd_arr = np.asarray(jd, dtype=np.float64)
    decl, eot, _ = _noaa_core(jd_arr)
    minutes_utc = np.mod(jd_arr - 0.5, 1.0) * 1440.0
    tst = np.mod(minutes_utc + eot + 4.0 * lon_deg, 1440.0)
    ha = np.where(tst / 4.0 < 0.0, tst / 4.0 + 180.0, tst / 4.0 - 180.0)
    lat = np.deg2rad(lat_deg)
    d = np.deg2rad(decl)
    h = np.deg2rad(ha)
    cos_zen = np.sin(lat) * np.sin(d) + np.cos(lat) * np.cos(d) * np.cos(h)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zen = np.arccos(cos_zen)
    sin_zen = np.sin(zen)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = (np.sin(lat) * cos_zen - np.sin(d)) / (np.cos(lat) * sin_zen)
    cos_az = np.clip(np.nan_to_num(cos_az, nan=1.0), -1.0, 1.0)
    az_base = np.rad2deg(np.arccos(cos_az))
    az = np.where(ha > 0.0, np.mod(az_base + 180.0, 360.0), np.mod(540.0 - az_base, 360.0))
    return SunPosition(
        elevation_deg=np.asarray(90.0 - np.rad2deg(zen), dtype=np.float64),
        azimuth_deg=np.asarray(az, dtype=np.float64),
        declination_deg=np.asarray(decl, dtype=np.float64),
        equation_of_time_min=np.asarray(eot, dtype=np.float64),
        hour_angle_deg=np.asarray(ha, dtype=np.float64),
    )


def sun_position_utc(lat_deg: float, lon_deg: float, when: datetime) -> SunPosition:
    return sun_position(lat_deg, lon_deg, julian_day(when))


def solar_noon_utc(lat_deg: float, lon_deg: float, day_utc: datetime) -> datetime:
    """The UTC instant of solar noon on the given (aware) day: 720 − 4·lon − EoT minutes."""
    del lat_deg
    midnight = day_utc.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    _, eot, _ = _noaa_core(julian_day(midnight + timedelta(hours=12)))
    minutes = 720.0 - 4.0 * lon_deg - float(eot)
    return midnight + timedelta(minutes=minutes)


def sun_direction(elevation_deg: Any, azimuth_deg: Any) -> NDArray[np.float64]:
    """Unit vector toward the sun in ENU (east, north, up); shape (..., 3)."""
    el = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    az = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def solar_loading(
    normal: Any,
    sun: Any,
    dni_w_m2: Any,
    dhi_w_m2: Any,
    sky_view_factor: Any,
    shadow: Any = 1.0,
) -> NDArray[np.float64]:
    """Q_sol = S · max(0, n·s) · DNI + V_s · DHI  (W m⁻²); normals and sun vectors (..., 3)."""
    n = np.asarray(normal, dtype=np.float64)
    s = np.asarray(sun, dtype=np.float64)
    cos_inc = np.maximum(0.0, np.sum(n * s, axis=-1))
    shade = np.asarray(shadow, dtype=np.float64)
    if np.any((shade < 0.0) | (shade > 1.0)):
        raise ValueError("shadow must lie in [0, 1] (1 = lit)")
    v_s = np.asarray(sky_view_factor, dtype=np.float64)
    if np.any((v_s < 0.0) | (v_s > 1.0)):
        raise ValueError("sky_view_factor must lie in [0, 1]")
    dni = np.asarray(dni_w_m2, dtype=np.float64)
    dhi = np.asarray(dhi_w_m2, dtype=np.float64)
    if np.any(dni < 0.0) or np.any(dhi < 0.0):
        raise ValueError("irradiance must be non-negative")
    return np.asarray(shade * cos_inc * dni + v_s * dhi, dtype=np.float64)


def absorbed_solar(alpha_sol: Any, q_sol_w_m2: Any) -> NDArray[np.float64]:
    """α_sol · Q_sol -- written in this form, never as a complement (§6.1, spec issue S2)."""
    a = np.asarray(alpha_sol, dtype=np.float64)
    if np.any((a < 0.0) | (a > 1.0)):
        raise ValueError("solar absorptivity must lie in [0, 1]")
    return np.asarray(a * np.asarray(q_sol_w_m2, dtype=np.float64), dtype=np.float64)
