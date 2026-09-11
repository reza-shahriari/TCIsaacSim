"""Solar geometry and loading (M6.4): NOAA known answers, declination at the solstice, the
equinox-noon identity, sunrise azimuth at 40 N, a southern-hemisphere noon, and the facet
cosine / shadow identities of the loading term."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from irsim.thermal.solar import (
    absorbed_solar,
    julian_day,
    solar_loading,
    solar_noon_utc,
    sun_direction,
    sun_position,
    sun_position_utc,
)

UTC = timezone.utc


def test_julian_day_reference() -> None:
    assert float(julian_day(datetime(2000, 1, 1, 12, tzinfo=UTC))) == 2451545.0
    with pytest.raises(ValueError, match="aware"):
        julian_day(datetime(2000, 1, 1, 12))


def test_solstice_declination_and_equinox_noon() -> None:
    d = sun_position_utc(0.0, 0.0, datetime(2024, 6, 20, 20, 51, tzinfo=UTC)).declination_deg
    assert float(d) == pytest.approx(23.44, abs=0.1)
    d = sun_position_utc(0.0, 0.0, datetime(2024, 12, 21, 9, 20, tzinfo=UTC)).declination_deg
    assert float(d) == pytest.approx(-23.44, abs=0.1)
    # equinox: noon elevation = 90 - |lat| at every latitude
    for lat in (0.0, 35.0, -45.0, 60.0):
        noon = solar_noon_utc(lat, 0.0, datetime(2024, 3, 20, tzinfo=UTC))
        el = float(sun_position_utc(lat, 0.0, noon).elevation_deg)
        assert el == pytest.approx(90.0 - abs(lat), abs=0.5), lat


def _spencer_oracle(lat_deg: float, lon_deg: float, when: datetime) -> tuple[float, float]:
    """Independent implementation: Spencer (1971) Fourier series for declination and the
    equation of time, then plain spherical trigonometry. Accurate to ~0.3 deg / 0.5 min."""
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    g = 2.0 * np.pi * (doy - 1 + (hour - 12.0) / 24.0) / 365.0
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    tst = (hour * 60.0 + eot + 4.0 * lon_deg) % 1440.0
    ha = np.deg2rad(tst / 4.0 - 180.0)
    lat = np.deg2rad(lat_deg)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    el = np.arcsin(sin_el)
    cos_az = (np.sin(decl) - np.sin(lat) * sin_el) / (np.cos(lat) * np.cos(el))
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    if ha > 0:
        az = 360.0 - az
    return float(np.degrees(el)), float(az)


@pytest.mark.parametrize(
    "lat, lon, when",
    [
        (40.0, -105.0, datetime(2010, 6, 21, 14, 0, tzinfo=UTC)),  # Boulder, morning
        (40.0, -105.0, datetime(2010, 6, 21, 23, 30, tzinfo=UTC)),  # Boulder, evening
        (51.5, -0.13, datetime(2024, 12, 21, 11, 0, tzinfo=UTC)),  # London, winter noon-ish
        (-33.87, 151.21, datetime(2024, 12, 21, 4, 0, tzinfo=UTC)),  # Sydney, afternoon
        (69.65, 18.96, datetime(2024, 6, 21, 22, 45, tzinfo=UTC)),  # Tromsø, midnight sun
        (1.35, 103.82, datetime(2024, 3, 20, 5, 0, tzinfo=UTC)),  # Singapore, equinox noon
    ],
)
def test_noaa_vs_independent_spencer_oracle(lat: float, lon: float, when: datetime) -> None:
    p = sun_position_utc(lat, lon, when)
    el, az = _spencer_oracle(lat, lon, when)
    assert float(p.elevation_deg) == pytest.approx(el, abs=0.5), (float(p.elevation_deg), el)
    # compare directions, not azimuth angles: azimuth is ill-conditioned near the zenith
    # (Singapore at the equinox has the sun 3 deg from overhead)
    ours = sun_direction(p.elevation_deg, p.azimuth_deg)
    theirs = sun_direction(el, az)
    separation = np.degrees(np.arccos(np.clip(np.dot(ours, theirs), -1.0, 1.0)))
    assert float(separation) < 0.5, (float(p.azimuth_deg), az, float(separation))


def test_boulder_solstice_noon_and_sunrise_azimuth() -> None:
    """Boulder, CO (40 N, 105 W) on 2010-06-21: solar noon at 19:01:4x UTC (720 + 420 − EoT
    with EoT = −1.8 min), noon elevation 90 − 40 + 23.44, sunrise azimuth cos⁻¹(sin δ / cos φ)."""
    noon = solar_noon_utc(40.0, -105.0, datetime(2010, 6, 21, tzinfo=UTC))
    assert abs((noon - datetime(2010, 6, 21, 19, 1, 45, tzinfo=UTC)).total_seconds()) < 60.0
    p = sun_position_utc(40.0, -105.0, noon)
    assert float(p.elevation_deg) == pytest.approx(90.0 - 40.0 + 23.44, abs=0.5)
    assert float(p.azimuth_deg) == pytest.approx(180.0, abs=1.0)
    # Tromsø midnight sun: elevation at solar midnight = δ + φ − 90 = +3.1 deg, due north
    p = sun_position_utc(
        69.65,
        18.96,
        solar_noon_utc(69.65, 18.96, datetime(2024, 6, 21, tzinfo=UTC)) + timedelta(hours=12),
    )
    assert float(p.elevation_deg) == pytest.approx(23.44 + 69.65 - 90.0, abs=0.5)
    assert abs(((float(p.azimuth_deg) + 180.0) % 360.0) - 180.0) < 2.0
    seconds = np.array(
        [
            (datetime(2010, 6, 21, 11, 20, tzinfo=UTC) + timedelta(minutes=k)).timestamp()
            for k in range(30)
        ]
    )
    track = sun_position(40.0, -105.0, julian_day(seconds))
    k = int(np.argmin(np.abs(track.elevation_deg)))
    assert abs(float(track.elevation_deg[k])) < 0.3
    expect_az = np.degrees(np.arccos(np.sin(np.deg2rad(23.44)) / np.cos(np.deg2rad(40.0))))
    assert float(track.azimuth_deg[k]) == pytest.approx(expect_az, abs=1.0)


def test_southern_hemisphere_noon_faces_north() -> None:
    noon = solar_noon_utc(-33.87, 151.21, datetime(2024, 12, 21, tzinfo=UTC))
    p = sun_position_utc(-33.87, 151.21, noon)
    assert float(p.elevation_deg) == pytest.approx(90.0 - abs(-33.87 + 23.44), abs=0.5)
    assert abs(((float(p.azimuth_deg) + 180.0) % 360.0) - 180.0) < 2.0, "sun to the north"
    with pytest.raises(ValueError):
        sun_position(95.0, 0.0, 2451545.0)


def test_facet_cosine_and_shadow_identities() -> None:
    s = sun_direction(30.0, 135.0)
    assert np.linalg.norm(s) == pytest.approx(1.0, rel=1e-12) and s[2] == pytest.approx(0.5)
    dni, dhi, vs = 800.0, 100.0, 0.7
    facing = solar_loading(s, s, dni, dhi, vs)
    assert float(facing) == pytest.approx(dni + vs * dhi, rel=1e-12)
    up = np.array([0.0, 0.0, 1.0])
    assert float(solar_loading(up, s, dni, dhi, vs)) == pytest.approx(
        np.sin(np.deg2rad(30)) * dni + vs * dhi, rel=1e-12
    )
    perp = np.array(
        [np.cos(np.deg2rad(135.0)), -np.sin(np.deg2rad(135.0)), 0.0]
    )  # horizontal, 90 deg from az
    assert float(solar_loading(perp, s, dni, dhi, vs)) == pytest.approx(vs * dhi, rel=1e-12)
    assert float(solar_loading(-s, s, dni, dhi, vs)) == pytest.approx(vs * dhi, rel=1e-12), (
        "no negative loading"
    )
    assert float(solar_loading(s, s, dni, dhi, vs, shadow=0.0)) == pytest.approx(
        vs * dhi, rel=1e-12
    )
    assert float(solar_loading(s, s, dni, dhi, 0.0, shadow=0.0)) == 0.0
    # vectorised over facets
    normals = np.stack([s, up, perp, -s])
    q = solar_loading(normals, s, dni, dhi, vs)
    assert q.shape == (4,) and q[0] > q[1] > q[2] == q[3]
    assert float(absorbed_solar(0.9, 500.0)) == 450.0
    with pytest.raises(ValueError):
        absorbed_solar(1.2, 1.0)
    with pytest.raises(ValueError):
        solar_loading(s, s, dni, dhi, vs, shadow=2.0)
