"""Weather CSV round trip, unit-header conversion, malformed files, the synthetic clear day's
analytic properties, and the committed sample's provenance (M6.2)."""

from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest

from irsim.thermal import WEATHER_FIELDS
from irsim.thermal.weather_io import (
    WEATHER_CSV_MAGIC,
    load_weather_csv,
    synthetic_clear_day,
    write_weather_csv,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = REPO / "data" / "weather" / "clear_midlat_summer_48h.csv"
START = datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc)


def _generator_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "gen_weather", REPO / "scripts" / "generate_weather.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_write_read_round_trip_is_bit_exact(tmp_path: pathlib.Path) -> None:
    w = synthetic_clear_day(START, hours=30.0, utc_offset_h=2.0, step_s=1800.0)
    w2_path = tmp_path / "w.csv"
    write_weather_csv(w, w2_path, comment="round trip\nsecond line")
    w2 = load_weather_csv(w2_path)
    assert w2.content_hash == w.content_hash and w2 == w
    assert w2.epoch_utc == w.epoch_utc
    for name in ("time_s", *WEATHER_FIELDS):
        np.testing.assert_array_equal(getattr(w2, name), getattr(w, name))
    text = w2_path.read_text()
    assert text.startswith(WEATHER_CSV_MAGIC) and "# round trip\n# second line\n" in text


def test_half_sine_irradiance_integral_and_diurnal_shape() -> None:
    peak, rise, set_ = 850.0, 5.5, 20.5
    w = synthetic_clear_day(
        START,
        hours=24.0,
        dni_peak_w_m2=peak,
        sunrise_hour_local=rise,
        sunset_hour_local=set_,
        step_s=60.0,
    )
    daily_j = float(np.trapezoid(w.dni_w_m2, w.time_s))
    expect = 2.0 * peak * (set_ - rise) * 3600.0 / np.pi
    assert abs(daily_j / expect - 1.0) < 1e-3, daily_j / expect
    assert w.dni_w_m2.max() == pytest.approx(peak, rel=1e-4)
    h = (w.time_s / 3600.0) % 24.0
    assert h[np.argmax(w.t_air_k)] == pytest.approx(15.0, abs=0.02)
    assert h[np.argmin(w.t_air_k)] == pytest.approx(3.0, abs=0.02)
    assert np.all(w.dni_w_m2[(h < rise) | (h > set_)] == 0.0)
    # constant dew point: RH highest when coldest, never above 1
    assert np.corrcoef(w.t_air_k, w.rh_fraction)[0, 1] < -0.99 and w.rh_fraction.max() <= 1.0


def test_celsius_and_percent_headers_convert_exactly_once(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "c.csv"
    p.write_text(
        f"{WEATHER_CSV_MAGIC}\n"
        "time_utc,t_air_c,rh_percent,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
        "2024-06-21T00:00:00Z,20.0,55,1,0,0,0,23000,0\n"
        "2024-06-21T01:00:00+00:00,21.5,60,1,0,0,0,23000,0\n"
        "2024-06-21T04:00:00+02:00,22.0,65,1,0,0,0,inf,0\n"
    )
    w = load_weather_csv(p)
    assert w.t_air_k[0] == 20.0 + 273.15 and w.t_air_k[1] == 21.5 + 273.15
    assert w.rh_fraction[0] == 0.55 and w.rh_fraction[2] == pytest.approx(0.65)
    np.testing.assert_array_equal(w.time_s, [0.0, 3600.0, 7200.0])  # +02:00 handled
    assert np.isinf(w.visibility_m[2])


@pytest.mark.parametrize(
    "rows, msg",
    [
        (
            "time_utc,t_air_k,rh_fraction\n2024-06-21T00:00:00Z,290,0.5\n2024-06-21T01:00:00Z,291,0.5\n",
            "missing",
        ),
        (
            "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h,albedo\n"
            "2024-06-21T00:00:00Z,290,0.5,1,0,0,0,23000,0,0.2\n2024-06-21T01:00:00Z,291,0.5,1,0,0,0,23000,0,0.2\n",
            "unknown column",
        ),
        (
            "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
            "2024-06-21T00:00:00Z,290,0.5,1,0,0,0,23000,0\n2024-06-21T01:00:00Z,warm,0.5,1,0,0,0,23000,0\n",
            "not a number",
        ),
        (
            "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
            "2024-06-21T00:00:00,290,0.5,1,0,0,0,23000,0\n2024-06-21T01:00:00,291,0.5,1,0,0,0,23000,0\n",
            "UTC offset",
        ),
        (
            "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
            "2024-06-21T01:00:00Z,290,0.5,1,0,0,0,23000,0\n2024-06-21T00:00:00Z,291,0.5,1,0,0,0,23000,0\n",
            "increasing",
        ),
        (
            "time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h\n"
            "2024-06-21T00:00:00Z,17,0.5,1,0,0,0,23000,0\n2024-06-21T01:00:00Z,18,0.5,1,0,0,0,23000,0\n",
            "celsius",
        ),
    ],
)
def test_malformed_files_raise(tmp_path: pathlib.Path, rows: str, msg: str) -> None:
    p = tmp_path / "bad.csv"
    p.write_text(f"{WEATHER_CSV_MAGIC}\n{rows}")
    with pytest.raises(ValueError, match=msg):
        load_weather_csv(p)
    p.write_text("time_utc,t_air_k\n")
    with pytest.raises(ValueError, match="first line"):
        load_weather_csv(p)


def test_committed_sample_matches_its_generator_bit_exactly(tmp_path: pathlib.Path) -> None:
    """data/weather/clear_midlat_summer_48h.csv is exactly what scripts/generate_weather.py
    writes today: the file and its documented recipe cannot drift apart."""
    gen = _generator_module()
    fresh = gen.build()
    committed = load_weather_csv(SAMPLE)
    assert committed.content_hash == fresh.content_hash
    out = tmp_path / "regen.csv"
    write_weather_csv(fresh, out, comment=gen.SAMPLE_COMMENT)
    assert out.read_text() == SAMPLE.read_text()
    assert committed.duration_s == 48 * 3600.0 and committed.n == 49
    assert committed.start_utc == START
    assert "SYNTHETIC" in SAMPLE.read_text().splitlines()[1]
