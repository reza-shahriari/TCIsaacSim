"""WeatherSeries (M6.1): interpolation identities, unit guards, immutability, the hash, and the
no-file-anywhere rule that makes CLAUDE.md #6 enforceable."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from irsim.atmosphere.humidity import vapour_pressure_hpa
from irsim.radiometry.constants import SOLAR_CONSTANT_W_M2
from irsim.thermal import WEATHER_FIELDS, WeatherSample, WeatherSeries, seconds_since

EPOCH = datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc)


def _series(n: int = 25) -> WeatherSeries:
    t = np.arange(n) * 3600.0
    h = t / 3600.0
    return WeatherSeries.from_arrays(
        EPOCH,
        t,
        t_air_k=293.15 + 6.0 * np.sin(2 * np.pi * (h - 9) / 24),
        rh_fraction=0.5 + 0.2 * np.cos(2 * np.pi * (h - 9) / 24),
        wind_speed_m_s=2.0 + 0.5 * np.sin(h),
        cloud_fraction=np.clip(0.2 + 0.1 * np.sin(h / 3), 0, 1),
        dni_w_m2=np.clip(800 * np.sin(np.pi * (h - 5.5) / 15), 0, None) * ((h > 5.5) & (h < 20.5)),
        dhi_w_m2=np.clip(100 * np.sin(np.pi * (h - 5.5) / 15), 0, None) * ((h > 5.5) & (h < 20.5)),
        visibility_m=np.full(n, 23000.0),
        precip_mm_h=np.zeros(n),
    )


def test_interpolation_identities_at_nodes_and_midpoints() -> None:
    w = _series()
    for i in (0, 7, 24):
        s = w.at(w.time_s[i])
        for name in WEATHER_FIELDS:
            assert getattr(s, name) == pytest.approx(getattr(w, name)[i], rel=1e-12, abs=1e-12)
    for i in (3, 12, 23):
        mid = 0.5 * (w.time_s[i] + w.time_s[i + 1])
        s = w.at(mid)
        for name in WEATHER_FIELDS:
            expect = 0.5 * (getattr(w, name)[i] + getattr(w, name)[i + 1])
            assert getattr(s, name) == pytest.approx(expect, rel=1e-12, abs=1e-12)
    arrays = w.interpolate(w.time_s)
    for name in WEATHER_FIELDS:
        np.testing.assert_array_equal(arrays[name], getattr(w, name))
    assert w.at_datetime(EPOCH + timedelta(hours=7)) == w.at(7 * 3600.0)


def test_extrapolation_refused_and_datetime_helpers() -> None:
    w = _series()
    with pytest.raises(ValueError, match="extrapolation"):
        w.at(-1.0)
    with pytest.raises(ValueError, match="extrapolation"):
        w.interpolate([0.0, w.duration_s + 0.5])
    assert w.duration_s == 24 * 3600.0 and w.n == 25
    assert w.start_utc == EPOCH and w.end_utc == EPOCH + timedelta(days=1)
    assert w.seconds_of(EPOCH + timedelta(hours=3)) == 3 * 3600.0
    with pytest.raises(ValueError, match="aware"):
        seconds_since(EPOCH, datetime(2024, 6, 21, 3))
    with pytest.raises(ValueError, match="aware"):
        WeatherSeries.constant(w.at(0.0), 10.0, epoch_utc=datetime(2024, 1, 1))


@pytest.mark.parametrize(
    "field, value, hint",
    [
        ("t_air_k", 15.0, "celsius"),
        ("rh_fraction", 55.0, "percentage"),
        ("dni_w_m2", -1.0, "non-negative"),
        ("dhi_w_m2", SOLAR_CONSTANT_W_M2 * 1.5, "solar constant"),
        ("wind_speed_m_s", -0.1, "non-negative"),
        ("cloud_fraction", 1.5, "fraction"),
        ("visibility_m", 0.0, "positive"),
        ("precip_mm_h", -1.0, "non-negative"),
    ],
)
def test_unit_and_range_guards(field: str, value: float, hint: str) -> None:
    w = _series()
    cols = {name: getattr(w, name).copy() for name in WEATHER_FIELDS}
    cols[field][3] = value
    with pytest.raises(ValueError, match=hint):
        WeatherSeries.from_arrays(EPOCH, w.time_s, **cols)


def test_structural_guards() -> None:
    w = _series()
    cols = dict(w.columns())
    t = w.time_s.copy()
    t[5] = t[4]
    with pytest.raises(ValueError, match="increasing"):
        WeatherSeries.from_arrays(EPOCH, t, **cols)
    with pytest.raises(ValueError, match="missing"):
        WeatherSeries.from_arrays(
            EPOCH, w.time_s, **{k: v for k, v in cols.items() if k != "dni_w_m2"}
        )
    with pytest.raises(ValueError, match="unknown"):
        WeatherSeries.from_arrays(EPOCH, w.time_s, air_temperature_k=cols["t_air_k"], **cols)
    bad = dict(cols)
    bad["t_air_k"] = cols["t_air_k"][:-1]
    with pytest.raises(ValueError, match="shape"):
        WeatherSeries.from_arrays(EPOCH, w.time_s, **bad)
    nan = dict(cols)
    nan["rh_fraction"] = cols["rh_fraction"].copy()
    nan["rh_fraction"][0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        WeatherSeries.from_arrays(EPOCH, w.time_s, **nan)
    with pytest.raises(ValueError, match="two samples"):
        WeatherSeries.from_arrays(EPOCH, [0.0], **{k: v[:1] for k, v in cols.items()})


def test_immutable_and_hash_sensitive_to_10mK() -> None:
    w = _series()
    with pytest.raises(ValueError, match="read-only"):
        w.t_air_k[0] = 300.0
    cols = {name: getattr(w, name).copy() for name in WEATHER_FIELDS}
    same = WeatherSeries.from_arrays(EPOCH, w.time_s.copy(), **cols)
    assert same.content_hash == w.content_hash and same == w
    cols["t_air_k"][10] += 0.01
    assert WeatherSeries.from_arrays(EPOCH, w.time_s, **cols).content_hash != w.content_hash
    later = WeatherSeries.from_arrays(EPOCH + timedelta(hours=1), w.time_s, **w.columns())
    assert later.content_hash != w.content_hash


def test_sample_derived_humidity_and_constant_series() -> None:
    s = WeatherSample(303.15, 0.80, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0)
    assert s.vapour_pressure_hpa == pytest.approx(vapour_pressure_hpa(303.15, 0.80))
    assert s.absolute_humidity_g_m3 == pytest.approx(24.3, rel=1e-2)
    w = WeatherSeries.constant(s, 3600.0)
    assert w.at(1234.5) == s
    with pytest.raises(ValueError):
        WeatherSeries.constant(s, 0.0)


def test_weather_series_never_opens_a_file() -> None:
    """CLAUDE.md #6 by construction: no constructor or method takes a path; the loader lives in
    weather_io and returns the object that everything shares."""
    for name, member in inspect.getmembers(WeatherSeries):
        if name.startswith("_") and name != "__init__":
            continue
        if not callable(member):
            continue
        params = inspect.signature(member).parameters
        for p in params:
            assert "path" not in p and "file" not in p, f"WeatherSeries.{name}({p}) takes a file"
    import ast

    import irsim.thermal.weather as mod

    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            assert not any(n.split(".")[0] in {"csv", "io", "pathlib"} for n in names), names
        if isinstance(node, ast.Call):
            f = node.func
            called = (
                f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            )
            assert called not in {"open", "read_text", "read_bytes", "load", "safe_load"}, called
