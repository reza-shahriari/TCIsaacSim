"""Weather CSV loader/writer and the synthetic clear-day generator (§6.4, ADR 0032).

Project CSV format, version 1:

    # irsim weather v1          (first line; further '#' lines are comments)
    time_utc,t_air_k,rh_fraction,wind_speed_m_s,cloud_fraction,dni_w_m2,dhi_w_m2,visibility_m,precip_mm_h
    2024-06-21T00:00:00Z,288.15,0.60,2.0,0.1,0.0,0.0,23000,0.0

Column names carry the unit. Accepted alternates, converted **once** at load: ``t_air_c``
(+273.15) and ``rh_percent`` (/100). Timestamps are ISO-8601 with an explicit UTC offset or
``Z``; naive timestamps raise. Floats are written with the shortest round-trip representation
so write → read is bit-exact (the content hash survives a round trip).

``synthetic_clear_day`` builds a documented clear-sky day: a sinusoidal air temperature with the
maximum at 15:00 local, relative humidity from a constant dew point (so RH rises at night as a
real clear night does), half-sine direct/diffuse irradiance between sunrise and sunset, constant
wind, cloud and visibility. It is a fixture generator, not a weather model.

docs/physics-model.md §6.4, Appendix A #8
"""

from __future__ import annotations

import csv
import os
import pathlib
from datetime import datetime, timezone

import numpy as np

from irsim.atmosphere.humidity import saturation_vapour_pressure_hpa
from irsim.thermal.weather import WEATHER_FIELDS, WeatherSeries

__all__ = [
    "WEATHER_CSV_MAGIC",
    "load_weather_csv",
    "write_weather_csv",
    "synthetic_clear_day",
]

WEATHER_CSV_MAGIC = "# irsim weather v1"
TIME_COLUMN = "time_utc"
# alternate header -> (canonical name, converter applied exactly once at load)
ALTERNATE_COLUMNS: dict[str, tuple[str, float, float]] = {
    "t_air_c": ("t_air_k", 1.0, 273.15),  # scale, offset
    "rh_percent": ("rh_fraction", 0.01, 0.0),
}


def _parse_time(text: str, line: int) -> datetime:
    text = text.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"line {line}: {text!r} is not an ISO-8601 timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"line {line}: timestamp {text!r} has no UTC offset (use 'Z')")
    return dt.astimezone(timezone.utc)


def load_weather_csv(path: str | os.PathLike[str]) -> WeatherSeries:
    """Read a project weather CSV into the shared :class:`WeatherSeries`."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != WEATHER_CSV_MAGIC:
        raise ValueError(f"{p}: first line must be {WEATHER_CSV_MAGIC!r}")
    body = [(i + 1, ln) for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#")]
    if not body:
        raise ValueError(f"{p}: no header row")
    header = [h.strip() for h in next(csv.reader([body[0][1]]))]
    if header[0] != TIME_COLUMN:
        raise ValueError(f"{p}: first column must be {TIME_COLUMN!r}, got {header[0]!r}")
    canonical: list[tuple[str, float, float]] = []
    for name in header[1:]:
        if name in WEATHER_FIELDS:
            canonical.append((name, 1.0, 0.0))
        elif name in ALTERNATE_COLUMNS:
            canonical.append(ALTERNATE_COLUMNS[name])
        else:
            raise ValueError(f"{p}: unknown column {name!r} (allowed: {WEATHER_FIELDS})")
    names = [c[0] for c in canonical]
    if sorted(names) != sorted(WEATHER_FIELDS):
        missing = sorted(set(WEATHER_FIELDS) - set(names))
        raise ValueError(f"{p}: missing columns {missing} (or a column given twice)")
    times: list[datetime] = []
    cols: dict[str, list[float]] = {name: [] for name in WEATHER_FIELDS}
    for line, text in body[1:]:
        row = next(csv.reader([text]))
        if len(row) != len(header):
            raise ValueError(f"{p} line {line}: {len(row)} fields, header has {len(header)}")
        times.append(_parse_time(row[0], line))
        for (name, scale, offset), cell in zip(canonical, row[1:], strict=True):
            try:
                value = float(cell)
            except ValueError as exc:
                raise ValueError(f"{p} line {line}: {name} = {cell!r} is not a number") from exc
            cols[name].append(value * scale + offset if (scale, offset) != (1.0, 0.0) else value)
    if len(times) < 2:
        raise ValueError(f"{p}: at least two rows are needed")
    epoch = times[0]
    time_s = np.array([(t - epoch).total_seconds() for t in times], dtype=np.float64)
    return WeatherSeries.from_arrays(
        epoch, time_s, **{name: np.asarray(v, dtype=np.float64) for name, v in cols.items()}
    )


def _format_time(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    spec = "seconds" if dt.microsecond == 0 else "microseconds"
    return dt.isoformat(timespec=spec).replace("+00:00", "Z")


def write_weather_csv(
    series: WeatherSeries, path: str | os.PathLike[str], comment: str | None = None
) -> None:
    """Write the project CSV; floats as shortest round-trip repr so a reload is bit-exact."""
    p = pathlib.Path(path)
    out = [WEATHER_CSV_MAGIC]
    if comment:
        out.extend(f"# {ln}" for ln in comment.splitlines())
    out.append(",".join((TIME_COLUMN, *WEATHER_FIELDS)))
    for i in range(series.n):
        when = series.datetime_at(float(series.time_s[i]))
        values = (repr(float(getattr(series, name)[i])) for name in WEATHER_FIELDS)
        out.append(",".join((_format_time(when), *values)))
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def synthetic_clear_day(
    start_utc: datetime,
    hours: float = 48.0,
    *,
    utc_offset_h: float = 0.0,
    t_mean_k: float = 293.15,
    t_swing_k: float = 10.0,
    t_max_hour_local: float = 15.0,
    dewpoint_k: float = 283.15,
    wind_speed_m_s: float = 2.0,
    cloud_fraction: float = 0.05,
    dni_peak_w_m2: float = 850.0,
    dhi_peak_w_m2: float = 100.0,
    sunrise_hour_local: float = 5.5,
    sunset_hour_local: float = 20.5,
    visibility_m: float = 23000.0,
    step_s: float = 3600.0,
) -> WeatherSeries:
    """A documented clear-sky diurnal cycle (fixture generator; see the module docstring).

    Irradiance is a half sine between sunrise and sunset, so its daily integral is
    2 E_peak (t_set − t_rise)/π exactly; temperature is
    t_mean + ½ swing · sin(2π (h − t_max + 6)/24) (maximum at ``t_max_hour_local``, minimum
    12 h earlier); RH = e_s(T_dew)/e_s(T_air) ≤ 1.
    """
    if start_utc.tzinfo is None or start_utc.utcoffset() is None:
        raise ValueError("start_utc must be timezone-aware")
    if hours <= 0 or step_s <= 0 or not 0 <= sunrise_hour_local < sunset_hour_local <= 24:
        raise ValueError("hours, step_s positive; 0 <= sunrise < sunset <= 24")
    n = int(round(hours * 3600.0 / step_s)) + 1
    t = np.arange(n, dtype=np.float64) * step_s
    start = start_utc.astimezone(timezone.utc)
    h0 = start.hour + start.minute / 60.0 + start.second / 3600.0 + utc_offset_h
    h_local = np.mod(h0 + t / 3600.0, 24.0)
    t_air = t_mean_k + 0.5 * t_swing_k * np.sin(
        2.0 * np.pi * (h_local - t_max_hour_local + 6.0) / 24.0
    )
    e_dew = saturation_vapour_pressure_hpa(dewpoint_k - 273.15)
    rh = np.array(
        [min(1.0, e_dew / saturation_vapour_pressure_hpa(float(tk) - 273.15)) for tk in t_air]
    )
    day = (h_local > sunrise_hour_local) & (h_local < sunset_hour_local)
    phase = np.pi * (h_local - sunrise_hour_local) / (sunset_hour_local - sunrise_hour_local)
    shape = np.where(day, np.sin(np.clip(phase, 0.0, np.pi)), 0.0)
    return WeatherSeries.from_arrays(
        start,
        t,
        t_air_k=t_air,
        rh_fraction=rh,
        wind_speed_m_s=np.full(n, float(wind_speed_m_s)),
        cloud_fraction=np.full(n, float(cloud_fraction)),
        dni_w_m2=dni_peak_w_m2 * shape,
        dhi_w_m2=dhi_peak_w_m2 * shape,
        visibility_m=np.full(n, float(visibility_m)),
        precip_mm_h=np.zeros(n),
    )
