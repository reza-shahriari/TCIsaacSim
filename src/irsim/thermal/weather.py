"""WeatherSeries: the one weather object the thermal solver and the atmosphere share (§6.4, §7.3).

CLAUDE.md #6: nothing may run summer weather in the temperature model and winter weather in the
atmosphere. The mechanism is *injection of one instance*: this class never opens a file (the CSV
loader in :mod:`irsim.thermal.weather_io` builds it), no consumer takes a path, and the
``Scene`` (M6.17) passes the same object to every consumer. Identity checks in the tests hold the
line.

Fields (SI, hourly is enough, linearly interpolated by :meth:`at`):

    time_s          seconds since ``epoch_utc`` (aware UTC datetime), strictly increasing
    t_air_k         air temperature, kelvin           (15 raises: that is a Celsius value)
    rh_fraction     relative humidity, 0..1           (55 raises: that is a percentage)
    wind_speed_m_s  ≥ 0
    cloud_fraction  0..1
    dni_w_m2        direct normal irradiance, 0..SOLAR_CONSTANT
    dhi_w_m2        diffuse horizontal irradiance, 0..SOLAR_CONSTANT
    visibility_m    meteorological optical range, > 0 (∞ allowed: no aerosol)
    precip_mm_h     ≥ 0

Arrays are copied in, float64 and read-only; :attr:`content_hash` keys spin-up caches and goldens
(a 0.01 K change changes it). Extrapolation is refused: asking for a time outside the series is
an error, never a clamp, because a clamped night is a wrong night.

docs/physics-model.md §6.4, §7.3, Appendix A #8 (weather is prescribed, not simulated)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.humidity import absolute_humidity_g_m3, vapour_pressure_hpa
from irsim.radiometry.constants import SOLAR_CONSTANT_W_M2

__all__ = ["WEATHER_FIELDS", "WeatherSample", "WeatherSeries", "seconds_since"]

WEATHER_FIELDS: tuple[str, ...] = (
    "t_air_k",
    "rh_fraction",
    "wind_speed_m_s",
    "cloud_fraction",
    "dni_w_m2",
    "dhi_w_m2",
    "visibility_m",
    "precip_mm_h",
)
T_AIR_MIN_K = 150.0
T_AIR_MAX_K = 350.0


def seconds_since(epoch_utc: datetime, when: datetime) -> float:
    """Seconds from an aware epoch to an aware datetime (naive datetimes are refused)."""
    for name, dt in (("epoch_utc", epoch_utc), ("when", when)):
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware (naive datetimes are ambiguous)")
    return (when - epoch_utc).total_seconds()


@dataclass(frozen=True)
class WeatherSample:
    """The weather at one instant (the interpolated state consumers read)."""

    t_air_k: float
    rh_fraction: float
    wind_speed_m_s: float
    cloud_fraction: float
    dni_w_m2: float
    dhi_w_m2: float
    visibility_m: float
    precip_mm_h: float

    @property
    def vapour_pressure_hpa(self) -> float:
        """e = RH · e_s(T_air) (Magnus, M8.2) -- the input of the sky emissivity (M6.5)."""
        return vapour_pressure_hpa(self.t_air_k, self.rh_fraction)

    @property
    def absolute_humidity_g_m3(self) -> float:
        return absolute_humidity_g_m3(self.t_air_k, self.rh_fraction)


def _check_range(name: str, arr: NDArray[np.float64], lo: float, hi: float, hint: str) -> None:
    if np.any(arr < lo) or np.any(arr > hi):
        bad = float(arr[(arr < lo) | (arr > hi)][0])
        raise ValueError(f"{name} = {bad} outside [{lo}, {hi}]: {hint}")


@dataclass(frozen=True)
class WeatherSeries:
    """Validated, immutable, linearly interpolated weather. Build with :meth:`from_arrays`."""

    epoch_utc: datetime
    time_s: NDArray[np.float64]
    t_air_k: NDArray[np.float64]
    rh_fraction: NDArray[np.float64]
    wind_speed_m_s: NDArray[np.float64]
    cloud_fraction: NDArray[np.float64]
    dni_w_m2: NDArray[np.float64]
    dhi_w_m2: NDArray[np.float64]
    visibility_m: NDArray[np.float64]
    precip_mm_h: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.epoch_utc.tzinfo is None or self.epoch_utc.utcoffset() is None:
            raise ValueError("epoch_utc must be timezone-aware")
        t = np.array(self.time_s, dtype=np.float64, copy=True)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("a WeatherSeries needs at least two samples along one time axis")
        if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
            raise ValueError("time_s must be finite and strictly increasing")
        object.__setattr__(self, "time_s", t)
        for name in WEATHER_FIELDS:
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if arr.shape != t.shape:
                raise ValueError(f"{name} has shape {arr.shape}, time_s has {t.shape}")
            if np.any(np.isnan(arr)):
                raise ValueError(f"{name} contains NaN")
            object.__setattr__(self, name, arr)
        _check_range("t_air_k", self.t_air_k, T_AIR_MIN_K, T_AIR_MAX_K, "kelvin, not celsius")
        _check_range("rh_fraction", self.rh_fraction, 0.0, 1.0, "a fraction, not a percentage")
        _check_range("wind_speed_m_s", self.wind_speed_m_s, 0.0, 150.0, "m/s, non-negative")
        _check_range("cloud_fraction", self.cloud_fraction, 0.0, 1.0, "a fraction")
        for name in ("dni_w_m2", "dhi_w_m2"):
            _check_range(
                name,
                getattr(self, name),
                0.0,
                SOLAR_CONSTANT_W_M2,
                "surface irradiance is non-negative and below the solar constant",
            )
        if np.any(self.visibility_m <= 0.0):
            raise ValueError("visibility_m must be positive (inf = no aerosol)")
        _check_range("precip_mm_h", self.precip_mm_h, 0.0, 1000.0, "mm/h, non-negative")
        for name in ("time_s", *WEATHER_FIELDS):
            arr = getattr(self, name)
            arr.setflags(write=False)

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, epoch_utc: datetime, time_s: Any, **columns: Any) -> WeatherSeries:
        missing = set(WEATHER_FIELDS) - columns.keys()
        unknown = columns.keys() - set(WEATHER_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"weather columns: missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        return cls(epoch_utc=epoch_utc, time_s=np.asarray(time_s, dtype=np.float64), **columns)

    @classmethod
    def constant(
        cls, sample: WeatherSample, duration_s: float, epoch_utc: datetime | None = None
    ) -> WeatherSeries:
        """Two-node series holding one state for ``duration_s`` (tests, steady-state runs)."""
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        epoch = epoch_utc or datetime(2000, 1, 1, tzinfo=timezone.utc)
        cols = {name: np.full(2, getattr(sample, name)) for name in WEATHER_FIELDS}
        return cls.from_arrays(epoch, np.array([0.0, duration_s]), **cols)

    # -- access ---------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def start_utc(self) -> datetime:
        return self.epoch_utc + timedelta(seconds=float(self.time_s[0]))

    @property
    def end_utc(self) -> datetime:
        return self.epoch_utc + timedelta(seconds=float(self.time_s[-1]))

    def datetime_at(self, t_s: float) -> datetime:
        return self.epoch_utc + timedelta(seconds=float(t_s))

    def seconds_of(self, when: datetime) -> float:
        return seconds_since(self.epoch_utc, when)

    def _check_inside(self, t: NDArray[np.float64]) -> None:
        if np.any(t < self.time_s[0]) or np.any(t > self.time_s[-1]):
            raise ValueError(
                f"time outside the weather series [{self.time_s[0]}, {self.time_s[-1]}] s: "
                "extrapolation is refused (extend the file or the spin-up window instead)"
            )

    def interpolate(self, t_s: Any) -> dict[str, NDArray[np.float64]]:
        """Linear interpolation of every field at an array of times (inside the series)."""
        t = np.asarray(t_s, dtype=np.float64)
        if not np.all(np.isfinite(t)):
            raise ValueError("times must be finite")
        self._check_inside(t)
        return {name: np.interp(t, self.time_s, getattr(self, name)) for name in WEATHER_FIELDS}

    def at(self, t_s: float) -> WeatherSample:
        """The weather at one time (seconds since ``epoch_utc``)."""
        values = self.interpolate(np.asarray(float(t_s)))
        return WeatherSample(**{name: float(values[name][()]) for name in WEATHER_FIELDS})

    def at_datetime(self, when: datetime) -> WeatherSample:
        return self.at(self.seconds_of(when))

    def columns(self) -> Mapping[str, NDArray[np.float64]]:
        return {name: getattr(self, name) for name in WEATHER_FIELDS}

    @property
    def content_hash(self) -> str:
        """SHA-256 over the epoch and the float64 bytes of every column, in field order."""
        h = hashlib.sha256()
        h.update(self.epoch_utc.astimezone(timezone.utc).isoformat().encode())
        for name in ("time_s", *WEATHER_FIELDS):
            h.update(name.encode())
            h.update(np.ascontiguousarray(getattr(self, name), dtype="<f8").tobytes())
        return h.hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WeatherSeries):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __hash__(self) -> int:
        return hash(self.content_hash)


assert tuple(f.name for f in fields(WeatherSample)) == WEATHER_FIELDS
