"""Thermal: weather, convection, solar loading, longwave down, temperature solvers.

docs/physics-model.md §6
"""

from irsim.thermal.weather import WEATHER_FIELDS, WeatherSample, WeatherSeries, seconds_since
from irsim.thermal.weather_io import load_weather_csv, synthetic_clear_day, write_weather_csv

__all__ = [
    "WEATHER_FIELDS",
    "WeatherSample",
    "WeatherSeries",
    "seconds_since",
    "load_weather_csv",
    "synthetic_clear_day",
    "write_weather_csv",
]
