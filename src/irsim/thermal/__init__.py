"""Thermal: weather, convection, solar loading, longwave down, temperature solvers.

docs/physics-model.md §6
"""

from irsim.thermal.weather import WEATHER_FIELDS, WeatherSample, WeatherSeries, seconds_since

__all__ = ["WEATHER_FIELDS", "WeatherSample", "WeatherSeries", "seconds_since"]
