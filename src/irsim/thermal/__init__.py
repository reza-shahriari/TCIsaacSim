"""Thermal: weather, convection, solar loading, longwave down, temperature solvers.

docs/physics-model.md §6
"""

from irsim.thermal.convection import (
    DEFAULT_CONVECTION,
    ConvectionParams,
    convection_coefficient,
    free_forced_crossover_k,
    relative_air_speed,
)
from irsim.thermal.longwave import (
    clear_sky_emissivity,
    longwave_down,
    longwave_down_from_sample,
    sky_emissivity,
)
from irsim.thermal.solar import (
    SunPosition,
    absorbed_solar,
    julian_day,
    solar_loading,
    solar_noon_utc,
    sun_direction,
    sun_position,
    sun_position_utc,
)
from irsim.thermal.solvers import (
    SOLVER_TYPES,
    NewtonCoolingSolver,
    PrescribedSolver,
    SolverState,
    TemperatureSolver,
)
from irsim.thermal.weather import WEATHER_FIELDS, WeatherSample, WeatherSeries, seconds_since
from irsim.thermal.weather_io import load_weather_csv, synthetic_clear_day, write_weather_csv

__all__ = [
    "SOLVER_TYPES",
    "NewtonCoolingSolver",
    "PrescribedSolver",
    "SolverState",
    "TemperatureSolver",
    "clear_sky_emissivity",
    "longwave_down",
    "longwave_down_from_sample",
    "sky_emissivity",
    "SunPosition",
    "absorbed_solar",
    "julian_day",
    "solar_loading",
    "solar_noon_utc",
    "sun_direction",
    "sun_position",
    "sun_position_utc",
    "DEFAULT_CONVECTION",
    "ConvectionParams",
    "convection_coefficient",
    "free_forced_crossover_k",
    "relative_air_speed",
    "WEATHER_FIELDS",
    "WeatherSample",
    "WeatherSeries",
    "seconds_since",
    "load_weather_csv",
    "synthetic_clear_day",
    "write_weather_csv",
]
