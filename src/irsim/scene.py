"""Scene: one WeatherSeries, loaded once, injected into every consumer (CLAUDE.md #6, ADR 0032).

``Scene.from_config`` is the only code path that turns a weather *file* into a ``WeatherSeries``
for a simulation. It builds the ``Atmosphere`` (M8.5) and the target solvers (M6.6) with that
one object and refuses, at construction, any consumer holding a different one. Phase-2
consumers -- the sky model (MS.2), the housing temperature (M9.3), the FPA thermal node (M9.2),
the environment solver (M6.12) -- register the same way: they take the object and expose it as
``.weather`` so the identity check below covers them without new code here.

Time base: consumers work in seconds on the weather's axis; ``t0_s`` is the scene start on
that axis, and ``t_rel_s`` in the helpers is seconds since the scene start.

docs/physics-model.md §6.4, §7.3, §12.2
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.model import Atmosphere
from irsim.atmosphere.sky import SkyModel
from irsim.config.environment import EnvironmentSpec, load_environment_preset
from irsim.config.loader import resolve_data_dir
from irsim.config.scene import SceneConfig, SceneSpec, TargetSpec, load_scene_config
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.thermal.aerial import AERIAL_HEAT_SOURCES, airframe_solver, heat_source_solver
from irsim.thermal.solvers import NewtonCoolingSolver, PrescribedSolver, TemperatureSolver
from irsim.thermal.weather import WeatherSample, WeatherSeries
from irsim.thermal.weather_io import load_weather_csv

__all__ = ["Scene", "build_target"]


def build_target(spec: TargetSpec, weather: WeatherSeries, t0_s: float) -> TemperatureSolver:
    """A solver for one target spec, on the scene's one shared ``WeatherSeries`` (CLAUDE.md #6).

    ``heat_source`` and ``airframe`` are §6.6's aerial nodes (ADR 0072). They come back as
    ``PrescribedSolver``s like everything else -- the difference is only that their schedule is
    *derived* from the throttle profile and the shared weather, refined until piecewise-linear
    interpolation reproduces the analytic law to 1 mK, rather than typed into the config.
    """
    if spec.solver == "newton":
        assert spec.t0_k is not None and spec.tau_s is not None
        return NewtonCoolingSolver(spec.t0_k, spec.tau_s, weather, t0_s=t0_s)
    if spec.solver in ("airframe", "heat_source"):
        if spec.solver == "airframe":
            derived = airframe_solver(weather, offset_k=spec.offset_k or 0.0)
        else:
            assert spec.source is not None and spec.throttle_s is not None
            assert spec.throttle is not None
            # The profile is written in seconds after the *scene start*; the solvers live on the
            # weather's absolute axis, so it is offset here and nowhere else.
            derived = heat_source_solver(
                AERIAL_HEAT_SOURCES[spec.source],
                weather,
                t0_s + np.asarray(spec.throttle_s, dtype=np.float64),
                spec.throttle,
            )
        # Neither constructor takes t0_s, and an airframe node's schedule spans the whole weather
        # file, so without this its initial reading is the temperature at the *file's* start
        # rather than the scene's -- hours out, and perfectly plausible.
        derived.advance(t0_s, 0.0)
        return derived
    assert spec.schedule_s is not None and spec.schedule_k is not None
    times = t0_s + np.asarray(spec.schedule_s, dtype=np.float64)
    return PrescribedSolver(times, np.asarray(spec.schedule_k, dtype=np.float64), t0_s=t0_s)


@dataclass(frozen=True)
class Scene:
    spec: SceneSpec
    weather: WeatherSeries
    atmosphere: Atmosphere
    targets: Mapping[str, TemperatureSolver]
    t0_s: float
    layered: LayeredAtmosphere | None = None  # MS.1 model on the same weather
    environment: EnvironmentSpec | None = None
    sky_models: Mapping[str, SkyModel] = field(default_factory=dict)  # per band (MS.2)
    extra_consumers: Mapping[str, Any] = field(
        default_factory=dict
    )  # phase-2 objects with .weather

    def __post_init__(self) -> None:
        for name, obj in self.consumers.items():
            w = getattr(obj, "weather", None)
            if w is None:
                continue
            if w is not self.weather:
                raise ValueError(
                    f"consumer {name!r} holds a different WeatherSeries than the scene "
                    "(CLAUDE.md #6: one weather object, injected everywhere)"
                )
        if not self.weather.time_s[0] <= self.t0_s <= self.weather.time_s[-1]:
            raise ValueError("scene start is outside the weather series")

    @property
    def consumers(self) -> dict[str, Any]:
        out: dict[str, Any] = {"atmosphere": self.atmosphere}
        if self.layered is not None:
            out["layered"] = self.layered
        out.update({f"sky:{k}": v for k, v in self.sky_models.items()})
        out.update({f"target:{k}": v for k, v in self.targets.items()})
        out.update(self.extra_consumers)
        return out

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: SceneConfig | SceneSpec,
        luts: Mapping[str, BandLUT] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        quantity: Quantity = "lb",
    ) -> Scene:
        spec = config.scene if isinstance(config, SceneConfig) else config
        weather = load_weather_csv(resolve_data_dir(data_dir) / spec.weather_file)  # once
        t0_s = weather.seconds_of(spec.start_utc)
        preset = load_atmosphere_preset(spec.atmosphere_preset)
        atmosphere = Atmosphere(preset, weather, luts)
        targets = {t.name: build_target(t, weather, t0_s) for t in spec.targets}
        layered = None
        environment = None
        sky_models: dict[str, SkyModel] = {}
        if spec.environment_preset is not None:
            environment = load_environment_preset(spec.environment_preset)
            layered = LayeredAtmosphere(preset, weather, luts)
            for band, lut in (luts or {}).items():
                sky_models[band] = SkyModel(layered, environment, band, lut, quantity)
        return cls(
            spec=spec,
            weather=weather,
            atmosphere=atmosphere,
            targets=targets,
            t0_s=t0_s,
            layered=layered,
            environment=environment,
            sky_models=sky_models,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        luts: Mapping[str, BandLUT] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        quantity: Quantity = "lb",
    ) -> Scene:
        return cls.from_config(load_scene_config(path), luts, data_dir, quantity)

    # -- time helpers ---------------------------------------------------------------------
    def weather_at(self, t_rel_s: float) -> WeatherSample:
        return self.weather.at(self.t0_s + float(t_rel_s))

    def advance_targets(self, t_rel_s: float, dt_s: float) -> dict[str, float]:
        """Step every target from t_rel to t_rel + dt; returns the new temperatures."""
        t = self.t0_s + float(t_rel_s)
        return {name: solver.advance(t, float(dt_s)) for name, solver in self.targets.items()}
