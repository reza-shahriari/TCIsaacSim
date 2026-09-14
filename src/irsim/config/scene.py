"""Scene config (phase 1): where, when, which weather file, which atmosphere, which targets.

The only place a weather *file* is named. ``Scene.from_config`` (irsim.scene) loads it once and
injects the resulting ``WeatherSeries`` into every consumer; no consumer takes a path
(CLAUDE.md #6, ADR 0032). Targets are the phase-1 solvers of M6.6 (sky targets first: aircraft,
drones, birds with scripted or relaxing temperatures); the environment solver and the §12.3
thermal block arrive with M6.12.

Four solver kinds. ``newton`` and ``prescribed`` are the raw M6.6 solvers. ``heat_source`` and
``airframe`` are the **aerial** model of §6.6 / ADR 0072 -- T = T_air(t) + ΔT_max u(t)^n for a
motor, speed controller or battery pack, and T = T_air(t) + offset for an unpowered skin. Those two
existed in :mod:`irsim.thermal.aerial` from M6.6 and were reachable only from tests: no scene could
ask for a motor. Naming a throttle profile here rather than a temperature schedule is the point --
a flight is authored as *what the pilot did*, and the temperatures follow from the model rather
than from numbers someone typed (ADR 0074).

docs/physics-model.md §6.5, §6.6, §12.2
"""

from __future__ import annotations

import os
import pathlib
from datetime import datetime
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "SCENE_SCHEMA_VERSION",
    "SiteSpec",
    "TargetSpec",
    "SceneSpec",
    "SceneConfig",
    "load_scene_config",
]

SCENE_SCHEMA_VERSION = 3  # v3: aerial heat_source/airframe target solvers (ADR 0074)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SiteSpec(_Frozen):
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)  # east positive
    altitude_m: float = Field(default=0.0, ge=-500.0, le=9000.0)


class TargetSpec(_Frozen):
    """One temperature node.

    * ``newton`` relaxes toward the weather's T_air from ``t0_k`` with ``tau_s``.
    * ``prescribed`` follows ``schedule_s`` (seconds after the scene start) → ``schedule_k``.
    * ``heat_source`` is §6.6's aerial dissipating node: ``source`` names one of
      :data:`irsim.thermal.aerial.AERIAL_HEAT_SOURCES` (``motor``, ``esc``, ``battery``) and
      ``throttle_s`` → ``throttle`` is the pilot's throttle fraction over time. The temperature is
      T_air(t) + ΔT_max u(t)^n, derived, never authored.
    * ``airframe`` is an unpowered skin: T_air(t) + ``offset_k``.

    **The heat-source law is a steady-state relation** (ADR 0072): there is no thermal time
    constant in it, so the node follows the throttle instantaneously. It is only defensible where
    the throttle varies slowly against a real motor's time constant, which is minutes -- a profile
    that swings the throttle in seconds will produce a temperature swing no metal could follow.
    ADR 0074 records this and why the demo is filmed as a time-lapse because of it.
    """

    name: str = Field(min_length=1)
    solver: Literal["newton", "prescribed", "heat_source", "airframe"]
    t0_k: float | None = Field(default=None, gt=0.0)
    tau_s: float | None = Field(default=None, gt=0.0)
    schedule_s: list[float] | None = None
    schedule_k: list[float] | None = None
    source: str | None = None
    throttle_s: list[float] | None = None
    throttle: list[float] | None = None
    offset_k: float | None = None

    @model_validator(mode="after")
    def _fields_for_solver(self) -> TargetSpec:
        if self.solver in ("heat_source", "airframe"):
            return self._aerial_fields()
        if self.source is not None or self.throttle is not None or self.throttle_s is not None:
            raise ValueError(f"target {self.name!r}: {self.solver} takes no throttle profile")
        if self.offset_k is not None:
            raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
        if self.solver == "newton":
            if self.t0_k is None or self.tau_s is None:
                raise ValueError(f"target {self.name!r}: newton needs t0_k and tau_s")
            if self.schedule_s is not None or self.schedule_k is not None:
                raise ValueError(f"target {self.name!r}: newton takes no schedule")
        else:
            if not self.schedule_s or not self.schedule_k:
                raise ValueError(
                    f"target {self.name!r}: prescribed needs schedule_s and schedule_k"
                )
            if len(self.schedule_s) != len(self.schedule_k):
                raise ValueError(
                    f"target {self.name!r}: schedule_s and schedule_k differ in length"
                )
            if any(b <= a for a, b in zip(self.schedule_s[:-1], self.schedule_s[1:], strict=True)):
                raise ValueError(f"target {self.name!r}: schedule_s must be strictly increasing")
            if any(t <= 0.0 for t in self.schedule_k):
                raise ValueError(f"target {self.name!r}: schedule_k must be positive kelvin")
            if self.t0_k is not None or self.tau_s is not None:
                raise ValueError(f"target {self.name!r}: prescribed takes no t0_k/tau_s")
        return self

    def _aerial_fields(self) -> TargetSpec:
        """Validate the ADR 0072 solvers. The heat-source names are checked against the model."""
        from irsim.thermal.aerial import AERIAL_HEAT_SOURCES

        if self.t0_k is not None or self.tau_s is not None:
            raise ValueError(f"target {self.name!r}: {self.solver} takes no t0_k/tau_s")
        if self.schedule_s is not None or self.schedule_k is not None:
            raise ValueError(
                f"target {self.name!r}: {self.solver} derives its schedule from the model; "
                "authoring temperatures directly is what this solver exists to replace"
            )
        if self.solver == "airframe":
            if self.source is not None or self.throttle is not None:
                raise ValueError(f"target {self.name!r}: airframe takes no source or throttle")
            return self

        if self.source not in AERIAL_HEAT_SOURCES:
            raise ValueError(
                f"target {self.name!r}: source must be one of "
                f"{sorted(AERIAL_HEAT_SOURCES)}, got {self.source!r}"
            )
        if self.offset_k is not None:
            raise ValueError(f"target {self.name!r}: only airframe takes offset_k")
        if not self.throttle_s or not self.throttle:
            raise ValueError(f"target {self.name!r}: heat_source needs throttle_s and throttle")
        if len(self.throttle_s) != len(self.throttle):
            raise ValueError(f"target {self.name!r}: throttle_s and throttle differ in length")
        if any(b <= a for a, b in zip(self.throttle_s[:-1], self.throttle_s[1:], strict=True)):
            raise ValueError(f"target {self.name!r}: throttle_s must be strictly increasing")
        if any(not 0.0 <= u <= 1.0 for u in self.throttle):
            raise ValueError(f"target {self.name!r}: throttle must lie in [0, 1]")
        return self


class SceneSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    weather_file: str = Field(min_length=1)  # relative to the data root (ADR 0008)
    atmosphere_preset: str = Field(min_length=1)  # name in configs/atmospheres
    environment_preset: str | None = None  # configs/environments (sky, ground, solar terms)
    site: SiteSpec
    start_utc: datetime
    targets: list[TargetSpec] = Field(default_factory=list)

    @field_validator("start_utc")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("start_utc must carry a UTC offset (e.g. 2024-06-21T04:00:00Z)")
        return v

    @field_validator("targets")
    @classmethod
    def _unique_names(cls, targets: list[TargetSpec]) -> list[TargetSpec]:
        names = [t.name for t in targets]
        if len(set(names)) != len(names):
            raise ValueError(f"target names must be unique: {names}")
        return targets


class SceneConfig(_Frozen):
    schema_version: int
    scene: SceneSpec

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != SCENE_SCHEMA_VERSION:
            raise ValueError(f"scene schema_version {v} != {SCENE_SCHEMA_VERSION}")
        return v


def load_scene_config(path: str | os.PathLike[str]) -> SceneConfig:
    raw = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    return SceneConfig.model_validate(raw)
