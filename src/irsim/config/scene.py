"""Scene config (phase 1): where, when, which weather file, which atmosphere, which targets.

The only place a weather *file* is named. ``Scene.from_config`` (irsim.scene) loads it once and
injects the resulting ``WeatherSeries`` into every consumer; no consumer takes a path
(CLAUDE.md #6, ADR 0032). Targets are the phase-1 prescribed/Newton solvers of M6.6 (sky
targets first: aircraft, drones, birds with scripted or relaxing temperatures); the
environment solver and the §12.3 thermal block arrive with M6.12.

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

SCENE_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SiteSpec(_Frozen):
    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)  # east positive
    altitude_m: float = Field(default=0.0, ge=-500.0, le=9000.0)


class TargetSpec(_Frozen):
    """One temperature node. ``newton`` relaxes toward the weather's T_air from ``t0_k`` with
    ``tau_s``; ``prescribed`` follows ``schedule_s`` (seconds after the scene start) →
    ``schedule_k``."""

    name: str = Field(min_length=1)
    solver: Literal["newton", "prescribed"]
    t0_k: float | None = Field(default=None, gt=0.0)
    tau_s: float | None = Field(default=None, gt=0.0)
    schedule_s: list[float] | None = None
    schedule_k: list[float] | None = None

    @model_validator(mode="after")
    def _fields_for_solver(self) -> TargetSpec:
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


class SceneSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    weather_file: str = Field(min_length=1)  # relative to the data root (ADR 0008)
    atmosphere_preset: str = Field(min_length=1)  # name in configs/atmospheres
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
