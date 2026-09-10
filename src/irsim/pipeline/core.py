"""Pipeline scaffolding: the stage protocol, the immutable configuration, the mutable state.

A ``Stage`` is a pure function of (planes, config, state) → planes. Planes are the G-buffer dict
contract (:mod:`irsim.config.gbuffer`) extended by the stages' own outputs (``radiance`` at the
supersampled grid, ``flux`` at the detector grid, ``dn16``...). Everything that carries
temperature or radiance is float32 or better (non-negotiable #2).

docs/physics-model.md §13.4, §13.6
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.detector.params import FpaParams, fpa_params_from_config
from irsim.materials.table import MaterialTable
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.lut_files import load_band_lut_for_config

__all__ = ["Planes", "Stage", "PipelineConfig", "PipelineState"]

Planes = dict[str, NDArray[Any]]


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a frame needs that does not change between frames."""

    sensor: SensorConfig
    lut: BandLUT
    materials: MaterialTable
    fpa: FpaParams
    supersample: int

    @classmethod
    def from_sensor(
        cls,
        sensor: SensorConfig,
        materials: MaterialTable,
        lut: BandLUT | None = None,
        lut_dir: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
    ) -> PipelineConfig:
        """Assemble from a validated sensor config; the LUT is given or loaded from ``lut_dir``."""
        if lut is None:
            if lut_dir is None:
                raise ValueError("give a BandLUT or a lut_dir to load one from (make luts)")
            lut = load_band_lut_for_config(sensor, lut_dir, data_dir)
        return cls(
            sensor=sensor,
            lut=lut,
            materials=materials,
            fpa=fpa_params_from_config(sensor),
            supersample=sensor.sensor.optics.supersample_factor,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | os.PathLike[str],
        materials: MaterialTable,
        lut_dir: str | os.PathLike[str],
        data_dir: str | os.PathLike[str] | None = None,
    ) -> PipelineConfig:
        return cls.from_sensor(
            load_sensor_config(path, data_dir), materials, None, lut_dir, data_dir
        )


@dataclass
class PipelineState:
    """Per-camera state that persists across frames: frame counter, housing temperature,
    and the buffers later stages keep (bolometer IIR, drift, FFC reference)."""

    frame_index: int = 0
    housing_temp_k: float = 300.0
    buffers: dict[str, NDArray[Any]] = field(default_factory=dict)

    def advance(self) -> None:
        self.frame_index += 1


class Stage(Protocol):
    """One pipeline stage: reads the planes it needs, returns the planes it produces."""

    name: str

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes: ...


def require_fp32_or_better(plane: NDArray[Any], name: str) -> NDArray[np.floating]:
    if plane.dtype == np.float16:
        raise TypeError(f"{name} is float16 (CLAUDE.md non-negotiable #2); use float32 or better")
    if not np.issubdtype(plane.dtype, np.floating):
        raise TypeError(f"{name} must be a float plane, got {plane.dtype}")
    return plane
