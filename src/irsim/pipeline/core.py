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

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.atmosphere.sky import SkyModel
from irsim.config.loader import load_sensor_config
from irsim.config.sensor import SensorConfig
from irsim.detector.anchor import anchor_noise
from irsim.detector.bolometer import MicrobolometerDetector
from irsim.detector.params import BolometerParams, FpaParams, PhotonParams, fpa_params_from_config
from irsim.detector.photon import PhotonDetector
from irsim.detector.response import Detector
from irsim.isp.radiometric import RadiometricCalibration
from irsim.materials.table import MaterialTable
from irsim.noise.stage import NoiseStage
from irsim.optics.psf import optical_psf
from irsim.radiometry.lut import BandLUT, Quantity
from irsim.radiometry.lut_files import load_band_lut_for_config

__all__ = ["Planes", "Stage", "PipelineConfig", "PipelineState", "RADIOMETRIC_RANGE_K"]

Planes = dict[str, NDArray[Any]]

# Scene-temperature span that fills the ADC for the calibrated transfer (ADR 0021): -40..+200 C.
RADIOMETRIC_RANGE_K: tuple[float, float] = (233.15, 473.15)


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a frame needs that does not change between frames."""

    sensor: SensorConfig
    lut: BandLUT
    materials: MaterialTable
    fpa: FpaParams
    supersample: int
    calibration: RadiometricCalibration | None
    t_housing_cal_k: float
    detector: Detector
    noise: NoiseStage
    sensor_seed: int
    noise_enabled: bool
    psf: NDArray[np.float64] | None  # optical PSF at the k× pitch; None = no optical blur
    atmosphere: Atmosphere | LayeredAtmosphere | None = None  # stage 2; None = identity
    tau_override: float | None = (
        None  # L1 fallback: constant τ, path radiance at the weather's T_air
    )
    sky: SkyModel | None = None  # stage-1 reflected term (M7.13); None = emission only

    @property
    def quantity(self) -> Quantity:
        """Which LUT table the radiance chain runs on: energy for bolometers, photon otherwise."""
        return "lb" if self.fpa.type == "bolometer" else "lb_q"

    @classmethod
    def from_sensor(
        cls,
        sensor: SensorConfig,
        materials: MaterialTable,
        lut: BandLUT | None = None,
        lut_dir: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        t_housing_cal_k: float | None = None,
        radiometric_range_k: tuple[float, float] = RADIOMETRIC_RANGE_K,
        sensor_seed: int = 0,
        noise_enabled: bool = True,
        psf_enabled: bool = True,
        reference_wavelength_um: float | None = None,
        atmosphere: Atmosphere | LayeredAtmosphere | None = None,
        tau_override: float | None = None,
        sky: SkyModel | None = None,
    ) -> PipelineConfig:
        """Assemble from a validated sensor config; the LUT is given or loaded from ``lut_dir``.

        The calibrated transfer (ADR 0021) is built for bolometer cameras with the housing at
        ``t_housing_cal_k`` (default: ``optics.housing_temp_k`` if fixed, else 300 K). The
        detector's noise is anchored to ``noise.netd_mk_at_300k`` (ADR 0025) and the correlated
        3-D stage seeded with ``sensor_seed`` (ADR 0022); ``noise_enabled=False`` runs the ideal
        chain (the ablation switch of ME.8). The optical PSF (diffraction at the band-
        representative wavelength -- ``reference_wavelength_um``, else
        ``mtf.reference_wavelength_um``, else the band centre -- times the aberration Gaussian) is
        built at the supersampled pitch (ADR 0059); ``psf_enabled=False`` skips it. Stage 2
        runs when an ``Atmosphere`` (M8.5, bound to the scene's WeatherSeries) is given; the
        frame time on the weather axis is ``PipelineState.t_s``. ``tau_override`` is the L1
        fallback: a constant τ at every distance, path radiance still at the weather's T_air.
        """
        if tau_override is not None:
            if atmosphere is None:
                raise ValueError("tau_override needs an Atmosphere (its weather gives T_air)")
            if not 0.0 <= tau_override <= 1.0:
                raise ValueError("tau_override must lie in [0, 1]")
        if atmosphere is not None and sensor.sensor.band.band_id not in atmosphere.preset.bands:
            raise ValueError(
                f"atmosphere preset has no band {sensor.sensor.band.band_id!r} "
                f"(has {sorted(atmosphere.preset.bands)})"
            )
        if tau_override is not None and isinstance(atmosphere, LayeredAtmosphere):
            raise ValueError(
                "tau_override is the grey L1 fallback; use the grey Atmosphere with it"
            )
        if sky is not None:
            if sky.band != sensor.sensor.band.band_id:
                raise ValueError(
                    f"sky model is for band {sky.band!r}, sensor is {sensor.sensor.band.band_id!r}"
                )
            if atmosphere is not None and sky.weather is not atmosphere.weather:
                raise ValueError("sky model and atmosphere hold different WeatherSeries (#6)")
            if sky.environment.ground.mode == "solver":
                raise ValueError("ground.mode 'solver' needs the environment solver (M6.12)")
            expected_q = "lb" if fpa_params_from_config(sensor).type == "bolometer" else "lb_q"
            if sky.quantity != expected_q:
                raise ValueError(
                    f"sky model built in the {sky.quantity!r} form; "
                    f"this sensor runs on {expected_q!r}"
                )
        if lut is None:
            if lut_dir is None:
                raise ValueError("give a BandLUT or a lut_dir to load one from (make luts)")
            lut = load_band_lut_for_config(sensor, lut_dir, data_dir)
        fpa = fpa_params_from_config(sensor)
        if t_housing_cal_k is None:
            fixed = sensor.sensor.optics.housing_temp_k
            t_housing_cal_k = fixed if fixed is not None else 300.0
        calibration = None
        budget = anchor_noise(sensor.sensor, lut)
        detector: Detector
        if isinstance(fpa, BolometerParams):
            calibration = RadiometricCalibration.from_scene_range(
                sensor.sensor, lut, radiometric_range_k[0], radiometric_range_k[1], t_housing_cal_k
            )
            detector = MicrobolometerDetector(fpa, calibration.transfer, budget)
        elif isinstance(fpa, PhotonParams):
            detector = PhotonDetector(fpa, budget)
        else:  # pragma: no cover
            raise TypeError(f"unknown FPA params {type(fpa).__name__}")
        spec = sensor.sensor
        psf = None
        if psf_enabled:
            lam = (
                reference_wavelength_um
                if reference_wavelength_um is not None
                else spec.reference_wavelength_um
            )
            psf = optical_psf(
                lam,
                spec.optics.f_number,
                spec.optics.mtf.aberration_sigma_um,
                spec.fpa.pitch_um,
                spec.optics.supersample_factor,
            )
        return cls(
            sensor=sensor,
            lut=lut,
            materials=materials,
            fpa=fpa,
            supersample=spec.optics.supersample_factor,
            calibration=calibration,
            t_housing_cal_k=t_housing_cal_k,
            detector=detector,
            noise=NoiseStage.from_sensor(spec, sensor_seed, enabled=noise_enabled),
            sensor_seed=sensor_seed,
            noise_enabled=noise_enabled,
            psf=psf,
            atmosphere=atmosphere,
            tau_override=tau_override,
            sky=sky,
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
    t_s: float = (
        0.0  # frame time on the scene weather's axis (Scene.t0_s + t_rel); stage 2 reads it
    )
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
