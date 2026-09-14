"""The night term of stage 1: airglow and moonlight as one incident band radiance (§5.5).

    l_night = [ E_airglow,B · c(cloud) · v(t) + E_moon,B · Φ(phase) · sin(el) · τ ] / π

in the LUT's own quantity, so ADR 0063's bundle adds it to the environment and solar terms and
stage 1 reflects the sum once.

**There is no shadow argument and no sun direction, and that is the physics, not an omission.**
§5.5: airglow "is not blocked in the same way as moonlight -- cloud attenuates it, but it does
not have a directional shadow". It is emitted at ~87 km across the whole sky; a wall casts no
airglow shadow. Adding an `S` here would let a scene darken the night sky by geometry, which is
the mistake this signature prevents rather than documents.

The moon is reflected sunlight and therefore *does* have a direction, a phase and a horizon. It
is first order here: the sun's spectral shape (the lunar reddening -- the moon's albedo roughly
doubles between 0.5 µm and 1.6 µm -- is deferred, ADR 0065), Lane & Irvine's phase law, and a
sin(elevation) projection onto the horizontal. Stars and artificial lighting are deferred.

docs/physics-model.md §5.5, §5.2; ADR 0063, ADR 0065
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.environment import EnvironmentSpec
from irsim.config.sensor import SensorConfig
from irsim.radiometry.constants import SOLAR_CONSTANT_W_M2
from irsim.radiometry.lut import Quantity
from irsim.radiometry.night import (
    AIRGLOW_SHAPE_FILE,
    FULL_MOON_IRRADIANCE_W_M2,
    airglow_variation,
    cloud_attenuation,
    load_airglow_spectrum,
    lunar_phase_factor,
)
from irsim.radiometry.solar import TOA_FILE, load_solar_spectrum
from irsim.radiometry.spectral_response import load_spectral_response

__all__ = ["NightIllumination"]


@dataclass(frozen=True)
class NightIllumination:
    """Band-integrated night sources for one sensor and one environment preset."""

    airglow_band: float
    moon_band_full: float
    k_cloud: float
    moon_enabled: bool
    moon_phase_fraction: float
    quantity: Quantity
    band_id: str
    airglow_seed: int = 0

    @classmethod
    def for_sensor(
        cls,
        sensor: SensorConfig,
        environment: EnvironmentSpec,
        quantity: Quantity,
        data_dir: str | os.PathLike[str] | None = None,
        airglow_seed: int = 0,
    ) -> NightIllumination:
        """Integrate the airglow shape and the solar shape against this sensor's own R(λ).

        The moon borrows the solar spectrum and is scaled by the **magnitude** ratio, so the
        lunar albedo never has to be authored: it cancels between the shape and the level.
        """
        from irsim.config.loader import resolve_data_dir

        root = resolve_data_dir(data_dir)
        response = load_spectral_response(sensor.sensor.band.spectral_response)
        night = environment.night
        shape_file = night.airglow_shape_file or AIRGLOW_SHAPE_FILE
        airglow = load_airglow_spectrum(root / shape_file)
        solar = load_solar_spectrum(root / TOA_FILE)
        moon_scale = FULL_MOON_IRRADIANCE_W_M2 / SOLAR_CONSTANT_W_M2
        return cls(
            airglow_band=airglow.band(response, night.airglow_w_m2, quantity),
            moon_band_full=moon_scale * solar.band(response, quantity),
            k_cloud=float(night.k_cloud),
            moon_enabled=bool(night.moon.enabled),
            moon_phase_fraction=float(night.moon.phase_fraction),
            quantity=quantity,
            band_id=sensor.sensor.band.band_id,
            airglow_seed=airglow_seed,
        )

    # -- the two sources ------------------------------------------------------------------

    def airglow_irradiance(self, cloud_fraction: Any = 0.0, t_s: Any = None) -> NDArray[np.float64]:
        """E_airglow,B after cloud and the slow drift. Never zero while the level is non-zero."""
        out = self.airglow_band * cloud_attenuation(cloud_fraction, self.k_cloud)
        if t_s is not None:
            out = out * airglow_variation(t_s, self.airglow_seed)
        return np.asarray(out, dtype=np.float64)

    def moon_irradiance(
        self,
        elevation_deg: Any = 90.0,
        tau_moon: Any = 1.0,
        phase_fraction: Any = None,
    ) -> NDArray[np.float64]:
        """E_moon,B on a horizontal surface: level × Φ(phase) × sin(el), 0 below the horizon."""
        if not self.moon_enabled:
            return np.asarray(0.0)
        phase = self.moon_phase_fraction if phase_fraction is None else phase_fraction
        el = np.asarray(elevation_deg, dtype=np.float64)
        tau = np.asarray(tau_moon, dtype=np.float64)
        if np.any((tau < 0.0) | (tau > 1.0)):
            raise ValueError("tau_moon must lie in [0, 1]")
        above = np.where(el > 0.0, np.sin(np.deg2rad(np.clip(el, 0.0, 90.0))), 0.0)
        return np.asarray(
            self.moon_band_full * lunar_phase_factor(phase) * above * tau, dtype=np.float64
        )

    # -- the bundle's plane ---------------------------------------------------------------

    def incident_radiance(
        self,
        cloud_fraction: Any = 0.0,
        *,
        t_s: Any = None,
        moon_elevation_deg: Any = 90.0,
        tau_moon: Any = 1.0,
        phase_fraction: Any = None,
    ) -> NDArray[np.float64]:
        """(E_airglow + E_moon) / π -- the bundle's ``l_night`` plane, isotropic by construction."""
        total = self.airglow_irradiance(cloud_fraction, t_s) + self.moon_irradiance(
            moon_elevation_deg, tau_moon, phase_fraction
        )
        return np.asarray(total / math.pi, dtype=np.float64)
