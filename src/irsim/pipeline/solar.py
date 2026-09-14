"""The reflected-solar term of stage 1: the direct beam as an incident band radiance (§5.4).

§5.4 writes the reflected radiance as

    L_sun = (ρ_B / π) E_B τ_sun(θ_zen) cos θ_s S

with S the shadow mask. This module produces everything except ρ_B, as the **equivalent isotropic
incident radiance** the M11.2 bundle carries,

    l_sun = E_B τ_sun cos θ_s S / π,

so stage 1 applies ρ exactly once, alongside the environment and night terms, and the Lambertian
identity ρ = 1 → E_B/π is an identity of the code rather than of the algebra (ADR 0063). A caller
who has read §5.4 and expects to apply ρ themselves would double-count it; the ADR says so, and
the plane is named ``l_sun`` in the same units as every other incident term to make the mistake
hard to write.

Three exact zeros, and they are exact on purpose -- a horizon that leaks 1e-12 of a 300 W m⁻²
beam is a horizon that shows up as a seam after AGC:

* the sun below the horizon (elevation ≤ 0),
* a surface facing away from it (cos θ_s ≤ 0),
* full shadow (S = 0).

τ_sun uses **Kasten–Young** air mass rather than ADR 0051's plane-parallel sec θ. The band-
averaged slant path keeps sec θ and nothing measured against ADR 0051 moves; but the direct beam
needs an answer at low sun, and 09:00 is not an error condition.

docs/physics-model.md §5.4, §5.2, §7.1; ADR 0063, ADR 0064
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.extinction import airmass_kasten_young
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.sensor import SensorConfig
from irsim.radiometry.lut import Quantity
from irsim.radiometry.solar import TOA_FILE, SolarSpectrum, load_solar_spectrum
from irsim.radiometry.spectral_response import load_spectral_response

__all__ = ["SolarIllumination", "beam_transmittance"]


def beam_transmittance(
    preset: AtmospherePreset, band: str, elevation_deg: float | NDArray[np.floating]
) -> NDArray[np.float64]:
    """τ_sun = τ_zenith^m(θ), Kasten–Young air mass; **exactly 0** at or below the horizon."""
    if band not in preset.solar.zenith_transmittance:
        raise KeyError(f"preset has no solar transmittance for band {band!r}")
    tau_zenith = float(preset.solar.zenith_transmittance[band])
    el = np.asarray(elevation_deg, dtype=np.float64)
    out = np.zeros_like(el)
    up = el > 0.0
    if np.any(up):
        zenith = np.deg2rad(90.0 - el[up] if el.ndim else np.asarray([90.0 - float(el)]))
        masses = np.asarray([airmass_kasten_young(float(z)) for z in np.atleast_1d(zenith)])
        out[up] = tau_zenith**masses if el.ndim else tau_zenith ** masses[0]
    return np.asarray(out)


@dataclass(frozen=True)
class SolarIllumination:
    """The band-integrated extraterrestrial irradiance for one sensor, in one quantity."""

    e_band: float
    quantity: Quantity
    band_id: str
    spectrum_sha256: str
    spectrum_path: str

    @classmethod
    def from_spectrum(
        cls, spectrum: SolarSpectrum, response: Any, quantity: Quantity, band_id: str = ""
    ) -> SolarIllumination:
        return cls(
            e_band=spectrum.band(response, quantity),
            quantity=quantity,
            band_id=band_id,
            spectrum_sha256=spectrum.sha256,
            spectrum_path=spectrum.source_path,
        )

    @classmethod
    def for_sensor(
        cls,
        sensor: SensorConfig,
        quantity: Quantity,
        data_dir: str | os.PathLike[str] | None = None,
        spectrum_file: str = TOA_FILE,
    ) -> SolarIllumination:
        """Integrate the configured spectrum against this sensor's own R(λ)."""
        from irsim.config.loader import resolve_data_dir

        root = resolve_data_dir(data_dir)
        spectrum = load_solar_spectrum(root / spectrum_file)
        response = load_spectral_response(sensor.sensor.band.spectral_response)
        return cls.from_spectrum(spectrum, response, quantity, sensor.sensor.band.band_id)

    def incident_radiance(
        self,
        cos_incidence: Any,
        *,
        elevation_deg: Any = 90.0,
        tau_sun: Any = 1.0,
        shadow: Any = 1.0,
    ) -> NDArray[np.float64]:
        """E_B τ_sun max(0, cos θ_s) S / π -- the bundle's ``l_sun`` plane."""
        cos_s = np.asarray(cos_incidence, dtype=np.float64)
        if np.any(cos_s < -1.0 - 1e-9) or np.any(cos_s > 1.0 + 1e-9):
            raise ValueError("cos_incidence must lie in [-1, 1]")
        shade = np.asarray(shadow, dtype=np.float64)
        if np.any((shade < 0.0) | (shade > 1.0)):
            raise ValueError("shadow must lie in [0, 1] (1 = lit)")
        tau = np.asarray(tau_sun, dtype=np.float64)
        if np.any((tau < 0.0) | (tau > 1.0)):
            raise ValueError("tau_sun must lie in [0, 1]")
        el = np.asarray(elevation_deg, dtype=np.float64)
        lit = np.where(el > 0.0, 1.0, 0.0)
        return np.asarray(
            self.e_band * tau * np.maximum(cos_s, 0.0) * shade * lit / math.pi,
            dtype=np.float64,
        )
