"""The M11.2 illumination bundle on the Isaac render path (M10.22).

docs/physics-model.md §5.2, §5.4, §5.5; ADR 0063 (the bundle), 0064 (solar), 0065 (night).

**Why this module exists.** M11.3 and M11.4 built the reflected-sunlight and night-sky terms, and
M11.2 built the bundle that carries them; `run_frame` reads them out of the plane dict as
``l_sun`` and ``l_night``. Nothing in the engine glue ever wrote those planes. Every Isaac render
this project has produced was therefore **emission only** -- correct for LWIR, where reflected
sunlight is 0.35 % of the signal, and catastrophically wrong for SWIR or NIR, where it is
essentially all of it. A NIR frame rendered without this module is black.

That is the same failure ADR 0077 and ADR 0082 record: a mechanism exists, is unit-tested, and
nothing calls it. This module is the call.

**The sun is the renderer's sun.** The direction here is the same vector
:func:`irsim_isaac.stage.add_sky_dome` aims the ``DistantLight`` along -- NOAA geometry at the
scene's own site and clock, converted to stage axes by the same function. If they were computed
separately, the shadows in the companion visible frame would fall away from the sun the
radiometry was using, and nobody would notice until the two were overlaid.

**What is not modelled: cast shadows.** ``shadow`` is 1 everywhere, so the only shadowing is
*self*-shadowing through max(0, n·s): a surface turned away from the sun gets nothing, a surface
facing it gets all of it, and a surface standing in another object's shadow is lit anyway. For the
aerial scenes this is exact -- there is nothing above a drone to shadow it. For a vessel it is
optimistic on the shaded side of the superstructure. Making it right needs an occlusion query
along the sun direction, which the renderer on this build does not provide (ADR 0014 addendum);
the honest interim is to say so rather than to render a plausible shadow nobody computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.extinction import solar_transmittance
from irsim.config.bands import enabled_illumination_terms
from irsim.config.sensor import SensorConfig
from irsim.pipeline.night import NightIllumination
from irsim.pipeline.solar import SolarIllumination
from irsim.radiometry.lut import Quantity

__all__ = ["SceneIllumination", "sun_direction_stage", "sun_cos_incidence"]


def sun_direction_stage(
    elevation_deg: float, azimuth_deg: float, heading_deg: float
) -> NDArray[np.float64]:
    """Unit vector **toward** the sun in stage axes (+X right, +Y up, -Z forward).

    The same expression :func:`irsim_isaac.stage.add_sky_dome` uses to aim the distant light, so
    the radiometry and the rendered shadows cannot disagree about where the sun is. ``heading_deg``
    is the compass bearing the stage's -Z points along, which is what ties the stage's arbitrary
    world frame to a compass azimuth.
    """
    el = math.radians(float(elevation_deg))
    az = math.radians(float(azimuth_deg) - float(heading_deg))
    return np.array(
        [math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)],
        dtype=np.float64,
    )


def sun_cos_incidence(normal_world: Any, direction: NDArray[np.float64]) -> NDArray[np.float64]:
    """max(0, n·s) per pixel, for unit normals already oriented to the viewer.

    Clamped at zero rather than kept signed: a negative cosine is a surface facing away from the
    sun, which receives no direct beam at all, and letting it through as a negative radiance would
    *subtract* sunlight from the reflected term.
    """
    n = np.asarray(normal_world, dtype=np.float64)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(f"normal_world must be (H, W, 3), got {n.shape}")
    return np.asarray(np.maximum(np.einsum("ijk,k->ij", n, direction), 0.0), dtype=np.float64)


@dataclass(frozen=True)
class SceneIllumination:
    """Builds a frame's ``l_sun`` and ``l_night`` planes for one camera on one stage.

    ``None`` from :meth:`for_camera` means this band's regime enables neither term, which is the
    LWIR case: building the planes and letting :meth:`Illumination.for_regime` drop them would
    give the same picture, and not building them keeps an LWIR render bit-identical to what it was
    before this module existed.
    """

    solar: SolarIllumination | None
    night: NightIllumination | None
    band_id: str
    quantity: Quantity
    heading_deg: float
    #: Band transmittance of the whole atmosphere along the *solar* slant path, as a function of
    #: the sun's zenith angle. Held as a callable so the preset is consulted once per frame and
    #: the atmosphere module stays the only place that knows the extinction law.
    tau_sun_for_zenith: Any

    @classmethod
    def for_camera(
        cls,
        sensor: SensorConfig,
        scene: Any,
        quantity: Quantity,
        *,
        heading_deg: float = 0.0,
        data_dir: Any = None,
    ) -> SceneIllumination | None:
        """Assemble from the scene's own atmosphere preset and the sensor's own R(λ).

        Returns ``None`` for a band whose regime enables no source term.
        """
        band = sensor.sensor.band
        terms = enabled_illumination_terms(band.regime)
        if not ({"solar", "night"} & set(terms)):
            return None
        solar = (
            SolarIllumination.for_sensor(sensor, quantity, data_dir=data_dir)
            if "solar" in terms
            else None
        )
        # The night terms are an *environment* setting (airglow level, moon phase, cloud
        # coupling), so a scene with no environment preset gets no night term rather than a
        # default one: §5.5's levels vary by two orders of magnitude between sites and nights,
        # and a made-up default would be a number nobody chose.
        night = (
            NightIllumination.for_sensor(sensor, scene.environment, quantity, data_dir=data_dir)
            if "night" in terms and scene.environment is not None
            else None
        )
        if solar is None and night is None:
            return None
        preset = scene.atmosphere_preset
        band_id = band.band_id

        def tau(zenith_rad: float) -> float:
            return float(solar_transmittance(preset, band_id, zenith_rad))

        return cls(
            solar=solar,
            night=night,
            band_id=band_id,
            quantity=quantity,
            heading_deg=float(heading_deg),
            tau_sun_for_zenith=tau,
        )

    def planes(
        self,
        normal_world: Any,
        *,
        sun_elevation_deg: float,
        sun_azimuth_deg: float,
        sky_mask: Any = None,
        cloud_fraction: float = 0.0,
        moon_phase_fraction: float | None = None,
        moon_elevation_deg: float = 0.0,
    ) -> dict[str, NDArray[np.float64]]:
        """``{"l_sun": ..., "l_night": ...}`` for this frame; keys absent when the term is off.

        Sky pixels are set to zero in both planes. A sky pixel's radiance comes from the sky model
        and the atmosphere, not from a surface reflecting anything, and leaving an incident term
        on it would have stage 1 reflect sunlight off the sky.
        """
        n = np.asarray(normal_world, dtype=np.float64)
        shape = n.shape[:2]
        out: dict[str, NDArray[np.float64]] = {}
        mask = None if sky_mask is None else np.asarray(sky_mask, dtype=bool)
        if mask is not None and mask.shape != shape:
            raise ValueError(f"sky_mask {mask.shape} does not match the normal plane {shape}")

        if self.solar is not None:
            direction = sun_direction_stage(sun_elevation_deg, sun_azimuth_deg, self.heading_deg)
            zenith = math.radians(90.0 - float(sun_elevation_deg))
            tau = self.tau_sun_for_zenith(zenith) if sun_elevation_deg > 0.0 else 0.0
            l_sun = self.solar.incident_radiance(
                sun_cos_incidence(n, direction),
                elevation_deg=sun_elevation_deg,
                tau_sun=tau,
                shadow=1.0,  # no cast shadows on this build -- see the module docstring
            )
            out["l_sun"] = np.asarray(l_sun, dtype=np.float64)

        if self.night is not None:
            kwargs: dict[str, Any] = {"cloud_fraction": cloud_fraction}
            if moon_phase_fraction is not None:
                kwargs["phase_fraction"] = moon_phase_fraction
                kwargs["moon_elevation_deg"] = moon_elevation_deg
            value = float(np.asarray(self.night.incident_radiance(**kwargs)).reshape(()))
            out["l_night"] = np.full(shape, value, dtype=np.float64)

        for key, plane in out.items():
            if mask is not None:
                out[key] = np.where(mask, 0.0, plane)
        return out
