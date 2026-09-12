"""Aerial thermal bridge: target solver temperatures -> the G-buffer's ``temperature_k`` plane.

docs/physics-model.md §13.1, §6.5; roadmap M10.18 (phase 1); ADR 0014 (why this is a table and
not an emissive colour), ADR 0060 (the coupling), CLAUDE.md non-negotiable #6 (one weather object).

The renderer cannot carry a temperature. Every colour AOV on this build is float16 and
exposure-scaled, which at 300 K quantises to about 100 mK against a 50 mK NETD (ADR 0014), so the
surface temperature reaches the kernels the other way round: the renderer transports an exact
integer **instance id** per pixel, and this module keeps a **float32 table indexed by that id**,
filled from the M6.6 solvers that the M6.17 :class:`~irsim.scene.Scene` built on the one shared
``WeatherSeries``. Temperature is then exact at facet granularity with no quantisation anywhere.

**Thermal time and render time are different clocks.** Surface temperature moves on a scale of
minutes; frames arrive every few tens of milliseconds. Advancing a solver per frame would be
wasted work and would make the result depend on the frame rate. The solvers therefore tick on a
coarse schedule (1 Hz by default) and the bridge linearly interpolates between the bracketing
ticks at render time. For the phase-1 aerial targets the interpolation error is negligible against
the 10 mK budget -- a Newton node with tau = 900 s curves by ~3 microkelvin across a 1 s tick --
and it is bounded and reportable rather than hidden: see :meth:`AerialThermalBridge.tick_error_k`.

**The sky is not a prim.** Phase 1 renders aerial targets against sky, and ADR 0044/MS.2 already
give the apparent sky temperature as a function of elevation. Background pixels (instance id 0,
``sky_mask`` set) therefore take ``T_sky(theta)`` evaluated from each pixel's own ray direction --
no emissive dome geometry, which would only reintroduce the fp16 colour path this design exists to
avoid, and would be wrong at the horizon where the elevation gradient is steepest. A background ray
pointing *below* the horizon is not sky at all; it takes the environment preset's ground
temperature instead (see :meth:`AerialThermalBridge.background_temperature_k`).

Phase 2's full :mod:`thermal_bridge` (M10.3) replaces the per-prim solver map with a ThermalField
over a facet mesh; this module deliberately does less.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sky import SkyModel
from irsim.pipeline.environment import ground_temperature_k
from irsim.scene import Scene
from irsim_isaac.pipeline.material_ids import BACKGROUND_INSTANCE_ID, labels_to_paths

__all__ = [
    "DEFAULT_TICK_HZ",
    "TickBracket",
    "AerialThermalBridge",
    "elevation_from_rays",
]

#: Thermal tick rate. Surface temperature is a minutes-scale quantity; 1 Hz is already far finer
#: than anything the energy balance resolves, and it decouples the result from the frame rate.
DEFAULT_TICK_HZ = 1.0


@dataclass(frozen=True)
class TickBracket:
    """The two solver ticks a render time falls between, per target."""

    t_prev_s: float
    t_next_s: float
    prev_k: dict[str, float]
    next_k: dict[str, float]

    def interpolate(self, t_rel_s: float) -> dict[str, float]:
        span = self.t_next_s - self.t_prev_s
        if span <= 0.0:
            return dict(self.next_k)
        frac = (float(t_rel_s) - self.t_prev_s) / span
        frac = min(max(frac, 0.0), 1.0)
        return {
            name: self.prev_k[name] + frac * (self.next_k[name] - self.prev_k[name])
            for name in self.next_k
        }


def elevation_from_rays(ray_dirs: Any, up: Any = (0.0, 1.0, 0.0)) -> NDArray[np.float64]:
    """Elevation angle (radians) of each per-pixel ray above the horizon.

    ``ray_dirs`` are the unit vectors :func:`irsim_isaac.pipeline.gbuffer_isaac.ray_directions`
    produces -- pointing **away** from the camera -- so a ray aimed at the zenith gives +pi/2 and
    one aimed at the ground -pi/2. The sky model is a function of this angle (ADR 0044), which is
    why the sky is evaluated per pixel rather than as one number for the frame: across a 50 degree
    field the clear-sky apparent temperature changes by tens of kelvin.
    """
    d = np.asarray(ray_dirs, dtype=np.float64)
    if d.ndim != 3 or d.shape[2] != 3:
        raise ValueError(f"ray_dirs must be (H, W, 3), got {d.shape}")
    u = np.asarray(up, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        raise ValueError("up must be a non-zero vector")
    return np.asarray(np.arcsin(np.clip(np.sum(d * (u / norm), axis=2), -1.0, 1.0)))


class AerialThermalBridge:
    """Per-prim solver temperatures and the sky model, assembled into ``temperature_k``.

    ``prim_to_target`` maps a USD prim path to the name of one of ``scene.targets``. Every prim
    that is not in the map keeps the sky/background treatment if it is background and otherwise
    raises when ``strict`` -- a rendered prim with no thermal node has no temperature at all, and
    guessing one is the silent failure this project exists to avoid.
    """

    def __init__(
        self,
        scene: Scene,
        prim_to_target: Mapping[str, str],
        *,
        band: str | None = None,
        sky: SkyModel | None = None,
        tick_hz: float = DEFAULT_TICK_HZ,
    ) -> None:
        unknown = set(prim_to_target.values()) - set(scene.targets)
        if unknown:
            raise ValueError(
                f"prim_to_target names targets that the scene does not define: {sorted(unknown)}; "
                f"the scene has {sorted(scene.targets)}"
            )
        if tick_hz <= 0.0:
            raise ValueError("tick_hz must be positive")

        if sky is None and band is not None:
            sky = scene.sky_models.get(band)
        if sky is not None and sky.weather is not scene.weather:
            raise ValueError(
                "the sky model holds a different WeatherSeries than the scene "
                "(CLAUDE.md #6: one weather object, injected everywhere) -- a scene cannot run "
                "summer weather in the target solvers and winter weather in the sky"
            )

        self.scene = scene
        self.prim_to_target = dict(prim_to_target)
        self.sky = sky
        self.band = band
        self.tick_s = 1.0 / float(tick_hz)
        self._t_rel_s = 0.0

        initial = {name: float(s.temperature()) for name, s in scene.targets.items()}
        self._bracket = TickBracket(0.0, 0.0, dict(initial), dict(initial))
        self._advance_one_tick()

    # -- the thermal clock ----------------------------------------------------------------

    @property
    def t_rel_s(self) -> float:
        """Current render time, seconds since the scene start."""
        return self._t_rel_s

    @property
    def bracket(self) -> TickBracket:
        return self._bracket

    def _advance_one_tick(self) -> None:
        b = self._bracket
        t_next = b.t_next_s + self.tick_s
        new = self.scene.advance_targets(b.t_next_s, self.tick_s)
        self._bracket = TickBracket(b.t_next_s, t_next, dict(b.next_k), {**b.next_k, **new})

    def advance_to(self, t_rel_s: float) -> dict[str, float]:
        """Move render time to ``t_rel_s``, ticking the solvers only as far as needed.

        Time may not run backwards: the solvers are stateful (a Newton node integrates), so a
        rewind would silently produce a different history than a forward run of the same scene.
        """
        t = float(t_rel_s)
        if t < self._t_rel_s - 1e-9:
            raise ValueError(
                f"cannot rewind the thermal clock from {self._t_rel_s} s to {t} s: the solvers "
                "are stateful, so replaying a frame needs a fresh Scene"
            )
        while t > self._bracket.t_next_s + 1e-12:
            self._advance_one_tick()
        self._t_rel_s = t
        return self.temperatures()

    def temperatures(self) -> dict[str, float]:
        """Target temperatures interpolated to the current render time."""
        return self._bracket.interpolate(self._t_rel_s)

    def tick_error_k(self, name: str) -> float:
        """Bound on the interpolation error for one target across the current tick.

        Linear interpolation of a function with curvature ``f''`` over a step ``h`` is wrong by at
        most ``f'' h^2 / 8``. Estimated here from the bracket itself, so it is a reported number
        rather than an assumption; an exponential node with tau = 900 s and a 1 s tick gives a few
        microkelvin, which is why 1 Hz is enough for phase 1.
        """
        solver = self.scene.targets[name]
        tau = float(getattr(solver, "tau_s", 0.0) or 0.0)
        if tau <= 0.0:
            return 0.0
        delta = abs(self._bracket.next_k[name] - self._bracket.prev_k[name])
        # |f''| = |f'| / tau for an exponential, and |f'| ~ delta / tick
        return float(delta / tau * self.tick_s / 8.0)

    # -- the facet table ------------------------------------------------------------------

    def facet_temperatures(self) -> dict[str, float]:
        """Prim path -> temperature at the current render time."""
        temps = self.temperatures()
        return {path: temps[target] for path, target in self.prim_to_target.items()}

    def facet_table(
        self,
        id_to_labels: Mapping[Any, Any] | None,
        *,
        fill_k: float = 0.0,
        strict: bool = True,
    ) -> NDArray[np.float32]:
        """Float32 temperature indexed by instance id -- the array a Warp kernel binds.

        ``fill_k`` is what an id with no thermal node gets. It is 0 K rather than something
        plausible on purpose: if it ever reaches a radiance kernel the result is obviously,
        loudly wrong instead of a believable image of the wrong scene. With ``strict`` (the
        default) such an id raises here instead.
        """
        paths = labels_to_paths(id_to_labels)
        by_path = self.facet_temperatures()
        size = (max(paths) if paths else 0) + 1
        table = np.full(size, float(fill_k), dtype=np.float32)
        missing: list[str] = []
        for ident, path in paths.items():
            if ident == BACKGROUND_INSTANCE_ID:
                continue
            if path in by_path:
                table[ident] = np.float32(by_path[path])
            else:
                missing.append(path)
        if strict and missing:
            raise KeyError(
                f"rendered prims have no thermal node: {sorted(missing)}. Add them to "
                "prim_to_target or to the scene's targets -- a surface with no temperature "
                "cannot be given a plausible one."
            )
        return table

    # -- the plane ------------------------------------------------------------------------

    def temperature_plane(
        self,
        instance_ids: Any,
        id_to_labels: Mapping[Any, Any] | None,
        *,
        sky_mask: Any | None = None,
        elevation_rad: Any | None = None,
        fill_k: float = 0.0,
        strict: bool = True,
    ) -> NDArray[np.float32]:
        """The G-buffer's ``temperature_k``: facet temperatures on geometry, T_sky on the sky.

        ``elevation_rad`` is the per-pixel ray elevation (:func:`elevation_from_rays`). Without a
        sky model, or without elevations, the masked pixels keep ``fill_k`` and the caller is
        asserting it will supply the sky itself.
        """
        ids = np.asarray(instance_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"instance_ids must be an integer plane, got {ids.dtype}")
        table = self.facet_table(id_to_labels, fill_k=fill_k, strict=strict)
        clipped = np.clip(ids, 0, table.size - 1)
        plane: NDArray[np.float32] = np.asarray(table[clipped], dtype=np.float32)

        mask = None if sky_mask is None else np.asarray(sky_mask, dtype=bool)
        if mask is None:
            mask = ids == BACKGROUND_INSTANCE_ID
        if mask.any() and self.sky is not None and elevation_rad is not None:
            plane = np.asarray(
                np.where(mask, self.background_temperature_k(elevation_rad), plane),
                dtype=np.float32,
            )
        return plane

    def background_temperature_k(self, elevation_rad: Any) -> NDArray[np.float64]:
        """Apparent temperature of a pixel that hit no geometry, split at the horizon.

        A ray with positive elevation that hits nothing is looking at sky, and takes MS.2's
        ``T_sky(theta)`` (ADR 0044). A ray with **negative** elevation that hits nothing is looking
        at *ground* beyond the scene, not at sky: the sky model is only defined on [0, 90] degrees
        and extrapolating it downwards would report a cold sky where the ground is, inverting the
        contrast of anything silhouetted against it. Those pixels take ``T_ground`` from the
        environment preset's ground mode, the same quantity ADR 0045's reflected term uses -- one
        temperature for the whole ground, which is all phase 1 claims.
        """
        if self.sky is None:
            raise ValueError("this bridge has no sky model; pass band= or sky= at construction")
        t_abs = self.scene.t0_s + self._t_rel_s
        elev = np.asarray(elevation_rad, dtype=np.float64)
        out = np.full(elev.shape, float(ground_temperature_k(self.sky, t_abs)), dtype=np.float64)
        above = elev >= 0.0
        if above.any():
            out[above] = self.sky.apparent_temperature_k(t_abs, elev[above])
        return out

    def sky_temperature(self, elevation_rad: Any) -> NDArray[np.float64]:
        """MS.2's apparent sky temperature; elevations must be above the horizon."""
        if self.sky is None:
            raise ValueError("this bridge has no sky model; pass band= or sky= at construction")
        t_abs = self.scene.t0_s + self._t_rel_s
        return np.asarray(
            self.sky.apparent_temperature_k(t_abs, np.asarray(elevation_rad, dtype=np.float64))
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"AerialThermalBridge(t_rel={self._t_rel_s:.3f}s, tick={self.tick_s:.3f}s, "
            f"targets={sorted(self.scene.targets)}, prims={len(self.prim_to_target)}, "
            f"sky={'yes' if self.sky is not None else 'no'})"
        )
