"""Stage 5 — the correlated 3-D noise on top of the detector's per-pixel noise (ADR 0026).

The detector response already carries σ_TVH (Poisson shot + Gaussian read for photon FPAs, the
anchored Gaussian for bolometers) and reports it per pixel as ``sigma_dn``. This stage adds the
six **correlated** NVESD components in the same pre-quantisation unit (DN) with σ_i = r_i · σ_TVH
from the configured ``noise.ratios_3d`` (§10.2), using the frame mean of ``sigma_dn`` as σ_TVH
(row/column/frame noise is readout-driven and scene-independent, so a scalar is the right
scale; for bolometers ``sigma_dn`` is constant anyway). Fixed patterns are unit-variance fields
generated once per sensor and scaled at apply time; the temporal terms come from the per-frame
counter-based streams, so any frame is reproducible in isolation (ADR 0022). Drift of the fixed
patterns (§10.3), NUC residual and bad pixels are M9. Quantisation happens after this stage.

docs/physics-model.md §10.2, §13.4 SPG#5, §16.4 step 4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import SensorSpec
from irsim.detector.response import Detector
from irsim.noise.three_d import FixedPattern, Sigmas7, synthesize_frame

__all__ = ["NoiseStage", "measure_from_uniform_scene"]

Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class NoiseStage:
    """Correlated 3-D noise for one sensor: ratios (RATIO_ORDER), unit fixed patterns, seed."""

    sensor_seed: int
    shape: tuple[int, int]
    ratios: tuple[float, ...]
    unit_fixed: FixedPattern
    enabled: bool = True

    @classmethod
    def from_sensor(cls, sensor: SensorSpec, sensor_seed: int, enabled: bool = True) -> NoiseStage:
        ratios = sensor.noise.sigma_ratios()
        shape = sensor.fpa_shape
        # fixed patterns with sigma_TVH = 1: scaled by the actual sigma at apply time
        unit = FixedPattern.generate(shape, Sigmas7.from_ratios(1.0, ratios), sensor_seed)
        return cls(
            sensor_seed=sensor_seed, shape=shape, ratios=ratios, unit_fixed=unit, enabled=enabled
        )

    def sigmas(self, sigma_tvh: float) -> Sigmas7:
        """The six correlated sigmas for this σ_TVH; the TVH slot is zero (the detector owns it)."""
        s = Sigmas7.from_ratios(sigma_tvh, self.ratios)
        return Sigmas7(t=s.t, v=s.v, h=s.h, tv=s.tv, th=s.th, vh=s.vh, tvh=0.0)

    def apply(
        self,
        signal_dn: Float32Array,
        sigma_dn: Float32Array,
        frame_index: int,
        fixed_override: FixedPattern | None = None,
    ) -> Float32Array:
        """Add the correlated components to the detector's signal (DN units, float32).

        ``fixed_override`` replaces this stage's own unit fixed pattern for one frame. M9.4's
        drift owns a *breathing* pattern, so on a wired chain the pattern changes every frame and
        the stage must be told which one to use rather than holding a stale copy. It is still in
        sigma_TVH = 1 units and is scaled here exactly like the static one.
        """
        signal = np.asarray(signal_dn)
        if signal.dtype != np.float32:
            raise TypeError(f"signal_dn must be float32, got {signal.dtype}")
        if signal.shape != self.shape:
            raise ValueError(f"signal shape {signal.shape} != stage shape {self.shape}")
        if not self.enabled:
            return signal
        sigma_tvh = float(np.asarray(sigma_dn, dtype=np.float64).mean())
        if sigma_tvh <= 0.0:
            return signal
        sig = self.sigmas(sigma_tvh)
        scale = np.float32(sigma_tvh)
        unit = self.unit_fixed if fixed_override is None else fixed_override
        if unit.shape != self.shape:
            raise ValueError(f"fixed pattern shape {unit.shape} != stage shape {self.shape}")
        fixed = FixedPattern(
            v=(unit.v * scale).astype(np.float32),
            h=(unit.h * scale).astype(np.float32),
            vh=(unit.vh * scale).astype(np.float32),
        )
        return np.asarray(
            signal + synthesize_frame(self.shape, sig, fixed, self.sensor_seed, frame_index),
            dtype=np.float32,
        )


def measure_from_uniform_scene(
    detector: Detector,
    stage: NoiseStage,
    flux: NDArray[np.floating],
    n_frames: int,
    first_frame: int = 0,
) -> tuple[NDArray[np.float32], float]:
    """Run ``n_frames`` of a uniform flux through detector + stage; return the (T, V, H) cube of
    pre-quantisation signals and the σ_TVH (DN) the stage used -- the Tier 2 3-D noise bench."""
    if n_frames < 2:
        raise ValueError("need at least two frames")
    frames = []
    sigma_tvh = 0.0
    for f in range(first_frame, first_frame + n_frames):
        out = detector.response(flux, f, stage.sensor_seed)
        sigma_tvh = float(out.sigma_dn.astype(np.float64).mean())
        frames.append(stage.apply(out.signal_dn, out.sigma_dn, f))
    return np.stack(frames).astype(np.float32), sigma_tvh
