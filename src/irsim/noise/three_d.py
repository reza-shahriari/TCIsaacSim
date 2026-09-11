"""Reference synthesis of the seven NVESD 3-D noise components (docs/physics-model.md §10.2).

    n[t, v, h] = T[t] + V[v] + H[h] + TV[t, v] + TH[t, h] + VH[v, h] + TVH[t, v, h]

Each term is Gaussian with its own σ. The fixed terms (V, H, VH) are drawn once per sensor; the
temporal terms (T, TV, TH, TVH) are drawn per frame from the counter-based streams of
:mod:`irsim.noise.seeding`, so a frame is reproducible in isolation. The synthesiser is
**unit-agnostic**: the caller passes σ in the unit of the signal it will be added to (radiance
for the bolometer chain, electrons for photon detectors) -- never kelvin (non-negotiable #3).
Drift of the fixed patterns (§10.3 "pattern breathing") is M9.4; this module is purely additive.

Directional structure, not magnitude, is what makes thermal imagery look thermal (§10.2), so the
axis conventions are tested: V is a per-row value (constant along h), H a per-column value
(constant along v).

docs/physics-model.md §10.2, §10.1 (FPN row)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import RATIO_ORDER
from irsim.noise.seeding import NoiseStream, field_normal, hash_normal, stream_key

__all__ = ["Sigmas7", "FixedPattern", "synthesize_frame", "FIXED_PATTERN_FRAME_INDEX"]

# Fixed patterns are per sensor, not per frame: their streams are keyed with this frame index.
FIXED_PATTERN_FRAME_INDEX = 0

Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class Sigmas7:
    """The seven component standard deviations in one physical unit, order = RATIO_ORDER."""

    t: float
    v: float
    h: float
    tv: float
    th: float
    vh: float
    tvh: float

    def __post_init__(self) -> None:
        for name in RATIO_ORDER:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"sigma {name} must be finite and non-negative, got {value}")

    @classmethod
    def from_ratios(cls, sigma_tvh: float, ratios: tuple[float, ...]) -> Sigmas7:
        """σ_i = r_i · σ_TVH for the ratio vector in RATIO_ORDER (§10.2 'in NETD units')."""
        if len(ratios) != 7:
            raise ValueError("expected seven ratios in RATIO_ORDER")
        if sigma_tvh < 0.0:
            raise ValueError("sigma_tvh must be non-negative")
        return cls(*(float(r) * float(sigma_tvh) for r in ratios))

    def as_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, k)) for k in RATIO_ORDER)

    @property
    def total(self) -> float:
        """σ_total = √Σσ² (§10.2)."""
        return float(np.sqrt(sum(s * s for s in self.as_vector())))


@dataclass(frozen=True)
class FixedPattern:
    """The per-sensor fixed terms: V[h_rows], H[w_cols], VH[h, w], float32, unit of the signal."""

    v: Float32Array
    h: Float32Array
    vh: Float32Array

    @classmethod
    def generate(cls, shape: tuple[int, int], sigmas: Sigmas7, sensor_seed: int) -> FixedPattern:
        rows, cols = shape
        f = FIXED_PATTERN_FRAME_INDEX
        v = np.float32(sigmas.v) * hash_normal(
            stream_key(sensor_seed, f, NoiseStream.V_FIXED), np.arange(rows)
        )
        h = np.float32(sigmas.h) * hash_normal(
            stream_key(sensor_seed, f, NoiseStream.H_FIXED), np.arange(cols)
        )
        vh = np.float32(sigmas.vh) * field_normal(
            stream_key(sensor_seed, f, NoiseStream.VH_FIXED), shape
        )
        return cls(v=v.astype(np.float32), h=h.astype(np.float32), vh=vh.astype(np.float32))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.v.size), int(self.h.size))


def synthesize_frame(
    shape: tuple[int, int],
    sigmas: Sigmas7,
    fixed: FixedPattern,
    sensor_seed: int,
    frame_index: int,
) -> Float32Array:
    """One frame of 3-D noise, float32, in the unit of ``sigmas``. Purely additive."""
    rows, cols = shape
    if fixed.shape != (rows, cols):
        raise ValueError(f"fixed pattern shape {fixed.shape} != frame shape {shape}")
    k_t = stream_key(sensor_seed, frame_index, NoiseStream.T)
    k_tv = stream_key(sensor_seed, frame_index, NoiseStream.TV)
    k_th = stream_key(sensor_seed, frame_index, NoiseStream.TH)
    k_tvh = stream_key(sensor_seed, frame_index, NoiseStream.TVH)
    frame = np.zeros(shape, dtype=np.float32)
    frame += fixed.v[:, None]
    frame += fixed.h[None, :]
    frame += fixed.vh
    # temporal terms; a zero sigma skips the draw (no behaviour change, saves the hashing)
    if sigmas.t > 0:
        frame += np.float32(sigmas.t) * hash_normal(k_t, np.array([0]))[0]
    if sigmas.tv > 0:
        frame += (np.float32(sigmas.tv) * hash_normal(k_tv, np.arange(rows)))[:, None]
    if sigmas.th > 0:
        frame += (np.float32(sigmas.th) * hash_normal(k_th, np.arange(cols)))[None, :]
    if sigmas.tvh > 0:
        frame += np.float32(sigmas.tvh) * field_normal(k_tvh, shape)
    out = np.asarray(frame, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("synthesised noise is not finite")
    return out
