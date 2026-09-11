"""Deterministic random streams for the noise model, reproducible bit-for-bit on CPU and GPU.

Two mechanisms, one contract (ADR 0022):

**Counter-based per-element hashing (the default for every Gaussian term).** Each random value is
a pure function of ``(sensor_seed, frame_index, stream, element_index)``: a 64-bit splitmix
mixer produces a uniform, and two uniforms give a normal by Box–Muller in float32. Nothing is
sequential, so

* any single pixel can be regenerated in isolation and the result is independent of traversal
  order or tiling (a GPU kernel evaluating the same integer mixer gets the same bits);
* frame 500 needs no history;
* frames, streams and sensors are independent by construction;
* ``frame_index`` is the sensor's own frame counter (``PipelineState.frame_index``), never the
  renderer's ``rtx.frameId``, which advances several times per orchestrator step.

The integer hash and the uniform are bit-exact across implementations. The normal involves
``log``/``cos``, whose last-ulp behaviour differs between libms; a CPU/GPU comparison asserts
exact equality on the uniforms and a few-ulp tolerance on the normals.

**Sequential PCG64 generators** (``noise_rng`` / ``sensor_rng``) remain for draws that have no
cheap counter-based form -- Poisson shot noise and the bad-pixel map. Those are statistically
equivalent on the GPU, not bit-identical. No global state anywhere: ``np.random.seed`` and the
legacy ``np.random.*`` API are never used.

docs/physics-model.md §13.2; sensor-noise-chain skill "Seed all noise deterministically"
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NoiseStream",
    "MAX_SEED",
    "stream_key",
    "hash_u64",
    "hash_uniform",
    "hash_normal",
    "field_normal",
    "noise_rng",
    "sensor_rng",
]

MAX_SEED = 2**63 - 1
_U64 = np.uint64
_GOLDEN = _U64(0x9E3779B97F4A7C15)
_M1 = _U64(0xBF58476D1CE4E5B9)
_M2 = _U64(0x94D049BB133111EB)
_TWO_POW_M24 = np.float32(2.0**-24)
_TWO_PI = np.float32(2.0 * np.pi)


class NoiseStream(IntEnum):
    """Enumerated random streams. Values are part of the reproducibility contract: never
    renumber."""

    TVH = 1  # per-pixel temporal (and read noise in the detector)
    TV = 2  # temporal row noise
    TH = 3  # temporal column noise
    T = 4  # frame bounce
    VH_FIXED = 10  # fixed 2-D pattern (once per sensor)
    V_FIXED = 11  # fixed row pattern
    H_FIXED = 12  # fixed column pattern
    VH_DRIFT = 20  # slow random walk of the fixed pattern
    V_DRIFT = 21
    H_DRIFT = 22
    BAD_PIXEL_MAP = 30  # defect map (once per sensor)
    RTS = 31  # random telegraph / blinking state
    SHOT = 40  # Poisson shot noise (photon detectors)
    DARK = 41  # dark-current shot noise


def _check(value: int, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not 0 <= int(value) <= MAX_SEED:
        raise ValueError(f"{name} must lie in [0, 2^63 - 1], got {value}")
    return int(value)


def _mix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """splitmix64 finaliser on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x ^ (x >> _U64(30))
        x = x * _M1
        x = x ^ (x >> _U64(27))
        x = x * _M2
        x = x ^ (x >> _U64(31))
    return x


def stream_key(sensor_seed: int, frame_index: int, stream: NoiseStream) -> int:
    """The 64-bit key of one (sensor, frame, stream) triple; element hashes derive from it."""
    s = _U64(_check(sensor_seed, "sensor_seed"))
    f = _U64(_check(frame_index, "frame_index"))
    st = _U64(int(NoiseStream(stream)))
    with np.errstate(over="ignore"):
        k = _mix64(np.asarray(s + _GOLDEN, dtype=_U64))
        k = _mix64(np.asarray((k ^ f) + _GOLDEN, dtype=_U64))
        k = _mix64(np.asarray((k ^ st) + _GOLDEN, dtype=_U64))
    return int(k)


def hash_u64(key: int, index: NDArray[np.integer] | int, lane: int = 0) -> NDArray[np.uint64]:
    """Bit-exact 64-bit hash of each element index (with a sub-lane) under ``key``."""
    idx = np.asarray(index, dtype=np.int64)
    if np.any(idx < 0):
        raise ValueError("element index must be non-negative")
    with np.errstate(over="ignore"):
        counter = (idx.astype(_U64) << _U64(2)) | _U64(lane & 3)
        return _mix64(_U64(key) + counter * _GOLDEN)


def hash_uniform(key: int, index: NDArray[np.integer] | int, lane: int = 0) -> NDArray[np.float32]:
    """Uniform in (0, 1] from the top 24 bits: (bits + 1) · 2⁻²⁴, exact in float32."""
    top = (hash_u64(key, index, lane) >> _U64(40)).astype(np.float32)
    return np.asarray((top + np.float32(1.0)) * _TWO_POW_M24, dtype=np.float32)


def hash_normal(key: int, index: NDArray[np.integer] | int) -> NDArray[np.float32]:
    """Standard normal per element by Box–Muller on lanes 0 and 1, all float32."""
    u1 = hash_uniform(key, index, 0)
    u2 = hash_uniform(key, index, 1)
    r = np.sqrt(np.float32(-2.0) * np.log(u1), dtype=np.float32)
    return np.asarray(r * np.cos(_TWO_PI * u2, dtype=np.float32), dtype=np.float32)


def field_normal(key: int, shape: tuple[int, ...]) -> NDArray[np.float32]:
    """Standard-normal field of ``shape`` with element index = flat C-order index."""
    n = int(np.prod(shape))
    return hash_normal(key, np.arange(n, dtype=np.int64)).reshape(shape)


def noise_rng(sensor_seed: int, frame_index: int, stream: NoiseStream) -> np.random.Generator:
    """Sequential PCG64 for draws with no counter-based form (Poisson). Statistically
    equivalent on the GPU only."""
    seq = np.random.SeedSequence(
        [
            _check(sensor_seed, "sensor_seed"),
            _check(frame_index, "frame_index"),
            int(NoiseStream(stream)),
        ]
    )
    return np.random.Generator(np.random.PCG64(seq))


def sensor_rng(sensor_seed: int, stream: NoiseStream) -> np.random.Generator:
    """Once-per-sensor sequential generator (bad-pixel map)."""
    seq = np.random.SeedSequence([_check(sensor_seed, "sensor_seed"), int(NoiseStream(stream))])
    return np.random.Generator(np.random.PCG64(seq))
