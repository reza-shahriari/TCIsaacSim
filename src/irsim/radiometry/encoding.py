"""Temperature encoding for the render G-buffer.

The renderer does not carry kelvin: it carries ``c = (T - T_ref) / T_span`` in a **float32** AOV,
with ``T_ref = 200 K`` and ``T_span = 800 K`` (:mod:`irsim.radiometry.constants`). This module is
the single definition of that contract, shared by the CPU reference path, the Warp/SPG kernels and
the Isaac adapter, so the encoder and decoder can never drift apart.

Why the fuss: at 300 K, float16 has a spacing of 2^8 * 2^-10 = 0.25 K, five times coarser than a
50 mK NETD. A float16 buffer anywhere on this path destroys the sensor's sensitivity while still
producing an image that looks fine. Encoding into [0, 1] does **not** rescue float16 (spacing near
c = 0.125 is 2^-14 -> 0.05 K; near c = 1 it is 2^-11 -> 0.4 K); only float32 does. Both functions
therefore refuse float16 (and integer/unorm) inputs. The negative controls in
tests/unit/test_temperature_encoding.py measure exactly these fp16 errors.

If a renderer offers no float32 target, the §13.3 fallback splits ``c`` into a coarse and a fine
fp16 channel (:func:`encode_temperature_fp16_pair`); that codec round-trips under 10 mK too.

docs/physics-model.md §13.3, §3.2 (precision trap); CLAUDE.md non-negotiable #2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import T_ENCODE_REF_K, T_ENCODE_SPAN_K

__all__ = [
    "encode_temperature",
    "decode_temperature",
    "encode_temperature_fp16_pair",
    "decode_temperature_fp16_pair",
    "FP16_PAIR_COARSE_STEPS",
]

Float32Array = NDArray[np.float32]

# Coarse channel quantises c to 1/256: multiples of 2^-8 in [0, 1] are exact in fp16, and the fine
# remainder (scaled to [0, 1)) then carries 2^-11 relative -> 800 K * 2^-11 / 256 = 1.5 mK.
FP16_PAIR_COARSE_STEPS = 256


def _require_fp32_or_better(x: object, what: str) -> NDArray[np.floating]:
    """Accept float32/float64 arrays or Python floats; refuse float16 and integer buffers.

    Integer (unorm) buffers are refused because an 8-bit unorm over 200-1000 K is 3.1 K per code.
    """
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError(
            f"{what} is float16: at 300 K its spacing is 0.25 K, five times a 50 mK NETD "
            "(CLAUDE.md non-negotiable #2). Use float32 or better."
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(
            f"{what} has dtype {arr.dtype}; temperature and encoded temperature must be float32 "
            "or float64 (an integer/unorm buffer is 3 K per code over 200-1000 K)."
        )
    return arr


def encode_temperature(temperature_k: object) -> Float32Array:
    """Kelvin -> encoded ``c = (T - 200) / 800`` as float32.

    The arithmetic is done in float64 and cast once, so the only rounding is the final float32
    representation of ``c`` (about 6e-8 relative -> < 0.05 mK anywhere in 200-1000 K).

    docs/physics-model.md §13.3
    """
    t = _require_fp32_or_better(temperature_k, "temperature_k").astype(np.float64)
    c = (t - T_ENCODE_REF_K) / T_ENCODE_SPAN_K
    return np.asarray(c, dtype=np.float32)


def decode_temperature(encoded: object) -> Float32Array:
    """Encoded ``c`` (float32 AOV) -> kelvin as float32. Inverse of :func:`encode_temperature`.

    docs/physics-model.md §13.3
    """
    c = _require_fp32_or_better(encoded, "encoded temperature").astype(np.float64)
    t = c * T_ENCODE_SPAN_K + T_ENCODE_REF_K
    return np.asarray(t, dtype=np.float32)


def encode_temperature_fp16_pair(
    temperature_k: object,
) -> tuple[NDArray[np.float16], NDArray[np.float16]]:
    """The §13.3 fallback for renderers with no float32 target: ``c`` split into two fp16 channels.

    ``coarse = floor(c * 256) / 256`` (exact in fp16) and ``fine = c * 256 - floor(c * 256)`` in
    [0, 1). Values outside [0, 1] are clamped (200-1000 K covers every scene this project models).
    Returns ``(coarse, fine)``; store them in the R and G channels of the fp16 target.
    """
    t = _require_fp32_or_better(temperature_k, "temperature_k").astype(np.float64)
    c = np.clip((t - T_ENCODE_REF_K) / T_ENCODE_SPAN_K, 0.0, 1.0)
    scaled = c * FP16_PAIR_COARSE_STEPS
    coarse_steps = np.floor(scaled)
    fine = scaled - coarse_steps
    coarse = coarse_steps / FP16_PAIR_COARSE_STEPS
    return coarse.astype(np.float16), fine.astype(np.float16)


def decode_temperature_fp16_pair(
    coarse: NDArray[np.float16], fine: NDArray[np.float16]
) -> Float32Array:
    """Inverse of :func:`encode_temperature_fp16_pair`; output is float32 kelvin."""
    c = (
        np.asarray(coarse, dtype=np.float64)
        + np.asarray(fine, dtype=np.float64) / FP16_PAIR_COARSE_STEPS
    )
    return np.asarray(c * T_ENCODE_SPAN_K + T_ENCODE_REF_K, dtype=np.float32)
