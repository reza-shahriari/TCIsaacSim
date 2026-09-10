"""Non-negotiable #2 at the encode/decode boundary (docs/physics-model.md §13.3, §15 Tier 1).

The positive test alone proves nothing -- a round trip through *any* dtype "passes" if the
tolerance is loose enough. The negative controls establish that the same measurement detects the
failure mode it exists for: raw kelvin, or even encoded c, forced through float16 lands far
outside 10 mK. If a negative control ever passes the 10 mK bound, the round-trip test has lost
its sensitivity and must be redesigned before anything else is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.radiometry.constants import (
    LUT_DT_K,
    LUT_N,
    LUT_T_MAX_K,
    LUT_T_MIN_K,
    T_ENCODE_REF_K,
    T_ENCODE_SPAN_K,
)
from irsim.radiometry.encoding import (
    decode_temperature,
    decode_temperature_fp16_pair,
    encode_temperature,
    encode_temperature_fp16_pair,
)

RAMP = np.arange(200.0, 1000.0 + 1e-9, 0.01)  # 80 001 points, off any fp32 grid
TOL_MK = 10.0  # 1/5 of a 50 mK NETD (ir-sim-testing skill)


def _max_err_mk(decoded: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(decoded.astype(np.float64) - truth)) * 1e3)


def test_constants_are_the_spec_values() -> None:
    assert T_ENCODE_REF_K == 200.0 and T_ENCODE_SPAN_K == 800.0
    assert (LUT_T_MIN_K, LUT_T_MAX_K, LUT_DT_K) == (200.0, 1000.0, 0.05)
    assert LUT_N == round((LUT_T_MAX_K - LUT_T_MIN_K) / LUT_DT_K) + 1 == 16001
    assert encode_temperature(np.float64(200.0)) == 0.0
    assert encode_temperature(np.float64(1000.0)) == 1.0


def test_fp32_round_trip_under_10mK() -> None:
    c = encode_temperature(RAMP)
    assert c.dtype == np.float32
    out = decode_temperature(c)
    assert out.dtype == np.float32
    err = _max_err_mk(out, RAMP)
    assert err < TOL_MK, f"fp32 round trip max error {err:.3f} mK"
    # The floor is float32 representation: c near 1 has spacing 6e-8 (0.024 mK half-step) and a
    # 1000 K float32 output has spacing 0.061 mK (0.03 mK half-step); measured 0.054 mK in total.
    assert err < 0.1, f"expected ~0.05 mK from float32 representation, got {err:.3f} mK"


def test_negative_control_raw_kelvin_through_fp16() -> None:
    """Raw temperature through an fp16 buffer: >= 0.1 K error at 300 K (0.25 K spacing, §3.2)."""
    t = np.linspace(295.0, 305.0, 10_001)
    err = _max_err_mk(t.astype(np.float16).astype(np.float32), t)
    assert err >= 100.0, f"fp16 kelvin error only {err:.1f} mK -- the control has lost its teeth"
    assert err == pytest.approx(125.0, rel=0.05), "expected half the 0.25 K spacing"


def test_negative_control_encoded_c_through_fp16() -> None:
    """Encoding does not rescue fp16: >= 40 mK at 300 K and >= 150 mK over 800-1000 K."""
    near_300 = np.linspace(295.0, 305.0, 10_001)
    err_300 = _max_err_mk(
        decode_temperature(encode_temperature(near_300).astype(np.float16).astype(np.float32)),
        near_300,
    )
    assert err_300 >= 40.0, f"{err_300:.1f} mK"
    hot = np.linspace(800.0, 1000.0, 20_001)
    err_hot = _max_err_mk(
        decode_temperature(encode_temperature(hot).astype(np.float16).astype(np.float32)), hot
    )
    assert err_hot >= 150.0, f"{err_hot:.1f} mK"
    assert err_300 > TOL_MK and err_hot > TOL_MK


@pytest.mark.parametrize(
    "fn", [encode_temperature, decode_temperature, encode_temperature_fp16_pair]
)
def test_float16_input_raises(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TypeError, match="float16"):
        fn(np.array([300.0], dtype=np.float16))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
def test_integer_unorm_input_raises(dtype: type) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        encode_temperature(np.array([128], dtype=dtype))
    with pytest.raises(TypeError, match="float32 or float64"):
        decode_temperature(np.array([128], dtype=dtype))


def test_float64_input_accepted_and_output_is_float32() -> None:
    out = decode_temperature(encode_temperature(np.array([300.0], dtype=np.float64)))
    assert out.dtype == np.float32 and out[0] == pytest.approx(300.0, abs=1e-4)


def test_fp16_pair_codec_round_trip_under_10mK() -> None:
    """The §13.3 fallback. Channels are genuinely fp16; the pair still lands under 10 mK,
    while a single fp16 channel of the same c fails the bound (previous negative control)."""
    coarse, fine = encode_temperature_fp16_pair(RAMP)
    assert coarse.dtype == np.float16 and fine.dtype == np.float16
    out = decode_temperature_fp16_pair(coarse, fine)
    assert out.dtype == np.float32
    err = _max_err_mk(out, RAMP)
    assert err < TOL_MK, f"fp16-pair round trip max error {err:.3f} mK"
    # coarse channel is exactly representable: decoding coarse alone is a clean 3.125 K staircase
    staircase = decode_temperature_fp16_pair(coarse, np.zeros_like(fine))
    assert np.all(staircase <= RAMP.astype(np.float32) + 1e-3)
    assert _max_err_mk(staircase, RAMP) == pytest.approx(3125.0, rel=0.01)


def test_fp16_pair_clamps_outside_range() -> None:
    coarse, fine = encode_temperature_fp16_pair(np.array([100.0, 1500.0]))
    out = decode_temperature_fp16_pair(coarse, fine)
    assert out[0] == pytest.approx(200.0, abs=1e-3) and out[1] == pytest.approx(1000.0, abs=1e-3)
