"""Display-branch assembly (M5.5, ADR 0031): identity settings on a ramp, the exact `agc: none`
shift, the config hash, dtype boundaries, and the operator order."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import IspSpec, SensorConfig
from irsim.isp import agc_none, isp_config_hash, run_display_branch

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _isp(**over: Any) -> IspSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["isp"].update(over)
    return SensorConfig.model_validate(d).sensor.isp


def test_identity_settings_on_a_ramp_is_the_rank_map() -> None:
    dn = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    out = run_display_branch(
        dn,
        _isp(
            agc="linear",
            clip_percentiles=[0.0, 1.0],
            gamma=1.0,
            dde_gain=0.0,
            palette="gray",
            polarity="white_hot",
        ),
        16,
    )
    rank = dn.astype(np.float64) / (dn.size - 1)
    expected = np.rint(255.0 * rank)
    assert np.max(np.abs(out.display8[..., 0].astype(np.float64) - expected)) <= 1.0
    assert out.display8.shape == (256, 256, 4) and out.display8.dtype == np.uint8
    assert out.y.dtype == np.float32


@pytest.mark.parametrize(("bits", "shift"), [(16, 8), (14, 6)])
def test_agc_none_is_the_exact_bit_shift(bits: int, shift: int) -> None:
    rng = np.random.default_rng(1)
    dn = rng.integers(0, 2**bits, size=(32, 32)).astype(np.uint16)
    out = run_display_branch(dn, _isp(agc="none", gamma=1.0, dde_gain=0.0, palette="gray"), bits)
    assert np.array_equal(out.display8[..., 0].astype(np.int64), dn.astype(np.int64) >> shift)
    assert np.array_equal(
        agc_none(dn, bits), ((dn >> shift).astype(np.float32) / np.float32(255.0))
    )


def test_hash_changes_with_plateau_or_palette() -> None:
    base = isp_config_hash(_isp(), 16)
    assert isp_config_hash(_isp(plateau=0.02), 16) != base
    assert isp_config_hash(_isp(palette="ironbow"), 16) != base
    assert isp_config_hash(_isp(), 14) != base
    assert isp_config_hash(_isp(), 16) == base
    assert run_display_branch(np.zeros((4, 4), np.uint16), _isp(), 16).isp_hash == base


def test_dtype_boundaries() -> None:
    with pytest.raises(TypeError, match="uint16"):
        run_display_branch(np.zeros((4, 4), np.float16), _isp(), 16)
    with pytest.raises(TypeError, match="uint16"):
        run_display_branch(np.zeros((4, 4), np.float32), _isp(), 16)
    with pytest.raises(ValueError):
        run_display_branch(np.full((4, 4), 20000, np.uint16), _isp(), 14)


def test_operator_order_gamma_before_dde_and_polarity_last() -> None:
    """A vertical step: DDE overshoot sits on the gamma-mapped step (order AGC -> gamma -> DDE),
    and black-hot is applied last (the halo flips with the polarity)."""
    dn = np.full((16, 16), 10000, np.uint16)
    dn[:, 8:] = 30000
    isp = _isp(
        agc="linear",
        clip_percentiles=[0.0, 1.0],
        gamma=2.0,
        dde_gain=0.6,
        palette="gray",
        polarity="white_hot",
    )
    y = run_display_branch(dn, isp, 16).y.astype(np.float64)
    # after linear AGC the step is 0 -> 1 (percentiles at the ends), gamma 2 keeps 0/1 fixed;
    # the DDE overshoot next to the edge is +0.1 on the high side, clipped at 1 -> 1.0, and the
    # undershoot -0.1 on the low side clipped at 0 -> 0.0; the halo shows on a mid-grey step:
    dn2 = np.full((16, 16), 20000, np.uint16)
    dn2[:, 8:] = 40000
    isp2 = _isp(agc="none", gamma=2.0, dde_gain=0.6, palette="gray")
    y2 = run_display_branch(dn2, isp2, 16).y.astype(np.float64)
    lo, hi = np.sqrt(20000 / 256 // 1 / 255), np.sqrt(40000 / 256 // 1 / 255)
    assert abs(y2[8, 7] - (lo - 0.6 * (hi - lo) / 3)) < 1e-6
    assert abs(y2[8, 8] - (hi + 0.6 * (hi - lo) / 3)) < 1e-6
    bh = run_display_branch(
        dn2, _isp(agc="none", gamma=2.0, dde_gain=0.6, palette="gray", polarity="black_hot"), 16
    )
    wh = run_display_branch(dn2, isp2, 16)
    assert np.array_equal(bh.display8[..., 0].astype(int), 255 - wh.display8[..., 0].astype(int))
    assert y.shape == (16, 16)
