"""Polarity, palettes and 8-bit quantisation (M5.4, ADR 0030)."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.isp.palette import PALETTES, palette_table, quantise_display, to_display8


def _luma(table: np.ndarray) -> np.ndarray:
    t = table.astype(np.float64)
    return 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]


def test_gray_white_hot_exact() -> None:
    y = (np.arange(1024) / 1023.0).astype(np.float32).reshape(32, 32)
    out = to_display8(y, "white_hot", "gray")
    expected = np.rint(255.0 * y.astype(np.float64)).astype(np.uint8)
    assert out.shape == (32, 32, 4) and out.dtype == np.uint8
    for c in range(3):
        assert np.array_equal(out[..., c], expected)
    assert np.all(out[..., 3] == 255)


def test_black_hot_is_complement() -> None:
    y = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    for name in PALETTES:
        wh = to_display8(y, "white_hot", name)
        bh = to_display8(y, "black_hot", name)
        dn8 = quantise_display(y)
        assert np.array_equal(bh[..., :3], palette_table(name)[255 - dn8])
        if name == "gray":
            assert np.array_equal(bh[..., 0].astype(int), 255 - wh[..., 0].astype(int))


def test_palette_tables_endpoints_luma_and_distinctness() -> None:
    for name, table in PALETTES.items():
        assert table.shape == (256, 3) and table.dtype == np.uint8, name
    g = palette_table("gray")
    assert tuple(g[0]) == (0, 0, 0) and tuple(g[255]) == (255, 255, 255)
    for name in ("ironbow", "lava", "gray"):
        assert np.all(np.diff(_luma(palette_table(name))) >= -1e-9), f"{name} luma not monotone"
    rainbow = palette_table("rainbow")
    assert len({tuple(r) for r in rainbow.tolist()}) == 256
    with pytest.raises(ValueError):
        palette_table("viridis")


def test_quantisation_half_lsb_and_no_wrap() -> None:
    y = np.linspace(0, 1, 5001, dtype=np.float32)
    dn8 = quantise_display(y)
    assert np.all(np.abs(255.0 * y.astype(np.float64) - dn8) <= 0.5 + 1e-9)
    assert quantise_display(np.array([-0.01, 1.01, 0.5])).tolist() == [0, 255, 128]
    with pytest.raises(TypeError, match="float16"):
        quantise_display(np.zeros(3, np.float16))
    with pytest.raises(ValueError):
        to_display8(y[:4], "warm_hot", "gray")  # type: ignore[arg-type]
