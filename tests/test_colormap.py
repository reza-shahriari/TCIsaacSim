"""Palette lookup behavior. Not itemized in the original Phase 1 checklist --
added while implementing, alongside test_blur.py."""
import numpy as np
from thermal_physics.colormap import apply_palette


def test_white_hot_is_identity_grayscale():
    dn = np.array([[0, 128, 255]], dtype=np.uint8)
    rgb = apply_palette(dn, "white_hot")
    assert np.array_equal(rgb[..., 0], dn)
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])


def test_black_hot_inverts_white_hot():
    dn = np.array([[0, 100, 255]], dtype=np.uint8)
    white = apply_palette(dn, "white_hot")
    black = apply_palette(dn, "black_hot")
    assert np.array_equal(white[..., 0], 255 - black[..., 0])


def test_ironbow_dark_to_light():
    dn = np.array([[0, 255]], dtype=np.uint8)
    rgb = apply_palette(dn, "ironbow")
    assert rgb.shape == (1, 2, 3)
    assert rgb[0, 0].sum() < rgb[0, 1].sum()  # dark end darker than light end


def test_unknown_palette_raises():
    try:
        apply_palette(np.zeros((2, 2), dtype=np.uint8), "not_a_real_palette")
        assert False, "expected ValueError"
    except ValueError:
        pass
