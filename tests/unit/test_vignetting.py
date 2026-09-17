"""Field angle and cos⁴ vignetting (M3.2, ADR 0015): Boson known answers, symmetry, an independent
per-pixel derivation, and supersample consistency."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.optics.vignetting import (
    cos4_at_radius,
    cos4_field,
    field_angle_map,
    field_radius_map_mm,
    format_corner_cos4,
)

W, H, PITCH, F = 640, 512, 12.0, 14.0  # Boson 640 with the 14 mm lens


def test_boson_format_corner_and_edge_known_answers() -> None:
    """At the format corner (3.84, 3.072 mm) cos⁴ = 0.79240; at the horizontal edge centre
    (3.84, 0) 0.86496 -- the roadmap's hand values, to 1e-4."""
    assert format_corner_cos4(W, H, PITCH, F) == pytest.approx(0.79240, abs=1e-4)
    assert float(cos4_at_radius(W * PITCH * 1e-3 / 2.0, F)) == pytest.approx(0.86496, abs=1e-4)
    assert float(cos4_at_radius(0.0, F)) == 1.0


def test_pixel_centre_map_matches_direct_formula_and_is_slightly_inside_the_corner() -> None:
    field = cos4_field(W, H, PITCH, F)
    assert field.shape == (H, W) and field.dtype == np.float32
    # independent derivation at a few pixel centres: (f²/(f²+r²))² with r from (i+0.5, j+0.5)
    for i, j in ((0, 0), (255, 319), (100, 600), (511, 639)):
        x = (j + 0.5 - W / 2) * PITCH * 1e-3
        y = (i + 0.5 - H / 2) * PITCH * 1e-3
        direct = (F**2 / (F**2 + x * x + y * y)) ** 2
        assert abs(float(field[i, j]) - direct) < 1e-6
    # corner pixel centre sits half a pixel inside the format corner: a little brighter
    assert 0.79240 < float(field[0, 0]) < 0.7935
    assert float(field.max()) == pytest.approx(1.0, abs=2e-6)


def test_symmetry_catches_half_pixel_errors() -> None:
    field = cos4_field(W, H, PITCH, F).astype(np.float64)
    assert np.max(np.abs(field - field[:, ::-1])) < 1e-7
    assert np.max(np.abs(field - field[::-1, :])) < 1e-7


def test_monotone_in_radius() -> None:
    r = field_radius_map_mm(W, H, PITCH)
    field = cos4_field(W, H, PITCH, F).astype(np.float64)
    order = np.argsort(r.ravel())
    assert np.all(np.diff(field.ravel()[order]) <= 1e-7)
    assert float(field.ravel()[order][-1]) < float(field.ravel()[order][0])


@pytest.mark.slow  # GT.1: over a second on its own
def test_supersampled_field_box_averages_to_native() -> None:
    native = cos4_field(W, H, PITCH, F).astype(np.float64)
    ss = cos4_field(W, H, PITCH, F, supersample=4).astype(np.float64)
    assert ss.shape == (4 * H, 4 * W)
    boxed = ss.reshape(H, 4, W, 4).mean(axis=(1, 3))
    assert np.max(np.abs(boxed - native)) < 1e-4


def test_field_angle_map_is_atan_r_over_f() -> None:
    theta = field_angle_map(W, H, PITCH, F)
    assert theta.dtype == np.float32 and theta.shape == (H, W)
    r = field_radius_map_mm(W, H, PITCH)
    np.testing.assert_allclose(theta, np.arctan(r / F), rtol=1e-6)
    hfov_half = math.atan(W * PITCH * 1e-3 / 2 / F)
    assert float(theta[H // 2, 0]) < hfov_half < float(theta[0, 0])


def test_disabled_returns_ones_and_measured_map_multiplies() -> None:
    ones = cos4_field(4, 3, PITCH, F, enabled=False)
    assert ones.shape == (3, 4) and np.all(ones == 1.0) and ones.dtype == np.float32
    meas = np.full((3, 4), 0.5)
    combined = cos4_field(4, 3, PITCH, F, enabled=False, measured_map=meas)
    assert np.all(combined == 0.5)
    with pytest.raises(ValueError, match="shape"):
        cos4_field(4, 3, PITCH, F, measured_map=np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        cos4_field(4, 3, PITCH, F, measured_map=np.full((3, 4), 1.5))


def test_principal_point_offset_moves_the_peak() -> None:
    field = cos4_field(64, 64, PITCH, F, principal_point_px=(16.0, 48.0))
    i, j = np.unravel_index(int(np.argmax(field)), field.shape)
    assert (i, j) in {(47, 15), (47, 16), (48, 15), (48, 16)}


@pytest.mark.parametrize(
    "kwargs", [{"focal_length_mm": 0.0}, {"pitch_um": 0.0}, {"supersample": 0}]
)
def test_bad_geometry_raises(kwargs: dict[str, float]) -> None:
    args = {"width": W, "height": H, "pitch_um": PITCH, "focal_length_mm": F}
    args.update(kwargs)
    with pytest.raises(ValueError):
        cos4_field(**args)  # type: ignore[arg-type]
