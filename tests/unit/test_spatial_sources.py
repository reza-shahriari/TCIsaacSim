"""ADR 0088 — heat sources that vary across a surface, and the sky they block.

Two things carry the weight here.

`test_corner_form_matches_a_brute_force_quadrature` checks the configuration factor against a
direct numerical integration of `c^2 / (pi (x^2 + y^2 + c^2)^2)` rather than against a remembered
constant. A view factor wrong by a factor of two produces a bonnet gradient of exactly the right
*shape* and half the right size, which no picture would reveal.

`test_a_cold_sky_warms_the_ground_more_than_the_engine_does` is the physics this module exists for.
A hot body over a surface both radiates onto it and **removes the sky it was seeing**; keeping only
the first term invents energy, and under a clear night sky the second term is the larger one. That
is why a parked car leaves a warm car-shaped patch on asphalt before its engine has ever run.

docs/physics-model.md §6.1, §6.6; ADR 0087, ADR 0088
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.spatial_sources import (
    RadiantRectangle,
    clamp_view_factor_sum,
    corner_view_factor,
    occluded_longwave_flux,
    patch_view_factors,
    view_factor_to_parallel_rectangle,
)
from irsim.thermal.surface_field import PlanarPatch

UP = np.array([0.0, 0.0, 1.0])
EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])


def _brute_corner(x: float, y: float, c: float, n: int = 2000) -> float:
    """The configuration-factor integral itself, evaluated by midpoint quadrature."""
    gx = (np.arange(n) + 0.5) * x / n
    gy = (np.arange(n) + 0.5) * y / n
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    integrand = c**2 / (np.pi * (grid_x**2 + grid_y**2 + c**2) ** 2)
    return float(integrand.sum() * (x / n) * (y / n))


# ---------------------------------------------------------------------------------------------
# the geometry kernel
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "y", "c"), [(1.0, 1.0, 1.0), (2.0, 3.0, 1.0), (0.5, 0.4, 1.0), (1.2, 0.3, 0.7)]
)
def test_corner_form_matches_a_brute_force_quadrature(x: float, y: float, c: float) -> None:
    assert float(corner_view_factor(x, y, c)) == pytest.approx(_brute_corner(x, y, c), rel=1e-4)


def test_a_quarter_infinite_plane_gives_one_quarter() -> None:
    """The limit that fixes the constant: four quadrants of an infinite plane must sum to 1."""
    assert float(corner_view_factor(1e6, 1e6, 1.0)) == pytest.approx(0.25, abs=1e-5)


def test_the_corner_form_is_odd_in_each_argument() -> None:
    """What makes the four-corner superposition valid for an offset rectangle."""
    assert float(corner_view_factor(-2.0, 3.0, 1.0)) == pytest.approx(
        -float(corner_view_factor(2.0, 3.0, 1.0))
    )
    assert float(corner_view_factor(-2.0, -3.0, 1.0)) == pytest.approx(
        float(corner_view_factor(2.0, 3.0, 1.0))
    )


def test_superposition_matches_brute_force_for_an_offset_rectangle() -> None:
    """The case the demo actually uses: a cell out at the wing, not under the block."""
    rect = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.9]), u_axis=EX, v_axis=EY, half_u_m=0.6, half_v_m=0.4
    )
    point = np.array([[0.75, 0.2, 0.0]])  # beyond the block in u, inside it in v
    got = float(view_factor_to_parallel_rectangle(point, rect)[0])

    # Brute force in the point's own frame: the rectangle spans [-1.35, -0.15] x [-0.6, 0.2].
    n = 3000
    gx = np.linspace(-1.35, -0.15, n)
    gy = np.linspace(-0.6, 0.2, n)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    integrand = 0.9**2 / (np.pi * (grid_x**2 + grid_y**2 + 0.9**2) ** 2)
    expected = float(integrand.mean() * 1.2 * 0.8)
    assert got == pytest.approx(expected, rel=2e-3)


def test_a_large_close_radiator_approaches_a_half() -> None:
    """A cell directly under a wide, near parallel plane sees half a hemisphere, not all of it."""
    rect = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.05]), u_axis=EX, v_axis=EY, half_u_m=50.0, half_v_m=50.0
    )
    f = float(view_factor_to_parallel_rectangle(np.array([[0.0, 0.0, 0.0]]), rect)[0])
    assert f == pytest.approx(1.0, abs=1e-3)


def test_the_view_factor_falls_with_distance_from_the_source() -> None:
    rect = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.4]), u_axis=EX, v_axis=EY, half_u_m=0.5, half_v_m=0.5
    )
    points = np.stack([np.linspace(0.0, 4.0, 12), np.zeros(12), np.zeros(12)], axis=-1)
    f = view_factor_to_parallel_rectangle(points, rect)
    assert np.all(np.diff(f) < 0.0)
    assert f[0] > 10.0 * f[-1]


# ---------------------------------------------------------------------------------------------
# the physics
# ---------------------------------------------------------------------------------------------


def test_a_cold_sky_warms_the_ground_more_than_the_engine_does() -> None:
    """The occlusion term, and why the demo scene is overcast.

    A 295 K underbody over asphalt under a **clear** 245 K sky: most of the added flux is the sky
    the car removed, not the car's own emission. Under an **overcast** sky at 288 K the occlusion
    nearly cancels and what is left really is the body's own warmth. A model keeping only the
    source term would report the same number for both nights.
    """
    f = np.array([0.6])
    eps = 0.95
    clear = float(occluded_longwave_flux(f, 295.0, eps, longwave_down_w_m2=SIGMA_SB * 245.0**4)[0])
    overcast = float(
        occluded_longwave_flux(f, 295.0, eps, longwave_down_w_m2=SIGMA_SB * 288.0**4)[0]
    )
    source_only = float(occluded_longwave_flux(f, 295.0, eps)[0])

    assert clear > overcast > 0.0
    # Without the occlusion term all three of these would be the *same* number, so each of the
    # three comparisons below is a test of it and not of the Stefan-Boltzmann law.
    assert clear > 5.0 * overcast
    # It is not a rounding correction: it removes about half the naive answer on a clear night,
    # and about nine tenths of it under overcast.
    assert 0.45 < clear / source_only < 0.60
    assert overcast / source_only < 0.15


def test_a_body_no_warmer_than_the_sky_it_blocks_adds_nothing() -> None:
    """The sign convention, stated as an identity rather than a tolerance."""
    f = np.array([0.5])
    q = occluded_longwave_flux(f, 290.0, 0.9, longwave_down_w_m2=SIGMA_SB * 290.0**4)
    assert float(q[0]) == pytest.approx(0.0, abs=1e-9)


def test_a_body_colder_than_the_sky_cools_the_surface() -> None:
    f = np.array([0.5])
    q = occluded_longwave_flux(f, 260.0, 0.9, longwave_down_w_m2=SIGMA_SB * 290.0**4)
    assert float(q[0]) < 0.0


def test_the_source_emissivity_scales_only_the_source_term() -> None:
    f = np.array([0.4])
    black = float(occluded_longwave_flux(f, 350.0, 0.9, source_emissivity=1.0)[0])
    grey = float(occluded_longwave_flux(f, 350.0, 0.9, source_emissivity=0.5)[0])
    assert grey == pytest.approx(0.5 * black, rel=1e-12)


def test_a_tilted_surface_loses_proportionally_less_sky() -> None:
    """A wall already seeing half the sky can only have half as much taken away."""
    f = np.array([0.3])
    full = occluded_longwave_flux(f, 300.0, 0.9, longwave_down_w_m2=200.0, sky_view=1.0)
    half = occluded_longwave_flux(f, 300.0, 0.9, longwave_down_w_m2=100.0, sky_view=0.5)
    assert float(full[0]) == pytest.approx(float(half[0]), rel=1e-12)


# ---------------------------------------------------------------------------------------------
# the patch join
# ---------------------------------------------------------------------------------------------


def test_patch_view_factors_peak_over_the_source_and_fall_at_the_edges() -> None:
    """The bonnet gradient, as geometry: hottest over the block, coolest at the wings."""
    patch = PlanarPatch(
        origin_m=np.array([-1.2, -0.8, 0.9]),
        u_axis=EX,
        v_axis=EY,
        n_u=16,
        n_v=8,
        du_m=0.15,
        dv_m=0.2,
        thickness_m=0.1,
    )
    block = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.35]), u_axis=EX, v_axis=EY, half_u_m=0.35, half_v_m=0.3
    )
    f = patch_view_factors(patch, block).reshape(patch.shape)

    centre = f[patch.n_v // 2, patch.n_u // 2]
    corner = f[0, 0]
    # A 0.70 x 0.60 m block 0.55 m below fills more than a quarter of the cell's hemisphere.
    assert centre > 0.25, f"a cell directly over the block should see a lot of it, got {centre}"
    assert corner < 0.1 * centre, "the wings should be nearly cold"
    # Monotone away from the block along the centre row, which is what makes it a gradient.
    row = f[patch.n_v // 2, patch.n_u // 2 :]
    assert np.all(np.diff(row) < 0.0)


def test_a_non_parallel_patch_raises_rather_than_returning_a_plausible_number() -> None:
    wall = PlanarPatch(origin_m=np.zeros(3), u_axis=EX, v_axis=UP, n_u=4, n_v=4, du_m=0.2, dv_m=0.2)
    block = RadiantRectangle(
        centre_m=np.array([0.0, 0.0, 0.5]), u_axis=EX, v_axis=EY, half_u_m=0.3, half_v_m=0.3
    )
    with pytest.raises(ValueError, match="parallel"):
        patch_view_factors(wall, block)


def test_a_point_in_the_radiators_plane_raises() -> None:
    rect = RadiantRectangle(centre_m=np.zeros(3), u_axis=EX, v_axis=EY, half_u_m=0.5, half_v_m=0.5)
    with pytest.raises(ValueError, match="singular"):
        view_factor_to_parallel_rectangle(np.array([[2.0, 0.0, 0.0]]), rect)


# ---------------------------------------------------------------------------------------------
# clamping nested radiators (ADR 0090, roadmap PT.3)
# ---------------------------------------------------------------------------------------------


def test_clamp_view_factor_sum_is_a_no_op_when_the_sum_is_already_under_one() -> None:
    a = np.array([0.2, 0.5, 0.9])
    b = np.array([0.1, 0.3, 0.05])
    out_a, out_b = clamp_view_factor_sum([a, b])
    np.testing.assert_array_equal(out_a, a)
    np.testing.assert_array_equal(out_b, b)


def test_clamp_view_factor_sum_caps_the_total_at_one() -> None:
    """The measured failure: three radiators, one cell over-counted by all three."""
    underbody = np.array([0.9, 0.3])
    engine_bay = np.array([0.5, 0.2])
    exhaust = np.array([0.1, 0.05])
    with pytest.warns(UserWarning, match="overlap"):
        out = clamp_view_factor_sum([underbody, engine_bay, exhaust])
    total = sum(out)
    assert np.all(total <= 1.0 + 1e-6), total
    # the untouched cell (index 1, sum 0.55) must be bit-identical
    assert out[0][1] == pytest.approx(0.3)
    assert out[1][1] == pytest.approx(0.2)
    assert out[2][1] == pytest.approx(0.05)


def test_clamp_view_factor_sum_preserves_relative_weight() -> None:
    """Scaling is proportional: the radiator with more raw view factor still dominates after."""
    underbody = np.array([0.9])
    engine_bay = np.array([0.6])
    out_u, out_e = clamp_view_factor_sum([underbody, engine_bay])
    assert out_u[0] == pytest.approx(0.9 / 1.5)
    assert out_e[0] == pytest.approx(0.6 / 1.5)
    assert out_u[0] + out_e[0] == pytest.approx(1.0)


def test_clamp_view_factor_sum_scales_the_occlusion_term_the_same_as_the_source_term() -> None:
    """occluded_longwave_flux is linear in view_factors, so clamping bounds both consistently."""
    raw = [np.array([0.9]), np.array([0.6])]
    with pytest.warns(UserWarning):
        clamped = clamp_view_factor_sum(raw)
    ratio = float(clamped[0][0] / raw[0][0])
    flux_raw = occluded_longwave_flux(
        raw[0], 340.0, 0.9, source_emissivity=0.9, longwave_down_w_m2=250.0, sky_view=1.0
    )
    flux_clamped = occluded_longwave_flux(
        clamped[0], 340.0, 0.9, source_emissivity=0.9, longwave_down_w_m2=250.0, sky_view=1.0
    )
    assert float(flux_clamped[0]) == pytest.approx(ratio * float(flux_raw[0]))


def test_clamp_view_factor_sum_rejects_an_input_already_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        clamp_view_factor_sum([np.array([1.5]), np.array([0.1])])
