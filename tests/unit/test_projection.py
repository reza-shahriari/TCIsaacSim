"""Lens projection (M10.9a): frame conventions, the OpenCV model, and the inverse.

The projection model is the oracle the engine's lens is checked against, so a test that merely
calls it is worthless -- if it agreed with itself while disagreeing with OpenCV, every in-sim
reprojection check would pass against the wrong lens. The tests here therefore pin it to things
decided outside this module: hand-evaluated OpenCV arithmetic, the equidistant limit of the
fisheye model, and ``irsim.optics.vignetting``'s independently-written pinhole geometry.

docs/physics-model.md §8.4; ADR 0015.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.config.sensor import DistortionSpec
from irsim.optics.projection import (
    FTHETA_UNVERIFIED,
    Intrinsics,
    distort_normalised,
    normalise,
    opencv_pinhole_coeffs,
    project,
    project_usd,
    undistort_normalised,
    usd_camera_to_opencv,
)
from irsim.optics.vignetting import cos4_at_radius, field_angle_map

BOSON = {"focal_length_mm": 14.0, "pitch_um": 12.0, "width": 640, "height": 512}
BOSON_FX_PX = 14.0 / 0.012  # 1166.666... px

#: The M10.9a verification coefficients: a visible barrel lens (~3 % at the Boson's corner).
BARREL = DistortionSpec(model="brown_conrady", coeffs=[-0.28, 0.09, 0.0, 0.0, 0.0])
PINHOLE = DistortionSpec(model="brown_conrady", coeffs=[0.0] * 5)

#: The row's budget for an in-sim reprojection. The engine-free round trip must be far inside it,
#: so that whatever a rendered grid measures is the *engine's* disagreement and not ours.
REPROJECTION_BUDGET_PX = 0.2


def boson_intrinsics(supersample: int = 1) -> Intrinsics:
    f_px = BOSON_FX_PX * supersample
    w = BOSON["width"] * supersample
    h = BOSON["height"] * supersample
    return Intrinsics(f_px, f_px, w / 2.0, h / 2.0, int(w), int(h))


def rays_from_pixels(u: np.ndarray, v: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """The ideal (undistorted) camera-space ray that images at each pixel, OpenCV frame."""
    x = (u - intr.cx_px) / intr.fx_px
    y = (v - intr.cy_px) / intr.fy_px
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def frame_grid(intr: Intrinsics, n: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """A grid of pixel centres spanning the whole format, corners included."""
    u = np.linspace(0.5, intr.width - 0.5, n)
    v = np.linspace(0.5, intr.height - 0.5, n)
    return np.meshgrid(u, v)


# --- frames -------------------------------------------------------------------------------------


def test_usd_to_opencv_flips_exactly_two_axes() -> None:
    """USD (+Y up, −Z forward) → OpenCV (+Y down, +Z forward) is (X, −Y, −Z), nothing else."""
    p = np.array([[1.0, 2.0, -3.0], [-0.5, -0.25, -10.0]])
    assert np.array_equal(usd_camera_to_opencv(p), np.array([[1.0, -2.0, 3.0], [-0.5, 0.25, 10.0]]))


def test_a_ray_above_the_axis_images_above_the_centre() -> None:
    """+Y is up in USD and v counts down, so an upward ray must land at v < cy.

    This is the sign the whole G-buffer hangs off: get it wrong and every image is flipped
    vertically, which is invisible on a symmetric test scene and wrong on every real one.
    """
    intr = boson_intrinsics()
    u, v = project_usd(np.array([[0.0, 1.0, -10.0]]), intr, PINHOLE)
    assert u[0] == pytest.approx(intr.cx_px, abs=1e-9)
    assert v[0] < intr.cy_px


def test_points_behind_the_camera_have_no_image() -> None:
    """z <= 0 in the OpenCV frame is behind the lens: NaN, never a plausible pixel."""
    pts = np.array([[1.0, 0.0, -2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
    x, _ = normalise(pts)
    assert np.isnan(x[0]) and np.isnan(x[1]) and x[2] == pytest.approx(0.5)


# --- intrinsics ---------------------------------------------------------------------------------


def test_boson_focal_length_in_pixels() -> None:
    """f/pitch = 14 mm / 12 µm = 1166.67 px, giving a 30.68° horizontal field.

    The YAML calls this "the 32 deg HFOV variant"; 32° is the nominal figure for the lens and
    30.68° is what the 640 × 12 µm format actually subtends through it. The number here is the
    one the geometry has to produce.
    """
    intr = boson_intrinsics()
    assert intr.fx_px == pytest.approx(1166.6667, abs=1e-3)
    hfov = 2.0 * math.atan(BOSON["width"] / 2.0 / intr.fx_px)
    assert math.degrees(hfov) == pytest.approx(30.676, abs=0.001)


def test_supersampling_scales_the_intrinsics_but_not_the_field_of_view() -> None:
    """4× subdivides pixels, it does not change the lens: f_px and c scale, the angles do not."""
    one, four = boson_intrinsics(1), boson_intrinsics(4)
    assert four.fx_px == pytest.approx(4.0 * one.fx_px)
    assert four.cx_px == pytest.approx(4.0 * one.cx_px)
    corner_1 = math.atan(math.hypot(one.cx_px, one.cy_px) / one.fx_px)
    corner_4 = math.atan(math.hypot(four.cx_px, four.cy_px) / four.fx_px)
    assert corner_4 == pytest.approx(corner_1, abs=1e-15)


def test_field_angle_agrees_with_the_vignetting_module_at_the_corner_pixel_centre() -> None:
    """The field angle this module implies must equal ``field_angle_map``'s.

    Two modules, two conventions to get wrong: pixel centres at (i + 0.5) and the principal point
    at the format *corner* (W/2). The tolerance is set by ``field_angle_map`` returning float32
    (2e-8 rad here), not by the geometry; half a pixel of disagreement at the Boson's corner
    would be 3.8e-4 rad, four thousand times larger, so this still separates the conventions.
    """
    intr = boson_intrinsics()
    angles = field_angle_map(
        BOSON["width"], BOSON["height"], BOSON["pitch_um"], BOSON["focal_length_mm"]
    )
    ray = rays_from_pixels(np.array([0.5]), np.array([0.5]), intr)
    theta = math.atan(math.hypot(ray[0, 0], ray[0, 1]))
    assert theta == pytest.approx(float(angles[0, 0]), abs=1e-7)


def test_cos4_of_the_projected_radius_is_cos4_of_the_field_angle() -> None:
    """r = f tanθ, so cos⁴ from the focal-plane radius must equal cos⁴θ -- the pinhole identity."""
    intr = boson_intrinsics()
    for theta_deg in (0.0, 5.0, 10.0, 15.0, 19.0):
        theta = math.radians(theta_deg)
        u, _ = project_usd(np.array([[math.sin(theta), 0.0, -math.cos(theta)]]), intr, PINHOLE)
        radius_mm = (u[0] - intr.cx_px) * BOSON["pitch_um"] * 1e-3
        assert float(cos4_at_radius(radius_mm, BOSON["focal_length_mm"])) == pytest.approx(
            math.cos(theta) ** 4, abs=1e-12
        )


# --- the OpenCV rational-polynomial model -------------------------------------------------------


def test_zero_coefficients_reproduce_the_pinhole_exactly() -> None:
    """A zero-coefficient lens must be the identity on normalised coordinates, to the last bit.

    Stated on the distortion itself rather than on the pixel round trip: the latter divides by
    ``fx`` and multiplies back, which is not bit-exact for every pixel and would make an exact
    assertion a statement about floating point instead of about the lens. The pixel path is
    checked just below, in pixels, where the budget lives.
    """
    x = np.array([-0.3, -0.05, 0.0, 0.12, 0.351])
    y = np.array([0.28, -0.19, 0.0, 0.02, -0.27])
    xd, yd = distort_normalised(x, y, PINHOLE)
    assert np.array_equal(xd, x) and np.array_equal(yd, y)


def test_pinhole_pixels_survive_the_projection_round_trip() -> None:
    """Ray → pixel for a zero-coefficient lens returns the pixel it was built from, < 1e-9 px."""
    intr = boson_intrinsics()
    u0, v0 = frame_grid(intr)
    u, v = project(rays_from_pixels(u0, v0, intr), intr, PINHOLE)
    assert float(np.abs(u - u0).max()) < 1e-9
    assert float(np.abs(v - v0).max()) < 1e-9


def test_radial_term_matches_hand_evaluated_opencv() -> None:
    """x_d = x(1 + k1 r² + k2 r⁴) at x=0.2, y=0.1, k1=−0.28, k2=0.09, evaluated by hand.

    r² = 0.05, r⁴ = 0.0025 → factor 1 − 0.014 + 0.000225 = 0.986225.
    """
    xd, yd = distort_normalised(np.array([0.2]), np.array([0.1]), BARREL)
    assert xd[0] == pytest.approx(0.2 * 0.986225, rel=0, abs=1e-15)
    assert yd[0] == pytest.approx(0.1 * 0.986225, rel=0, abs=1e-15)


def test_tangential_terms_are_asymmetric_in_the_two_axes() -> None:
    """p1 adds 2p1·xy to x and p1(r² + 2y²) to y -- a decentred lens, not a radial one.

    Swapping p1 and p2 (or symmetrising them) is the classic transcription error; it leaves a
    radially symmetric test pattern looking right and shifts every off-axis point.
    """
    spec = DistortionSpec(model="brown_conrady", coeffs=[0.0, 0.0, 0.01, 0.0, 0.0])
    xd, yd = distort_normalised(np.array([0.2]), np.array([0.1]), spec)
    assert xd[0] == pytest.approx(0.2 + 2 * 0.01 * 0.2 * 0.1, abs=1e-15)
    assert yd[0] == pytest.approx(0.1 + 0.01 * (0.05 + 2 * 0.01), abs=1e-15)


@pytest.mark.parametrize(("k1", "inward"), [(-0.28, True), (0.28, False)])
def test_sign_of_k1_sets_barrel_versus_pincushion(k1: float, inward: bool) -> None:
    """k1 < 0 pulls the field edge toward the centre (barrel); k1 > 0 pushes it out."""
    spec = DistortionSpec(model="brown_conrady", coeffs=[k1, 0.0, 0.0, 0.0, 0.0])
    xd, _ = distort_normalised(np.array([0.35]), np.array([0.0]), spec)
    assert bool(xd[0] < 0.35) is inward


def test_barrel_distortion_moves_the_boson_corner_by_about_thirteen_pixels() -> None:
    """A 3 % radial shrink at r_n = 0.351 is ~13.6 px at the corner: big enough to matter."""
    intr = boson_intrinsics()
    corner = rays_from_pixels(np.array([0.5]), np.array([0.5]), intr)
    u_ideal, v_ideal = project(corner, intr, PINHOLE)
    u, v = project(corner, intr, BARREL)
    shift = math.hypot(u[0] - u_ideal[0], v[0] - v_ideal[0])
    assert shift == pytest.approx(13.6, abs=0.5)
    assert math.hypot(u[0] - intr.cx_px, v[0] - intr.cy_px) < math.hypot(
        u_ideal[0] - intr.cx_px, v_ideal[0] - intr.cy_px
    )


def test_round_trip_is_far_inside_the_reprojection_budget() -> None:
    """undistort → distort over the whole frame returns the same pixel to < 1e-6 px.

    The M10.9a budget for an in-sim grid is 0.2 px; the model's own inverse has to be orders
    below that, or a rendered-grid disagreement could not be attributed to the engine.
    """
    intr = boson_intrinsics()
    u0, v0 = frame_grid(intr, n=17)
    xd = (u0 - intr.cx_px) / intr.fx_px
    yd = (v0 - intr.cy_px) / intr.fy_px
    x, y = undistort_normalised(xd, yd, BARREL)
    assert np.all(np.isfinite(x)), "the fixed point failed to converge inside the format"
    xr, yr = distort_normalised(x, y, BARREL)
    err_px = np.hypot((xr - xd) * intr.fx_px, (yr - yd) * intr.fy_px)
    assert float(err_px.max()) < 1e-6
    assert float(err_px.max()) < REPROJECTION_BUDGET_PX / 100.0


# --- the fisheye model --------------------------------------------------------------------------


def test_kannala_brandt_with_no_coefficients_is_the_equidistant_projection() -> None:
    """θ_d = θ, so r_d = θ: the defining property of a fisheye, not of a rectilinear lens.

    At 60° a rectilinear lens images at r = tan 60° = 1.732 and a fisheye at θ = 1.047 -- a
    66 % difference, so a model that silently fell back to the pinhole could not pass this.
    """
    spec = DistortionSpec(model="kannala_brandt", coeffs=[0.0] * 4)
    for theta_deg in (1.0, 20.0, 45.0, 60.0, 85.0):
        theta = math.radians(theta_deg)
        r = math.tan(theta)
        xd, yd = distort_normalised(np.array([r]), np.array([0.0]), spec)
        assert math.hypot(xd[0], yd[0]) == pytest.approx(theta, abs=1e-12)


def test_kannala_brandt_coefficients_enter_as_odd_powers_of_theta() -> None:
    """θ_d = θ + k1θ³: hand-evaluated at θ = 0.5 rad with k1 = 0.1 → 0.5125."""
    spec = DistortionSpec(model="kannala_brandt", coeffs=[0.1, 0.0, 0.0, 0.0])
    xd, _ = distort_normalised(np.array([math.tan(0.5)]), np.array([0.0]), spec)
    assert xd[0] == pytest.approx(0.5 + 0.1 * 0.125, abs=1e-12)


def test_kannala_brandt_inverse_recovers_the_ray() -> None:
    """Newton on θ_d(θ) must return the original direction to 1e-12 across a wide field."""
    spec = DistortionSpec(model="kannala_brandt", coeffs=[-0.02, 0.004, 0.0, 0.0])
    theta = np.radians(np.array([0.0, 5.0, 25.0, 50.0, 75.0]))
    r = np.tan(theta)
    xd, yd = distort_normalised(r, np.zeros_like(r), spec)
    x, y = undistort_normalised(xd, yd, spec)
    assert np.allclose(x, r, atol=1e-12) and np.allclose(y, 0.0, atol=1e-12)


def test_fisheye_coefficients_are_refused_by_the_pinhole_coefficient_map() -> None:
    """A polynomial in θ is not a polynomial in r; writing one into the other is silent nonsense."""
    spec = DistortionSpec(model="kannala_brandt", coeffs=[0.1, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="not a rational-polynomial model"):
        opencv_pinhole_coeffs(spec)


def test_brown_conrady_coefficients_map_positionally_onto_opencv() -> None:
    """[k1, k2, p1, p2, k3] are OpenCV's first five, zero-padded to twelve -- no reordering."""
    values = opencv_pinhole_coeffs(
        DistortionSpec(model="brown_conrady", coeffs=[1.0, 2.0, 3.0, 4.0, 5.0])
    )
    assert values["k1"] == 1.0
    assert values["k2"] == 2.0
    assert values["p1"] == 3.0
    assert values["p2"] == 4.0
    assert values["k3"] == 5.0
    assert all(values[name] == 0.0 for name in ("k4", "k5", "k6", "s1", "s2", "s3", "s4"))


# --- the model we will not guess ----------------------------------------------------------------


def test_ftheta_is_refused_rather_than_approximated() -> None:
    """f-theta's polynomial convention is unverified on this build, so both entry points raise."""
    spec = DistortionSpec(model="ftheta", coeffs=[0.0, 500.0])
    with pytest.raises(NotImplementedError, match="not verified"):
        project(np.array([[0.0, 0.0, 1.0]]), boson_intrinsics(), spec)
    with pytest.raises(NotImplementedError):
        distort_normalised(np.array([0.1]), np.array([0.0]), spec)
    assert "M10.9b" in FTHETA_UNVERIFIED
