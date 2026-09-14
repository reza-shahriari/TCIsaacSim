"""A spinning rotor as a time-averaged veil (§8.3, §9.2; ADR 0074, ADR 0077).

ADR 0074 left propellers off the quadrotor rather than draw a blade frozen at one azimuth, which
is a picture no thermal camera takes. The model that replaces that absence makes two claims worth
testing harder than "it draws a disc":

1. **The shutter moves the disc's flux around without changing how much there is.** Coverage is a
   running mean of the blade-passage pulse train, and a running mean cannot move the mean of a
   periodic function -- so a bolometer's smooth annulus and a cooled detector's resolved arcs carry
   the same total. If that failed, the rotor would be a brightness knob disguised as geometry.

2. **Viewing tilt enters only through the projected area of a pitched plate.** The blade area that
   lands in the image is ``N S k(tilt, pitch)`` whatever the foreshortening, because the ellipse
   shrinking and the per-pixel coverage rising are the same fact counted once.

Both are area identities, so both are checked by summing the rasterised map and comparing against
closed-form geometry -- not against a previous run of the same code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.optics.rotor import (
    RotorDisc,
    azimuthal_duty,
    coverage_map,
    projection_factor,
    swept_angle_rad,
    veil_radiance,
)
from irsim.radiometry.planck import band_radiance_tophat

DISC = RotorDisc()

#: Where the demo stage looks at the rotor plane: the camera is tilted 15 degrees up at a level
#: quadrotor, so the disc axis is 75 degrees off the view ray -- nearly edge-on, which is the
#: regime where the pitch term is load-bearing rather than decorative.
DEMO_TILT_DEG = 75.0

AZIMUTHS = np.linspace(0.0, 2.0 * np.pi, 100_001)[:-1]


def blade_spacings(disc: RotorDisc, swept_rad: float) -> float:
    return swept_rad / (2.0 * math.pi / disc.blades)


# --------------------------------------------------------------------------------------------
# The invariant: the shutter changes the picture, never the total.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "swept_deg", [0.0, 1.0, 5.0, 36.0, 90.0, 180.0, 300.0, 360.0, 540.0, 720.0, 3600.0]
)
def test_azimuthal_mean_is_the_solidity_for_every_window(swept_deg: float) -> None:
    """Mean coverage is the local solidity whatever the integration window.

    This is the claim that makes the two detector families comparable: a 2 ms cooled integration
    resolves arcs five times the mean and a 16.7 ms bolometer frame smears them into an annulus,
    and the flux is the same either way.
    """
    radius = 0.75 * DISC.radius_m
    width = float(DISC.blade_width_rad(radius))
    sigma = float(DISC.local_solidity(radius))
    duty = azimuthal_duty(AZIMUTHS, math.radians(swept_deg), width, DISC.blades)
    assert duty.mean() == pytest.approx(sigma, rel=1e-3)


@pytest.mark.parametrize("spacings", [1, 2, 3, 8])
def test_a_whole_number_of_blade_spacings_is_exactly_flat(spacings: int) -> None:
    """No ripple when the window holds a whole number of blade passes -- and ripple otherwise."""
    radius = 0.75 * DISC.radius_m
    width = float(DISC.blade_width_rad(radius))
    swept = spacings * 2.0 * math.pi / DISC.blades
    duty = azimuthal_duty(AZIMUTHS, swept, width, DISC.blades)
    assert float(np.ptp(duty)) < 1e-9
    assert duty.mean() == pytest.approx(float(DISC.local_solidity(radius)), rel=1e-6)


def test_a_fractional_window_bands_the_disc_by_one_whole_blade_pass() -> None:
    """At 1.67 spacings some azimuths are crossed twice and some once -- the banding is real.

    A bolometer frame on a 3000 rpm prop is exactly this case, so the annulus it draws is not
    uniform, and the two brightnesses stand in the ratio 2:1 because they differ by one whole
    blade pass, not by a fraction of one.
    """
    radius = 0.75 * DISC.radius_m
    width = float(DISC.blade_width_rad(radius))
    swept = swept_angle_rad(3000.0, 1.0 / 60.0)
    assert blade_spacings(DISC, swept) == pytest.approx(5.0 / 3.0)
    duty = azimuthal_duty(AZIMUTHS, swept, width, DISC.blades)
    assert duty.max() / duty.min() == pytest.approx(2.0, rel=1e-3)
    # one and two passes spread over the same window
    assert duty.min() == pytest.approx(width / swept, rel=1e-3)
    assert duty.max() == pytest.approx(2.0 * width / swept, rel=1e-3)


def test_a_short_window_resolves_one_arc_per_blade() -> None:
    """A cooled 2 ms integration leaves the blades separate, each smeared by the swept angle."""
    radius = 0.75 * DISC.radius_m
    width = float(DISC.blade_width_rad(radius))
    swept = swept_angle_rad(3000.0, 0.002)
    assert blade_spacings(DISC, swept) < 0.25
    duty = azimuthal_duty(AZIMUTHS, swept, width, DISC.blades)
    lit = duty > 1e-9
    # count runs of covered azimuth around the circle
    edges = int(np.count_nonzero(lit != np.roll(lit, 1)))
    assert edges // 2 == DISC.blades
    # each arc is the swept angle plus the blade's own width
    span = 2.0 * np.pi * float(np.count_nonzero(lit)) / lit.size / DISC.blades
    assert span == pytest.approx(swept + width, rel=5e-3)


def test_a_frozen_shutter_shows_the_blade_itself() -> None:
    """Zero window: the indicator, not an average -- coverage 1 on the blade and 0 beside it."""
    radius = 0.75 * DISC.radius_m
    width = float(DISC.blade_width_rad(radius))
    duty = azimuthal_duty(AZIMUTHS, 0.0, width, DISC.blades)
    assert set(np.unique(duty)) <= {0.0, 1.0}
    assert duty.mean() == pytest.approx(float(DISC.local_solidity(radius)), rel=1e-3)


# --------------------------------------------------------------------------------------------
# The projected area of a pitched plate
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 18.0, 45.0, 70.0])
def test_face_on_projection_is_the_cosine_of_the_pitch(pitch_deg: float) -> None:
    k = float(projection_factor(0.0, math.radians(pitch_deg)))
    assert k == pytest.approx(math.cos(math.radians(pitch_deg)), rel=1e-9)


@pytest.mark.parametrize("pitch_deg", [0.0, 10.0, 18.0, 45.0, 70.0])
def test_edge_on_projection_is_two_over_pi_times_the_sine(pitch_deg: float) -> None:
    """The closed form of the azimuthal mean of |sin|, and the reason pitch is a parameter."""
    k = float(projection_factor(math.pi / 2.0, math.radians(pitch_deg)))
    assert k == pytest.approx((2.0 / math.pi) * math.sin(math.radians(pitch_deg)), rel=1e-9)


def test_a_feathered_blade_is_invisible_edge_on_and_a_pitched_one_is_not() -> None:
    """Zero pitch edge-on projects nothing; 18 degrees of pitch keeps a fifth of the blade area."""
    assert float(projection_factor(math.pi / 2.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(projection_factor(math.pi / 2.0, math.radians(18.0))) > 0.19


def test_projection_factor_is_continuous_where_the_branch_changes() -> None:
    """|A| = |B| is the boundary between the two closed forms; they must agree on it."""
    pitch = math.radians(30.0)
    # A = B when cos(pitch)cos(tilt) = sin(pitch)sin(tilt), i.e. tilt = 90 - pitch
    tilt = math.pi / 2.0 - pitch
    below = float(projection_factor(tilt - 1e-7, pitch))
    above = float(projection_factor(tilt + 1e-7, pitch))
    assert below == pytest.approx(above, rel=1e-5)


def test_projection_factor_never_exceeds_one_or_falls_below_zero() -> None:
    tilt, pitch = np.meshgrid(np.linspace(0.0, np.pi / 2, 91), np.linspace(0.0, 1.5, 61))
    k = projection_factor(tilt, pitch)
    assert np.all(k >= 0.0) and np.all(k <= 1.0 + 1e-12)


# --------------------------------------------------------------------------------------------
# Geometry of the disc itself
# --------------------------------------------------------------------------------------------


def test_solidity_integrates_to_the_blade_planform_area() -> None:
    """``integral sigma(r) 2 pi r dr = N S`` -- the definition, checked against the chord model."""
    r = np.linspace(DISC.root_m, DISC.radius_m, 200_001)
    swept = float(np.trapezoid(DISC.local_solidity(r) * 2.0 * np.pi * r, r))
    assert swept == pytest.approx(DISC.blades * DISC.blade_area_m2(), rel=1e-6)


def test_a_heavy_lift_prop_covers_only_a_few_percent() -> None:
    """The number ADR 0074 predicted: a faint annulus, not a solid disc."""
    sigma = float(DISC.local_solidity(0.75 * DISC.radius_m))
    assert 0.02 < sigma < 0.06


def test_the_chord_is_zero_off_the_blade_span() -> None:
    assert float(DISC.chord_m(DISC.root_m * 0.5)) == 0.0
    assert float(DISC.chord_m(DISC.radius_m * 1.01)) == 0.0
    assert float(DISC.chord_m(DISC.root_m)) == pytest.approx(DISC.chord_root_m)
    assert float(DISC.chord_m(DISC.radius_m)) == pytest.approx(DISC.chord_tip_m)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius_m": 0.0},
        {"root_m": -0.1},
        {"root_m": 1.0},
        {"blades": 0},
        {"chord_tip_m": 0.0},
        {"pitch_deg": 90.0},
        {"pitch_deg": -1.0},
        {"root_m": 0.005, "chord_root_m": 0.05},  # blades overlap at the root
    ],
)
def test_impossible_rotors_are_refused(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RotorDisc(**kwargs)


# --------------------------------------------------------------------------------------------
# The rasterised map
# --------------------------------------------------------------------------------------------


def rasterise(tilt_deg: float, *, semi_major: float = 60.0, swept_rad: float = 40.0, **kw):
    cos_tilt = math.cos(math.radians(tilt_deg))
    return coverage_map(
        DISC,
        (200, 200),
        (100.0, 100.0),
        semi_major,
        semi_major * cos_tilt,
        swept_rad,
        **kw,
    )


@pytest.mark.parametrize("tilt_deg", [0.0, 30.0, 60.0, DEMO_TILT_DEG])
def test_the_blade_area_that_lands_in_the_image_depends_on_tilt_only_through_k(
    tilt_deg: float,
) -> None:
    """Sum of coverage = ``N S k`` in pixel units, at every foreshortening.

    The ellipse shrinking with tilt and the per-pixel coverage rising are the same fact, and this
    is where a double count or a missing cosine would show up as a disc that brightens or fades
    for no reason as the aircraft banks.
    """
    semi_major = 60.0
    alpha = rasterise(tilt_deg, semi_major=semi_major)
    k = float(projection_factor(math.radians(tilt_deg), math.radians(DISC.pitch_deg)))
    expected = k * DISC.blades * DISC.blade_area_m2() * (semi_major / DISC.radius_m) ** 2
    assert alpha.sum() == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize("tilt_deg", [0.0, 20.0, 45.0, 60.0, 71.9])
def test_coverage_is_tilt_invariant_until_the_disc_is_within_a_pitch_of_edge_on(
    tilt_deg: float,
) -> None:
    """``k / cos(tilt)`` is exactly ``cos(pitch)`` while ``tilt < 90 - pitch``.

    The blade's projected area and the ellipse's area shrink by the same cosine, so over most of
    the range a banking rotor does not change brightness at all -- only its shape. This was
    written the other way round first (as "coverage rises with tilt"), which the rasteriser
    contradicted: at 0 and 45 degrees the peak was the same number to three digits, differing
    only by where the pixel centres fell.
    """
    pitch = math.radians(DISC.pitch_deg)
    tilt = math.radians(tilt_deg)
    assert tilt_deg < 90.0 - DISC.pitch_deg
    ratio = float(projection_factor(tilt, pitch)) / math.cos(tilt)
    assert ratio == pytest.approx(math.cos(pitch), rel=1e-9)


def test_past_that_crossover_the_annulus_densifies_towards_edge_on() -> None:
    """Within ``pitch`` degrees of edge-on the same blade area is packed into a vanishing ellipse.

    The demo stage sits just past the crossover -- 75 degrees of tilt against 18 of pitch, where
    the threshold is 72 -- so its rotors are very slightly denser than the flat ``cos(pitch)``,
    which is the only reason this branch is exercised by the scene at all.
    """
    pitch = math.radians(DISC.pitch_deg)

    def density(tilt_deg: float) -> float:
        tilt = math.radians(tilt_deg)
        return float(projection_factor(tilt, pitch)) / math.cos(tilt)

    past = [density(t) for t in (72.0, 75.0, 80.0, 85.0, 89.0)]
    assert past == sorted(past)
    assert past[0] == pytest.approx(math.cos(pitch), rel=1e-3)
    assert density(DEMO_TILT_DEG) == pytest.approx(1.0, abs=0.02)
    assert density(89.9) > 100.0


def test_the_root_cutout_is_a_hole_and_outside_the_tip_is_empty() -> None:
    alpha = rasterise(0.0, semi_major=60.0, swept_rad=100.0)
    rows, cols = np.indices(alpha.shape, dtype=float)
    radius_px = np.hypot(cols + 0.5 - 100.0, rows + 0.5 - 100.0)
    root_px = 60.0 * DISC.root_m / DISC.radius_m
    assert float(alpha[radius_px < root_px * 0.9].max()) == 0.0
    assert float(alpha[radius_px > 60.0 * 1.05].max()) == 0.0
    assert float(alpha[(radius_px > root_px * 1.2) & (radius_px < 55.0)].max()) > 0.0


def test_coverage_is_clipped_at_one_at_grazing_incidence() -> None:
    """Beyond a certain tilt the blades stack up along the ray and the annulus is opaque."""
    alpha = rasterise(89.5, semi_major=200.0, swept_rad=100.0)
    assert alpha.max() == pytest.approx(1.0)
    assert np.all(alpha <= 1.0)


def test_an_exactly_edge_on_disc_draws_nothing() -> None:
    alpha = coverage_map(DISC, (64, 64), (32.0, 32.0), 20.0, 0.0, 40.0)
    assert not alpha.any()


def test_rotating_the_ellipse_moves_the_disc_and_keeps_its_area() -> None:
    flat = rasterise(60.0, swept_rad=100.0)
    turned = rasterise(60.0, swept_rad=100.0, rotation_deg=90.0)
    assert turned.sum() == pytest.approx(flat.sum(), rel=0.03)
    assert turned.T.sum() == pytest.approx(flat.sum(), rel=0.03)


def test_phase_turns_the_arcs_without_changing_the_total() -> None:
    short = swept_angle_rad(3000.0, 0.002)
    a = rasterise(0.0, swept_rad=short)
    b = rasterise(0.0, swept_rad=short, phase_rad=math.pi / 3.0)
    assert b.sum() == pytest.approx(a.sum(), rel=0.03)
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semi_major_px": 0.0},
        {"semi_minor_px": -1.0},
        {"semi_minor_px": 90.0},  # minor axis longer than major
        {"shape": (0, 10)},
    ],
)
def test_impossible_ellipses_are_refused(kwargs: dict) -> None:
    args = {
        "shape": (64, 64),
        "centre_px": (32.0, 32.0),
        "semi_major_px": 20.0,
        "semi_minor_px": 10.0,
        "swept_rad": 40.0,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        coverage_map(DISC, **args)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Compositing, in radiance
# --------------------------------------------------------------------------------------------


def test_the_veil_is_linear_in_radiance() -> None:
    bg = np.full((4, 4), 3.0)
    alpha = np.full((4, 4), 0.25)
    out = veil_radiance(bg, alpha, 11.0)
    assert np.allclose(out, 0.25 * 11.0 + 0.75 * 3.0)


def test_blending_temperatures_instead_of_radiance_reads_the_disc_cold() -> None:
    """Planck is convex in T, so the two blends are different numbers and one of them is wrong.

    A 3 % veil of airframe-temperature blade over a 230 K sky is a small effect, and averaging
    apparent temperatures under-reports it -- in the direction that would hide the disc. The gap
    is the reason CLAUDE.md #3's rule about noise applies to compositing too.
    """
    t_blade, t_sky, alpha = 290.0, 230.0, 0.032
    lb = band_radiance_tophat(8.0, 12.0, t_blade)
    ls = band_radiance_tophat(8.0, 12.0, t_sky)
    mixed = float(veil_radiance(np.array([ls]), np.array([alpha]), lb)[0])

    grid = np.linspace(t_sky, t_blade, 60_001)
    curve = np.array([band_radiance_tophat(8.0, 12.0, float(t)) for t in grid[::200]])
    apparent = float(np.interp(mixed, curve, grid[::200]))
    naive = alpha * t_blade + (1.0 - alpha) * t_sky

    assert apparent > naive + 0.1  # the radiance blend reads warmer, by a measurable margin
    assert apparent == pytest.approx(233.0, abs=1.0)


def test_an_occluder_in_front_of_the_disc_removes_the_veil() -> None:
    """Without this the blade would be painted over the motor bell it is bolted to."""
    bg = np.full((4, 4), 3.0)
    alpha = np.full((4, 4), 0.5)
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    out = veil_radiance(bg, alpha, 11.0, occluded=mask)
    assert np.allclose(out[:2], 3.0)
    assert np.allclose(out[2:], 7.0)


def test_zero_coverage_returns_the_background_untouched() -> None:
    rng = np.random.default_rng(7)
    bg = rng.normal(size=(16, 16))
    out = veil_radiance(bg, np.zeros((16, 16)), 100.0)
    assert np.array_equal(out, bg)


@pytest.mark.parametrize(
    "alpha,occluded",
    [(np.full((3, 3), 1.5), None), (np.full((2, 2), 0.5), None), (np.full((3, 3), 0.5), "bad")],
)
def test_bad_composites_are_refused(alpha: np.ndarray, occluded: object) -> None:
    bg = np.zeros((3, 3))
    mask = np.zeros((5, 5), dtype=bool) if occluded == "bad" else None
    with pytest.raises(ValueError):
        veil_radiance(bg, alpha, 1.0, occluded=mask)


# --------------------------------------------------------------------------------------------
# The two detector families, on one rotor
# --------------------------------------------------------------------------------------------


def test_a_bolometer_draws_an_annulus_and_a_cooled_detector_draws_arcs() -> None:
    """ADR 0077's split again: the window length alone decides, and the totals agree.

    The bolometer has no shutter, so it integrates the whole 16.7 ms frame and the blade crosses
    every azimuth; a 2 ms cooled integration turns through a fifth of a blade spacing and leaves
    them separate. Same rotor, same rpm, same flux.
    """
    rpm = 3000.0
    bolometer = rasterise(0.0, swept_rad=swept_angle_rad(rpm, 1.0 / 60.0))
    cooled = rasterise(0.0, swept_rad=swept_angle_rad(rpm, 0.002))

    assert cooled.sum() == pytest.approx(bolometer.sum(), rel=0.03)
    # the annulus is continuous in azimuth; the arcs are not
    lit_bolo = np.count_nonzero(bolometer > 0.0)
    lit_cooled = np.count_nonzero(cooled > 0.0)
    assert lit_cooled < 0.35 * lit_bolo
    assert cooled.max() > 3.0 * bolometer.max()
