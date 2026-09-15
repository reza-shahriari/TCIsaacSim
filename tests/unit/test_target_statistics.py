"""ME.4 — statistics that survive an unknown AGC (§5.3, §8.3, §15 T4; ADR 0068).

The public data for sky targets is 8-bit, post-AGC, codec-compressed video. An AGC has already
rescaled every level in it, so **a statistic that depends on the absolute level is useless** — and
the ones this module keeps are the ones that are not: a normalised profile shape, a gradient in
units of the flat-sky σ, a power-law exponent, and a contrast *ratio*.

The exception is the interesting one. **Edge asymmetry is not scale-free and does not need to be**:
a bolometer's membrane carries a moving target's history across frames, so the trailing side of its
profile is longer than the leading one, and that is a property of the shape. It is how a clip of
unknown provenance can be asked what its detector's time constant was.

docs/physics-model.md §5.3, §8.3, §15 T4; ADR 0068
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.validation.targets import (
    Box,
    asymmetry_vs_tau_curve,
    clutter_slope,
    edge_asymmetry,
    edge_spread_function,
    profile_centre_index,
    size_from_range_px,
    sky_profile,
    target_statistics,
)

RNG = np.random.default_rng(20260915)


def _gaussian_target(
    shape: tuple[int, int], centre: tuple[float, float], sigma: float, peak: float
) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    return peak * np.exp(-0.5 * (((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) / sigma**2))


# ---------------------------------------------------------------------------------------------
# SCR
# ---------------------------------------------------------------------------------------------


def test_a_synthetic_target_of_known_scr_is_measured_within_five_percent() -> None:
    """A flat-top target on Gaussian noise: the SCR is (amplitude)/σ by construction."""
    noise_sigma, amplitude = 4.0, 30.0
    frame = RNG.normal(120.0, noise_sigma, (128, 128))
    box = Box(58, 58, 12, 12)
    rows, cols = box.slice()
    frame[rows, cols] += amplitude
    stats = target_statistics(frame, box, ring_margin=6)
    assert stats.scr == pytest.approx(amplitude / noise_sigma, rel=0.05)
    assert stats.polarity == 1
    assert stats.area_px == 144


def test_polarity_flips_with_the_sign_of_the_contrast() -> None:
    """A target colder than its background is the other half of §15's polarity statistic."""
    frame = RNG.normal(120.0, 4.0, (128, 128))
    box = Box(58, 58, 12, 12)
    rows, cols = box.slice()
    frame[rows, cols] -= 30.0
    stats = target_statistics(frame, box, ring_margin=6)
    assert stats.polarity == -1
    assert stats.scr < -5.0


def test_scr_is_invariant_under_an_affine_display_mapping() -> None:
    """The point of using a ratio: an unknown AGC gain and offset must not move it."""
    frame = RNG.normal(120.0, 4.0, (128, 128))
    box = Box(58, 58, 12, 12)
    rows, cols = box.slice()
    frame[rows, cols] += 30.0
    plain = target_statistics(frame, box, ring_margin=6).scr
    stretched = target_statistics(2.7 * frame - 40.0, box, ring_margin=6).scr
    assert stretched == pytest.approx(plain, rel=1e-9)


def test_the_ring_is_local_rather_than_frame_wide() -> None:
    """A sky target's background is the sky right behind it. A frame-wide background would mix in
    the horizon, the ground and any cloud on the other side of the picture."""
    frame = np.full((128, 128), 100.0)
    frame[:40] = 30.0  # a bright/dark region far from the target
    box = Box(80, 80, 10, 10)
    rows, cols = box.slice()
    frame[rows, cols] = 140.0
    stats = target_statistics(frame, box, ring_margin=5)
    assert stats.mean_ring == pytest.approx(100.0, abs=1e-9)


def test_a_box_outside_the_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the frame"):
        target_statistics(np.zeros((32, 32)), Box(30, 30, 8, 8))
    with pytest.raises(ValueError, match="positive"):
        Box(0, 0, 0, 5)


# ---------------------------------------------------------------------------------------------
# the sky
# ---------------------------------------------------------------------------------------------


def test_the_sky_profile_shape_is_invariant_under_gain_and_offset() -> None:
    """Normalised by its own horizon-to-zenith span, so an AGC that stretched the sky twice as
    hard would leave the shape untouched."""
    rows = np.linspace(0.0, 1.0, 96)[:, None] ** 1.5
    frame = 40.0 + 60.0 * np.tile(rows, (1, 64))
    shape, _ = sky_profile(frame, horizon_row=95, zenith_row=0)
    stretched, _ = sky_profile(2.7 * frame - 15.0, horizon_row=95, zenith_row=0)
    assert np.allclose(shape, stretched, atol=1e-9)
    assert shape[0] == pytest.approx(0.0, abs=1e-12)
    assert shape[-1] == pytest.approx(1.0, abs=1e-12)


def test_the_sky_gradient_is_reported_in_sigma_per_row() -> None:
    """ "Is the gradient visible?" rather than "how many codes does it span?" -- so a noisier sky
    with the same gradient reports a smaller number."""
    rows = np.linspace(0.0, 1.0, 96)[:, None]
    base = 40.0 + 60.0 * np.tile(rows, (1, 64))
    _, quiet = sky_profile(base + RNG.normal(0.0, 0.5, base.shape), 95, 0)
    _, noisy = sky_profile(base + RNG.normal(0.0, 4.0, base.shape), 95, 0)
    assert quiet > noisy
    assert quiet / noisy == pytest.approx(8.0, rel=0.3)


def test_a_flat_sky_has_no_span_to_normalise_by() -> None:
    with pytest.raises(ValueError, match="no horizon-to-zenith span"):
        sky_profile(np.full((64, 64), 100.0), 63, 0)


# ---------------------------------------------------------------------------------------------
# clutter
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [1.5, 2.0, 2.5])
def test_a_power_law_clutter_field_recovers_its_own_slope(beta: float) -> None:
    """Within 0.1 — synthesised in the Fourier domain with a known exponent."""
    n = 192
    rows, cols = np.indices((n, n))
    radius = np.hypot(rows - n / 2.0, cols - n / 2.0)
    radius[int(n / 2), int(n / 2)] = 1.0
    amplitude = radius ** (-beta / 2.0)
    phase = RNG.uniform(0.0, 2.0 * np.pi, (n, n))
    field = np.real(np.fft.ifft2(np.fft.ifftshift(amplitude * np.exp(1j * phase))))
    assert clutter_slope(field) == pytest.approx(beta, abs=0.1)


def test_the_slope_is_invariant_under_gain() -> None:
    n = 128
    rows, cols = np.indices((n, n))
    radius = np.hypot(rows - n / 2.0, cols - n / 2.0)
    radius[int(n / 2), int(n / 2)] = 1.0
    phase = RNG.uniform(0.0, 2.0 * np.pi, (n, n))
    field = np.real(np.fft.ifft2(np.fft.ifftshift(radius**-1.0 * np.exp(1j * phase))))
    assert clutter_slope(5.0 * field + 77.0) == pytest.approx(clutter_slope(field), abs=1e-6)


def test_a_patch_too_small_to_fit_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 8x8"):
        clutter_slope(np.zeros((4, 4)))


# ---------------------------------------------------------------------------------------------
# the smear fingerprint
# ---------------------------------------------------------------------------------------------


def test_a_symmetric_target_has_no_edge_asymmetry() -> None:
    frame = _gaussian_target((64, 64), (32.0, 32.0), 3.0, 80.0) + 20.0
    profile = edge_spread_function(frame, Box(26, 26, 12, 12), axis=1, margin=10)
    assert abs(edge_asymmetry(profile)) < 0.02


def test_asymmetry_rises_monotonically_with_the_time_constant() -> None:
    """⚠️ **There is no closed-form inversion from asymmetry to τ**, and the roadmap's "recovers τ
    within 2 %" is not achievable. What is reliable is monotonicity at a fixed target and window.

    The obvious inversion treats the target as a point and the trail as a geometric tail. For an
    extended target (σ = 3 px) the core carries far more energy than the tail, and the point-source
    formula returns **0.45 frames for a true τ of 4**. So the statistic is a *fingerprint* -- is
    this clip smeared, and roughly how much -- inverted by interpolation against a curve built for
    the measurement's own target size, speed and window.

    Interpolation is not exact either: the curve is convex, so a linear read of a τ = 3 probe
    against knots at 2 and 4 returns 3.25. That is the accuracy this statistic supports, and it is
    the reason the curve is the deliverable rather than a number.
    """
    taus = np.array([0.5, 1.0, 2.0, 4.0])
    curve = asymmetry_vs_tau_curve(taus, velocity_px_per_frame=2.0, target_sigma_px=3.0)
    assert np.all(np.diff(curve) > 0.0), curve
    assert curve == pytest.approx([0.083, 0.261, 0.496, 0.700], abs=0.01)
    probe = asymmetry_vs_tau_curve([3.0], 2.0, 3.0)[0]
    assert curve[-2] < probe < curve[-1]
    assert float(np.interp(probe, curve, taus)) == pytest.approx(3.0, rel=0.15)


def test_the_curve_is_only_valid_for_the_geometry_it_was_built_with() -> None:
    """Target size and speed both move the whole curve, so a curve is not transferable.

    This is the practical warning that goes with the interpolation: reading a σ = 3 px measurement
    against a σ = 1.5 px curve, or a 2 px/frame target against a 1 px/frame curve, is a larger
    error than anything the statistic itself contributes.
    """
    reference = asymmetry_vs_tau_curve([0.5, 1.0, 2.0, 4.0], 2.0, 3.0)
    smaller = asymmetry_vs_tau_curve([0.5, 1.0, 2.0, 4.0], 2.0, 1.5)
    slower = asymmetry_vs_tau_curve([0.5, 1.0, 2.0, 4.0], 1.0, 3.0)
    # A smaller target smears further relative to its own size, a slower one less far.
    assert np.all(smaller > reference), (smaller, reference)
    assert np.all(slower < reference), (slower, reference)
    # Reading tau=2 off the wrong curve: the error dwarfs the interpolation error above.
    taus = np.array([0.5, 1.0, 2.0, 4.0])
    assert float(np.interp(reference[2], slower, taus)) > 3.0


def test_a_point_like_target_needs_the_geometric_split() -> None:
    """The finding that shaped this module: the *centroid* split is not usable.

    A long faint trail drags the centroid into itself, which leaves the compact core on the
    leading side of the split -- so the statistic reports the trail on the wrong side, and its
    magnitude is not monotone in τ either. Splitting at the box centre instead
    (:func:`profile_centre_index`) is monotone in both regimes, including the sub-pixel target
    where a real sky detection lives.
    """
    point_like = asymmetry_vs_tau_curve([0.5, 1.0, 2.0, 4.0], 2.0, target_sigma_px=0.6)
    assert np.all(np.diff(point_like) > 0.0), point_like
    assert point_like == pytest.approx([0.317, 0.633, 0.820, 0.913], abs=0.01)

    # The same frames, split at the centroid instead. Build one directly so the comparison is of
    # the split alone: a rightward trail on a sigma = 3 px target, ten frames of history.
    frame = np.zeros((32, 200))
    rows, cols = np.mgrid[0:32, 0:200]
    for k in range(20):
        frame += (0.8**k) * np.exp(
            -0.5 * (((rows - 16.0) ** 2 + (cols - 40.0 - 2.0 * k) ** 2) / 9.0)
        )
    box = Box(36, 13, 9, 7)
    profile = edge_spread_function(frame, box, axis=1, margin=60)
    geometric = edge_asymmetry(profile, centre=profile_centre_index(box, 60, axis=1))
    centroid = edge_asymmetry(profile)
    assert geometric > 0.5, geometric  # the trail is unambiguously to the right
    assert centroid < 0.0, centroid  # ...and the centroid split says it is to the left


def test_too_narrow_a_window_compresses_the_curve() -> None:
    """⚠️ A real trap: a margin too small to contain the trail truncates it, and the top of the
    curve flattens. It stays monotone -- it just stops resolving. Between τ = 4 and τ = 16 a
    120 px window gains 0.212 of asymmetry and a 6 px window only 0.110, so the same DN8
    measurement error inverts to nearly twice the spread in τ."""
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    narrow = asymmetry_vs_tau_curve(taus, 2.0, 3.0, margin_px=6)
    wide = asymmetry_vs_tau_curve(taus, 2.0, 3.0, margin_px=120)
    assert np.all(np.diff(narrow) > 0.0), narrow
    assert np.all(np.diff(wide) > 0.0), wide
    # Short and long tau agree; it is the resolving power in between that is lost.
    assert narrow[0] == pytest.approx(wide[0], abs=0.01)
    assert (wide[-1] - wide[3]) / (narrow[-1] - narrow[3]) > 1.8


def test_a_flat_profile_and_a_negative_asymmetry_are_refused() -> None:
    with pytest.raises(ValueError, match="flat"):
        edge_asymmetry(np.full(20, 5.0))
    with pytest.raises(ValueError, match="positive"):
        asymmetry_vs_tau_curve([-1.0], 2.0, 3.0)
    with pytest.raises(ValueError, match="positive"):
        asymmetry_vs_tau_curve([1.0], 0.0, 3.0)
    with pytest.raises(ValueError, match="five samples of window"):
        asymmetry_vs_tau_curve([1.0], 2.0, 3.0, margin_px=2)
    with pytest.raises(ValueError, match="non-empty"):
        asymmetry_vs_tau_curve([], 2.0, 3.0)


# ---------------------------------------------------------------------------------------------
# size vs range
# ---------------------------------------------------------------------------------------------


def test_apparent_size_follows_f_over_r_p_within_three_percent() -> None:
    """The relation a size-vs-range scatter has to lie on: 0.5 m target, 14 mm lens, 12 µm pitch."""
    ranges = np.array([120.0, 250.0, 500.0, 1000.0, 2500.0])
    px = size_from_range_px(0.5, ranges, focal_length_mm=14.0, pitch_um=12.0)
    assert np.allclose(px * ranges, px[0] * ranges[0], rtol=1e-12), "size × range must be constant"
    assert px[0] == pytest.approx(14.0e-3 * 0.5 / (120.0 * 12e-6), rel=0.03)
    assert px[-1] < 1.0, "a 0.5 m drone at 2.5 km is sub-pixel on this camera"
    with pytest.raises(ValueError, match="positive"):
        size_from_range_px(0.5, -10.0, 14.0, 12.0)
