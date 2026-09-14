"""Level A Fresnel reflectance from a complex index (M7.4, §4.2).

These tests are built so that a wrong branch, a dropped k, or a sign slip fails one of them:

* the normal-incidence closed form over five decades of n and k, to 1e-12;
* the Brewster null, which only exists if r_par is formed with n~^2 rather than n~;
* water at 10 µm against the [R1] validation numbers -- the material the sea model rests on;
* the absence of a total-internal-reflection knee in an absorbing medium, which is the observable
  signature of the branch choice, plus a direct demonstration that the rejected branch is not
  passive (it returns R > 1).

docs/physics-model.md §4.2 Level A [R1]
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.fresnel import directional_emissivity_fresnel, fresnel_reflectance

#: Water at 10 µm: the pair [R1] validates against, and the reason this module exists -- a sea
#: surface is a Fresnel reflector seen almost edge-on.
WATER_10UM = (1.218, 0.0508)


def _normal_incidence(n: float, k: float) -> float:
    """((n-1)^2 + k^2) / ((n+1)^2 + k^2): the textbook normal-incidence reflectance."""
    return ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)


def test_normal_incidence_matches_the_closed_form_over_five_decades() -> None:
    """R(0) is the one case with an exact answer, so it is checked to machine precision.

    n from a metal-like 0.05 to a semiconductor-like 30, k from a dielectric 0 to a metallic 100.
    """
    n = np.geomspace(0.05, 30.0, 37)[:, None]
    k = np.concatenate(([0.0], np.geomspace(1e-3, 100.0, 36)))[None, :]

    r_unpol, r_perp, r_par = fresnel_reflectance(n, k, 1.0)
    expected = ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)

    assert np.max(np.abs(r_unpol - expected)) < 1e-12
    # At normal incidence the two polarisations are indistinguishable; if they are not, the
    # parallel branch has an angle-dependent factor that should have cancelled.
    assert np.max(np.abs(r_perp - r_par)) < 1e-12


def test_brewster_null_for_a_lossless_dielectric() -> None:
    """tan(theta_B) = n kills R_par exactly; R_perp does not vanish there."""
    n = 1.5
    cos_b = np.cos(np.arctan(n))
    r_unpol, r_perp, r_par = fresnel_reflectance(n, 0.0, cos_b)

    assert float(r_par) < 1e-12
    assert float(r_perp) > 0.1
    assert float(r_unpol) == pytest.approx(0.5 * float(r_perp), abs=1e-12)


def test_index_matched_medium_never_reflects() -> None:
    """n = 1, k = 0 is vacuum: zero reflectance at every angle, not just at normal incidence."""
    mu = np.cos(np.radians(np.linspace(0.0, 89.9, 91)))
    r_unpol, r_perp, r_par = fresnel_reflectance(1.0, 0.0, mu)

    assert np.max(np.abs(r_unpol)) < 1e-15
    assert np.max(np.abs(r_perp)) < 1e-15
    assert np.max(np.abs(r_par)) < 1e-15


def test_water_at_10um_matches_the_published_validation_values() -> None:
    """[R1]'s water numbers: R(0) = 0.01018, eps(60) = 0.961, eps(80) = 0.697.

    The 60-to-80 degree drop is the whole point. A Level C constant emissivity would report 0.99
    at both, and a sea surface built on it would be 40 K wrong at grazing incidence.
    """
    n, k = WATER_10UM

    assert float(fresnel_reflectance(n, k, 1.0)[0]) == pytest.approx(0.01018, abs=1e-4)
    eps_60 = float(directional_emissivity_fresnel(n, k, np.cos(np.radians(60.0))))
    eps_80 = float(directional_emissivity_fresnel(n, k, np.cos(np.radians(80.0))))
    assert eps_60 == pytest.approx(0.961, abs=5e-3)
    assert eps_80 == pytest.approx(0.697, abs=5e-3)
    assert eps_80 < eps_60 - 0.2


def test_water_emissivity_falls_monotonically_toward_grazing() -> None:
    """Dielectric emissivity is flat near normal and collapses at the limb; it never rises."""
    n, k = WATER_10UM
    theta = np.radians(np.linspace(0.0, 89.5, 180))
    eps = directional_emissivity_fresnel(n, k, np.cos(theta))

    assert np.all(np.diff(eps) < 1e-12)
    assert eps[0] == pytest.approx(0.9898, abs=1e-3)  # near-blackbody looking straight down
    assert eps[-1] < 0.1  # and a mirror at the horizon


def test_absorbing_medium_has_no_total_internal_reflection_knee() -> None:
    """[R1]'s counter-intuitive result: with k > 0, R < 1 at every angle, with no kink.

    A lossless n = 0.5 hits R = 1 exactly past its critical angle; adding k = 0.5 removes the
    discontinuity entirely. A model that keeps the knee has taken the wrong square root.
    """
    theta = np.radians(np.linspace(0.0, 89.99, 2000))
    mu = np.cos(theta)

    r_lossy = fresnel_reflectance(0.5, 0.5, mu)[0]
    assert np.max(r_lossy) < 1.0 - 1e-4

    # Bounded slope: no knee. dR/dtheta stays finite and comparable to the lossless case away
    # from its critical angle -- 30 per radian is generous and still an order below a step.
    d_r = np.abs(np.gradient(r_lossy, theta))
    assert np.max(d_r) < 30.0

    r_lossless = fresnel_reflectance(0.5, 0.0, mu)[0]
    assert np.max(r_lossless) == pytest.approx(1.0, abs=1e-12)


def test_the_rejected_branch_is_not_passive() -> None:
    """Why the A/B form is not cosmetic: -(A + jB) returns reflectances above 1.

    This recomputes the same water case with the other square root, which is what an
    implementation that does not think about the branch can land on. Energy conservation fails,
    so the branch is observable and this test would catch a regression to it.
    """
    n, k = WATER_10UM
    mu = np.cos(np.radians(45.0))

    n2_re, n2_im = n * n - k * k, 2.0 * n * k
    z_re, z_im = n2_re - (1.0 - mu * mu), n2_im
    z_abs = np.hypot(z_re, z_im)
    A = np.sqrt(0.5 * (z_abs + z_re))
    B = np.sign(z_im) * np.sqrt(0.5 * (z_abs - z_re))

    # -(A + jB): the other root of the same Z.
    r_perp_wrong = ((mu + A) ** 2 + B**2) / ((mu - A) ** 2 + B**2)

    assert r_perp_wrong > 1.0
    assert float(fresnel_reflectance(n, k, mu)[1]) < 1.0


def test_reflectance_is_continuous_across_an_index_sweep() -> None:
    """n from 0.3 to 3.0 at 70 degrees: continuous, and absorption removes the cusp entirely.

    Continuity is tested by refinement rather than by a fixed threshold, because at k = 0 the
    lossless critical angle (n = sin 70 = 0.94) is a genuine square-root cusp -- the value is
    continuous but the slope is not, and any fixed tolerance either fails on real physics or is
    too loose to catch a real discontinuity. Refining the grid four-fold must shrink the largest
    step: by about 2 for a sqrt cusp, by 4 where the function is smooth, by 1 for a jump. A wrong
    branch produces the jump, and only the jump fails to shrink.
    """

    def max_step(n_points: int, k: float) -> float:
        n = np.linspace(0.3, 3.0, n_points)
        r = fresnel_reflectance(n, k, np.cos(np.radians(70.0)))[0]
        assert np.all(np.isfinite(r))
        assert np.all((r >= 0.0) & (r <= 1.0))
        return float(np.max(np.abs(np.diff(r))))

    # Lossless: a cusp, so the step shrinks like sqrt(dn) -- clearly more than a jump would.
    assert max_step(5401, 0.0) / max_step(21601, 0.0) > 1.8

    # Absorbing: no cusp at all, the step shrinks linearly with dn. This is the same [R1] result
    # as the no-TIR-knee test, seen along the index axis instead of the angle axis.
    for k in (0.01, 1.0):
        assert max_step(5401, k) / max_step(21601, k) > 3.5
        assert max_step(5401, k) < 2e-2


def test_back_faces_use_the_absolute_cosine() -> None:
    """A facet pointing away is the same surface from the other side; R is even in cos theta."""
    n, k = WATER_10UM
    mu = np.cos(np.radians([0.0, 25.0, 55.0, 85.0]))

    front = fresnel_reflectance(n, k, mu)[0]
    back = fresnel_reflectance(n, k, -mu)[0]
    assert np.allclose(front, back, atol=0.0, rtol=0.0)


def test_broadcasts_a_spectrum_against_a_cosine_plane() -> None:
    """Spectral (n, k) against a G-buffer of cosines in one call, shapes broadcast as NumPy does."""
    n = np.array([1.2, 1.3, 1.4])[:, None, None]
    k = np.array([0.05, 0.06, 0.07])[:, None, None]
    mu = np.cos(np.radians(np.linspace(0.0, 80.0, 12))).reshape(1, 3, 4)

    r = fresnel_reflectance(n, k, mu)[0]
    assert r.shape == (3, 3, 4)
    assert np.all((r >= 0.0) & (r <= 1.0))
    assert r.dtype == np.float64


@pytest.mark.parametrize(
    "n, k, mu, message",
    [
        (0.0, 0.1, 1.0, "n must be positive"),
        (-1.2, 0.1, 1.0, "n must be positive"),
        (1.2, -0.05, 1.0, "non-negative"),
        (1.2, 0.05, 1.5, r"\[-1, 1\]"),
    ],
)
def test_unphysical_inputs_raise(n: float, k: float, mu: float, message: str) -> None:
    """A negative k is a gain medium and would flip the branch; it fails loudly instead."""
    with pytest.raises(ValueError, match=message):
        fresnel_reflectance(n, k, mu)
