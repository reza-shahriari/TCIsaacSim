"""M7.6 — Level B: ε(θ) = ε₀[1 − a(1 − cos θ)^p], and the fit that bakes it (§4.2).

§4.2's instruction is "fit (a, p) once per material class against Level A and bake it", so Level A
is the oracle here and Level B is a compression of it. Two instructions in a shader, and close
enough over the angles a camera spends its time looking at.

The whole step turns on one thing the obvious implementation gets wrong: **an absolute residual
bar does not catch a metal.** Fresnel ε *rises* with angle for aluminium, which Level B cannot
represent at any a ≥ 0 — but aluminium's ε is 0.019 at normal and 0.030 at 70°, so a completely
wrong shape fits to an RMS of **0.0034**, well inside the 0.02 the roadmap asks for. The fit would
return a flat curve, the limb of every metal panel would render darker when it should be brighter,
and nothing would have complained. So direction is checked before error, on a relative scale.

docs/physics-model.md §4.2 Level B, §13.5; ADR 0042
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.materials.angular import (
    FIT_ANGLES_DEG,
    FIT_MAX_ANGLE_DEG,
    MAX_RMS_RESIDUAL,
    bake_angular_models,
    emissivity_empirical,
    fit_empirical,
    fit_from_nk,
)
from irsim.materials.fresnel import directional_emissivity_fresnel

#: (n, k) at 10 µm. Water is the checked-in Segelstein value; the rest are literature proxies of
#: the right class, which is all a *class* model needs.
NK_10UM = {
    "water": (1.218, 0.0508),
    "glass": (2.40, 0.15),
    "paint": (1.55, 0.02),
    "aluminium": (25.0, 68.0),
}
DIELECTRICS = ("water", "glass", "paint")
COS = np.cos(np.deg2rad(FIT_ANGLES_DEG))


# ---------------------------------------------------------------------------------------------
# the form
# ---------------------------------------------------------------------------------------------


def test_normal_incidence_returns_epsilon_zero_exactly() -> None:
    """ε(0) = ε₀ for every (a, p): (1 − cos 0)^p = 0, and it must be exactly 0, not nearly."""
    for a in (0.0, 0.25, 0.9):
        for p in (4.0, 5.0, 6.0):
            assert float(emissivity_empirical(0.87, a, p, 1.0)) == 0.87


def test_a_zero_is_a_constant_emissivity() -> None:
    """The Level C limit, reachable from inside Level B, so asphalt needs no special case."""
    values = emissivity_empirical(0.95, 0.0, 5.0, COS)
    assert np.allclose(values, 0.95, atol=0.0, rtol=0.0)


def test_a_back_facing_normal_uses_the_absolute_cosine() -> None:
    """Two sides of a thin panel have the same emissivity; which way the normal was authored is
    not physics. Same convention as fresnel_reflectance."""
    assert float(emissivity_empirical(0.9, 0.3, 4.0, -0.4)) == float(
        emissivity_empirical(0.9, 0.3, 4.0, 0.4)
    )


def test_the_falloff_is_monotone_and_bounded() -> None:
    values = emissivity_empirical(0.95, 0.5, 4.0, COS)
    assert np.all(np.diff(values) <= 1e-15), "ε must not rise with angle at a >= 0"
    assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0


def test_a_negative_a_is_refused_rather_than_evaluated() -> None:
    with pytest.raises(ValueError, match="rising"):
        emissivity_empirical(0.9, -0.1, 4.0, 0.5)
    with pytest.raises(ValueError, match="p must be positive"):
        emissivity_empirical(0.9, 0.1, 0.0, 0.5)


# ---------------------------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("material", DIELECTRICS)
def test_the_dielectric_proxies_fit_inside_the_class_tolerance(material: str) -> None:
    n, k = NK_10UM[material]
    fit = fit_from_nk(n, k, material)
    assert fit.rms_residual < MAX_RMS_RESIDUAL
    assert fit.max_residual < 0.02
    assert fit.p in (4.0, 5.0, 6.0), "p must come from §4.2's candidate set"
    assert 0.0 < fit.a <= 1.0
    assert fit.epsilon_0 == pytest.approx(
        float(directional_emissivity_fresnel(n, k, 1.0)), rel=1e-9
    )


def test_aluminium_is_refused_and_the_residual_would_not_have_caught_it() -> None:
    """The trap, both halves of it, in one test."""
    n, k = NK_10UM["aluminium"]
    with pytest.raises(ValueError, match="which is a metal"):
        fit_from_nk(n, k, "aluminium")

    # and here is why a residual bar alone is not enough: force the fit past the shape check by
    # asking for the best (a, p) directly, and look at how small its error is.
    epsilon = directional_emissivity_fresnel(n, k, COS)
    flat = emissivity_empirical(float(epsilon[0]), 0.0, 4.0, COS)
    rms = float(np.sqrt(np.mean((epsilon - flat) ** 2)))
    assert rms < MAX_RMS_RESIDUAL, rms
    assert rms == pytest.approx(0.0034, abs=0.001)
    assert float(epsilon[-1]) > float(epsilon[0]), "aluminium's ε must rise with angle"


def test_epsilon_zero_is_taken_from_the_data_and_not_fitted() -> None:
    """Letting the fit move ε₀ would make the normal-incidence value depend on how well the
    angular model happened to fit, which is backwards: ε₀ is what every other consumer uses."""
    epsilon = directional_emissivity_fresnel(*NK_10UM["water"], COS)
    fit = fit_empirical(COS, epsilon, "water")
    assert fit.epsilon_0 == float(epsilon[0])


def test_a_flat_curve_fits_with_a_equal_to_zero() -> None:
    """Rough dielectrics — asphalt, vegetation — are `a → 0` in §4.2's own words."""
    fit = fit_empirical(COS, np.full_like(COS, 0.95), "asphalt")
    assert fit.a == 0.0 and fit.is_flat
    assert fit.rms_residual == pytest.approx(0.0, abs=1e-12)


def test_the_fit_reproduces_its_own_parameters() -> None:
    """Round trip: generate from known (a, p), recover them."""
    for a, p in ((0.25, 4.0), (0.35, 5.0), (0.15, 6.0)):
        synthetic = emissivity_empirical(0.92, a, p, COS)
        fit = fit_empirical(COS, synthetic, f"synthetic a={a} p={p}")
        assert fit.p == p
        assert fit.a == pytest.approx(a, rel=1e-9)
        assert fit.rms_residual < 1e-12


def test_the_fit_stops_at_seventy_degrees() -> None:
    """§4.2's own bound. Beyond it Fresnel turns over towards ε → 0 and the forms diverge fast;
    including that region would drag the fit away from the angles that matter."""
    assert FIT_MAX_ANGLE_DEG == 70.0
    assert float(FIT_ANGLES_DEG.max()) == 70.0
    wide = np.cos(np.deg2rad(np.linspace(0.0, 89.0, 90)))
    epsilon = directional_emissivity_fresnel(*NK_10UM["water"], wide)
    fit = fit_empirical(wide, epsilon, "water")
    assert fit.n_samples < wide.size, "samples beyond 70 deg were not excluded"
    narrow = fit_from_nk(*NK_10UM["water"], "water")
    assert fit.a == pytest.approx(narrow.a, rel=0.05)


def test_too_few_samples_inside_the_range_is_refused() -> None:
    grazing = np.cos(np.deg2rad(np.array([80.0, 84.0, 88.0])))
    with pytest.raises(ValueError, match="fewer than three samples"):
        fit_empirical(grazing, np.array([0.7, 0.5, 0.3]), "grazing only")


# ---------------------------------------------------------------------------------------------
# baking a whole library
# ---------------------------------------------------------------------------------------------


def test_baking_returns_the_refusals_instead_of_swallowing_them() -> None:
    """A silently missing entry becomes a silently constant ε, the failure Level B fixes."""
    curves = {name: (COS, directional_emissivity_fresnel(*NK_10UM[name], COS)) for name in NK_10UM}
    fits, refused = bake_angular_models(curves, strict=False)
    assert set(fits) == set(DIELECTRICS)
    assert set(refused) == {"aluminium"}
    assert "fresnel" in refused["aluminium"]
    with pytest.raises(ValueError, match="which is a metal"):
        bake_angular_models(curves, strict=True)


def test_every_baked_fit_reproduces_level_a_where_a_camera_looks() -> None:
    """The acceptance criterion in the form that matters: ε per angle, not an RMS."""
    for name in DIELECTRICS:
        fit = fit_from_nk(*NK_10UM[name], name)
        cos_theta = np.cos(np.deg2rad(np.linspace(0.0, FIT_MAX_ANGLE_DEG, 71)))
        level_a = directional_emissivity_fresnel(*NK_10UM[name], cos_theta)
        level_b = emissivity_empirical(fit.epsilon_0, fit.a, fit.p, cos_theta)
        assert float(np.max(np.abs(level_a - level_b))) < 0.02, name
        # and the limb really is darker than the centre, for every dielectric
        assert float(level_b[-1]) < float(level_b[0]) - 0.05, name


def test_the_fitted_parameters_land_in_section_4_2s_quoted_ranges() -> None:
    """§4.2 says a ≈ 0.15–0.35 and p ≈ 4–6 "for painted metals and plastics".

    Measured here for the paint proxy: **p = 4** and **a = 0.78**, which is outside the quoted a
    range. Recorded rather than clamped: a is a shape parameter that trades against p, and §4.2's
    figures are for a fit over a different angular range and a different Level A parameterisation.
    What the acceptance criterion actually asks for — agreement with Level A to 0.02 over 0–70° —
    is met, and that is the claim being made.
    """
    fit = fit_from_nk(*NK_10UM["paint"], "paint")
    assert fit.p == 4.0
    assert fit.a == pytest.approx(0.78, abs=0.05)
    assert not (0.15 <= fit.a <= 0.35), "if this starts passing, §4.2's range has been reached"
    assert math.isclose(
        float(emissivity_empirical(fit.epsilon_0, fit.a, fit.p, math.cos(math.radians(60.0)))),
        float(directional_emissivity_fresnel(*NK_10UM["paint"], math.cos(math.radians(60.0)))),
        abs_tol=0.02,
    )
