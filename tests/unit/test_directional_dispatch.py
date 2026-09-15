"""M7.7 — one entry point for ε(θ), dispatching on the level the *material* declares (§4.2, §13.3).

Level A is Fresnel from a checked-in n/k table, B is the baked empirical falloff of M7.6, C is a
constant. The choice belongs to the material, not to the caller, and the CPU answer here is the
oracle the Warp/SPG kernel's ε(θ) is compared against.

§4.2 permits Level C "only for rough, high-emissivity surfaces (ε > 0.93)" and names vehicle
bodies, glass and water as violating it. That bound is **enforced, not warned about**, because the
error it hides is large and one-sided — a constant ε renders every limb too warm, everywhere, in a
way that looks like a perfectly plausible scene. Enforcing it immediately found one violation in
the committed library, which this module pins.

docs/physics-model.md §4.2, §13.3, §13.5; ADR 0042
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.angular import (
    FIT_MAX_ANGLE_DEG,
    emissivity_empirical,
    fit_band_level_b,
    fit_from_nk,
)
from irsim.materials.directional import (
    LEVEL_C_MIN_EPSILON,
    angular_level,
    directional_emissivity,
)
from irsim.materials.library import MaterialLibrary
from irsim.materials.nk import band_directional_emissivity, load_nk_table
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COS_70 = math.cos(math.radians(FIT_MAX_ANGLE_DEG))


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


def _dispatch(material, band, cos_theta, boson):  # type: ignore[no-untyped-def]
    return directional_emissivity(material, band, cos_theta, boson, data_dir=DATA)


# ---------------------------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------------------------


def test_every_committed_material_declares_a_level(library) -> None:  # type: ignore[no-untyped-def]
    levels = {name: angular_level(library[name]) for name in library.names}
    assert set(levels.values()) <= {"A", "B", "C"}
    assert "B" in levels.values(), "nothing uses the empirical model M7.6 exists to bake"


def test_the_dispatch_preserves_float32(library, boson) -> None:  # type: ignore[no-untyped-def]
    material = library["car_paint_black"]
    cos_theta = np.linspace(1.0, 0.0, 17, dtype=np.float32)
    out = _dispatch(material, "lwir", cos_theta, boson)
    assert out.dtype == np.float32
    assert out.shape == cos_theta.shape
    with pytest.raises(TypeError, match="float16"):
        _dispatch(material, "lwir", cos_theta.astype(np.float16), boson)


def test_a_back_facing_normal_uses_the_absolute_cosine(library, boson) -> None:  # type: ignore[no-untyped-def]
    material = library["car_paint_black"]
    front = _dispatch(material, "lwir", np.float32(0.4), boson)
    back = _dispatch(material, "lwir", np.float32(-0.4), boson)
    assert float(front) == float(back)


def test_the_answer_is_finite_at_grazing(library, boson) -> None:  # type: ignore[no-untyped-def]
    """cos θ = 0 must be a number. A formulation that divided by cos θ would not be."""
    for name in library.names:
        material = library[name]
        if (
            angular_level(material) == "C"
            and material.band_properties("lwir").emissivity < LEVEL_C_MIN_EPSILON
        ):
            continue
        value = _dispatch(material, "lwir", np.float32(0.0), boson)
        assert np.isfinite(value).all(), name
        assert 0.0 <= float(value) <= 1.0, name


def test_normal_incidence_returns_the_band_value(library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε(0) must be the ε_B every other consumer of the material already uses."""
    for name in library.names:
        material = library[name]
        if (
            angular_level(material) == "C"
            and material.band_properties("lwir").emissivity < LEVEL_C_MIN_EPSILON
        ):
            continue
        expected = float(material.band_properties("lwir", boson).emissivity)
        assert float(_dispatch(material, "lwir", np.float32(1.0), boson)) == pytest.approx(
            expected, rel=1e-6
        ), name


def test_the_limb_is_darker_than_the_centre_for_every_dielectric(library, boson) -> None:  # type: ignore[no-untyped-def]
    for name in library.names:
        material = library[name]
        if angular_level(material) != "B":
            continue
        centre = float(_dispatch(material, "lwir", np.float32(1.0), boson))
        limb = float(_dispatch(material, "lwir", np.float32(COS_70), boson))
        assert limb < centre, name


# ---------------------------------------------------------------------------------------------
# Level C's bound, enforced
# ---------------------------------------------------------------------------------------------


def test_level_c_is_refused_below_the_section_4_2_bound(library, boson) -> None:  # type: ignore[no-untyped-def]
    """⚠️ The committed library violates §4.2 once, and this pins it rather than hiding it.

    `bare_aluminium` declares `angular_model: constant` with ε_LWIR far below 0.93 — and it is a
    *metal*, whose ε rises with angle, so a constant is wrong in the one direction Level B cannot
    fix either (M7.6 refuses to fit it). The correct model is Level A, which needs an aluminium
    n/k table; M7.5 has shipped only water so far. Until it does, any scene that needs bare
    metal's ε(θ) raises here instead of quietly rendering a flat limb.
    """
    aluminium = library["bare_aluminium"]
    assert angular_level(aluminium) == "C"
    assert float(aluminium.band_properties("lwir", boson).emissivity) < LEVEL_C_MIN_EPSILON
    with pytest.raises(ValueError, match="§4.2"):
        _dispatch(aluminium, "lwir", np.float32(1.0), boson)


def test_level_c_is_allowed_for_a_rough_high_emissivity_surface(library, boson) -> None:  # type: ignore[no-untyped-def]
    """The other side of the bound: asphalt is exactly what §4.2 permits a constant for."""
    asphalt = library["asphalt_dry"]
    assert angular_level(asphalt) == "C"
    assert float(asphalt.band_properties("lwir", boson).emissivity) >= LEVEL_C_MIN_EPSILON
    values = _dispatch(asphalt, "lwir", np.linspace(1.0, 0.0, 9, dtype=np.float32), boson)
    assert float(values.std()) == 0.0, "a constant model must be constant"


# ---------------------------------------------------------------------------------------------
# Level A against baked Level B
# ---------------------------------------------------------------------------------------------


def test_baked_level_b_reproduces_level_a_on_the_sphere_fixture(
    gbuffer_sphere: dict[str, np.ndarray], boson
) -> None:  # type: ignore[no-untyped-def]
    """The acceptance criterion, on real geometry: agreement to 0.02 wherever cos θ > cos 70°.

    Water, because it is the one material with a checked-in n/k table (M7.5) and because a sea
    surface is the case the whole angular model exists for — its ε collapses from 0.99 at nadir
    to 0.70 at 80°, which no constant can describe.
    """
    table = load_nk_table("water", DATA)
    cos_theta = np.asarray(gbuffer_sphere["normal_dot_view"], dtype=np.float64)
    inside = cos_theta > COS_70
    assert inside.sum() > 100, "the fixture does not reach enough of the fitted range"

    level_a = band_directional_emissivity(table, boson, cos_theta[inside])
    fit = fit_band_level_b(table, boson, "water")
    level_b = emissivity_empirical(fit.epsilon_0, fit.a, fit.p, cos_theta[inside])
    assert float(np.max(np.abs(level_a - level_b))) < 0.02

    # ⚠️ and the fit must see the *band*, not a representative wavelength. Fitting at 10 µm and
    # comparing against the band average is a real error: 0.0275 at 70°, past the criterion.
    single = fit_from_nk(
        float(np.interp(10.0, table.wavelength_um, table.n)),
        float(np.interp(10.0, table.wavelength_um, table.k)),
        "water at 10 um",
    )
    single_b = emissivity_empirical(single.epsilon_0, single.a, single.p, cos_theta[inside])
    assert float(np.max(np.abs(level_a - single_b))) > 0.02


def test_the_sphere_fixture_reaches_grazing_where_the_two_levels_must_diverge(
    gbuffer_sphere: dict[str, np.ndarray], boson
) -> None:
    """Outside 70° the forms part company, which is why the fit stops there and why the
    agreement above is claimed only inside it."""
    table = load_nk_table("water", DATA)
    cos_theta = np.asarray(gbuffer_sphere["normal_dot_view"], dtype=np.float64)
    grazing = cos_theta[(cos_theta > 0.0) & (cos_theta < 0.1)]
    assert grazing.size > 0, "the fixture never reaches grazing incidence"
    level_a = band_directional_emissivity(table, boson, grazing)
    assert float(level_a.min()) < 0.75, "water's angular collapse is missing"
