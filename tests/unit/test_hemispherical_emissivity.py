"""M7.8 — the ε a thermal solver needs, which is not the ε a camera sees (§6.1, §6.2, §4.2).

A camera measures ε(θ) along one ray; an energy balance radiates into the whole sky. The two
differ by more than a rounding — water reads **0.990** at normal and **0.951** hemispherically,
because the cos θ sin θ weight peaks at 45° where water's collapse has already begun — and the
difference is systematic and one-signed, so it does not average out over a scene.

**The sign flips for metals**, which is the part that would survive a long time if it were wrong:
aluminium's ε rises with angle, so its ε_hemi is *above* its normal value. Any model that assumed
ε_hemi ≤ ε(0) would be right for every dielectric in the library and wrong for every metal.

docs/physics-model.md §6.1, §6.2, §4.2; ADR 0043
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.fresnel import directional_emissivity_fresnel
from irsim.materials.hemispherical import (
    NEGLIGIBLE_BAND_WEIGHT,
    band_hemispherical_emissivity,
    hemispherical_emissivity,
    total_hemispherical_emissivity,
)
from irsim.materials.library import MaterialLibrary
from irsim.radiometry.constants import SIGMA_SB
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.thermal.longwave import clear_sky_emissivity

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
#: [R1]'s water optical constants at 10 µm — the pair §4.2's Level A was checked against in M7.4.
WATER_R1_NK = (1.218, 0.0508)
ALUMINIUM_NK = (25.0, 68.0)


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


# ---------------------------------------------------------------------------------------------
# the integral
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 0.93, 1.0])
def test_a_constant_emissivity_integrates_to_itself(value: float) -> None:
    """2∫₀¹ ε µ dµ = ε for constant ε — exactly, since 2∫µ dµ = 1. To 1e-12, not approximately."""
    got = hemispherical_emissivity(lambda mu: np.full_like(mu, value))
    assert got == pytest.approx(value, abs=1e-12)


def test_the_quadrature_has_converged_far_below_the_node_count_used() -> None:
    """16 vs 64 nodes to 1e-6 — asserted, rather than 32 being asserted to be "enough"."""

    def epsilon(mu: np.ndarray) -> np.ndarray:
        return np.asarray(directional_emissivity_fresnel(*WATER_R1_NK, mu))

    coarse = hemispherical_emissivity(epsilon, 16)
    fine = hemispherical_emissivity(epsilon, 64)
    assert coarse == pytest.approx(fine, abs=1e-6)
    assert hemispherical_emissivity(epsilon, 8) == pytest.approx(fine, abs=1e-6)


def test_water_reads_lower_hemispherically_than_at_nadir() -> None:
    """The headline number: 0.951 against 0.990, on [R1]'s own optical constants."""
    normal = float(directional_emissivity_fresnel(*WATER_R1_NK, 1.0))
    hemi = hemispherical_emissivity(
        lambda mu: np.asarray(directional_emissivity_fresnel(*WATER_R1_NK, mu))
    )
    assert normal == pytest.approx(0.990, abs=0.002)
    assert hemi == pytest.approx(0.951, abs=0.002)
    assert hemi < normal - 0.03


def test_the_sign_flips_for_a_metal() -> None:
    """Aluminium's ε *rises* with angle, so ε_hemi is above ε(0). A model assuming otherwise
    would be right for every dielectric in the library and wrong for every metal."""
    normal = float(directional_emissivity_fresnel(*ALUMINIUM_NK, 1.0))
    hemi = hemispherical_emissivity(
        lambda mu: np.asarray(directional_emissivity_fresnel(*ALUMINIUM_NK, mu))
    )
    assert hemi > normal
    assert hemi / normal == pytest.approx(1.29, rel=0.05)


def test_a_callable_returning_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="one value per"):
        hemispherical_emissivity(lambda mu: np.zeros(3))
    with pytest.raises(ValueError, match="n_nodes"):
        hemispherical_emissivity(lambda mu: np.zeros_like(mu), 1)


# ---------------------------------------------------------------------------------------------
# through the material library
# ---------------------------------------------------------------------------------------------


def test_level_c_materials_integrate_to_their_own_constant(library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε_hemi == ε_B to 1e-12 when the material declares no angular variation."""
    from irsim.materials.directional import angular_level

    for name in library.names:
        material = library[name]
        if angular_level(material) != "C":
            continue
        epsilon_b = float(material.band_properties("lwir", boson).emissivity)
        if epsilon_b < 0.93:
            continue  # refused by §4.2's bound; covered in test_directional_dispatch
        # 1e-7, not 1e-12: the dispatch returns float32 (the G-buffer's dtype), so the
        # comparison is limited by float32 spacing near 1, not by the quadrature, which is exact
        # for a constant. The exact-to-1e-12 version of this identity is the pure-integral test
        # above, which never passes through float32.
        assert band_hemispherical_emissivity(
            material, "lwir", boson, data_dir=DATA
        ) == pytest.approx(epsilon_b, abs=1e-7), name


def test_level_b_materials_read_lower_hemispherically(library, boson) -> None:  # type: ignore[no-untyped-def]
    from irsim.materials.directional import angular_level

    checked = 0
    for name in library.names:
        material = library[name]
        if angular_level(material) != "B":
            continue
        normal = float(material.band_properties("lwir", boson).emissivity)
        hemi = band_hemispherical_emissivity(material, "lwir", boson, data_dir=DATA)
        assert hemi < normal, name
        assert hemi > normal - 0.10, name
        checked += 1
    assert checked >= 5


# ---------------------------------------------------------------------------------------------
# the total form, and how much of it is an assumption
# ---------------------------------------------------------------------------------------------


def test_the_total_form_reports_how_much_of_itself_is_extrapolated(library, boson) -> None:  # type: ignore[no-untyped-def]
    """⚠️ **61 % of Planck's weight at 300 K falls outside every configured band**, mostly beyond
    13.5 µm, and is filled by extending the nearest one.

    That is the standard assumption for a thermal solver and it is probably close to right for a
    dielectric — but it is an assumption, and a solver that cannot see how much of its ε rests on
    one cannot report its own uncertainty. So the number is returned, not buried.
    """
    result = total_hemispherical_emissivity(
        library["car_paint_black"], 300.0, responses={"lwir": boson}, data_dir=DATA
    )
    assert 0.0 < result.value < 1.0
    assert result.extrapolated_fraction == pytest.approx(0.607, abs=0.02)
    assert result.temperature_k == 300.0


def test_bands_with_no_planck_weight_are_skipped_rather_than_queried(library, boson) -> None:  # type: ignore[no-untyped-def]
    """At 300 K, NIR and SWIR hold ~1e-9 of the exitance between them.

    Querying a material's angular model there asks a question whose answer cannot move the energy
    balance — and it *can* fail, because §4.2's Level C bound is a statement about thermal-band
    roughness and a material may legally be Level-C-invalid in a band the solver never integrates.
    """
    result = total_hemispherical_emissivity(
        library["car_paint_black"], 300.0, responses={"lwir": boson}, data_dir=DATA
    )
    assert result.bands_used == ("lwir", "mwir")
    assert "nir" not in result.bands_used and "swir" not in result.bands_used
    # and at a temperature where SWIR *does* carry weight, it comes back
    hot = total_hemispherical_emissivity(library["car_paint_black"], 1500.0, data_dir=DATA)
    assert "swir" in hot.bands_used
    assert NEGLIGIBLE_BAND_WEIGHT == 1e-4


def test_the_total_is_bounded_by_the_band_values_it_was_built_from(library, boson) -> None:  # type: ignore[no-untyped-def]
    result = total_hemispherical_emissivity(
        library["human_skin"], 300.0, responses={"lwir": boson}, data_dir=DATA
    )
    per_band = [
        band_hemispherical_emissivity(library["human_skin"], band, None, data_dir=DATA)
        for band in result.bands_used
    ]
    assert min(per_band) - 1e-9 <= result.value <= max(per_band) + 1e-9


# ---------------------------------------------------------------------------------------------
# what the difference costs
# ---------------------------------------------------------------------------------------------


def _equilibrium_temperature(epsilon: float, t_air_k: float = 288.15, h: float = 8.0) -> float:
    """Steady state of ε σ (T⁴ − ε_sky T_air⁴) + h (T_air − T) = 0, by bisection.

    A single node under a clear sky with free convection — §6.1's balance with the solar term off,
    which is a clear night. The absolute numbers depend on h; the *difference* between two ε
    values does not, which is what is being measured.
    """
    from irsim.atmosphere.humidity import vapour_pressure_hpa

    eps_sky = clear_sky_emissivity(vapour_pressure_hpa(t_air_k, 0.5), t_air_k)

    def residual(t: float) -> float:
        return epsilon * SIGMA_SB * (t**4 - eps_sky * t_air_k**4) - h * (t_air_k - t)

    lo, hi = t_air_k - 40.0, t_air_k + 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_using_the_directional_value_instead_of_the_hemispherical_one_costs_kelvins() -> None:
    """ε 0.990 against 0.951 on a clear night: **0.230 K at worst**, systematic and one-signed.

    ⚠️ The roadmap asked for > 0.3 K and a convection-coupled single-node balance **cannot reach
    it at any h**, which is a property of the balance rather than a tolerance to be loosened. The
    difference vanishes at both limits — with no convection the radiative equilibrium is
    T = ε_sky^{1/4} T_air, independent of ε, and with very strong convection the surface is pinned
    to the air — so it has a maximum in between. This test finds that maximum (0.230 K near
    h = 4 W m⁻² K⁻¹) rather than asserting one point, which is the stronger statement and the one
    that survives a change of h.

    0.23 K is small. It is also in the same direction for every water pixel of every maritime
    scene, at every hour, which is why the solver must not be able to reach the directional band
    value at all.
    """
    deltas = {
        h: _equilibrium_temperature(0.951, h=h) - _equilibrium_temperature(0.990, h=h)
        for h in (0.05, 0.5, 2.0, 3.0, 4.0, 5.0, 8.0, 20.0, 2000.0)
    }
    assert all(d > 0.0 for d in deltas.values()), "a lower ε must always cool less"
    peak_h = max(deltas, key=lambda h: deltas[h])
    assert deltas[peak_h] == pytest.approx(0.230, abs=0.01), deltas
    assert peak_h == 4.0, deltas
    assert max(deltas.values()) < 0.3, "the roadmap's > 0.3 K target is unreachable here"
    # and it really does vanish at both ends, which is why a maximum exists
    assert deltas[0.05] < 0.02
    assert deltas[2000.0] < 0.01


def test_radiative_equilibrium_alone_does_not_depend_on_epsilon() -> None:
    """The reason the difference above has a maximum: with h = 0, ε cancels out of the balance.

    ε σ (T⁴ − ε_sky T_air⁴) = 0 has the same root for every ε > 0. Worth pinning, because it is
    the thing that makes "a higher ε means a colder surface" true only in the presence of a
    competing heat path.
    """
    hot = _equilibrium_temperature(0.99, h=1e-6)
    cold = _equilibrium_temperature(0.50, h=1e-6)
    assert hot == pytest.approx(cold, abs=1e-3)
