"""M7.14 — ε(θ) per pixel in stage 1, and the limb darkening it produces (§4.2, §13.5).

A uniform-temperature sphere under a cold sky does not read uniform: its limb reflects more of the
sky and emits less of itself, so it reads colder than its centre. The size of that effect is
available in closed form,

    ΔT_app ≈ (ε₀ − ε(θ)) · (L_B(T) − L_env) / (∂L_B/∂T),

and it is the cleanest available test of a directional model, because it goes to **zero** when the
environment is at the surface's own temperature — the isothermal enclosure again. A model that had
the angular dependence right but the reflected term wrong would pass the first half and fail the
second.

The prediction is **differential**, against the same scene rendered at ε₀, and that matters: under
a 220 K sky an ε = 0.90 surface at 300 K already reads **294.7 K** everywhere, from M7.13's
reflected term alone. Measuring the limb against the *kinetic* temperature would report a 5.3 K
error that has nothing to do with the angular model and would drown the ~7 K that does.

The directional path is **opt-in by packing**: stage 1 uses it only when the material table carries
an angle LUT (M7.10), so every scene and every golden written before this is bit-identical.

docs/physics-model.md §4.2, §13.5, §16.4 step 7; ADR 0042
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SPHERE_T_K = 300.0
COLD_SKY_K = 220.0


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


@pytest.fixture(scope="module")
def packed(library, boson):  # type: ignore[no-untyped-def]
    return MaterialTable.from_library(library, "lwir", boson, angle_lut=True, data_dir=DATA)


def _sphere_scene(packed, name: str, gbuffer_sphere):  # type: ignore[no-untyped-def]
    ids = np.where(
        gbuffer_sphere["material_id"] == 1, packed.id_for(name), packed.id_for(name)
    ).astype(np.int32)
    return ids, np.asarray(gbuffer_sphere["normal_dot_view"], dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# the identity, first
# ---------------------------------------------------------------------------------------------


def test_normal_incidence_gives_epsilon_zero_exactly(packed, tophat_lwir_lut: BandLUT) -> None:
    """n·v = 1 must return the authored band value, to 1e-12 in radiance."""
    ids = np.full((4, 4), packed.id_for("car_paint_black"), dtype=np.int32)
    t = np.full((4, 4), SPHERE_T_K, dtype=np.float32)
    env = np.full((4, 4), float(tophat_lwir_lut.lookup(COLD_SKY_K)[()]), dtype=np.float32)
    directional = band_radiance(
        t, ids, packed, tophat_lwir_lut, l_env=env, normal_dot_view=np.ones((4, 4), np.float32)
    )
    scalar = band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env)
    assert np.allclose(directional, scalar, rtol=1e-12, atol=0.0)


def test_the_directional_path_is_opt_in_by_packing(library, boson, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """A table without an angle LUT ignores `normal_dot_view` entirely — bit for bit.

    That is what keeps every golden written before M7.14 valid: the directional path appears
    because a caller packed the table for it, not because a G-buffer carries a plane it has
    always carried.
    """
    plain = MaterialTable.from_library(library, "lwir", boson)
    assert plain.angle_lut is None
    ids = np.full((8, 8), plain.id_for("glass_windshield"), dtype=np.int32)
    t = np.full((8, 8), SPHERE_T_K, dtype=np.float32)
    env = np.full((8, 8), float(tophat_lwir_lut.lookup(COLD_SKY_K)[()]), dtype=np.float32)
    cos_theta = np.linspace(1.0, 0.1, 64, dtype=np.float32).reshape(8, 8)
    with_plane = band_radiance(t, ids, plain, tophat_lwir_lut, l_env=env, normal_dot_view=cos_theta)
    without = band_radiance(t, ids, plain, tophat_lwir_lut, l_env=env)
    assert np.array_equal(with_plane, without)


def test_closure_holds_at_every_angle(packed, library) -> None:  # type: ignore[no-untyped-def]
    """ρ is re-derived at each angle, never carried: ε(θ) moves and τ does not, so ρ must absorb
    the difference or the pixel stops conserving energy the moment the surface tilts."""
    cos_theta = np.linspace(1.0, 0.0, 65, dtype=np.float32)
    for name in ("car_paint_black", "glass_windshield", "bare_aluminium", "water"):
        ids = np.full(cos_theta.shape, packed.id_for(name), dtype=np.int32)
        eps, rho, tau = packed.directional_properties_for(ids, cos_theta)
        closure = eps.astype(np.float64) + rho.astype(np.float64) + tau.astype(np.float64)
        assert np.all(np.abs(closure - 1.0) < 1e-6), name
        assert np.all(rho >= 0.0) and np.all(eps >= 0.0)


def test_the_sky_mask_still_reads_as_a_blackbody(packed) -> None:
    """A sky pixel carries an *apparent* temperature — a radiance dressed as a temperature — and
    has no surface to have an angle to."""
    ids = np.full((3, 3), packed.id_for("water"), dtype=np.int32)
    cos_theta = np.full((3, 3), 0.2, dtype=np.float32)
    sky = np.ones((3, 3), dtype=bool)
    eps, rho, tau = packed.directional_properties_for(ids, cos_theta, sky)
    assert np.all(eps == 1.0) and np.all(rho == 0.0) and np.all(tau == 0.0)


# ---------------------------------------------------------------------------------------------
# limb darkening on the sphere fixture
# ---------------------------------------------------------------------------------------------


def test_the_rim_reads_colder_than_the_centre_by_the_analytic_amount(
    packed, tophat_lwir_lut: BandLUT, gbuffer_sphere: dict[str, np.ndarray]
) -> None:
    """ΔT_app = (ε₀ − ε(θ))(L_B(T) − L_env)/(∂L_B/∂T), within 10 mK, on real geometry."""
    name = "car_paint_black"
    ids, cos_theta = _sphere_scene(packed, name, gbuffer_sphere)
    t = np.full(ids.shape, SPHERE_T_K, dtype=np.float32)
    l_env = float(tophat_lwir_lut.lookup(COLD_SKY_K)[()])
    env = np.full(ids.shape, l_env, dtype=np.float32)

    directional = tophat_lwir_lut.apparent_temperature(
        band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env, normal_dot_view=cos_theta)
    ).astype(np.float64)
    baseline = tophat_lwir_lut.apparent_temperature(
        band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env)
    ).astype(np.float64)
    # the ε₀ frame already sits well below the kinetic temperature, from the reflected term alone
    assert float(baseline.mean()) == pytest.approx(294.74, abs=0.05)

    eps, _, _ = packed.directional_properties_for(ids, cos_theta)
    eps0 = float(packed.emissivity[packed.id_for(name)])
    l_body = float(tophat_lwir_lut.lookup(SPHERE_T_K)[()])
    delta_l = -(eps0 - eps.astype(np.float64)) * (l_body - l_env)
    # Evaluate ∂L/∂T at the **midpoint** of the excursion, not at its start. The formula is a
    # first-order expansion and the limb moves 7.1 K inside 70°, over which Planck's slope is well
    # off constant; anchoring at the baseline leaves 254 mK of pure second-order term, twenty-five
    # times the 10 mK the criterion asks for. One midpoint iteration brings it to 9.0 mK.
    #
    # ⚠️ That is now a thin margin, and it got thin in M7.5: the excursion was ~1.4 K while the
    # painted materials carried an estimated (a = 0.25, p = 5), and the midpoint iteration had the
    # bound to itself. Re-fitting the class against Level A quintupled the excursion. If a future
    # material darkens its limb further, this needs a second iteration (or a secant step), not a
    # looser tolerance -- the tolerance is what says the closed form still describes the kernel.
    first = delta_l / float(tophat_lwir_lut.lookup(float(baseline.mean()), "dlb_dt")[()])
    slope = tophat_lwir_lut.lookup(baseline + 0.5 * first, "dlb_dt").astype(np.float64)
    predicted = delta_l / slope

    inside = cos_theta > np.cos(np.deg2rad(70.0))
    assert inside.sum() > 100
    measured = directional[inside] - baseline[inside]
    assert float(np.max(np.abs(measured - predicted[inside]))) < 0.010

    # and the effect is real: the rim is visibly colder than the centre
    rim = cos_theta < 0.4
    assert float(directional[rim].mean()) < float(directional[cos_theta > 0.99].mean()) - 0.3


def test_the_rim_effect_vanishes_in_an_isothermal_enclosure(
    packed, tophat_lwir_lut: BandLUT, gbuffer_sphere: dict[str, np.ndarray]
) -> None:
    """L_env = L_B(T): every angle must read T to under a millikelvin, for every ε(θ).

    This is the half a wrong reflected term fails. An implementation that varied ε with angle but
    forgot to move ρ with it would darken the limb here too, and look entirely plausible doing it.
    """
    ids, cos_theta = _sphere_scene(packed, "car_paint_black", gbuffer_sphere)
    t = np.full(ids.shape, SPHERE_T_K, dtype=np.float32)
    env = np.full(ids.shape, float(tophat_lwir_lut.lookup(SPHERE_T_K)[()]), dtype=np.float32)
    radiance = band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env, normal_dot_view=cos_theta)
    apparent = tophat_lwir_lut.apparent_temperature(radiance).astype(np.float64)
    assert float(np.max(np.abs(apparent - SPHERE_T_K))) < 0.001


def test_a_flat_material_darkens_its_limb_far_less_than_a_paint(
    packed, tophat_lwir_lut: BandLUT, gbuffer_sphere: dict[str, np.ndarray]
) -> None:
    """§4.2: a ≈ 0 for rough dielectrics. Asphalt's limb barely moves where paint's moves several
    kelvin — the difference a directional model is *for*.

    The paint figure was 1.44 K while the painted materials carried an *estimated* (a = 0.25,
    p = 5). M7.5 re-fitted the class against Level A on a measured acrylic, as §4.2 instructs, and
    the estimate turned out far too flat: (a = 0.75, p = 4) puts the same limb drop at 6.8 K.
    Asphalt is untouched, so the contrast between the two — the thing the test is really about —
    widens rather than moving.
    """
    drops = {}
    for name in ("car_paint_black", "asphalt_dry"):
        ids, cos_theta = _sphere_scene(packed, name, gbuffer_sphere)
        t = np.full(ids.shape, SPHERE_T_K, dtype=np.float32)
        env = np.full(ids.shape, float(tophat_lwir_lut.lookup(COLD_SKY_K)[()]), dtype=np.float32)
        apparent = tophat_lwir_lut.apparent_temperature(
            band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env, normal_dot_view=cos_theta)
        ).astype(np.float64)
        rim = (cos_theta > 0.3) & (cos_theta < 0.4)
        centre = cos_theta > 0.99
        drops[name] = float(apparent[centre].mean() - apparent[rim].mean())
    assert drops["car_paint_black"] > drops["asphalt_dry"]
    assert drops["asphalt_dry"] < 0.5, drops
    assert drops["car_paint_black"] > 2.5 * drops["asphalt_dry"], drops
    assert drops["car_paint_black"] == pytest.approx(6.80, abs=0.15), drops


def test_the_metal_brightens_its_limb_instead_of_darkening_it(
    packed, tophat_lwir_lut: BandLUT, gbuffer_sphere: dict[str, np.ndarray]
) -> None:
    """The sign flip, all the way through stage 1 and out the other side as apparent temperature.

    Bare aluminium's ε rises with angle, so its limb emits *more* and reflects *less* of the cold
    sky — it reads warmer than its centre, which is the opposite of every dielectric in the scene.
    """
    ids, cos_theta = _sphere_scene(packed, "bare_aluminium", gbuffer_sphere)
    t = np.full(ids.shape, SPHERE_T_K, dtype=np.float32)
    env = np.full(ids.shape, float(tophat_lwir_lut.lookup(COLD_SKY_K)[()]), dtype=np.float32)
    apparent = tophat_lwir_lut.apparent_temperature(
        band_radiance(t, ids, packed, tophat_lwir_lut, l_env=env, normal_dot_view=cos_theta)
    ).astype(np.float64)
    rim = (cos_theta > 0.3) & (cos_theta < 0.4)
    centre = cos_theta > 0.99
    assert float(apparent[rim].mean()) > float(apparent[centre].mean())
    # and the whole object reads far below its kinetic temperature, because ε is 0.09
    assert float(apparent[centre].mean()) < SPHERE_T_K - 50.0
