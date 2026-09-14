"""M11.7 — the reflection lobe: near-mirror sky in LWIR, and solar glint in MWIR (§4.3, §5.4).

§4.3's claim is qualitative and strong: an infrared reflection lobe is narrower than the same
material's visible one, so painted bodywork and glass show near-mirror sky reflections in LWIR
while asphalt stays Lambertian. These tests turn that into numbers, and then check the numbers
against the two limits where the answer is known independently — a perfect mirror reads the sky
in one direction, a perfect diffuser reads the V_s blend of M7.13.

The interesting part is that a microfacet kernel **does not** reach the second limit on its own:
GGX at α = 1 sits a total-variation distance of 0.30 from the cosine hemisphere and does not
converge as α grows. That is why the split is explicit (ADR 0067), and one test measures the gap
so the decision stays visible rather than becoming folklore.

docs/physics-model.md §4.3, §5.4, §12.3, §13.7; ADR 0067
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.lobe import (
    SOLID_ANGLE_SUN_SR,
    ggx_ndf,
    glint_radiance,
    hemisphere_kernel,
    mirror_direction,
    reflected_radiance,
    roughness_to_alpha,
    specular_weight,
)
from irsim.pipeline.specular import (
    incident_field,
    specular_glint_radiance,
    specular_reflected_radiance,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
UP = np.array([0.0, 0.0, 1.0])
VIEW_ELEVATION_DEG = 30.0


def _view(elevation_deg: float) -> np.ndarray:
    """A direction from the surface toward a camera at this elevation."""
    e = math.radians(elevation_deg)
    return np.array([math.cos(e), 0.0, math.sin(e)])


# ---------------------------------------------------------------------------------------------
# the lobe itself
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("roughness", [0.05, 0.2, 0.5])
def test_the_white_furnace_closes_at_the_specular_weight(roughness: float) -> None:
    """Lobe × cos integrates to w_s within 1e-3 — the whole kernel to exactly 1."""
    kernel = hemisphere_kernel(_view(VIEW_ELEVATION_DEG), UP, roughness)
    assert float(kernel.weights.sum()) == pytest.approx(1.0, abs=1e-12)
    half = kernel.directions.shape[0] // 2
    assert float(kernel.weights[:half].sum()) == pytest.approx(
        float(specular_weight(roughness)), abs=1e-3
    )
    assert float(kernel.weights[half:].sum()) == pytest.approx(
        1.0 - float(specular_weight(roughness)), abs=1e-3
    )
    assert np.all(kernel.weights >= 0.0)


def test_specular_plus_diffuse_is_the_whole_reflectance(  # noqa: D103
) -> None:
    """ρ_spec + ρ_diff = ρ = 1 − ε to 1e-12, for every roughness — energy conservation."""
    for roughness in (0.0, 0.03, 0.12, 0.4, 0.8, 1.0):
        rho = 0.37
        kernel = hemisphere_kernel(_view(VIEW_ELEVATION_DEG), UP, roughness)
        uniform = np.ones(kernel.directions.shape[0])
        assert float(reflected_radiance(rho, kernel, uniform)) == pytest.approx(rho, abs=1e-12)
        w_s = float(specular_weight(roughness))
        assert rho * w_s + rho * (1.0 - w_s) == pytest.approx(rho, abs=1e-12)


def test_the_two_limits_are_exact() -> None:
    """Roughness 0 reads one direction; roughness 1 reads the cosine hemisphere."""
    view = _view(VIEW_ELEVATION_DEG)
    mirror = mirror_direction(view, UP)
    assert math.degrees(math.asin(float(mirror[2]))) == pytest.approx(VIEW_ELEVATION_DEG, abs=1e-9)

    def field(kernel):  # type: ignore[no-untyped-def]
        el = np.degrees(np.arcsin(np.clip(kernel.directions[:, 2], -1.0, 1.0)))
        return 250.0 + 40.0 * np.cos(np.radians(el))

    sharp = hemisphere_kernel(view, UP, 0.0)
    exact_mirror = 250.0 + 40.0 * math.cos(math.radians(VIEW_ELEVATION_DEG))
    assert float(sharp.average(field(sharp))) == pytest.approx(exact_mirror, rel=1e-6)

    rough = hemisphere_kernel(view, UP, 1.0)
    # the cosine-weighted hemisphere mean of 250 + 40 cos(el), el measured from the horizon:
    # ∫(250 + 40 sin θ) cos θ sin θ dθ dφ / π = 250 + 40 · (2/3)
    assert float(rough.average(field(rough))) == pytest.approx(250.0 + 40.0 * 2.0 / 3.0, rel=1e-6)


def test_a_microfacet_kernel_does_not_become_lambertian_on_its_own() -> None:
    """The measurement behind ADR 0067's explicit split, kept so the decision stays visible."""
    view = _view(VIEW_ELEVATION_DEG)
    pure_ggx = hemisphere_kernel(view, UP, 0.0)  # w_s = 1, so the kernel is all GGX
    alpha_one = hemisphere_kernel(view, UP, 1.0)
    assert pure_ggx.specular_weight == 1.0
    assert alpha_one.specular_weight == 0.0
    # with w_s forced to 1 at alpha = 1 the kernel would still not be the cosine hemisphere:
    from irsim.materials.lobe import _cone_nodes, kernel_value  # noqa: PLC0415

    dirs, solid = _cone_nodes(mirror_direction(view, UP), -1.0, 40, 64)
    above = np.sum(dirs * UP, axis=-1) > 0.0
    raw = np.where(above, kernel_value(view, UP, dirs, 1.0) * solid, 0.0)
    ggx_w = raw / raw.sum()
    cos_i = np.maximum(np.sum(dirs * UP, axis=-1), 0.0)
    lam_w = (cos_i / math.pi) * solid
    lam_w = lam_w / lam_w.sum()
    total_variation = 0.5 * float(np.abs(ggx_w - lam_w).sum())
    assert total_variation > 0.2, total_variation


@pytest.mark.parametrize("roughness", [0.03, 0.12, 0.3, 0.7])
def test_the_lobe_narrows_as_the_surface_smooths(roughness: float) -> None:
    view = _view(VIEW_ELEVATION_DEG)
    kernel = hemisphere_kernel(view, UP, roughness)
    mirror = mirror_direction(view, UP)
    half = kernel.directions.shape[0] // 2
    angles = np.degrees(
        np.arccos(np.clip(np.sum(kernel.directions[:half] * mirror, axis=-1), -1.0, 1.0))
    )
    w = kernel.weights[:half]
    spread = float((angles * w).sum() / w.sum())
    assert spread < 90.0
    if roughness <= 0.12:
        assert spread < 5.0, f"roughness {roughness} spread {spread:.2f} deg"


def test_the_library_reproduces_section_4_3s_ordering() -> None:
    """Glass near-mirror, paint mostly specular, asphalt near-Lambertian — in LWIR, from the
    authored roughness values, without the weight having been fitted to them."""
    glass, paint, asphalt = 0.03, 0.12, 0.70
    assert float(specular_weight(glass)) == pytest.approx(0.941, abs=0.002)
    assert float(specular_weight(paint)) == pytest.approx(0.774, abs=0.002)
    assert float(specular_weight(asphalt)) == pytest.approx(0.090, abs=0.002)
    assert (
        float(specular_weight(glass))
        > float(specular_weight(paint))
        > float(specular_weight(asphalt))
    )


def test_alpha_is_the_square_of_the_authored_roughness() -> None:
    assert float(roughness_to_alpha(0.3)) == pytest.approx(0.09, rel=1e-12)
    assert float(roughness_to_alpha(0.0)) > 0.0  # floored, never zero
    with pytest.raises(ValueError, match="roughness"):
        roughness_to_alpha(1.5)


def test_the_ndf_is_normalised() -> None:
    """∫ D cos θ_h dω_h = 1 by construction; checked by quadrature, not asserted."""
    for alpha in (0.05, 0.2, 0.6):
        nodes, gl = np.polynomial.legendre.leggauss(400)
        cos_h = 0.5 * (nodes + 1.0)
        d_cos = 0.5 * gl
        total = float(2.0 * math.pi * np.sum(ggx_ndf(cos_h, alpha) * cos_h * d_cos))
        assert total == pytest.approx(1.0, rel=2e-3), (alpha, total)


def test_a_view_below_the_surface_is_refused() -> None:
    with pytest.raises(ValueError, match="below the surface"):
        mirror_direction(np.array([1.0, 0.0, -0.1]), UP)


# ---------------------------------------------------------------------------------------------
# against a real sky
# ---------------------------------------------------------------------------------------------


def test_a_flat_mirror_reads_the_sky_it_points_at(aerial_sky, tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """`Lb⁻¹(ε Lb(T_s) + (1−ε) Lb(T_sky(30°)))` within 0.1 K — the §4.3 headline as a number."""
    eps, t_surface = 0.10, 300.0
    rho = 1.0 - eps
    view = _view(VIEW_ELEVATION_DEG)
    reflected = specular_reflected_radiance(rho, aerial_sky, 0.0, view, UP, roughness=0.0)
    emitted = eps * float(tophat_lwir_lut.lookup(t_surface)[()])
    reading = float(tophat_lwir_lut.apparent_temperature(np.asarray(emitted + reflected))[()])

    l_sky_30 = float(aerial_sky.radiance(0.0, math.radians(VIEW_ELEVATION_DEG))[()])
    expected = float(tophat_lwir_lut.apparent_temperature(np.asarray(emitted + rho * l_sky_30))[()])
    assert abs(reading - expected) < 0.1, (reading, expected)
    # and it is a *cold* reading, which is the phenomenon: a shiny panel under a clear sky
    assert reading < t_surface - 30.0


def test_roughness_one_recovers_the_hemisphere_blend(aerial_sky) -> None:  # type: ignore[no-untyped-def]
    """To 1e-6 against the tilt-LUT answer the M7.13 environment term already uses."""
    rho = 0.9
    view = _view(VIEW_ELEVATION_DEG)
    diffuse = specular_reflected_radiance(rho, aerial_sky, 0.0, view, UP, roughness=1.0)
    l_env = float(aerial_sky.effective_radiance(0.0, 0.0)[()])  # an upward-facing surface
    assert diffuse == pytest.approx(rho * l_env, rel=1e-6)


def test_the_ground_half_of_the_field_is_not_forgotten(aerial_sky) -> None:  # type: ignore[no-untyped-def]
    """A near-vertical panel reflects ground over half its lobe and must not read as sky.

    ``up`` is the world axis, not the normal: passing the normal would give a vertical panel a
    sky in every direction, which is the mistake the argument exists to prevent.
    """
    vertical = np.array([0.0, -1.0, 0.0])
    view = np.array([0.0, -1.0, 0.0])
    kernel = hemisphere_kernel(view, vertical, 0.6)
    field = incident_field(aerial_sky, 0.0, kernel, up=UP)
    sin_el = np.sum(kernel.directions * UP, axis=-1)
    assert np.any(sin_el > 0.0) and np.any(sin_el < 0.0), "the panel sees both halves"
    assert field[sin_el < 0.0].max() > field[sin_el > 0.0].min(), "ground is warmer than sky"
    mixed = float(reflected_radiance(0.9, kernel, field))
    sky_only = float(reflected_radiance(0.9, kernel, np.where(sin_el > 0.0, field, field.min())))
    assert mixed > sky_only


# ---------------------------------------------------------------------------------------------
# glint
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mwir():  # type: ignore[no-untyped-def]
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_lut_for_config

    cfg = load_sensor_config(REPO / "configs/sensors/example_mwir_insb_640.yaml", DATA)
    return cfg, load_band_lut_for_config(cfg, DATA / "lut", DATA)


def test_midday_mwir_glint_is_a_thousand_times_a_300_k_scene(mwir) -> None:  # type: ignore[no-untyped-def]
    """§5.4's glint: a smooth panel angled at the sun reads hundreds of kelvin hot in MWIR."""
    from irsim.pipeline.solar import SolarIllumination

    cfg, lut = mwir
    sun_elevation = 60.0
    sun = _view(sun_elevation)
    # the camera sits where the sun's reflection goes: reflection is symmetric, so the view
    # direction is the mirror of the sun about the normal, NOT the sun direction itself
    view = mirror_direction(sun, UP)
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    glint = specular_glint_radiance(
        0.10, view, UP, sun, roughness=0.03, band_irradiance=solar.e_band, tau_sun=0.82
    )
    lb_300 = float(lut.lookup(300.0)[()])
    assert glint / lb_300 > 1e3, f"glint/Lb(300 K) = {glint / lb_300:.3e}"
    apparent = float(lut.apparent_temperature(np.asarray(glint))[()])
    assert apparent > 600.0, apparent


def test_the_glint_never_exceeds_the_sun_it_came_from(mwir) -> None:  # type: ignore[no-untyped-def]
    """The cap is physics, not a guard: a mirror shows you the sun, not something brighter."""
    from irsim.pipeline.solar import SolarIllumination

    cfg, _ = mwir
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    sun = _view(60.0)
    l_sun = solar.e_band * 1.0 / (SOLID_ANGLE_SUN_SR * math.sin(math.radians(60.0)))
    view = mirror_direction(sun, UP)
    for roughness in (0.0, 0.001, 0.01, 0.03):
        glint = specular_glint_radiance(
            1.0, view, UP, sun, roughness=roughness, band_irradiance=solar.e_band
        )
        assert glint <= l_sun * (1.0 + 1e-9), (roughness, glint, l_sun)


def test_the_glint_is_zero_away_from_the_specular_direction_and_at_night(mwir) -> None:  # type: ignore[no-untyped-def]
    from irsim.pipeline.solar import SolarIllumination

    cfg, lut = mwir
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    sun = _view(60.0)
    view = mirror_direction(sun, UP)
    off_axis = specular_glint_radiance(
        0.1, _view(10.0), UP, sun, roughness=0.03, band_irradiance=solar.e_band
    )
    on_axis = specular_glint_radiance(
        0.1, view, UP, sun, roughness=0.03, band_irradiance=solar.e_band
    )
    assert off_axis < 1e-6 * on_axis

    # at night there is no beam at all, so the same pixel is the emissive answer exactly
    night = specular_glint_radiance(0.1, view, UP, sun, roughness=0.03, band_irradiance=0.0)
    assert night == 0.0
    below = specular_glint_radiance(
        0.1, view, UP, _view(-5.0), roughness=0.03, band_irradiance=solar.e_band
    )
    assert below == 0.0


def test_a_rough_surface_spreads_the_glint_instead_of_concentrating_it(mwir) -> None:  # type: ignore[no-untyped-def]
    from irsim.pipeline.solar import SolarIllumination

    cfg, _ = mwir
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    sun = _view(60.0)
    view = mirror_direction(sun, UP)
    values = [
        specular_glint_radiance(0.1, view, UP, sun, roughness=r, band_irradiance=solar.e_band)
        for r in (0.03, 0.12, 0.3, 0.6)
    ]
    assert all(b < a for a, b in zip(values, values[1:], strict=False)), values
    assert values[0] / values[-1] > 100.0


def test_the_glint_scales_with_the_beam_and_the_reflectance(mwir) -> None:  # type: ignore[no-untyped-def]
    from irsim.pipeline.solar import SolarIllumination

    cfg, _ = mwir
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    sun = _view(45.0)
    view = mirror_direction(sun, UP)
    base = specular_glint_radiance(
        0.1, view, UP, sun, roughness=0.12, band_irradiance=solar.e_band, tau_sun=0.8
    )
    assert specular_glint_radiance(
        0.2, view, UP, sun, roughness=0.12, band_irradiance=solar.e_band, tau_sun=0.8
    ) == pytest.approx(2.0 * base, rel=1e-12)
    assert specular_glint_radiance(
        0.1, view, UP, sun, roughness=0.12, band_irradiance=solar.e_band, tau_sun=0.4
    ) == pytest.approx(0.5 * base, rel=1e-12)


def test_the_glint_helper_and_the_kernel_agree_on_normalisation(mwir) -> None:  # type: ignore[no-untyped-def]
    from irsim.pipeline.solar import SolarIllumination

    cfg, _ = mwir
    solar = SolarIllumination.for_sensor(cfg, "lb", DATA)
    sun = _view(50.0)
    view = mirror_direction(sun, UP)
    kernel = hemisphere_kernel(view, UP, 0.12)
    direct = glint_radiance(
        0.1,
        view,
        UP,
        sun,
        0.12,
        solar.e_band,
        kernel_normalisation=kernel.specular_normalisation,
    )
    through = specular_glint_radiance(
        0.1, view, UP, sun, roughness=0.12, band_irradiance=solar.e_band
    )
    assert direct == pytest.approx(through, rel=1e-12)
