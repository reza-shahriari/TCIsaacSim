"""M7.16 — Tier 3 reflection phenomenology: what a facet *reads* at 03:00, not what it *is*.

Every other thermal test in this repository asks what temperature a surface reaches. This one asks
what a camera reads, which is a different number: at 03:00 under a clear sky a painted roof is
**colder in the picture than it is in fact**, because a tenth of what leaves it is a reflection of
a −40 °C sky. That gap is the whole reason §5.3 says reflections are not to be skipped.

The cleanest statement available is the **roof/wall pair**: the same paint at the same kinetic
temperature, differing only in how much sky it sees (V_s = 1 against V_s = 1/2). Their apparent
temperatures then differ by exactly

    ΔT_app = (1 − ε) · 0.5 · (L_B(T_ground) − L_sky,eff) / (∂L_B/∂T)

with nothing else in it. Both facets are evaluated at the *same* kinetic temperature on purpose —
in the scene they differ by 2.2 K, and comparing them as they stand would measure the thermal
model and the reflection model at once.

⚠️ This bench is how §6.1's missing ε on `Q_LW↓` was found: the bare-aluminium panel it needed came
out **19 K above the air at 03:00**, which a mirror at night cannot be. See the fix in
`irsim.thermal.balance`.

docs/physics-model.md §15 T3, §5.3, §4.1, §16.4 step 7; ADR 0045
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.materials.table import MaterialTable
from irsim.pipeline.environment import environment_radiance, ground_temperature_k
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response
from irsim.scene import Scene
from irsim.thermal.scene_forcing import sky_view_for_tilt

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SCENE_PATH = REPO / "configs" / "scenes" / "thermal_facet_scene.yaml"
NIGHT_UTC_H = 3.0  # 05:00 local, an hour before sunrise


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


@pytest.fixture(scope="module")
def lut(boson):  # type: ignore[no-untyped-def]
    return BandLUT.build(boson, n=1601)


@pytest.fixture(scope="module")
def table(boson):  # type: ignore[no-untyped-def]
    library = MaterialLibrary.load(REPO / "configs" / "materials")
    return MaterialTable.from_library(library, "lwir", boson)


def _scene(lut: BandLUT, overcast: bool = False) -> Scene:
    from irsim.config.scene import load_scene_config
    from irsim.thermal.weather import WEATHER_FIELDS, WeatherSeries

    config = load_scene_config(SCENE_PATH)
    if not overcast:
        return Scene.from_config(config, luts={"lwir": lut})
    base = Scene.from_file(SCENE_PATH).weather
    columns = {
        field: (
            np.full_like(np.asarray(base.columns()[field]), 0.98)
            if field == "cloud_fraction"
            else np.asarray(base.columns()[field])
        )
        for field in WEATHER_FIELDS
    }
    weather = WeatherSeries.from_arrays(base.epoch_utc, base.time_s, **columns)
    return Scene.from_config(config, luts={"lwir": lut}, weather_override=weather)


def _apparent(
    scene: Scene,
    lut: BandLUT,
    table: MaterialTable,
    name: str,
    material: str,
    tilt_deg: float,
    t_kinetic_k: float | None = None,
) -> float:
    """Apparent temperature of one facet through stage 1, with the scene's own sky."""
    t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
    scene.thermal.advance_to(t_s)
    kinetic = scene.surface_temperature_k(name, t_s) if t_kinetic_k is None else t_kinetic_k
    sky = scene.sky_models["lwir"]
    v_s = float(sky_view_for_tilt(tilt_deg))
    l_env = environment_radiance(sky, lut, t_s, np.array([[v_s]]), "lb")
    radiance = band_radiance(
        np.full((1, 1), kinetic, dtype=np.float32),
        np.full((1, 1), table.id_for(material), dtype=np.int32),
        table,
        lut,
        l_env=l_env,
    )
    return float(lut.apparent_temperature(radiance)[0, 0])


# ---------------------------------------------------------------------------------------------
# a clear night
# ---------------------------------------------------------------------------------------------


def test_a_painted_roof_reads_colder_than_it_is(lut, table) -> None:  # type: ignore[no-untyped-def]
    """ε 0.90 facing straight up under a clear sky: at least 3 K below its kinetic temperature."""
    scene = _scene(lut)
    t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
    scene.thermal.advance_to(t_s)
    kinetic = scene.surface_temperature_k("roof_black", t_s)
    apparent = _apparent(scene, lut, table, "roof_black", "car_paint_black", 0.0)
    assert kinetic - apparent > 3.0, (kinetic, apparent)


def test_glass_reads_colder_too_and_rough_asphalt_barely_moves(lut, table) -> None:  # type: ignore[no-untyped-def]
    """The ordering is ε, not material: asphalt at 0.94 is within a kelvin of its own temperature
    while glass at 0.88 is not."""
    scene = _scene(lut)
    t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
    scene.thermal.advance_to(t_s)

    glass_gap = scene.surface_temperature_k("glass", t_s) - _apparent(
        scene, lut, table, "glass", "glass_windshield", 30.0
    )
    asphalt_gap = scene.surface_temperature_k("asphalt_sun", t_s) - _apparent(
        scene, lut, table, "asphalt_sun", "asphalt_dry", 0.0
    )
    assert glass_gap > 3.0
    # ⚠️ asphalt moves 2.0 K, not the "within 1 K" the roadmap asks for. At ε = 0.94 the reflected
    # fraction is 6 %, and this sky is 54 K below the air at zenith, so 6 % of that is 2 K -- the
    # criterion assumed a shallower sky depression than `clear_dry` actually specifies. Even the
    # roughest, blackest surface in the library is not reflection-free on a clear night.
    assert abs(asphalt_gap) == pytest.approx(2.0, abs=0.5)
    assert glass_gap > 1.8 * abs(asphalt_gap)


def test_bare_aluminium_reads_as_the_sky_rather_than_as_itself(lut, table) -> None:  # type: ignore[no-untyped-def]
    """ε 0.09: nine tenths of what leaves it is a reflection, so it reads within 5 K of the sky.

    §16.2 names this as one of the two rows most likely to break a naive simulator — "aluminium
    because ε = 0.09 means it is a mirror, not a surface". Its *kinetic* temperature is close to
    the air; its *apparent* temperature is forty degrees lower.
    """
    scene = _scene(lut)
    t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
    scene.thermal.advance_to(t_s)
    sky = scene.sky_models["lwir"]
    # The **effective** sky for an upward-facing surface (V_s = 1), not the zenith value: a flat
    # panel sees the whole hemisphere, and the horizon is 13 K warmer than the zenith. Comparing
    # against zenith is the mistake this line exists to avoid.
    t_sky = float(
        lut.apparent_temperature(
            np.asarray(sky.effective_radiance_from_sky_view(t_s, 1.0), dtype=np.float32)
        )[()]
    )
    kinetic = scene.surface_temperature_k("alu_panel", t_s)
    apparent = _apparent(scene, lut, table, "alu_panel", "bare_aluminium", 0.0)
    # 5.5 K, not the "within 5 K" the roadmap asks for -- and the residual is not slop, it is
    # exactly the tenth of its own emission the panel still contributes. The identity below pins
    # that: the reading is Lb⁻¹(ε Lb(T_s) + (1−ε) L_sky,eff) with nothing else in it.
    assert abs(apparent - t_sky) == pytest.approx(5.5, abs=1.0)
    assert abs(apparent - t_sky) < abs(kinetic - apparent) / 4.0
    assert kinetic - apparent > 20.0, "a mirror must read far colder than it is"

    epsilon = float(table.emissivity[table.id_for("bare_aluminium")])
    l_sky = float(sky.effective_radiance_from_sky_view(t_s, 1.0)[()])
    expected = float(
        lut.apparent_temperature(
            np.asarray(
                epsilon * float(lut.lookup(np.float32(kinetic))[()]) + (1.0 - epsilon) * l_sky,
                dtype=np.float32,
            )
        )[()]
    )
    assert apparent == pytest.approx(expected, abs=0.05)


def test_the_roof_wall_gap_is_the_closed_form_view_factor_difference(lut, table) -> None:  # type: ignore[no-untyped-def]
    """ΔT_app = (1 − ε)·0.5·(L_B(T_ground) − L_sky,eff)/(∂L_B/∂T), within **0.1 K**.

    Both facets are evaluated at the **same kinetic temperature** on purpose. In the scene they
    differ by 2.2 K, and comparing them as they stand would be measuring the thermal model and
    the reflection model at the same time; holding the temperature fixed leaves only V_s.
    """
    scene = _scene(lut)
    t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
    scene.thermal.advance_to(t_s)
    kinetic = scene.surface_temperature_k("roof_black", t_s)

    roof = _apparent(scene, lut, table, "roof_black", "car_paint_black", 0.0, kinetic)
    wall = _apparent(scene, lut, table, "wall_black", "car_paint_black", 90.0, kinetic)

    sky = scene.sky_models["lwir"]
    epsilon = float(table.emissivity[table.id_for("car_paint_black")])
    l_ground = float(lut.lookup(np.float64(ground_temperature_k(sky, t_s)))[()])
    # ⚠️ **L_sky,eff is not one number: it depends on the tilt.** The roadmap's form,
    # (1−ε)·0.5·(L_ground − L_sky,eff), assumes the roof and the wall see sky of the same
    # effective radiance and differ only in how much of it. They do not -- the tilt LUT is a
    # cosine-weighted average over the sky each facet can actually see, and a wall's half is the
    # warmer half near the horizon. Using one L_sky for both is 19 % low.
    l_sky_roof = float(sky.effective_radiance_from_sky_view(t_s, 1.0)[()])
    l_sky_wall = float(sky.effective_radiance_from_sky_view(t_s, 0.5)[()])
    delta_l = (1.0 - epsilon) * ((0.5 * l_sky_wall + 0.5 * l_ground) - l_sky_roof)
    naive = (1.0 - epsilon) * 0.5 * (l_ground - l_sky_roof)
    assert l_sky_wall > l_sky_roof, "a wall sees the warmer, near-horizon half of the sky"

    assert wall > roof, "the facet that sees less sky must read warmer"
    assert delta_l > 0.5, "the effect must be big enough to be worth a test"

    # In **radiance**, where the relation is exact: the wall's radiance is the roof's plus ΔL.
    l_roof = float(lut.lookup(np.float32(roof))[()])
    l_wall = float(lut.lookup(np.float32(wall))[()])
    assert (l_wall - l_roof) == pytest.approx(delta_l, rel=2e-3)
    assert naive / delta_l == pytest.approx(0.84, abs=0.05), "the one-sky form is ~19 % low"

    # With the tilt-dependent sky in it, the linearised form lands too: ΔT ≈ ΔL/(∂L_B/∂T) with
    # the slope at the midpoint of the two apparent temperatures.
    slope = float(lut.lookup(np.float64(0.5 * (roof + wall)), "dlb_dt")[()])
    assert (wall - roof) == pytest.approx(delta_l / slope, abs=0.1)


# ---------------------------------------------------------------------------------------------
# overcast
# ---------------------------------------------------------------------------------------------


def test_overcast_closes_the_gap_for_every_facet(lut, table) -> None:  # type: ignore[no-untyped-def]
    """With the sky at nearly ambient there is nothing cold to reflect, so every facet reads its
    own temperature — and the scene's spread of *apparent* temperatures collapses with it."""
    clear = _scene(lut)
    cloudy = _scene(lut, overcast=True)
    facets = [
        ("roof_black", "car_paint_black", 0.0),
        ("glass", "glass_windshield", 30.0),
        ("asphalt_sun", "asphalt_dry", 0.0),
        ("wall_black", "car_paint_black", 90.0),
        ("alu_panel", "bare_aluminium", 0.0),
    ]
    gaps = {}
    for scene, label in ((clear, "clear"), (cloudy, "overcast")):
        t_s = scene.t0_s + NIGHT_UTC_H * 3600.0
        scene.thermal.advance_to(t_s)
        gaps[label] = [
            scene.surface_temperature_k(name, t_s)
            - _apparent(scene, lut, table, name, material, tilt)
            for name, material, tilt in facets
        ]
    assert max(abs(g) for g in gaps["overcast"]) < 0.5 * max(abs(g) for g in gaps["clear"])
    assert float(np.std(gaps["overcast"])) < 0.5 * float(np.std(gaps["clear"]))
    # the mirror is where the difference is most dramatic
    assert abs(gaps["clear"][-1]) > 20.0
    assert abs(gaps["overcast"][-1]) < 0.5 * abs(gaps["clear"][-1])
