"""PT.1 — a wall half in sun, which is the case point-wise temperature exists for.

ADR 0087's own headline example is a wall spanning 10–20 K because part of it is lit and part is
shaded. Until this step nothing could produce it. A patch gave every cell its own temperature, but
`SceneSurfaceForcing` carried one `shaded` bool per **surface**, so the only spatial variation a
field could express came from `q_internal_w_m2` — an engine bay under a bonnet. That is why both
point-wise scenes shipped so far are pre-dawn: the machinery could not have rendered a sunlit one
any differently from a flat surface.

The load-bearing test is `test_a_wall_half_in_sun_splits_by_more_than_eight_kelvin`, with
`test_a_per_surface_shaded_bool_gives_one_temperature` as its negative control: the old
representation is run on the same wall and fails by the full step, which is what makes the new one
worth its cost.

docs/physics-model.md §6.1, §5.2; ADR 0087; roadmap PT.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.thermal.balance import SurfaceForcing, ThermalProperties, steady_state_temperature
from irsim.thermal.facets import FacetForcing, FacetProperties
from irsim.thermal.shadow import ShadowRectangle, cell_shadow, patch_solar_loading
from irsim.thermal.solar import solar_loading, sun_direction
from irsim.thermal.surface_field import PlanarPatch, PlanarThermalField

EX = np.array([1.0, 0.0, 0.0])
EY = np.array([0.0, 1.0, 0.0])
EZ = np.array([0.0, 0.0, 1.0])

#: §16.2 concrete, and a clear-day 14:00 insolation the weather files carry.
CONCRETE = {"heat_capacity_j_m2_k": 180_000.0, "emissivity": 0.92, "solar_absorptivity": 0.60}
T_AIR_K = 298.0
H_W_M2_K = 10.0
DNI_W_M2 = 850.0
DHI_W_M2 = 110.0
LW_DOWN_W_M2 = 330.0
#: A vertical wall sees half the sky and half the ground.
SKY_VIEW = 0.5


def _wall(n: int = 8) -> PlanarPatch:
    """A 4 m × 4 m vertical wall in the east–up plane, its outward normal pointing **north**.

    `u × v` is what fixes the normal, so the axes are ordered to face the sun rather than away
    from it — a wall facing away is a different (and much duller) test.
    """
    return PlanarPatch(
        origin_m=np.array([-2.0, 0.0, 0.0]),
        u_axis=EZ,
        v_axis=EX,
        n_u=n,
        n_v=n,
        du_m=0.5,
        dv_m=0.5,
        thickness_m=0.15,
    )


def _sun_at_14h() -> np.ndarray:
    """45° elevation, 20° east of north — a sun this wall can actually see."""
    return sun_direction(45.0, 20.0)


def _slab(half_u: float = 2.0) -> ShadowRectangle:
    """A horizontal slab standing off the wall, shading the half of it below the overhang."""
    return ShadowRectangle(
        centre_m=np.array([-1.0, 1.5, 3.0]),
        u_axis=EX,
        v_axis=EY,
        half_u_m=half_u,
        half_v_m=1.5,
    )


def _equilibrium(q_solar: float) -> float:
    """§6.1's own root for one cell's flux — the oracle, not a reimplementation of the solver."""
    return steady_state_temperature(
        ThermalProperties(**CONCRETE),
        SurfaceForcing(
            t_air_k=T_AIR_K,
            h_w_m2_k=H_W_M2_K,
            q_solar_w_m2=q_solar,
            q_longwave_down_w_m2=LW_DOWN_W_M2 * SKY_VIEW,
        ),
    )


def _solve(patch: PlanarPatch, q_solar: np.ndarray) -> np.ndarray:
    """Run the field to equilibrium: tiny capacity, long clock, so the transient is gone."""
    n = patch.n_cells
    props = FacetProperties(
        heat_capacity_j_m2_k=np.full(n, CONCRETE["heat_capacity_j_m2_k"]),
        emissivity=np.full(n, CONCRETE["emissivity"]),
        solar_absorptivity=np.full(n, CONCRETE["solar_absorptivity"]),
    )
    field = PlanarThermalField(
        patch,
        props,
        lambda _t: FacetForcing(
            t_air_k=T_AIR_K,
            h_w_m2_k=H_W_M2_K,
            q_solar_w_m2=q_solar,
            q_longwave_down_w_m2=LW_DOWN_W_M2 * SKY_VIEW,
        ),
        0.0,
        np.full(n, T_AIR_K),
        tick_s=10.0,
    )
    field.advance_to(200_000.0)
    return np.asarray(field.temperature_at(200_000.0), dtype=np.float64)


# --- the headline case --------------------------------------------------------------------------


def test_a_wall_half_in_sun_splits_by_more_than_eight_kelvin() -> None:
    patch, sun = _wall(), _sun_at_14h()
    lit = cell_shadow(patch, sun, [_slab()])
    assert 0.2 < lit.mean() < 0.8, f"the slab should shade part of the wall, not {lit.mean():.2f}"

    q = patch_solar_loading(patch, sun, DNI_W_M2, DHI_W_M2, SKY_VIEW, [_slab()])
    temperature = _solve(patch, q)

    step = float(temperature[lit == 1.0].mean() - temperature[lit == 0.0].mean())
    assert step >= 8.0, f"lit/shaded step is only {step:.2f} K at this insolation"


def test_every_cell_holds_its_own_equilibrium_to_a_millikelvin() -> None:
    """Not "the field varies" but "each cell is the right number" — the §6.1 root for its own flux.

    A model that merely smeared a gradient across the wall would pass a variation test and fail
    this one at every cell.
    """
    patch, sun = _wall(), _sun_at_14h()
    q = patch_solar_loading(patch, sun, DNI_W_M2, DHI_W_M2, SKY_VIEW, [_slab()])
    temperature = _solve(patch, q)

    expected = np.array([_equilibrium(float(flux)) for flux in q])
    assert np.max(np.abs(temperature - expected)) < 1e-3


def test_a_per_surface_shaded_bool_gives_one_temperature() -> None:
    """The negative control: the representation PT.1 replaces, on the same wall.

    `shaded=False` renders the whole wall at the lit temperature and `shaded=True` at the shaded
    one. Either way it is flat, and wrong across half the surface by the full step.
    """
    patch, sun = _wall(), _sun_at_14h()
    per_cell = patch_solar_loading(patch, sun, DNI_W_M2, DHI_W_M2, SKY_VIEW, [_slab()])
    field = _solve(patch, per_cell)

    for shaded in (False, True):
        flat_q = solar_loading(
            patch.normal, sun, DNI_W_M2, DHI_W_M2, SKY_VIEW, 0.0 if shaded else 1.0
        )
        flat = _solve(patch, np.full(patch.n_cells, float(flat_q)))
        assert float(np.ptp(flat)) < 1e-6, "the old representation is flat by construction"
        assert float(np.max(np.abs(flat - field))) >= 8.0


# --- the geometry ---------------------------------------------------------------------------------


def test_nothing_in_the_way_leaves_every_cell_lit() -> None:
    assert np.all(cell_shadow(_wall(), _sun_at_14h(), []) == 1.0)


def test_an_occluder_behind_the_surface_does_not_shade_it() -> None:
    """A slab on the far side is between the wall and the ground, not the wall and the sun."""
    behind = ShadowRectangle(
        centre_m=np.array([-1.0, -1.5, 3.0]), u_axis=EX, v_axis=EY, half_u_m=2.0, half_v_m=1.5
    )
    assert np.all(cell_shadow(_wall(), _sun_at_14h(), [behind]) == 1.0)


def test_a_big_enough_slab_shades_the_whole_wall() -> None:
    wide = ShadowRectangle(
        centre_m=np.array([0.0, 2.0, 5.0]), u_axis=EX, v_axis=EY, half_u_m=40.0, half_v_m=40.0
    )
    assert np.all(cell_shadow(_wall(), _sun_at_14h(), [wide]) == 0.0)


def test_two_occluders_union_rather_than_overwrite() -> None:
    """Shadow is not the last caster's opinion; a cell in either shadow is shaded."""
    patch, sun = _wall(), _sun_at_14h()
    left = ShadowRectangle(
        centre_m=np.array([-1.5, 1.5, 3.0]), u_axis=EX, v_axis=EY, half_u_m=0.6, half_v_m=1.5
    )
    right = ShadowRectangle(
        centre_m=np.array([1.5, 1.5, 3.0]), u_axis=EX, v_axis=EY, half_u_m=0.6, half_v_m=1.5
    )
    both = cell_shadow(patch, sun, [left, right])
    each = cell_shadow(patch, sun, [left]) * cell_shadow(patch, sun, [right])
    assert np.array_equal(both, each)
    assert both.mean() < cell_shadow(patch, sun, [left]).mean()


def test_a_back_facing_cell_is_reported_lit_and_zeroed_by_the_cosine() -> None:
    """`cell_shadow` answers about occlusion only; `max(0, n·s)` is what handles self-shadowing.

    Duplicating the cosine here would make a self-shadowing result look like an occlusion one, and
    the two need different fixes when a scene comes out wrong.
    """
    patch = _wall()
    away = -_sun_at_14h()
    assert np.all(cell_shadow(patch, away, []) == 1.0)
    q = patch_solar_loading(patch, away, DNI_W_M2, DHI_W_M2, SKY_VIEW, [])
    # Only the diffuse term survives.
    assert np.allclose(q, SKY_VIEW * DHI_W_M2)


def test_shadow_gates_the_beam_and_not_the_diffuse_sky() -> None:
    """Zeroing all solar in shade is the shortcut that renders shaded surfaces far too cold."""
    patch, sun = _wall(), _sun_at_14h()
    q = patch_solar_loading(patch, sun, DNI_W_M2, DHI_W_M2, SKY_VIEW, [_slab()])
    lit = cell_shadow(patch, sun, [_slab()])
    shaded_flux = q[lit == 0.0]
    assert shaded_flux.size and np.allclose(shaded_flux, SKY_VIEW * DHI_W_M2)
    assert float(shaded_flux[0]) > 0.0


def test_a_zero_length_sun_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="zero length"):
        cell_shadow(_wall(), np.zeros(3), [])
