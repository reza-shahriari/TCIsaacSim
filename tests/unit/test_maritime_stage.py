"""The sea ground mode, and the maritime stage's engine-free geometry (MM.5, MM.6).

Nothing here needs Isaac Sim: `irsim_isaac.maritime_demo` keeps its `pxr` imports inside the
functions that author prims, so the placement arithmetic — which is the part that can be wrong in a
way a picture would not reveal — is testable with plain pytest.

docs/physics-model.md §5.3, §12.2; ADR 0078
"""

from __future__ import annotations

import math

import pytest
import yaml

from irsim.config.environment import EnvironmentConfig, load_environment_preset
from irsim.pipeline.environment import ground_temperature_k
from irsim_isaac.maritime_demo import DEMO_VESSELS, EARTH_RADIUS_M, Vessel, _taper

# -- the sea ground mode (MM.5) -----------------------------------------------------------


def test_the_sea_preset_loads_and_carries_a_bulk_sst() -> None:
    env = load_environment_preset("sea_clear_day")
    assert env.ground.mode == "sea"
    assert env.ground.bulk_sst_k == pytest.approx(290.0)
    assert env.ground.fixed_temperature_k is None


def _preset(**ground):
    base = yaml.safe_load(
        (__import__("pathlib").Path("configs/environments/sea_clear_day.yaml")).read_text()
    )
    base["environment"]["ground"] = ground
    return base


def test_sea_without_a_bulk_sst_is_refused() -> None:
    """The sea model has to start from a measured SST; there is no sensible default for it."""
    with pytest.raises(ValueError, match="bulk_sst_k"):
        EnvironmentConfig(**_preset(mode="sea"))


def test_a_bulk_sst_on_dry_land_is_refused() -> None:
    """An SST on a ground preset means someone edited the mode and left the value behind."""
    with pytest.raises(ValueError, match="only applies to ground.mode 'sea'"):
        EnvironmentConfig(**_preset(mode="air", bulk_sst_k=290.0))


def test_the_reflected_environment_gets_the_sst_not_the_angular_profile() -> None:
    """One scalar is right *here* and wrong for the background, and the two must not be confused.

    `ground_temperature_k` feeds the reflected term of §5.3(a): what a vessel's hull sees below its
    own horizon is the water immediately around it, at steep incidence, where ε ≈ 0.99 and the
    water radiates essentially at its own temperature. So the SST is the correct answer.

    A **background** ray is a different quantity entirely — it crosses kilometres of sea at grazing
    incidence, where ε falls under 0.15 and the signal is mostly reflected sky. That one comes from
    `SeaModel` and needs an angle. The first version of this code raised here to force the
    distinction, which broke every maritime frame at the reflected term: the guard was right about
    the physics and wrong about which caller it was guarding.
    """
    sky = _FakeSky(load_environment_preset("sea_clear_day"))
    assert ground_temperature_k(sky, 0.0) == pytest.approx(290.0)


class _FakeSky:
    """Just enough of a SkyModel for `ground_temperature_k` to reach its ground branch."""

    def __init__(self, environment) -> None:
        self.environment = environment


# -- stage geometry (MM.6) ----------------------------------------------------------------


def test_a_vessel_sits_at_the_depression_its_range_implies() -> None:
    """The waterline angle, with the Earth's curve in it.

    At 5.2 km the curvature drop is 2.1 m against a 20 m eye height — a 10 % change in the
    depression angle, which at this scale is two Boson pixels of vertical placement. Dropping it
    would put every distant vessel visibly above its own horizon.
    """
    vessel = Vessel("v", 5200.0, 95.0, 0.0, "painted_composite", "a", "b", "c")
    drop = 5200.0**2 / (2.0 * EARTH_RADIUS_M)
    assert drop == pytest.approx(2.12, abs=0.02)

    flat = math.degrees(math.atan2(20.0, 5200.0))
    curved = vessel.depression_deg(20.0)
    assert curved == pytest.approx(math.degrees(math.atan2(20.0 + drop, 5200.0)), abs=1e-9)
    assert curved > flat
    assert curved / flat == pytest.approx(1.106, abs=0.01)


def test_vessels_are_resolved_at_every_demo_range() -> None:
    """All three span many pixels, so the renderer is the right model for them (no MS.6 handover).

    A 95 m freighter at 5.2 km and a 7 m skiff at 600 m subtend almost the same angle — which is
    the point of the range ladder: same pixels, very different background, because the sea behind
    them is at a completely different depression angle.
    """
    ifov_mrad = 0.857
    spans = {}
    for name, range_m, length_m, offset_m, material, _node in DEMO_VESSELS:
        v = Vessel(name, range_m, length_m, offset_m, material, "a", "b", "c")
        spans[name] = v.pixels_across(ifov_mrad)
        assert spans[name] > 5.0

    assert spans["skiff"] == pytest.approx(13.6, abs=1.0)
    assert spans["freighter"] == pytest.approx(21.3, abs=1.0)


def test_wave_displacement_is_tapered_off_where_the_mesh_cannot_resolve_it() -> None:
    """Below two samples per wavelength a displaced vertex is tessellation, not a wave.

    Without the taper the far rings — whose spacing runs to hundreds of metres — would carry
    displacement at their own scale and render as a field of giant fake swells, which looks like
    weather and is actually an artefact.
    """
    assert _taper(spacing_m=1.0, wavelength_m=42.0) == 1.0  # well resolved
    assert _taper(spacing_m=21.0, wavelength_m=42.0) == 0.0  # exactly Nyquist: gone
    assert _taper(spacing_m=400.0, wavelength_m=42.0) == 0.0  # far field: flat
    mid = _taper(spacing_m=14.0, wavelength_m=42.0)  # three samples per wavelength
    assert 0.0 < mid < 1.0
    # monotone in resolvability
    coarse = [_taper(s, 42.0) for s in (5.0, 10.0, 14.0, 18.0, 21.0)]
    assert all(a >= b for a, b in zip(coarse, coarse[1:], strict=False))
