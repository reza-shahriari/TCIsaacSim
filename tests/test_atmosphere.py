"""docs/isaacsim_checklist.md Phase 1, step 1.6."""
import pytest


def test_zero_range_is_identity():
    """At range=0, apply_atmosphere's output should exactly equal the input radiance."""
    pytest.skip("TODO: implement thermal_physics.atmosphere first")


def test_large_range_approaches_path_radiance():
    """As range -> large, output should approach L_bb(T_atm) regardless of the source."""
    pytest.skip("TODO: implement thermal_physics.atmosphere first")
