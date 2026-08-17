"""docs/isaacsim_checklist.md Phase 1, steps 1.2-1.5."""
import pytest


def test_planck_law_peak_wavelength():
    """Wien's law: at T=300K, L_bb(lambda,300) should peak near lambda ~= 2898/300 um."""
    pytest.skip("TODO: implement thermal_physics.emission.planck_radiance first")


def test_band_radiance_monotonic_in_temperature():
    """L_band(T) should be smooth and strictly increasing over 250-400K, no kinks."""
    pytest.skip("TODO: implement thermal_physics.emission.band_radiance first")


def test_surface_radiance_emissivity_edge_cases():
    """eps=1 -> output == L_bb(T); eps=0 -> output == L_bb(T_env) regardless of T."""
    pytest.skip("TODO: implement thermal_physics.emission.surface_leaving_radiance first")
