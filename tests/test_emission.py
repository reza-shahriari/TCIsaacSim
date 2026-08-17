"""docs/isaacsim_checklist.md Phase 1, steps 1.2-1.5."""
import numpy as np
from thermal_physics.emission import planck_radiance, band_radiance, surface_leaving_radiance


def test_planck_law_peak_wavelength():
    """Wien's law: at T=300K, L_bb(lambda,300) should peak near lambda ~= 2898/300 um."""
    wl = np.linspace(1e-6, 25e-6, 5000)
    L = planck_radiance(wl, 300.0)
    peak_um = wl[np.argmax(L)] * 1e6
    assert abs(peak_um - 2898.0 / 300.0) < 0.1


def test_band_radiance_monotonic_in_temperature():
    """L_band(T) should be strictly increasing over 250-400K, no kinks."""
    temps = np.linspace(250, 400, 30)
    values = [band_radiance(t, (8e-6, 14e-6)) for t in temps]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_band_radiance_vectorizes_over_array_temperature():
    """A (H, W) temperature field should produce a matching-shape output."""
    t_field = np.array([[290.0, 300.0], [310.0, 320.0]])
    out = band_radiance(t_field, (8e-6, 14e-6))
    assert out.shape == (2, 2)
    assert out[1, 1] > out[0, 0]


def test_surface_radiance_emissivity_edge_cases():
    """eps=1 -> output == L_bb(T); eps=0 -> output == L_bb(T_env) regardless of T."""
    band = (8e-6, 14e-6)
    full_emit = surface_leaving_radiance(320.0, 1.0, 290.0, band)
    full_reflect = surface_leaving_radiance(320.0, 0.0, 290.0, band)
    assert np.isclose(full_emit, band_radiance(320.0, band))
    assert np.isclose(full_reflect, band_radiance(290.0, band))
