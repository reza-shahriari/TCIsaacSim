"""docs/isaacsim_checklist.md Phase 1, step 1.7."""
import numpy as np
from thermal_physics.optics import focal_plane_irradiance


def test_irradiance_scales_with_inverse_f_number_squared():
    """Doubling f_number should quarter the irradiance, all else equal."""
    e1 = focal_plane_irradiance(radiance=100.0, f_number=1.0, tau_optics=1.0)
    e2 = focal_plane_irradiance(radiance=100.0, f_number=2.0, tau_optics=1.0)
    assert np.isclose(e2, e1 / 4)


def test_cos4_falloff_at_field_angle():
    on_axis = focal_plane_irradiance(100.0, 1.2, 0.8, field_angle_rad=0.0)
    off_axis = focal_plane_irradiance(100.0, 1.2, 0.8, field_angle_rad=np.deg2rad(20))
    assert off_axis < on_axis
    assert np.isclose(off_axis / on_axis, np.cos(np.deg2rad(20)) ** 4)
