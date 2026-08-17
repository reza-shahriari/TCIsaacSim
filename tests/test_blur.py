"""MTF boundary conditions and blur behavior. Not itemized in the original
Phase 1 checklist -- added while implementing, alongside test_colormap.py."""
import numpy as np
from thermal_physics.blur import diffraction_mtf, detector_mtf, apply_system_blur


def test_diffraction_mtf_boundary_conditions():
    assert np.isclose(diffraction_mtf(0.0, 10e-6, 1.2), 1.0)
    f_c = 1.0 / (10e-6 * 1.2)
    assert np.isclose(diffraction_mtf(f_c, 10e-6, 1.2), 0.0, atol=1e-6)
    assert diffraction_mtf(2 * f_c, 10e-6, 1.2) == 0.0


def test_detector_mtf_at_zero_frequency():
    assert np.isclose(detector_mtf(0.0, 12e-6), 1.0)


def test_blur_spreads_a_point_and_preserves_energy():
    img = np.zeros((21, 21))
    img[10, 10] = 1.0
    blurred = apply_system_blur(img, psf_sigma_px=2.0)
    assert np.isclose(blurred.sum(), img.sum(), rtol=1e-2)
    assert blurred[10, 10] < 1.0
    assert blurred[10, 11] > 0.0


def test_zero_sigma_blur_is_identity():
    img = np.random.default_rng(0).normal(size=(5, 5))
    assert np.array_equal(apply_system_blur(img, 0.0), img)
