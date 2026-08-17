"""docs/isaacsim_checklist.md Phase 1, steps 1.10-1.12 (end-to-end)."""
import pytest


def test_quantize_stays_in_bit_range():
    """Output must contain only integers in [0, 2**bits - 1], even for extreme inputs."""
    pytest.skip("TODO: implement thermal_physics.quantize first")


def test_agc_uses_close_to_full_display_range():
    """Output min/max should span close to 0-255 for a normal scene."""
    pytest.skip("TODO: implement thermal_physics.agc first")


def test_end_to_end_smoke():
    """render_frame() on a synthetic temperature gradient should produce a
    believable (H, W, 3) uint8 image -- the Phase 1 completion check."""
    pytest.skip("TODO: implement thermal_physics.pipeline.render_frame first")
