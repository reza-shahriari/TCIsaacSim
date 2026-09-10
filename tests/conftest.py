"""Shared fixtures.

Synthetic G-buffer fixtures live here so kernels can be tested without launching
Isaac Sim. See the ir-sim-testing skill.
"""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator. Every test that uses randomness must take this."""
    return np.random.default_rng(20260910)


@pytest.fixture
def gbuffer_ramp() -> dict[str, np.ndarray]:
    """256x256 G-buffer: linear 250-450 K temperature ramp, camera-facing normals,
    constant 50 m range, single material. The workhorse fixture for kernel tests."""
    h = w = 256
    t = np.linspace(250.0, 450.0, w, dtype=np.float32)
    return {
        "temperature_k": np.tile(t, (h, 1)),
        "normal_dot_view": np.ones((h, w), dtype=np.float32),
        "distance_m": np.full((h, w), 50.0, dtype=np.float32),
        "material_id": np.zeros((h, w), dtype=np.int32),
        "sky_view_factor": np.full((h, w), 0.5, dtype=np.float32),
    }


@pytest.fixture
def gbuffer_uniform() -> dict[str, np.ndarray]:
    """64x64 G-buffer at a uniform 300 K. Use for noise-decomposition tests."""
    h = w = 64
    return {
        "temperature_k": np.full((h, w), 300.0, dtype=np.float32),
        "normal_dot_view": np.ones((h, w), dtype=np.float32),
        "distance_m": np.full((h, w), 10.0, dtype=np.float32),
        "material_id": np.zeros((h, w), dtype=np.int32),
        "sky_view_factor": np.zeros((h, w), dtype=np.float32),
    }
