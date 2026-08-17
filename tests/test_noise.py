"""docs/isaacsim_checklist.md Phase 1, step 1.9."""
import numpy as np
from thermal_physics.noise import netd_noise_sigma, add_temporal_noise, add_fixed_pattern_noise


def test_temporal_noise_std_matches_netd():
    """Empirical std dev over >=1000 noise realizations should be close to
    the target sigma = NETD * dSignal/dT."""
    sigma = netd_noise_sigma(netd_k=0.04, d_signal_d_t=1000.0)
    rng = np.random.default_rng(0)
    samples = np.array([add_temporal_noise(np.array([100.0]), sigma, rng)[0] for _ in range(20000)])
    assert np.isclose(samples.std(), sigma, rtol=0.05)
    assert np.isclose(samples.mean(), 100.0, atol=sigma * 0.1)


def test_fixed_pattern_noise_is_frame_varying_but_structured():
    rng = np.random.default_rng(0)
    img = np.full((20, 20), 100.0)
    out = add_fixed_pattern_noise(img, gain_sigma=0.05, offset_sigma=2.0, rng=rng)
    assert out.shape == img.shape
    assert out.std() > 0  # not all pixels identical
    assert np.isclose(out.mean(), 100.0, atol=5.0)


def test_zero_sigma_noise_is_deterministic():
    out = add_temporal_noise(np.array([50.0, 60.0]), sigma=0.0)
    assert np.array_equal(out, np.array([50.0, 60.0]))
