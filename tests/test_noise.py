"""docs/isaacsim_checklist.md Phase 1, step 1.9."""
import pytest


def test_temporal_noise_std_matches_netd():
    """Empirical std dev over >=1000 noise realizations should be close to
    the target sigma = NETD * dSignal/dT."""
    pytest.skip("TODO: implement thermal_physics.noise first")
