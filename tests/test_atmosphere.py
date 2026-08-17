"""docs/isaacsim_checklist.md Phase 1, step 1.6."""
import numpy as np
from thermal_physics.atmosphere import transmittance, apply_atmosphere


def test_zero_range_is_identity():
    """At range=0, apply_atmosphere's output should exactly equal the input radiance."""
    out = apply_atmosphere(source_radiance=123.4, gamma_per_m=1e-4, range_m=0.0, l_bb_t_atm=999.0)
    assert np.isclose(out, 123.4)
    assert transmittance(1e-4, 0.0) == 1.0


def test_large_range_approaches_path_radiance():
    """As range -> large, output should approach L_bb(T_atm) regardless of the source."""
    out = apply_atmosphere(source_radiance=123.4, gamma_per_m=1e-4, range_m=1e9, l_bb_t_atm=999.0)
    assert np.isclose(out, 999.0)


def test_transmittance_monotonically_decreases_with_range():
    r = np.linspace(0, 10000, 50)
    tau = transmittance(1e-4, r)
    assert all(a >= b for a, b in zip(tau, tau[1:]))
    assert tau[0] == 1.0
