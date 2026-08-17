"""docs/isaacsim_checklist.md Phase 3 (step 3.1) and Phase 5 (steps 5.2, 5.4)."""
import pytest


def test_lumped_capacitance_matches_closed_form():
    """Stepping LumpedThermalNode under constant heat input should match
    T(t) = T_ambient + (Q_run/G_th)*(1 - exp(-t/tau)) within a small tolerance."""
    pytest.skip("TODO: implement thermal_physics.dynamics.lumped_capacitance first")


def test_water_damped_swing():
    """A water-parameterized node should show a smaller, phase-delayed swing
    than a ground-parameterized node under the same ambient cycle."""
    pytest.skip("TODO: implement thermal_physics.dynamics.water first")


def test_ice_enthalpy_plateau():
    """Constant heat input should hold T at the melting point for
    m*L_f/heat_rate seconds before it starts rising."""
    pytest.skip("TODO: implement thermal_physics.dynamics.ice first")


def test_fire_flicker_frequency():
    """flicker_signal's output should oscillate in the ~10-20 Hz range, not
    look smooth/static."""
    pytest.skip("TODO: implement thermal_physics.dynamics.fire first")
