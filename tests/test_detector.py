"""docs/isaacsim_checklist.md Phase 1, step 1.8."""
import numpy as np
from thermal_physics.detector import linear_detector_signal, bolometer_temperature_rise


def test_linear_detector_signal_monotonic():
    s1 = linear_detector_signal(100.0, k_resp=2.0)
    s2 = linear_detector_signal(200.0, k_resp=2.0)
    assert s2 > s1
    assert not np.isnan(s1) and not np.isnan(s2)


def test_bolometer_ode_matches_closed_form_step_response():
    """Stepping the bolometer thermal ODE under constant absorbed power should
    match the closed-form C_th/G_th exponential warm-up curve."""
    c_th, g_th, q = 1000.0, 10.0, 500.0
    dt, delta_t = 0.01, 0.0
    steps = int(300 / dt)
    for _ in range(steps):
        delta_t = bolometer_temperature_rise(q, c_th, g_th, dt, delta_t)
    tau = c_th / g_th
    closed_form = (q / g_th) * (1 - np.exp(-(steps * dt) / tau))
    assert np.isclose(delta_t, closed_form, rtol=1e-3)
