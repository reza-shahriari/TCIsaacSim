"""docs/isaacsim_checklist.md Phase 3 (step 3.1) and Phase 5 (steps 5.2, 5.4)."""
import numpy as np
from thermal_physics.dynamics.lumped_capacitance import LumpedThermalNode
from thermal_physics.dynamics.water import evaporative_loss_w
from thermal_physics.dynamics.ice import EnthalpyThermalNode, WATER_LATENT_HEAT_FUSION_J_PER_KG
from thermal_physics.dynamics.fire import flicker_signal, volumetric_emission


def test_lumped_capacitance_matches_closed_form():
    """Stepping LumpedThermalNode under constant heat input should match
    T(t) = T_ambient + (Q_run/G_th)*(1 - exp(-t/tau)) within a small tolerance."""
    c_th, g_th, t_amb, q_run = 1000.0, 10.0, 290.0, 500.0
    node = LumpedThermalNode(c_th, g_th, t_amb)
    dt, steps = 0.05, int(300 / 0.05)
    for _ in range(steps):
        node.step(dt, q_run)
    tau = c_th / g_th
    closed_form = t_amb + (q_run / g_th) * (1 - np.exp(-(steps * dt) / tau))
    assert np.isclose(node.temperature_k, closed_form, rtol=1e-3)


def test_lumped_capacitance_cools_toward_ambient():
    c_th, g_th, t_amb = 1000.0, 10.0, 290.0
    node = LumpedThermalNode(c_th, g_th, t_amb, initial_temp_k=340.0)
    for _ in range(int(500 / 0.05)):
        node.step(0.05, 0.0)
    assert abs(node.temperature_k - t_amb) < 0.5


def test_water_evaporation_increases_with_deficit():
    """Lower humidity (bigger vapor pressure deficit) should evaporate faster."""
    humid = evaporative_loss_w(1.0, 3.0, 300.0, 290.0, relative_humidity=0.9)
    dry = evaporative_loss_w(1.0, 3.0, 300.0, 290.0, relative_humidity=0.2)
    assert dry > humid


def test_ice_enthalpy_plateau():
    """Constant heat input should hold T at the melting point for
    m*L_f/heat_rate seconds before it starts rising."""
    mass, c_ice, c_water, t_melt = 1.0, 2100.0, 4186.0, 273.15
    node = EnthalpyThermalNode(mass, c_ice, c_water, WATER_LATENT_HEAT_FUSION_J_PER_KG,
                                t_melt, initial_temp_k=t_melt)
    heat_rate, dt = 50.0, 1.0
    expected_plateau_s = mass * WATER_LATENT_HEAT_FUSION_J_PER_KG / heat_rate

    plateau_steps = 0
    t = None
    for _ in range(int(expected_plateau_s) + 200):
        t, melt_fraction = node.step(dt, heat_rate)
        if np.isclose(t, t_melt, atol=1e-6):
            plateau_steps += 1

    assert abs(plateau_steps - expected_plateau_s) < 2
    assert t > t_melt  # finished melting by the end


def test_fire_flicker_frequency():
    """flicker_signal's output should oscillate near base_freq_hz, not look
    smooth/static."""
    base_freq = 15.0
    ts = np.linspace(0, 4, 800)
    values = np.array([flicker_signal(float(t), base_freq_hz=base_freq, seed=1) for t in ts])

    assert values.std() > 0.01  # not static
    fft_mag = np.abs(np.fft.rfft(values - values.mean()))
    freqs = np.fft.rfftfreq(len(ts), d=ts[1] - ts[0])
    peak_freq = freqs[np.argmax(fft_mag)]
    assert 10.0 <= peak_freq <= 20.0


def test_fire_volumetric_emission_thicker_soot_is_hotter_looking():
    thick = volumetric_emission(1500.0, kappa_soot=5.0, path_length_m=0.1, wavelength_m=10e-6)
    thin = volumetric_emission(1500.0, kappa_soot=0.01, path_length_m=0.1, wavelength_m=10e-6)
    assert thick > thin
