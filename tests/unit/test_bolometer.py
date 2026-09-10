"""Ideal bolometer static transfer and the shared quantiser (M3.7, ADR 0019)."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.detector import (
    BolometerParams,
    BolometerTransfer,
    absorbed_power_w,
    fpa_params_from_config,
    membrane_delta_t_k,
    quantise,
    static_responsivity_v_per_w,
)
from irsim.optics import pixel_power
from irsim.radiometry.planck import band_radiance_tophat, spectral_radiance

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


@pytest.fixture(scope="module")
def boson() -> BolometerParams:
    p = fpa_params_from_config(SensorConfig.model_validate(BOSON))
    assert isinstance(p, BolometerParams)
    return p


def test_pixel_power_matches_independent_trapezoid(boson: BolometerParams) -> None:
    """Φ(300 K, 7.5-13.5 µm top-hat, F/1, τ 0.92, A_d 1.296e-10) vs trapezoid of spectral_radiance
    × (π/5) × 0.92 × A_d to 1e-4; magnitude ≈ 1e-8 W (catches per-µm and π errors)."""
    lam = np.linspace(7.5, 13.5, 60_001)
    lb_trap = float(np.trapezoid(spectral_radiance(lam, 300.0), lam))
    expected = lb_trap * (np.pi / 5.0) * 0.92 * boson.active_area_m2
    phi = float(
        pixel_power(band_radiance_tophat(7.5, 13.5, 300.0), 1.0, 0.92, boson.active_area_m2)
    )
    assert abs(phi / expected - 1.0) < 1e-4
    assert 1e-9 < phi < 1e-8, phi  # 4.16e-9 W: the '≈1e-8 W' order of magnitude


def test_dn_is_linear_in_power(boson: BolometerParams) -> None:
    transfer = BolometerTransfer.from_power_range(1e-9, 5e-8, boson.bit_depth)
    phi = np.linspace(2e-9, 4e-8, 1000)
    s = transfer.signal_dn(phi).astype(np.float64)
    fit = np.polyfit(phi, s, 1)
    resid = s - np.polyval(fit, phi)
    assert np.max(np.abs(resid)) / np.max(np.abs(s)) < 1e-6  # float32 signal, exact in float64
    s64 = transfer.gain_dn_per_w * (phi - transfer.offset_w)
    assert np.max(np.abs(np.diff(s64, 2))) < 1e-9 * s64.max()
    assert transfer.dn(np.array([1e-9])).tolist() == [0] and transfer.dn(
        np.array([5e-8])
    ).tolist() == [65535]


def test_steady_state_matches_rk4_of_membrane_ode(boson: BolometerParams) -> None:
    """ΔT_ss = α Φ / G_th equals the ODE C dΔT/dt = αΦ − G ΔT integrated to 20 τ (RK4) to 1e-9."""
    phi = 2e-8
    c, g, a = boson.c_th_j_per_k, boson.g_th_w_per_k, boson.absorptance
    tau = c / g
    dt = tau / 200.0
    t_mem = 0.0

    def f(x: float) -> float:
        return (a * phi - g * x) / c

    for _ in range(int(20 * tau / dt)):
        k1 = f(t_mem)
        k2 = f(t_mem + 0.5 * dt * k1)
        k3 = f(t_mem + 0.5 * dt * k2)
        k4 = f(t_mem + dt * k3)
        t_mem += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs(t_mem / float(membrane_delta_t_k(phi, boson)) - 1.0) < 1e-8
    assert float(absorbed_power_w(phi, boson)) == pytest.approx(a * phi)


def test_responsivity_known_answer() -> None:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(
        absorptance=0.8,
        tcr_per_k=-0.021,
        bias_current_a=50e-6,
        resistance_ohm=1e5,
        g_th_w_per_k=1e-7,
    )
    p = fpa_params_from_config(SensorConfig.model_validate(d))
    assert isinstance(p, BolometerParams)
    assert static_responsivity_v_per_w(p) == pytest.approx(-8.4e5, rel=1e-12)
    d["sensor"]["fpa"]["tcr_per_k"] = 0.021
    assert static_responsivity_v_per_w(fpa_params_from_config(SensorConfig.model_validate(d))) > 0  # type: ignore[arg-type]


def test_photon_form_input_is_refused(boson: BolometerParams) -> None:
    """Lb_q is ~1e20× Lb: feeding a photon rate as watts must raise, not silently saturate."""
    with pytest.raises(ValueError, match="photon rate"):
        absorbed_power_w(np.array([1e12]), boson)
    with pytest.raises(ValueError, match="negative"):
        absorbed_power_w(np.array([-1e-9]), boson)
    with pytest.raises(TypeError, match="float16"):
        absorbed_power_w(np.array([1e-9], dtype=np.float16), boson)


def test_quantiser_floors_clips_and_never_wraps() -> None:
    s = np.array([-3.0, 0.0, 0.4, 0.6, 65534.999, 65535.0, 70000.0, 1e9], dtype=np.float64)
    dn = quantise(s, 16)
    assert dn.dtype == np.uint16
    assert dn.tolist() == [0, 0, 0, 0, 65534, 65535, 65535, 65535]
    assert quantise(np.array([300.7]), 8).tolist() == [255]
    with pytest.raises(TypeError, match="float16"):
        quantise(np.array([1.0], dtype=np.float16), 16)
    with pytest.raises(ValueError):
        quantise(np.array([np.nan]), 16)
    with pytest.raises(ValueError):
        quantise(np.array([1.0]), 17)


def test_transfer_inverse_and_bounds() -> None:
    t = BolometerTransfer.from_power_range(1e-9, 5e-8, 16)
    phi = np.array([1e-9, 2.5e-8, 5e-8])
    back = t.power_from_signal_w(t.signal_dn(phi))
    np.testing.assert_allclose(back, phi, rtol=1e-6)
    with pytest.raises(ValueError):
        BolometerTransfer.from_power_range(5e-8, 1e-9, 16)
    with pytest.raises(ValueError):
        BolometerTransfer(gain_dn_per_w=0.0, offset_w=0.0, bit_depth=16)
