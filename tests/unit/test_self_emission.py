"""Optics self-emission (M3.3, ADR 0016): the isothermal-enclosure identity, the N-element stack
against hand values and its reduction to the single-lens form, Kirchhoff closure of elements, and
the shutterless-drift known answer (+1 K housing -> +87 mK apparent)."""

from __future__ import annotations

import pytest

from irsim.optics import (
    OpticalElement,
    aperture_factor,
    pixel_power,
    self_emission_power,
    stack_self_radiance,
    stack_transmittance,
)
from irsim.radiometry.planck import band_radiance_tophat

A_D = 1.296e-10
LO, HI = 7.5, 13.5


def lb(t: float) -> float:
    return band_radiance_tophat(LO, HI, t)


def dlb_dt(t: float, d: float = 1e-3) -> float:
    return (lb(t + d) - lb(t - d)) / (2 * d)


@pytest.mark.parametrize("tau", [0.5, 0.8, 0.92, 1.0])
def test_isothermal_enclosure_identity(tau: float) -> None:
    """Scene and housing both at T: Φ_scene + Φ_self = A_d Ω_eff L_B(T), independent of τ_opt
    to 1e-12 (a wrong Ω, a wrong (1 − τ) or a stray cos⁴ all break this)."""
    t = 300.0
    phi_scene = float(pixel_power(lb(t), 1.0, tau, A_D))
    phi_self = self_emission_power(A_D, 1.0, tau, lb(t))
    expected = A_D * aperture_factor(1.0) * lb(t)
    assert abs((phi_scene + phi_self) / expected - 1.0) < 1e-12


def test_stack_reduces_to_single_lens_form() -> None:
    lens = OpticalElement(tau=0.92, temperature_k=305.0, name="lens")
    l_self = stack_self_radiance([lens], lb)
    assert (
        abs(
            l_self * A_D * aperture_factor(1.0) / self_emission_power(A_D, 1.0, 0.92, lb(305.0))
            - 1.0
        )
        < 1e-12
    )
    assert stack_transmittance([lens]) == 0.92


def test_two_element_stack_matches_hand_value() -> None:
    """Window then lens (scene → detector): ε_w L(T_w) τ_l + ε_l L(T_l)."""
    window = OpticalElement(tau=0.95, temperature_k=290.0, name="window")
    lens = OpticalElement(tau=0.92, temperature_k=305.0, name="lens")
    hand = 0.05 * lb(290.0) * 0.92 + 0.08 * lb(305.0)
    got = stack_self_radiance([window, lens], lb)
    assert abs(got / hand - 1.0) < 1e-12
    # order matters: downstream-only attenuation
    reversed_hand = 0.08 * lb(305.0) * 0.95 + 0.05 * lb(290.0)
    assert abs(stack_self_radiance([lens, window], lb) / reversed_hand - 1.0) < 1e-12
    assert stack_transmittance([window, lens]) == pytest.approx(0.95 * 0.92)


def test_kirchhoff_closure_of_elements() -> None:
    e = OpticalElement(tau=0.92, temperature_k=300.0, rho=0.05)
    assert e.emissivity == pytest.approx(0.03, abs=1e-12)
    assert e.tau + e.rho + e.emissivity == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="Kirchhoff"):
        OpticalElement(tau=0.92, temperature_k=300.0, rho=0.1)
    with pytest.raises(ValueError):
        OpticalElement(tau=1.2, temperature_k=300.0)
    with pytest.raises(ValueError):
        OpticalElement(tau=0.9, temperature_k=0.0)


def test_shutterless_drift_known_answer() -> None:
    """Boson-like τ 0.92, F/1: a +1 K housing step reads as +(1−τ)/τ K = +87 mK apparent
    temperature when the camera's transfer assumes the housing at its calibration temperature."""
    t_scene = t_housing = 300.0
    phi0 = float(pixel_power(lb(t_scene), 1.0, 0.92, A_D)) + self_emission_power(
        A_D, 1.0, 0.92, lb(t_housing)
    )
    phi1 = float(pixel_power(lb(t_scene), 1.0, 0.92, A_D)) + self_emission_power(
        A_D, 1.0, 0.92, lb(t_housing + 1.0)
    )
    # the radiometric branch attributes all of ΔΦ to the scene: ΔL = ΔΦ / (A_d Ω τ)
    delta_l = (phi1 - phi0) / (A_D * aperture_factor(1.0) * 0.92)
    delta_t_mk = delta_l / dlb_dt(t_scene) * 1e3
    assert delta_t_mk == pytest.approx(87.0, abs=1.0), delta_t_mk
    assert delta_t_mk == pytest.approx((1 - 0.92) / 0.92 * 1e3, abs=0.5)


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        self_emission_power(A_D, 1.0, 1.2, 1.0)
    with pytest.raises(ValueError):
        self_emission_power(A_D, 1.0, 0.9, -1.0)
    with pytest.raises(ValueError):
        self_emission_power(0.0, 1.0, 0.9, 1.0)
