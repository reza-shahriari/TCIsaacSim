"""M11.5 — the cold shield: the one §9.1 knob with no formula in the spec (S31, ADR 0066).

§9.1 says a mismatched cold stop raises background flux and degrades NETD, and stops there. The
geometry chosen here is Ω_admit = min(π, Ω_lens/η_cs), so the pixel's extra cone -- the part the
lens does not fill -- is filled by warm dewar structure. Two boundary identities pin it:
η_cs = 1 admits **exactly** nothing, and η_cs = 0 opens the full hemisphere, where an isothermal
enclosure reads A_d·π·L_B(T) through the two terms together.

The cost is not the offset (NUC removes offsets) but the **shot noise the offset carries**, so the
degradation is √((N_signal + N_bg)/N_signal) and the test holds it to that analytically.

docs/physics-model.md §9.1, §8.1, §2; ADR 0066, spec issue S31
"""

from __future__ import annotations

import itertools
import math
import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.config.sensor import PhotonFpa
from irsim.detector.cold_shield import (
    HEMISPHERE_PROJECTED_SR,
    admitted_solid_angle,
    background_electrons,
    background_power,
    excess_solid_angle,
    netd_degradation_factor,
)
from irsim.detector.netd import NoiseBudget, mean_signal, predict_netd_k
from irsim.detector.params import fpa_params_from_config
from irsim.optics.aperture import aperture_factor
from irsim.optics.self_emission import self_emission_power
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COARSE_N = 161
A_D = 1.0e-10  # a 10 um pixel, m²


@pytest.fixture(scope="module")
def mwir():  # type: ignore[no-untyped-def]
    return load_sensor_config(REPO / "configs/sensors/example_mwir_insb_640.yaml", DATA)


@pytest.fixture(scope="module")
def mwir_lut(mwir):  # type: ignore[no-untyped-def]
    return BandLUT.build(load_spectral_response(mwir.sensor.band.spectral_response), n=COARSE_N)


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("f_number", [1.0, 2.0, 4.0])
def test_a_matched_shield_admits_exactly_nothing(f_number: float) -> None:
    """Exactly 0.0, not 1e-17: a spurious Poisson term in every well-shielded camera's budget."""
    assert admitted_solid_angle(f_number, 1.0) == aperture_factor(f_number)
    assert excess_solid_angle(f_number, 1.0) == 0.0
    assert background_power(1234.0, f_number, 1.0, A_D) == 0.0


@pytest.mark.parametrize("f_number", [1.0, 2.0, 4.0])
def test_a_fully_open_shield_reaches_the_hemisphere_and_no_further(f_number: float) -> None:
    assert admitted_solid_angle(f_number, 0.0) == HEMISPHERE_PROJECTED_SR
    # and the cap binds well before eta reaches zero, at eta = Omega_lens/pi
    binding = aperture_factor(f_number) / HEMISPHERE_PROJECTED_SR
    assert admitted_solid_angle(f_number, binding * 0.5) == HEMISPHERE_PROJECTED_SR


def test_the_isothermal_enclosure_identity_holds_at_zero_efficiency() -> None:
    """Scene through the lens plus background around it = A_d π L_B(T), the enclosure answer.

    A pixel inside a cavity at one temperature must read that cavity, whatever the optics do. With
    τ = 1 and η_cs = 0 the two terms have to add to the full-hemisphere integral, and nothing in
    the code enforces that -- it falls out of Ω_admit's π cap only if the cap is the *projected*
    hemisphere and not 2π.
    """
    f_number, l_b = 2.0, 55.0
    scene = A_D * aperture_factor(f_number) * 1.0 * l_b
    background = background_power(l_b, f_number, 0.0, A_D)
    assert scene + background == pytest.approx(A_D * math.pi * l_b, rel=1e-12)


@pytest.mark.parametrize("f_number", [1.0, 2.0, 4.0])
def test_the_background_grows_monotonically_as_the_shield_opens(f_number: float) -> None:
    etas = [1.0, 0.95, 0.9, 0.7, 0.5, 0.2, 0.05, 0.0]
    excess = [excess_solid_angle(f_number, e) for e in etas]
    assert all(b >= a for a, b in itertools.pairwise(excess))
    assert excess[0] == 0.0 and excess[-1] > excess[1]


def test_out_of_range_efficiency_and_area_are_refused() -> None:
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="cold_shield_efficiency"):
            admitted_solid_angle(2.0, bad)
    with pytest.raises(ValueError, match="active_area_m2"):
        background_power(1.0, 2.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="negative"):
        background_power(-1.0, 2.0, 0.5, A_D)


def test_the_background_is_not_the_optics_self_emission(mwir, mwir_lut) -> None:  # type: ignore[no-untyped-def]
    """Two different warm surfaces in two different solid angles. Neither contains the other.

    Self-emission (§8.2) is the lens glowing *inside* the scene cone, weighted by (1 − τ); the
    cold-shield background is structure *outside* it, and it does not pass through the lens, so τ
    must not multiply it. Halving τ must move one and not the other.
    """
    f_number = mwir.sensor.optics.f_number
    l_b = float(mwir_lut.lookup(300.0)[()])
    self_hi = self_emission_power(A_D, f_number, 0.85, l_b)
    self_lo = self_emission_power(A_D, f_number, 0.425, l_b)
    assert self_lo > self_hi * 1.5, "self-emission must depend on transmittance"
    bg = background_power(l_b, f_number, 0.9, A_D)
    assert bg > 0.0
    # background_power has no transmittance argument at all -- the assertion is the signature
    import inspect

    assert "tau" not in inspect.signature(background_power).parameters


# ---------------------------------------------------------------------------------------------
# what it costs
# ---------------------------------------------------------------------------------------------


def test_the_degradation_factor_is_the_analytic_shot_ratio() -> None:
    assert netd_degradation_factor(1000.0, 0.0) == 1.0
    assert netd_degradation_factor(1000.0, 3000.0) == pytest.approx(2.0, rel=1e-12)
    with pytest.raises(ValueError):
        netd_degradation_factor(0.0, 10.0)
    with pytest.raises(ValueError):
        netd_degradation_factor(10.0, -1.0)


def test_netd_rises_with_the_analytic_factor_in_the_shot_limited_case(mwir, mwir_lut) -> None:  # type: ignore[no-untyped-def]
    """σ_gaussian = 0 isolates the shot term, which is the part the cold shield controls."""
    spec = mwir.sensor
    params = fpa_params_from_config(mwir)
    lb_q = float(mwir_lut.lookup(300.0, "lb_q")[()])
    n_signal = mean_signal(300.0, spec, mwir_lut)
    reference = predict_netd_k(
        300.0, spec, mwir_lut, NoiseBudget(kind="photon", sigma_gaussian=0.0)
    )
    previous = reference
    for eta in (0.95, 0.9, 0.8, 0.7, 0.5):
        n_bg = background_electrons(lb_q, params, spec.optics.f_number, eta)
        netd = predict_netd_k(
            300.0,
            spec,
            mwir_lut,
            NoiseBudget(kind="photon", sigma_gaussian=0.0, background_electrons=n_bg),
        )
        assert netd > previous, f"NETD did not rise at eta_cs = {eta}"
        assert netd / reference == pytest.approx(netd_degradation_factor(n_signal, n_bg), rel=1e-9)
        previous = netd


def test_netd_still_rises_monotonically_with_real_read_noise(mwir, mwir_lut) -> None:  # type: ignore[no-untyped-def]
    spec = mwir.sensor
    params = fpa_params_from_config(mwir)
    lb_q = float(mwir_lut.lookup(300.0, "lb_q")[()])
    sigma = 350.0
    values = []
    for eta in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        n_bg = background_electrons(lb_q, params, spec.optics.f_number, eta)
        values.append(
            predict_netd_k(
                300.0,
                spec,
                mwir_lut,
                NoiseBudget(kind="photon", sigma_gaussian=sigma, background_electrons=n_bg),
            )
        )
    assert all(b > a for a, b in itertools.pairwise(values)), values
    assert values[2] / values[0] == pytest.approx(1.060, rel=0.02), (
        f"eta_cs = 0.90 costs {values[2] / values[0]:.3f}x NETD"
    )


def test_a_badly_matched_shield_can_fill_the_well_on_its_own(mwir, mwir_lut) -> None:  # type: ignore[no-untyped-def]
    """The trade study §9.1 calls for, as a fact rather than a curve: at η_cs = 0.2 this camera
    saturates on its own dewar before it sees the scene."""
    params = fpa_params_from_config(mwir)
    lb_q = float(mwir_lut.lookup(300.0, "lb_q")[()])
    n_bg = background_electrons(lb_q, params, mwir.sensor.optics.f_number, 0.2)
    assert n_bg > params.well_capacity_e
    good = background_electrons(lb_q, params, mwir.sensor.optics.f_number, 0.9)
    assert good / params.well_capacity_e < 0.10


# ---------------------------------------------------------------------------------------------
# the MWIR config, and the cameras this must not touch
# ---------------------------------------------------------------------------------------------


def test_the_mwir_config_loads_classifies_and_is_cooled(mwir) -> None:  # type: ignore[no-untyped-def]
    spec = mwir.sensor
    assert spec.band.band_id == "mwir"
    assert spec.band.regime == "mixed"
    assert isinstance(spec.fpa, PhotonFpa)
    assert spec.fpa.fpa_temp_mode == "fixed" and spec.fpa.fpa_temp_k == 77.0
    assert 0.0 < spec.optics.cold_shield_efficiency < 1.0
    assert spec.outputs.apparent_temperature is True


def test_the_mwir_band_sits_on_both_sides_of_the_crossover(mwir_lut) -> None:  # type: ignore[no-untyped-def]
    """`mixed` earns its name: a 300 K scene emits usefully here, unlike SWIR."""
    assert float(mwir_lut.lookup(300.0)[()]) > 1.0
    assert float(mwir_lut.lookup(300.0)[()]) < float(mwir_lut.lookup(310.0)[()])


def test_an_uncooled_camera_is_untouched_by_any_of_this() -> None:
    """Every LWIR golden in this repository must be bit-identical after M11.5."""
    lwir = load_sensor_config(REPO / "configs/sensors/flir_boson_640_lwir.yaml", DATA)
    assert lwir.sensor.optics.cold_shield_efficiency == 1.0
    assert background_power(55.0, lwir.sensor.optics.f_number, 1.0, A_D) == 0.0


def test_the_pipeline_config_carries_the_background_only_for_the_cooled_camera(mwir) -> None:  # type: ignore[no-untyped-def]
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config

    materials = MaterialTable.constant(0.9, ids=(1,))
    cooled = PipelineConfig.from_sensor(
        mwir, materials, lut=load_band_lut_for_config(mwir, DATA / "lut", DATA)
    )
    assert cooled.detector.budget.background_electrons > 0.0  # type: ignore[union-attr]

    lwir = load_sensor_config(REPO / "configs/sensors/flir_boson_640_lwir.yaml", DATA)
    uncooled = PipelineConfig.from_sensor(
        lwir, materials, lut=load_band_lut_for_config(lwir, DATA / "lut", DATA)
    )
    assert uncooled.detector.budget.background_electrons == 0.0  # type: ignore[union-attr]


def test_the_background_scales_with_the_surround_radiance(mwir, mwir_lut) -> None:  # type: ignore[no-untyped-def]
    """A warm day costs more than a cold one, which is the other half of the trade study."""
    params = fpa_params_from_config(mwir)
    cold = background_electrons(float(mwir_lut.lookup(260.0, "lb_q")[()]), params, 2.0, 0.9)
    warm = background_electrons(float(mwir_lut.lookup(320.0, "lb_q")[()]), params, 2.0, 0.9)
    assert warm > 2.0 * cold
    assert np.isfinite([cold, warm]).all()
