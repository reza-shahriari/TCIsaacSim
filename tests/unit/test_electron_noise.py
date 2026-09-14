"""M11.6 — noise for a photon FPA, built from electrons rather than anchored to a NETD.

ADR 0025 made NETD the calibration handle: the chain sets the structure, the datasheet sets the
magnitude. That is right for a bolometer, whose datasheet quotes NETD and nothing else usable.
M11.1's SWIR camera is where it breaks in the open — a 300 K blackbody puts **1.24 photoelectrons
per pixel per frame** into 0.9–1.7 µm against 120 e⁻ of read noise, so §9.4 evaluated honestly
gives 976 K, and anchoring to that would scale the Gaussian term by 2e4 and invent the noise.

So for a photon FPA the electron datasheet is primary and NETD becomes the cross-check. What did
*not* change: Poisson terms are never rescaled, noise is added in electron space and never in
kelvin, and a camera that cannot reach its own claim raises instead of quietly existing.

docs/physics-model.md §10.1, §9.1, §9.4, §12.1; ADR 0025 and its M11.6 addendum
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.detector.netd import (
    NoiseBudget,
    mean_signal,
    predict_netd_k,
    signal_derivative_per_k,
)
from irsim.detector.params import fpa_params_from_config
from irsim.noise.electron import (
    InfeasibleNoiseConfigError,
    dark_electrons_for,
    electron_budget,
    electron_noise,
    shot_limited_netd_k,
)
from irsim.optics.aperture import aperture_factor
from irsim.radiometry.lut_files import load_band_lut_for_config

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
LUT_DIR = DATA / "lut"
#: The SWIR camera is measured where it actually works. At 300 K it collects one photoelectron a
#: frame, so a NETD bench there measures the read noise and nothing else.
SWIR_BENCH_K = 700.0
SAMPLES = 200_000


def _config(name: str):  # type: ignore[no-untyped-def]
    cfg = load_sensor_config(REPO / "configs/sensors" / f"{name}.yaml", DATA)
    return cfg, load_band_lut_for_config(cfg, LUT_DIR, DATA)


@pytest.fixture(scope="module")
def swir():  # type: ignore[no-untyped-def]
    return _config("example_swir_ingaas_640")


@pytest.fixture(scope="module")
def mwir():  # type: ignore[no-untyped-def]
    return _config("example_mwir_insb_640")


# ---------------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------------


def test_shot_noise_has_variance_equal_to_its_mean() -> None:
    """The defining property, checked on the generator rather than assumed of it."""
    rng = np.random.default_rng(11)
    for mean in (50.0, 5_000.0, 500_000.0):
        n = electron_noise(np.full(SAMPLES, mean), 0.0, 1e-3, 0.0, rng)
        assert float(n.var() / n.mean()) == pytest.approx(1.0, abs=0.02), mean


def test_one_poisson_draw_covers_signal_dark_and_background_together() -> None:
    """Sum of independent Poissons is Poisson in the sum, so two draws would be wrong, not safer."""
    rng = np.random.default_rng(3)
    signal, dark_e, bg = 4_000.0, 1_000.0, 500.0
    from irsim.radiometry.constants import Q_E

    i_dark = dark_e * Q_E / 1e-3
    n = electron_noise(np.full(SAMPLES, signal), i_dark, 1e-3, 0.0, rng, background_e=bg)
    total = signal + dark_e + bg
    assert float(n.mean()) == pytest.approx(total, rel=0.01)
    assert float(n.var()) == pytest.approx(total, rel=0.03)


def test_read_noise_adds_in_quadrature_and_is_gaussian() -> None:
    rng = np.random.default_rng(5)
    mean, sigma_read = 10_000.0, 120.0
    n = electron_noise(np.full(SAMPLES, mean), 0.0, 1e-3, sigma_read, rng)
    assert float(n.var()) == pytest.approx(mean + sigma_read**2, rel=0.03)


def test_negative_counts_are_clipped_and_bad_inputs_refused() -> None:
    rng = np.random.default_rng(7)
    n = electron_noise(np.full(2000, 1.0), 0.0, 1e-3, 500.0, rng)
    assert float(n.min()) >= 0.0
    with pytest.raises(ValueError, match="negative"):
        electron_noise(np.array([-1.0]), 0.0, 1e-3, 0.0, rng)
    with pytest.raises(ValueError, match="read noise"):
        electron_noise(np.array([1.0]), 0.0, 1e-3, -1.0, rng)
    with pytest.raises(ValueError, match="background"):
        electron_noise(np.array([1.0]), 0.0, 1e-3, 0.0, rng, background_e=-1.0)


# ---------------------------------------------------------------------------------------------
# the aperture factor, where 20 % hides
# ---------------------------------------------------------------------------------------------


def test_the_signal_derivative_uses_the_plus_one_aperture_factor(swir) -> None:  # type: ignore[no-untyped-def]
    """π/(4F²+1), not π/(4F²) — CLAUDE.md #5. At F/1 that is π/5 against π/4: **20 % less**.

    Rebuilt here from §9.1's own terms rather than read back from the same helper, so the test
    would catch the aperture factor being right in one place and wrong in this one.
    """
    cfg, lut = swir
    spec = cfg.sensor
    params = fpa_params_from_config(cfg)
    derivative = signal_derivative_per_k(SWIR_BENCH_K, spec, lut, f_number=1.0)
    geometry = params.active_area_m2 * spec.optics.transmittance
    d_lb_q = float(lut.lookup(SWIR_BENCH_K, "dlb_q_dt")[()])
    common = params.quantum_efficiency * params.integration_time_s * geometry * d_lb_q
    with_plus_one = common * (math.pi / 5.0)
    without = common * (math.pi / 4.0)
    assert aperture_factor(1.0) == pytest.approx(math.pi / 5.0, rel=1e-12)
    assert derivative == pytest.approx(with_plus_one, rel=1e-6)
    assert with_plus_one / without == pytest.approx(0.80, abs=0.005)


# ---------------------------------------------------------------------------------------------
# the budget
# ---------------------------------------------------------------------------------------------


def test_the_budget_uses_the_configured_read_noise_rather_than_solving_for_it(swir) -> None:  # type: ignore[no-untyped-def]
    cfg, lut = swir
    budget = electron_budget(cfg.sensor, lut)
    assert budget.sigma_gaussian == cfg.sensor.fpa.read_noise_e == 120.0
    assert budget.kind == "photon"
    assert budget.floors["read_noise_e_configured"] == 120.0


def test_the_dark_current_block_becomes_a_poisson_term(swir, mwir) -> None:  # type: ignore[no-untyped-def]
    """The reason InSb is cryocooled, as a number: 200 e⁻ a frame uncooled, 0.12 e⁻ at 77 K."""
    swir_cfg, swir_lut = swir
    mwir_cfg, mwir_lut = mwir
    warm = dark_electrons_for(fpa_params_from_config(swir_cfg))
    cold = dark_electrons_for(fpa_params_from_config(mwir_cfg))
    assert warm > 100.0
    assert cold < 1.0
    assert warm / cold > 100.0
    # and warming the InSb array back to room temperature is catastrophic
    room = dark_electrons_for(fpa_params_from_config(mwir_cfg), t_fpa_k=300.0)
    assert room / cold > 1e6


def test_a_reflective_band_skips_the_netd_cross_check(swir) -> None:  # type: ignore[no-untyped-def]
    """§9.4's definition is vacuous where a 300 K scene emits nothing — M11.1's open loop, closed.

    The SWIR file records 976 K because that is what §9.4 evaluates to, and says nothing may
    anchor to it. Here nothing does: the budget is built and the check is not run.
    """
    cfg, lut = swir
    assert cfg.sensor.band.regime == "reflective"
    budget = electron_budget(cfg.sensor, lut)  # would raise if the check ran
    assert predict_netd_k(300.0, cfg.sensor, lut, budget) > 100.0, "still a vacuous NETD"


def test_the_cooled_mwir_camera_passes_its_own_datasheet_claim(mwir) -> None:  # type: ignore[no-untyped-def]
    cfg, lut = mwir
    budget = electron_budget(cfg.sensor, lut)
    predicted = predict_netd_k(300.0, cfg.sensor, lut, budget)
    target = cfg.sensor.noise.netd_mk_at_300k * 1e-3
    assert predicted < target
    assert predicted * 1e3 == pytest.approx(18.4, rel=0.05)
    # and it is within a couple of percent of shot-limited, which is what cooled MWIR is sold on
    floor = shot_limited_netd_k(cfg.sensor, lut)
    assert 1.0 < predicted / floor < 1.05


def test_a_camera_that_cannot_reach_its_own_claim_raises(mwir) -> None:  # type: ignore[no-untyped-def]
    cfg, lut = mwir
    tight = cfg.sensor.model_copy(
        update={"noise": cfg.sensor.noise.model_copy(update={"netd_mk_at_300k": 5.0})}
    )
    with pytest.raises(InfeasibleNoiseConfigError, match="never"):
        electron_budget(tight, lut)
    # and the check can be switched off deliberately, which is not the same as it not existing
    assert electron_budget(tight, lut, check_netd=False).sigma_gaussian == 350.0


def test_a_bolometer_is_refused_and_told_why() -> None:
    cfg, lut = _config("flir_boson_640_lwir")
    with pytest.raises(TypeError, match="ADR 0025 anchor"):
        electron_budget(cfg.sensor, lut)


def test_a_photon_fpa_without_read_noise_is_refused(mwir) -> None:  # type: ignore[no-untyped-def]
    cfg, lut = mwir
    stripped = cfg.sensor.model_copy(
        update={"fpa": cfg.sensor.fpa.model_copy(update={"read_noise_e": None})}
    )
    with pytest.raises(ValueError, match="read_noise_e"):
        electron_budget(stripped, lut)


# ---------------------------------------------------------------------------------------------
# the synthetic bench
# ---------------------------------------------------------------------------------------------


def _bench_netd_k(cfg, lut, budget: NoiseBudget, temperature_k: float, seed: int) -> float:  # type: ignore[no-untyped-def]
    """Measure NETD the way §15 Tier 2 does: σ of a uniform scene over ∂S/∂T, in electrons."""
    params = fpa_params_from_config(cfg)
    rng = np.random.default_rng(seed)
    n_e = mean_signal(temperature_k, cfg.sensor, lut)
    frames = electron_noise(
        np.full(SAMPLES, n_e),
        0.0,
        params.integration_time_s,
        budget.sigma_gaussian,
        rng,
        background_e=budget.dark_electrons + budget.background_electrons,
    )
    sigma = float(frames.std())
    return sigma / signal_derivative_per_k(temperature_k, cfg.sensor, lut)


def test_the_synthetic_ingaas_bench_matches_the_prediction(swir) -> None:  # type: ignore[no-untyped-def]
    """Generator and predictor agree: the noise the camera makes is the noise it is described by."""
    cfg, lut = swir
    budget = electron_budget(cfg.sensor, lut)
    measured = _bench_netd_k(cfg, lut, budget, SWIR_BENCH_K, seed=101)
    predicted = predict_netd_k(SWIR_BENCH_K, cfg.sensor, lut, budget)
    assert measured == pytest.approx(predicted, rel=0.10), (
        f"bench {measured * 1e3:.1f} mK vs predicted {predicted * 1e3:.1f} mK"
    )


def test_the_synthetic_insb_bench_matches_the_prediction(mwir) -> None:  # type: ignore[no-untyped-def]
    cfg, lut = mwir
    budget = electron_budget(cfg.sensor, lut)
    measured = _bench_netd_k(cfg, lut, budget, 300.0, seed=202)
    predicted = predict_netd_k(300.0, cfg.sensor, lut, budget)
    assert measured == pytest.approx(predicted, rel=0.10), (
        f"bench {measured * 1e3:.2f} mK vs predicted {predicted * 1e3:.2f} mK"
    )


def test_the_bench_is_sensitive_to_the_thing_it_measures(mwir) -> None:  # type: ignore[no-untyped-def]
    """A bench that passes against any budget is measuring the predictor, not the camera."""
    cfg, lut = mwir
    honest = electron_budget(cfg.sensor, lut)
    inflated = NoiseBudget(
        kind="photon",
        sigma_gaussian=honest.sigma_gaussian * 10.0,
        dark_electrons=honest.dark_electrons,
    )
    measured = _bench_netd_k(cfg, lut, inflated, 300.0, seed=303)
    assert measured > 2.0 * predict_netd_k(300.0, cfg.sensor, lut, honest)
