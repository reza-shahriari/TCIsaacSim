"""Mean-reverting drift of the fixed pattern (M9.4): the statistics, not the API.

The whole point of choosing Ornstein-Uhlenbeck over a random walk is a property you cannot see in
one frame, so every test here measures a long run against a closed form:

* the lag-τ autocorrelation is e^{-1}, which a random walk cannot produce -- its lag-τ correlation
  tends to 1 as the run lengthens;
* the variance is stationary after 2000 frames, so the configured 3-D ratios survive a long
  sequence (a random walk destroys them silently, a little per frame, while every frame still
  looks plausible);
* the mean stays at zero, so drift adds no pedestal -- the pedestal is physical and comes from the
  housing and FPA nodes (M3.3, M9.3), which is exactly the separation ADR 0054 records;
* τ = ∞ is frozen bit-identical, the control case;
* and the step is *exact*: a coarse step and many fine steps over the same interval give the same
  stationary law, so the drift does not depend on the frame rate it happens to be run at.

docs/physics-model.md §10.3, §10.2. ADR 0054.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.config.sensor import RATIO_ORDER
from irsim.noise import DRIFTING_COMPONENTS, FixedPattern, FpnDrift, Sigmas7, drift_rng, ou_step

SHAPE = (64, 96)
SEED = 7
# One ratio vector with every fixed term present, so no test passes by a sigma being zero.
SIGMAS = Sigmas7(t=0.02, v=0.30, h=0.20, tv=0.05, th=0.04, vh=0.40, tvh=1.0)


def _fixed() -> FixedPattern:
    return FixedPattern.generate(SHAPE, SIGMAS, SEED)


def _sigma(hist: np.ndarray, stride: int) -> float:
    """Stationary sigma of a (frames, elements) history, averaged over decorrelated frames.

    A single frame's spatial std of a V or H term is an estimate from only a few dozen elements
    (+/- 7 % for 96 columns), and consecutive frames are correlated by construction -- tau is
    tens of frames. Averaging the *variance* over frames spaced more than 3 tau apart is what
    makes the 5 % assertions below a statement about the process rather than about the estimator.
    """
    frames = hist[::stride]
    return float(np.sqrt(np.mean(np.var(frames, axis=1))))


def _series(drift: FpnDrift, n_frames: int, dt_s: float, seed: int = SEED) -> dict[str, np.ndarray]:
    """Run the drift for ``n_frames`` and return each component's history."""
    rng = drift_rng(seed)
    pattern = _fixed()
    out: dict[str, list[np.ndarray]] = {c: [] for c in DRIFTING_COMPONENTS}
    for _ in range(n_frames):
        pattern = drift.advance(pattern, dt_s, rng)
        for c in DRIFTING_COMPONENTS:
            out[c].append(np.asarray(getattr(pattern, c), dtype=np.float64).ravel().copy())
    return {c: np.stack(v) for c, v in out.items()}  # (frames, elements)


# -- the defining property: mean reversion ---------------------------------------------------


@pytest.mark.parametrize("component", DRIFTING_COMPONENTS)
def test_lag_tau_autocorrelation_is_one_over_e(component: str) -> None:
    """rho(tau) = e^{-1} = 0.368 +/- 0.05. A random walk gives ~1 here; this is the whole ADR.

    Measured across the ensemble of elements at many starting frames, after discarding a burn-in,
    so the estimate is of the *process* and not of one unlucky trajectory.
    """
    tau, dt = 120.0, 6.0  # tau = 20 frames
    lag_frames = int(round(tau / dt))
    drift = FpnDrift(tau_s=tau, sigmas=SIGMAS)
    hist = _series(drift, n_frames=1200, dt_s=dt)[component]

    burn = 200
    a = hist[burn:-lag_frames]
    b = hist[burn + lag_frames :]
    a = a - a.mean()
    b = b - b.mean()
    rho = float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    assert rho == pytest.approx(math.exp(-1.0), abs=0.05)
    assert rho == pytest.approx(drift.autocorrelation(tau), abs=0.05)


@pytest.mark.parametrize("n_tau", [0.5, 1.0, 2.0, 3.0])
def test_the_autocorrelation_decays_exponentially_at_every_lag(n_tau: float) -> None:
    """Not just at tau: rho(k) = e^{-k/tau} across the curve, pinning the shape of the process."""
    tau, dt = 100.0, 5.0
    drift = FpnDrift(tau_s=tau, sigmas=SIGMAS)
    hist = _series(drift, n_frames=1500, dt_s=dt)["vh"]
    lag = int(round(n_tau * tau / dt))
    burn = 200
    a = hist[burn:-lag]
    b = hist[burn + lag :]
    a = a - a.mean()
    b = b - b.mean()
    rho = float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    assert rho == pytest.approx(math.exp(-n_tau), abs=0.05)


def test_a_random_walk_would_fail_the_autocorrelation_test() -> None:
    """The control: the test above genuinely discriminates, rather than passing on anything noisy.

    A random walk with a matched per-step innovation is run through the same estimator; its lag-tau
    correlation must land far above e^{-1}, confirming the assertion is not vacuous.
    """
    tau, dt, n = 120.0, 6.0, 1200
    lag = int(round(tau / dt))
    rng = np.random.default_rng(0)
    sigma = SIGMAS.vh
    innovation = sigma * math.sqrt(1.0 - math.exp(-2.0 * dt / tau))
    x = rng.standard_normal(SHAPE[0] * SHAPE[1]) * sigma
    hist = []
    for _ in range(n):
        x = x + innovation * rng.standard_normal(x.shape)  # no decay term: a random walk
        hist.append(x.copy())
    h = np.stack(hist)
    burn = 200
    a = h[burn:-lag]
    b = h[burn + lag :]
    a = a - a.mean()
    b = b - b.mean()
    rho = float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    assert rho > 0.9  # nowhere near e^{-1}


# -- stationarity ----------------------------------------------------------------------------


@pytest.mark.parametrize("component", DRIFTING_COMPONENTS)
def test_variance_is_stationary_after_2000_frames(component: str) -> None:
    """sigma stays at the configured value, so the 3-D ratios remain the sensor's identity."""
    tau, dt = 120.0, 6.0  # tau = 20 frames
    stride = 30  # 1.5 tau; adjacent samples correlate at 0.22, low enough to average over
    drift = FpnDrift(tau_s=tau, sigmas=SIGMAS)
    hist = _series(drift, n_frames=2000, dt_s=dt)[component]
    sigma_cfg = getattr(SIGMAS, component)

    # The roadmap's requirement: sigma is still the configured one after 2000 frames.
    late = _sigma(hist[-900:], stride)
    assert late == pytest.approx(sigma_cfg, rel=0.05)
    # And it has not trended: the start and the end of the run agree. The looser bound is the
    # estimator's, not the process's -- comparing two 30-sample estimates costs sqrt(2) in error.
    early = _sigma(hist[:900], stride)
    assert late == pytest.approx(early, rel=0.08)


def test_the_ratio_vector_survives_a_long_run() -> None:
    """sigma_V : sigma_H : sigma_VH after 2000 frames still matches the configured ratios.

    This is the failure a random walk produces and no single frame reveals: each frame looks like
    a plausible thermal image while the camera's fingerprint quietly walks away.
    """
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    hist = _series(drift, n_frames=2000, dt_s=6.0)
    for a, b in (("v", "h"), ("v", "vh"), ("h", "vh")):
        got = _sigma(hist[a][-1200:], 60) / _sigma(hist[b][-1200:], 60)
        want = getattr(SIGMAS, a) / getattr(SIGMAS, b)
        assert got == pytest.approx(want, rel=0.08)


@pytest.mark.parametrize("component", DRIFTING_COMPONENTS)
def test_the_mean_stays_at_zero(component: str) -> None:
    """Drift must add no pedestal: the DC level is physical (M3.3/M9.3), not stochastic."""
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    hist = _series(drift, n_frames=2000, dt_s=6.0)[component]
    sigma_cfg = getattr(SIGMAS, component)
    # Decorrelated frames only, so the standard error of the mean is the honest one.
    frames = hist[::60]
    n_eff = frames.size
    assert abs(float(frames.mean())) < 4.0 * sigma_cfg / math.sqrt(n_eff)


# -- the exact step --------------------------------------------------------------------------


def test_one_coarse_step_has_the_same_law_as_many_fine_ones() -> None:
    """The update is the closed-form OU solution, so the drift does not depend on the frame rate.

    A first-order (Euler-Maruyama) discretisation would give a different stationary variance at
    coarse dt, which is how the same camera at 9 Hz and 60 Hz would end up with different FPN.
    """
    tau, total = 120.0, 600.0
    sigma = SIGMAS.vh
    n_elem = 20000

    coarse = ou_step(
        np.zeros(n_elem), sigma, total, tau, np.random.default_rng(1)
    )  # one 5-tau step from zero
    x = np.zeros(n_elem)
    rng = np.random.default_rng(2)
    for _ in range(100):
        x = ou_step(x, sigma, total / 100.0, tau, rng)

    # From a zero start, both must have relaxed to the stationary sigma.
    assert float(np.std(coarse)) == pytest.approx(sigma, rel=0.05)
    assert float(np.std(x)) == pytest.approx(sigma, rel=0.05)
    assert abs(float(np.mean(coarse))) < 0.1 * sigma
    assert abs(float(np.mean(x))) < 0.1 * sigma


def test_the_stationary_variance_is_exactly_preserved_in_expectation() -> None:
    """Var(x') = sigma^2 e^{-2dt/tau} + sigma^2 (1 - e^{-2dt/tau}) = sigma^2, at any step size.

    Checked over a wide range of dt/tau, including dt >> tau where a decaying-only or an
    innovation-only bug would be most obvious.
    """
    sigma, tau, n = 0.4, 50.0, 40000
    rng = np.random.default_rng(11)
    start = sigma * rng.standard_normal(n)
    for dt in (0.1, 1.0, 50.0, 500.0):
        out = ou_step(start, sigma, dt, tau, np.random.default_rng(3))
        assert float(np.std(out)) == pytest.approx(sigma, rel=0.03)


# -- frozen ----------------------------------------------------------------------------------


def test_infinite_tau_is_frozen_bit_identical() -> None:
    """The control case, and the right model for an ideal sensor."""
    drift = FpnDrift(tau_s=float("inf"), sigmas=SIGMAS)
    assert drift.frozen
    start = _fixed()
    rng = drift_rng(SEED)
    pattern = start
    for _ in range(50):
        pattern = drift.advance(pattern, 6.0, rng)
    for c in DRIFTING_COMPONENTS:
        assert np.array_equal(getattr(pattern, c), getattr(start, c))
    assert drift.autocorrelation(1e9) == 1.0


def test_a_zero_step_changes_nothing() -> None:
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    start = _fixed()
    assert drift.advance(start, 0.0, drift_rng(SEED)) is start


def test_a_zero_sigma_component_stays_zero() -> None:
    """A camera configured with no column FPN must not acquire some by drifting."""
    sigmas = Sigmas7(t=0.0, v=0.3, h=0.0, tv=0.0, th=0.0, vh=0.4, tvh=1.0)
    drift = FpnDrift(tau_s=60.0, sigmas=sigmas)
    pattern = FixedPattern.generate(SHAPE, sigmas, SEED)
    rng = drift_rng(SEED)
    for _ in range(100):
        pattern = drift.advance(pattern, 6.0, rng)
    assert np.all(pattern.h == 0.0)
    assert float(pattern.vh.std()) > 0.0


# -- contracts -------------------------------------------------------------------------------


def test_dtype_and_shape_are_preserved() -> None:
    """float32 or better everywhere the signal flows (non-negotiable #2)."""
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    out = drift.advance(_fixed(), 6.0, drift_rng(SEED))
    assert out.v.dtype == out.h.dtype == out.vh.dtype == np.float32
    assert out.shape == SHAPE
    assert out.v.shape == (SHAPE[0],) and out.h.shape == (SHAPE[1],)


def test_advance_does_not_mutate_its_input() -> None:
    """A golden sequence must not depend on how many times the pattern was read."""
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    start = _fixed()
    before = {c: np.array(getattr(start, c), copy=True) for c in DRIFTING_COMPONENTS}
    drift.advance(start, 6.0, drift_rng(SEED))
    for c in DRIFTING_COMPONENTS:
        assert np.array_equal(getattr(start, c), before[c])


def test_the_run_is_reproducible_from_the_sensor_seed() -> None:
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    a = _series(drift, 20, 6.0, seed=SEED)
    b = _series(drift, 20, 6.0, seed=SEED)
    c = _series(drift, 20, 6.0, seed=SEED + 1)
    for comp in DRIFTING_COMPONENTS:
        assert np.array_equal(a[comp], b[comp])
        assert not np.array_equal(a[comp], c[comp])


def test_restricting_the_component_set_freezes_the_rest() -> None:
    """The ablation switch: breathe VH only, and the stripes must stay put."""
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS, components=("vh",))
    start = _fixed()
    pattern = start
    rng = drift_rng(SEED)
    for _ in range(20):
        pattern = drift.advance(pattern, 6.0, rng)
    assert np.array_equal(pattern.v, start.v)
    assert np.array_equal(pattern.h, start.h)
    assert not np.array_equal(pattern.vh, start.vh)


def test_the_global_term_cannot_be_drifted_here() -> None:
    """ADR 0054: the DC level is produced by the housing and FPA nodes, not by a random walk.

    Allowing 't' here would count the same drift twice -- once physically and once stochastically.
    """
    with pytest.raises(ValueError, match="global offset drifts physically"):
        FpnDrift(tau_s=120.0, sigmas=SIGMAS, components=("vh", "t"))


def test_a_non_positive_tau_is_refused() -> None:
    for tau in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="must be positive"):
            FpnDrift(tau_s=tau, sigmas=SIGMAS)


def test_a_negative_step_is_refused() -> None:
    drift = FpnDrift(tau_s=120.0, sigmas=SIGMAS)
    with pytest.raises(ValueError, match="non-negative"):
        drift.advance(_fixed(), -1.0, drift_rng(SEED))


def test_the_drifting_set_is_the_fixed_terms_of_the_ratio_vector() -> None:
    """A guard on the abstraction: every drifting name must be a real 3-D component."""
    assert set(DRIFTING_COMPONENTS) <= set(RATIO_ORDER)
    assert set(DRIFTING_COMPONENTS) == {"v", "h", "vh"}
