"""Beer-Lambert kernel (M8.1): multiplicativity, isothermal invariance, the contrast identity,
segment composition, convexity, the ramp fixture, and dtype guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere import apply_atmosphere, apply_tau_override, path_radiance, transmittance
from irsim.radiometry.planck import band_radiance_tophat

LB = lambda t: band_radiance_tophat(8.0, 12.0, t)  # noqa: E731


def test_multiplicativity_known_values_and_limits() -> None:
    g = math.log(2.0) / 100.0
    assert float(transmittance(100.0, g)) == pytest.approx(0.5, rel=1e-12)
    assert float(transmittance(0.0, g)) == 1.0
    assert float(transmittance(np.inf, g)) == 0.0
    assert float(transmittance(np.inf, 0.0)) == 1.0
    d1, d2 = 37.0, 163.0
    assert float(transmittance(d1 + d2, g)) == pytest.approx(
        float(transmittance(d1, g) * transmittance(d2, g)), rel=1e-12
    )
    with pytest.raises(ValueError):
        transmittance(10.0, -1e-3)
    with pytest.raises(ValueError):
        transmittance(-1.0, 1e-3)


@pytest.mark.parametrize("d", [0.0, 10.0, 200.0, 5000.0, np.inf])
@pytest.mark.parametrize("gamma", [0.0, 1e-3, 1e-1])
def test_isothermal_invariance(d: float, gamma: float) -> None:
    """L = L_B(T_air) → L' = L_B(T_air) at every distance and extinction (< 1e-6 mK)."""
    l_air = LB(290.0)
    out = float(apply_atmosphere(l_air, d, gamma, l_air))
    assert abs(out / l_air - 1.0) < 1e-12


def test_contrast_attenuation_identity_and_segments() -> None:
    l_air, d, g = LB(290.0), 200.0, 2e-3
    hi, lo = (
        float(apply_atmosphere(LB(310.0), d, g, l_air)),
        float(apply_atmosphere(LB(290.0), d, g, l_air)),
    )
    assert (hi - lo) / (LB(310.0) - LB(290.0)) == pytest.approx(
        float(transmittance(d, g)), rel=1e-12
    )
    first = apply_atmosphere(LB(310.0), 80.0, g, l_air)
    both = apply_atmosphere(first, 120.0, g, l_air)
    assert float(both) == pytest.approx(hi, rel=1e-12)


def test_convexity_and_path_radiance() -> None:
    rng = np.random.default_rng(3)
    lb = rng.uniform(10.0, 200.0, 1000)
    l_air = rng.uniform(10.0, 200.0, 1000)
    d = rng.uniform(0.0, 5000.0, 1000)
    out = apply_atmosphere(lb, d, 1e-3, l_air)
    assert np.all(out >= np.minimum(lb, l_air) - 1e-9)
    assert np.all(out <= np.maximum(lb, l_air) + 1e-9)
    tau = transmittance(d, 1e-3)
    np.testing.assert_allclose(path_radiance(tau, l_air), (1 - tau) * l_air, rtol=1e-12)


def test_ramp_fixture_contrast(gbuffer_ramp: dict[str, np.ndarray], tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    lb = tophat_lwir_lut.lookup(gbuffer_ramp["temperature_k"])
    out = apply_atmosphere(lb, gbuffer_ramp["distance_m"], 2e-3, LB(290.0))
    assert out.dtype == np.float32
    ratio = (out[0, -1] - out[0, 0]) / (lb[0, -1] - lb[0, 0])
    assert float(ratio) == pytest.approx(math.exp(-0.1), rel=1e-5)


def test_tau_override_and_dtype_guards() -> None:
    lb = np.array([50.0, 60.0], dtype=np.float32)
    out = apply_tau_override(lb, 0.8, 55.0)
    assert out.dtype == np.float32 and np.allclose(out, 0.8 * lb + 0.2 * 55.0)
    with pytest.raises(TypeError, match="float16"):
        apply_atmosphere(lb.astype(np.float16), 10.0, 1e-3, 55.0)
    with pytest.raises(ValueError):
        apply_tau_override(lb, 1.2, 55.0)
