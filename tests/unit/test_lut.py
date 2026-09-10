"""BandLUT (M1.6, ADR 0011): the fast path against the quadrature oracle, in millikelvin.

The LUT is only as trustworthy as this comparison. Off-grid temperatures are used throughout
(grid points would hide interpolation error), and the float32 kernel contract is exercised
exactly as the GPU will run it, including the clamp.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from irsim.radiometry.band_integration import band_radiance, d_band_radiance_dT
from irsim.radiometry.constants import LUT_DT_K, LUT_N, LUT_T_MAX_K, LUT_T_MIN_K
from irsim.radiometry.lut import QUANTITIES, BandLUT
from irsim.radiometry.planck import band_photon_radiance_tophat, band_radiance_tophat

OFF_GRID = np.arange(200.013, 1000.0, 0.7919)  # never on a 0.05 K node


def test_grid_and_dtypes(boson_lut: BandLUT) -> None:
    assert (boson_lut.t0_k, boson_lut.t1_k, boson_lut.n) == (LUT_T_MIN_K, LUT_T_MAX_K, LUT_N)
    assert boson_lut.dt_k == pytest.approx(LUT_DT_K, rel=1e-12)
    for q in QUANTITIES:
        assert boson_lut.table(q).dtype == np.float32 and boson_lut.table(q).shape == (LUT_N,)
    assert boson_lut.temperatures_k[6000] == pytest.approx(500.0, abs=1e-9)  # integer index lands


def test_lookup_vs_quadrature_under_5mK(boson_lut: BandLUT, boson_response) -> None:  # type: ignore[no-untyped-def]
    """Off-grid lookup vs the Simpson oracle, in mK via dLb/dT. Expect ~0.03 mK."""
    ref = band_radiance(boson_response, OFF_GRID)
    slope = d_band_radiance_dT(boson_response, OFF_GRID)
    err_mk = np.abs(boson_lut.lookup(OFF_GRID).astype(np.float64) - ref) / slope * 1e3
    worst = float(err_mk.max())
    print(f"LUT vs quadrature: max {worst:.4f} mK at {OFF_GRID[int(err_mk.argmax())]:.2f} K")
    assert worst < 5.0, f"{worst:.3f} mK exceeds the 5 mK budget (1/10 of the tightest NETD)"
    assert worst < 0.1, f"{worst:.3f} mK: expected ~0.03 mK from linear interpolation at 0.05 K"


def test_float32_cast_costs_under_0p1mK(tophat_lwir_lut: BandLUT) -> None:
    """The float32 path vs float64 interpolation of float64 closed-form values on the same grid.

    Measured 0.086 mK, dominated not by the float32 tables (~0.004 mK) but by the kernel's
    float32 index u ~ 16000 (spacing 1e-3 of a 0.05 K step = 0.05 mK) and the float32 input
    temperature near 1000 K (spacing 0.061 mK). Fifty times under the 5 mK budget (ADR 0011)."""
    temps = OFF_GRID
    grid = tophat_lwir_lut.temperatures_k
    exact = np.array([band_radiance_tophat(7.5, 13.5, float(t)) for t in grid])
    f64 = np.interp(temps, grid, exact)
    f32 = tophat_lwir_lut.lookup(temps).astype(np.float64)
    slope = np.array([d_band_radiance_dT_tophat(t) for t in temps])
    err_mk = np.abs(f32 - f64) / slope * 1e3
    assert err_mk.max() < 0.1, f"float32 costs {err_mk.max():.4f} mK"


def d_band_radiance_dT_tophat(t: float) -> float:
    d = 1e-3
    return (band_radiance_tophat(7.5, 13.5, t + d) - band_radiance_tophat(7.5, 13.5, t - d)) / (
        2 * d
    )


def test_tophat_lut_matches_closed_form(tophat_lwir_lut: BandLUT) -> None:
    for t in (250.0, 300.0, 373.0, 600.0, 999.0):
        lb = float(tophat_lwir_lut.lookup(t)[()])
        assert abs(lb / band_radiance_tophat(7.5, 13.5, t) - 1.0) < 1e-5
        lq = float(tophat_lwir_lut.lookup(t, "lb_q")[()])
        assert abs(lq / band_photon_radiance_tophat(7.5, 13.5, t) - 1.0) < 1e-5


def test_tables_monotone_and_positive(boson_lut: BandLUT) -> None:
    assert np.all(np.diff(boson_lut.lb) > 0), "Lb must be strictly increasing"
    assert np.all(np.diff(boson_lut.lb_q) > 0)
    assert np.all(boson_lut.dlb_dt > 0) and np.all(boson_lut.dlb_q_dt > 0)


def test_derivative_tables_match_finite_difference_of_radiance_tables(boson_lut: BandLUT) -> None:
    # +/-10 nodes (0.5 K): float32 rounding of a ~1500 W/m2/sr entry over a 0.1 K stencil is
    # 5e-4 relative; over 1 K it is 5e-5, while the truncation error stays ~1e-6.
    k = 10
    dt = boson_lut.dt_k * k
    for rad, der in (("lb", "dlb_dt"), ("lb_q", "dlb_q_dt")):
        r = boson_lut.table(rad).astype(np.float64)
        fd = (r[2 * k :] - r[: -2 * k]) / (2 * dt)
        d = boson_lut.table(der).astype(np.float64)[k:-k]
        rel = np.abs(fd / d - 1.0)
        assert rel.max() < 1e-4, f"{der}: {rel.max():.2e}"


def test_clamp_matches_kernel(boson_lut: BandLUT) -> None:
    assert boson_lut.lookup(199.0) == boson_lut.lookup(200.0)
    assert boson_lut.lookup(150.0) == boson_lut.lookup(200.0)
    # top end: u clamps to N - 1.001, i.e. 0.001 of a step below the last node
    top = boson_lut.lookup(1000.0)
    assert boson_lut.lookup(1500.0) == top
    assert abs(float(top) - float(boson_lut.lb[-1])) < 2e-3 * float(
        boson_lut.lb[-1] - boson_lut.lb[-2]
    )
    assert boson_lut.out_of_range(np.array([199.0, 200.0, 650.0, 1000.0, 1000.01])).tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_float16_input_refused(boson_lut: BandLUT) -> None:
    with pytest.raises(TypeError, match="float16"):
        boson_lut.lookup(np.array([300.0], dtype=np.float16))


def test_output_dtype_and_shapes(boson_lut: BandLUT) -> None:
    t = np.full((4, 5), 300.0, dtype=np.float32)
    out = boson_lut.lookup(t)
    assert out.dtype == np.float32 and out.shape == (4, 5)
    assert boson_lut.lookup(300.0).dtype == np.float32
    with pytest.raises(ValueError, match="quantity"):
        boson_lut.lookup(300.0, "watts")  # type: ignore[arg-type]


def test_build_is_fast_enough(boson_response) -> None:  # type: ignore[no-untyped-def]
    t = time.perf_counter()
    BandLUT.build(boson_response)
    elapsed = time.perf_counter() - t
    assert elapsed < 2.0, f"LUT build took {elapsed:.2f} s (budget 2 s for the session fixture)"


def test_constructor_rejects_non_float32_tables() -> None:
    z = np.zeros(16001, dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        BandLUT(200.0, 1000.0, 16001, z, z, z, z)


# --- inverse lookup (M1.7) -------------------------------------------------------------


def test_inverse_round_trip_under_1mK(boson_lut: BandLUT) -> None:
    """T -> Lb -> T_app on a 0.013 K off-grid sweep: < 1 mK (expected ~0.05 mK, float32)."""
    temps = np.arange(200.013, 1000.0, 0.013)
    for q in ("lb", "lb_q"):
        raw = boson_lut.apparent_temperature(boson_lut.lookup(temps, q), q)  # type: ignore[arg-type]
        assert raw.dtype == np.float32
        err_mk = np.abs(raw.astype(np.float64) - temps) * 1e3
        print(f"{q}: round trip max {err_mk.max():.4f} mK")
        assert err_mk.max() < 1.0, f"{q}: {err_mk.max():.3f} mK"


def test_grey_body_apparent_temperature_is_the_product(tophat_lwir_lut: BandLUT) -> None:
    """ε = 0.9 at 300 K in 7.5-13.5 µm reads 293.51 K apparent -- the 6.5 K gap *is* what a
    radiometric camera reports (§3.3). LUT inverse vs brentq on the closed form < 5 mK."""
    from scipy.optimize import brentq

    radiance = 0.9 * band_radiance_tophat(7.5, 13.5, 300.0)
    exact = brentq(lambda t: band_radiance_tophat(7.5, 13.5, t) - radiance, 250.0, 300.0, xtol=1e-9)
    assert exact == pytest.approx(293.51, abs=0.02)
    t_app = float(tophat_lwir_lut.apparent_temperature(np.float32(radiance))[()])
    assert abs(t_app - exact) * 1e3 < 5.0, f"{t_app} vs {exact}"
    assert t_app < 300.0 - 6.0, "apparent must sit well below kinetic for a grey body"


def test_inverse_clamps_and_flags_out_of_range(boson_lut: BandLUT) -> None:
    lo, hi = float(boson_lut.lb[0]), float(boson_lut.lb[-1])
    t = boson_lut.apparent_temperature(np.array([lo * 0.5, lo, hi, hi * 2.0], dtype=np.float32))
    assert t.tolist() == pytest.approx([200.0, 200.0, 1000.0, 1000.0], abs=1e-3)
    assert np.all(np.isfinite(t)), "an image must never carry NaN"
    flags = boson_lut.radiance_out_of_range(np.array([lo * 0.5, lo, 50.0, hi, hi * 2.0]))
    assert flags.tolist() == [True, False, False, False, True]


def test_inverse_refuses_float16_and_derivative_tables(boson_lut: BandLUT) -> None:
    with pytest.raises(TypeError, match="float16"):
        boson_lut.apparent_temperature(np.array([50.0], dtype=np.float16))
    with pytest.raises(ValueError, match="radiance table"):
        boson_lut.apparent_temperature(np.array([1.0]), "dlb_dt")  # type: ignore[arg-type]


def test_inverse_is_monotone_and_handles_exact_nodes(boson_lut: BandLUT) -> None:
    nodes = boson_lut.lb[::1000]
    back = boson_lut.apparent_temperature(nodes).astype(np.float64)
    np.testing.assert_allclose(back, boson_lut.temperatures_k[::1000], atol=1e-3)
    assert np.all(
        np.diff(boson_lut.apparent_temperature(np.linspace(nodes[0], nodes[-1], 5000))) >= 0
    )
