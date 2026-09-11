"""Grey-band approximation error (M8.7, ADR 0048): grey identity, the curve of growth, and the
bounds the L2 atmosphere accepts for structured gamma(lambda)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.atmosphere import (
    band_transmittance_spectral,
    effective_gamma,
    fit_grey_gamma,
    grey_fit_error,
)
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response


def _tophat(tmp_path: pathlib.Path, lo: float, hi: float) -> SpectralResponse:
    p = tmp_path / f"tophat_{lo}_{hi}.csv"
    p.write_text(f"# exact top-hat\n{lo},1.0\n{hi},1.0\n")
    return load_spectral_response(p)


def test_grey_identity(tmp_path: pathlib.Path) -> None:
    resp = _tophat(tmp_path, 8.0, 12.0)
    g = 3e-4
    d = np.array([50.0, 500.0, 2000.0])
    tau = band_transmittance_spectral(resp, lambda lam: np.full_like(lam, g), d)
    np.testing.assert_allclose(tau, np.exp(-g * d), rtol=1e-12)
    assert fit_grey_gamma(resp, lambda lam: np.full_like(lam, g)) == pytest.approx(g, rel=1e-9)


def test_curve_of_growth_for_structured_gamma(tmp_path: pathlib.Path) -> None:
    resp = _tophat(tmp_path, 8.0, 12.0)
    two_level = lambda lam: np.where(lam < 10.0, 2e-4, 1.2e-3)  # noqa: E731
    d = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    g_eff = effective_gamma(resp, two_level, d)
    assert np.all(np.diff(g_eff) < 0.0), "strong absorbers saturate first"
    # a band-averaged gamma (averaging gamma, not tau) would give the constant mean
    assert g_eff[0] > g_eff[-1] * 1.2


def test_grey_fit_errors_match_adr_0048(tmp_path: pathlib.Path) -> None:
    """The numbers recorded in ADR 0048 (scripts/validate_atmosphere_band_average.py): a grey
    gamma_B fitted over 0-300 m over-attenuates (the sign is fixed by convexity: the mean of
    exponentials exceeds the exponential of the mean) by 1.7 % at 500 m and 8.7 % at 1 km for the
    two-level LWIR band, and by 18 % / 42 % for a MWIR band with an opaque CO2 notch."""
    lwir = _tophat(tmp_path, 8.0, 12.0)
    err_lwir = grey_fit_error(lwir, lambda lam: np.where(lam < 10.0, 2e-4, 1.2e-3))
    assert err_lwir[500.0] == pytest.approx(-0.0165, abs=0.002), err_lwir
    assert err_lwir[1000.0] == pytest.approx(-0.0873, abs=0.005), err_lwir
    mwir = _tophat(tmp_path, 3.0, 5.0)
    notch = lambda lam: np.where((lam >= 4.2) & (lam <= 4.4), 5.0, 3e-4)  # noqa: E731
    err_mwir = grey_fit_error(mwir, notch)
    assert err_mwir[500.0] == pytest.approx(-0.182, abs=0.01), err_mwir
    assert err_mwir[1000.0] == pytest.approx(-0.417, abs=0.02), err_mwir
    # both fits over-attenuate: a grey model is conservative (too little contrast) at range
    assert all(e < 0.0 for e in (*err_lwir.values(), *err_mwir.values()))
