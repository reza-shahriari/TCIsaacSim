"""Band averaging (M1.5, ADR 0010): linearity, closure survival, and the size of the error the
grey-within-band approximation accepts."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import DEFAULT_DATA_DIR
from irsim.radiometry.band_average import band_average, tabulated
from irsim.radiometry.planck import band_radiance_tophat
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

BOSON_CSV = DEFAULT_DATA_DIR / "spectra" / "responses" / "boson_vox.csv"


@pytest.fixture(scope="module")
def boson() -> SpectralResponse:
    return load_spectral_response(BOSON_CSV)


@pytest.fixture
def tophat_8_12(tmp_path: pathlib.Path) -> SpectralResponse:
    p = tmp_path / "tophat.csv"
    p.write_text("8.0,1.0\n12.0,1.0\n")
    return load_spectral_response(p)


@pytest.mark.parametrize("form", ["energy", "photon"])
def test_constant_spectrum_averages_to_itself(boson: SpectralResponse, form: str) -> None:
    for value in (0.0, 0.37, 0.95, 1.0):
        avg = band_average(boson, lambda lam, v=value: np.full_like(lam, v), form=form)  # type: ignore[arg-type]
        assert abs(avg - value) < 1e-12


def test_step_spectrum_matches_ratio_of_tophats(tophat_8_12: SpectralResponse) -> None:
    """ε = 1 on 8-10 µm, 0 on 10-12 µm: the average is Lb(8-10)/Lb(8-12), a closed form.

    The step sits on a quadrature node and takes the midpoint value 0.5 there, so the Simpson
    errors of the two panels that share the node cancel and the comparison is at 1e-4 (a
    discontinuity inside a panel would cost O(h) ~ 1e-3, which is not a physics error)."""
    step = lambda lam: np.where(  # noqa: E731
        np.isclose(lam, 10.0, atol=1e-9), 0.5, np.where(lam < 10.0, 1.0, 0.0)
    )
    for T in (300.0, 500.0):
        expected = band_radiance_tophat(8.0, 10.0, T) / band_radiance_tophat(8.0, 12.0, T)
        got = band_average(tophat_8_12, step, t_ref_k=T)
        assert abs(got - expected) < 1e-4, f"T={T}: {got} vs {expected}"


def test_closure_survives_averaging(boson: SpectralResponse) -> None:
    """Pointwise ε + ρ + τ = 1 ⇒ ⟨ε⟩ + ⟨ρ⟩ + ⟨τ⟩ = 1 to 1e-9 (linearity; non-negotiable #4)."""
    lam = np.linspace(7.0, 14.0, 71)
    eps = tabulated(lam, 0.6 + 0.3 * np.sin(lam))
    tau = tabulated(lam, 0.05 * (1 + np.cos(2 * lam)))
    rho = lambda g: 1.0 - eps(g) - tau(g)  # noqa: E731
    for form in ("energy", "photon"):
        total = sum(band_average(boson, s, form=form) for s in (eps, rho, tau))  # type: ignore[arg-type]
        assert abs(total - 1.0) < 1e-9, f"{form}: {total}"


def test_sloped_emissivity_depends_on_reference_temperature(boson: SpectralResponse) -> None:
    """The grey approximation's accepted error: a slope of 0.1 across the band moves the average
    by > 1e-3 between 300 and 600 K because the Planck weighting shifts blueward (ADR 0010)."""
    sloped = tabulated(np.array([7.0, 14.0]), np.array([0.95, 0.85]))
    a300 = band_average(boson, sloped, t_ref_k=300.0)
    a600 = band_average(boson, sloped, t_ref_k=600.0)
    assert a600 > a300, "hotter weighting favours short wavelengths, where ε is higher"
    assert a600 - a300 > 1e-3, f"difference {a600 - a300:.2e}"
    assert a600 - a300 < 1e-2, "but it is a small correction, which is why grey-in-band is accepted"


def test_energy_and_photon_forms_differ_for_sloped_spectrum(boson: SpectralResponse) -> None:
    sloped = tabulated(np.array([7.0, 14.0]), np.array([0.95, 0.85]))
    e = band_average(boson, sloped, form="energy")
    q = band_average(boson, sloped, form="photon")
    assert e > q, "photon weighting (∝ λ B) favours long wavelengths, where ε is lower"
    assert 1e-4 < e - q < 1e-2


def test_bad_inputs(boson: SpectralResponse) -> None:
    with pytest.raises(ValueError, match="form"):
        band_average(boson, lambda lam: lam, form="watts")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        band_average(boson, lambda lam: np.full_like(lam, np.nan))
    with pytest.raises(ValueError, match="increasing"):
        tabulated(np.array([2.0, 1.0]), np.array([1.0, 1.0]))
