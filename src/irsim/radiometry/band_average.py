"""Band-weighted averaging of spectral material properties.

The material model is grey within a band (docs/physics-model.md Appendix A #3, §12.3): a spectral
ε(λ), ρ(λ) or τ(λ) is reduced to one scalar per band. This module is the **only sanctioned
route** for that reduction (ADR 0010):

    ⟨s⟩_B(T_ref) = ∫ R(λ) s(λ) B(λ, T_ref) dλ  /  ∫ R(λ) B(λ, T_ref) dλ

with B the energy-form Planck radiance for bolometer cameras and the photon form for photon
cameras, at a reference temperature (300 K by default). The average is **linear in s**, so if
ε + ρ + τ = 1 pointwise, the averages close to the same identity (non-negotiable #4 survives
the reduction). It is also **temperature dependent** when s(λ) slopes across the band; that
dependence is the error the grey approximation accepts, and a test documents its size.

docs/physics-model.md §4.1, §12.3, Appendix A #3
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.band_integration import quadrature_grid, simpson
from irsim.radiometry.planck import spectral_photon_radiance, spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["band_average", "tabulated", "WeightingForm", "T_REF_K"]

FloatArray = NDArray[np.float64]
WeightingForm = Literal["energy", "photon"]
Spectrum = Callable[[FloatArray], FloatArray]

T_REF_K = 300.0


def tabulated(wavelength_um: FloatArray, values: FloatArray) -> Spectrum:
    """A spectrum callable from a table, linearly interpolated and held constant beyond its ends.

    Holding the end values (rather than zeroing) is deliberate: a material file that stops at
    14 µm still has an emissivity at 14.2 µm, and the response there is already nearly zero.
    """
    lam = np.asarray(wavelength_um, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if lam.ndim != 1 or lam.shape != val.shape or np.any(np.diff(lam) <= 0):
        raise ValueError(
            "tabulated spectrum needs equal-length 1-D arrays with increasing wavelength"
        )

    def spectrum(grid_um: FloatArray) -> FloatArray:
        return np.asarray(np.interp(grid_um, lam, val), dtype=np.float64)

    return spectrum


def band_average(
    response: SpectralResponse,
    spectrum: Spectrum,
    t_ref_k: float = T_REF_K,
    form: WeightingForm = "energy",
) -> float:
    """Planck-weighted band average of ``spectrum`` under ``response`` at ``t_ref_k``.

    ``form='energy'`` weights by B(λ, T) (bolometers absorb power); ``form='photon'`` weights by
    B_q(λ, T) (photon detectors count photons). The two differ by the λ-weighting hc/λ, so a
    sloped spectrum averages differently -- use the form of the camera that will see it.
    """
    if form == "energy":
        planck = spectral_radiance
    elif form == "photon":
        planck = spectral_photon_radiance
    else:
        raise ValueError(f"form must be 'energy' or 'photon', got {form!r}")
    grid = quadrature_grid(response)
    dl = float(grid[1] - grid[0])
    weight = response.resampled(grid) * planck(grid, np.asarray(t_ref_k, dtype=np.float64))
    s = np.asarray(spectrum(grid), dtype=np.float64)
    if s.shape != grid.shape or not np.all(np.isfinite(s)):
        raise ValueError("spectrum must return a finite array on the quadrature grid")
    denominator = float(simpson(weight, dl))
    if denominator <= 0.0:
        raise ValueError("response carries no radiance at this temperature")
    return float(simpson(weight * s, dl)) / denominator
