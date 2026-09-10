"""Reference band integration: ∫R(λ) B(λ, T) dλ by composite Simpson quadrature.

This is the oracle. It is float64, written for clarity, vectorised over temperature, and every
faster path (the LUT, the GPU kernels) is tested against it -- never the other way round. It
uses NumPy only, so the oracle does not depend on a library's adaptive tolerances.

Grid (§3.2 b): 0.01 µm spacing, **edge-aligned** to the response file's support so that a file
whose first and last samples are its band edges (an exact top-hat) is integrated exactly, and
**odd-length** as composite Simpson requires. The response is resampled onto this grid by linear
interpolation and is zero outside the file's support.

Four quantities, one per LUT table (§13.5): band radiance (energy and photon form) and the
band-integrated thermal derivative in both forms (§3.4, §9.4).

docs/physics-model.md §3.2 (b), §3.4, §9.4
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.planck import (
    d_spectral_photon_radiance_dT,
    d_spectral_radiance_dT,
    spectral_photon_radiance,
    spectral_radiance,
)
from irsim.radiometry.spectral_response import RESAMPLE_DL_UM, SpectralResponse

__all__ = [
    "quadrature_grid",
    "simpson",
    "band_radiance",
    "band_photon_radiance",
    "d_band_radiance_dT",
    "d_band_photon_radiance_dT",
]

FloatArray = NDArray[np.float64]


def quadrature_grid(response: SpectralResponse, dl_um: float = RESAMPLE_DL_UM) -> FloatArray:
    """Odd-length uniform grid at ``dl_um`` spacing, starting at the response's first sample
    and ending at or one step beyond its last (where R is zero anyway)."""
    lo, hi = response.support_um
    n_intervals = int(np.ceil((hi - lo) / dl_um - 1e-9))
    if n_intervals % 2 == 1:
        n_intervals += 1
    n_intervals = max(n_intervals, 2)
    return lo + dl_um * np.arange(n_intervals + 1, dtype=np.float64)


def simpson(y: FloatArray, dx: float) -> FloatArray:
    """Composite Simpson over the last axis of ``y`` (odd length, uniform spacing ``dx``)."""
    n = y.shape[-1]
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs an odd number of samples >= 3, got {n}")
    return np.asarray(
        dx
        / 3.0
        * (y[..., 0] + y[..., -1] + 4.0 * y[..., 1:-1:2].sum(-1) + 2.0 * y[..., 2:-1:2].sum(-1)),
        dtype=np.float64,
    )


def _integrate(
    response: SpectralResponse,
    temperature_k: FloatArray | float,
    spectral_fn: object,
) -> FloatArray:
    grid = quadrature_grid(response)
    weights = response.resampled(grid)
    t = np.atleast_1d(np.asarray(temperature_k, dtype=np.float64))
    spectral = spectral_fn(grid[None, :], t[:, None])  # type: ignore[operator]
    out = simpson(weights[None, :] * spectral, float(grid[1] - grid[0]))
    return np.asarray(out, dtype=np.float64)


def band_radiance(response: SpectralResponse, temperature_k: FloatArray | float) -> FloatArray:
    """L_B(T) = ∫R(λ) B(λ, T) dλ in W m^-2 sr^-1, for each temperature. §3.2 (b)."""
    return _integrate(response, temperature_k, spectral_radiance)


def band_photon_radiance(
    response: SpectralResponse, temperature_k: FloatArray | float
) -> FloatArray:
    """L_{B,q}(T) = ∫R(λ) B_q(λ, T) dλ in photons s^-1 m^-2 sr^-1. §3.2 (b), §9.1."""
    return _integrate(response, temperature_k, spectral_photon_radiance)


def d_band_radiance_dT(response: SpectralResponse, temperature_k: FloatArray | float) -> FloatArray:
    """(∂L_B/∂T) = ∫R(λ) ∂B/∂T dλ in W m^-2 sr^-1 K^-1: what bolometer NETD divides by. §3.4."""
    return _integrate(response, temperature_k, d_spectral_radiance_dT)


def d_band_photon_radiance_dT(
    response: SpectralResponse, temperature_k: FloatArray | float
) -> FloatArray:
    """(∂L_{B,q}/∂T) in photons s^-1 m^-2 sr^-1 K^-1: what photon-detector NETD divides by. §9.4."""
    return _integrate(response, temperature_k, d_spectral_photon_radiance_dT)
