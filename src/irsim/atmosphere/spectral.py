"""Exact spectral band transmittance and the grey-band fit -- the evidence for ADR 0048.

The L2 atmosphere is a single grey γ_B per band (§7.2, Appendix A #2). The exact band
transmittance of a structured γ(λ) is

    τ_B(d) = ∫ R(λ) B(λ, T) e^{−γ(λ) d} dλ / ∫ R(λ) B(λ, T) dλ

which is *not* an exponential in d: strong lines saturate first, so the effective
γ_eff(d) = −ln τ_B(d)/d decreases with distance (the curve of growth). A grey γ_B fitted over
0–300 m therefore over-attenuates at longer ranges. These functions quantify that error for
synthetic γ(λ); the numbers are recorded in ADR 0048 and the README limitation line.

docs/physics-model.md §7.2, §7.4, Appendix A #2
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.band_integration import quadrature_grid, simpson
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["band_transmittance_spectral", "effective_gamma", "fit_grey_gamma", "grey_fit_error"]

GammaOfLambda = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def band_transmittance_spectral(
    response: SpectralResponse,
    gamma_of_lambda: GammaOfLambda,
    distance_m: NDArray[np.floating] | float,
    temperature_k: float = 300.0,
) -> NDArray[np.float64]:
    """Exact Planck-weighted band transmittance of a spectral γ(λ) at each distance."""
    grid = quadrature_grid(response)
    dl = float(grid[1] - grid[0])
    weight = response.resampled(grid) * spectral_radiance(
        grid, np.asarray(temperature_k, dtype=np.float64)
    )
    gamma = np.asarray(gamma_of_lambda(grid), dtype=np.float64)
    if gamma.shape != grid.shape or np.any(gamma < 0.0):
        raise ValueError("gamma(lambda) must be non-negative on the quadrature grid")
    d = np.atleast_1d(np.asarray(distance_m, dtype=np.float64))
    denominator = float(simpson(weight, dl))
    numer = simpson(weight[None, :] * np.exp(-gamma[None, :] * d[:, None]), dl)
    return np.asarray(numer / denominator, dtype=np.float64)


def effective_gamma(
    response: SpectralResponse,
    gamma_of_lambda: GammaOfLambda,
    distance_m: NDArray[np.floating],
    temperature_k: float = 300.0,
) -> NDArray[np.float64]:
    """γ_eff(d) = −ln τ_B(d) / d: constant for grey γ, decreasing for structured γ."""
    d = np.asarray(distance_m, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("distances must be positive")
    tau = band_transmittance_spectral(response, gamma_of_lambda, d, temperature_k)
    return np.asarray(-np.log(tau) / d, dtype=np.float64)


def fit_grey_gamma(
    response: SpectralResponse,
    gamma_of_lambda: GammaOfLambda,
    d_max_m: float = 300.0,
    n_points: int = 31,
    temperature_k: float = 300.0,
) -> float:
    """Least-squares grey γ_B (m⁻¹) over 0–d_max: minimise Σ (ln τ_B(d) + γ d)² on a grid."""
    d = np.linspace(0.0, d_max_m, n_points)[1:]
    tau = band_transmittance_spectral(response, gamma_of_lambda, d, temperature_k)
    return float(-np.sum(d * np.log(tau)) / np.sum(d * d))


def grey_fit_error(
    response: SpectralResponse,
    gamma_of_lambda: GammaOfLambda,
    distances_m: tuple[float, ...] = (500.0, 1000.0),
    d_fit_m: float = 300.0,
    temperature_k: float = 300.0,
) -> dict[float, float]:
    """Relative error exp(−γ_B d)/τ_B(d) − 1 of the 0–d_fit grey fit at the given distances."""
    gamma_b = fit_grey_gamma(response, gamma_of_lambda, d_fit_m, temperature_k=temperature_k)
    d = np.asarray(distances_m, dtype=np.float64)
    exact = band_transmittance_spectral(response, gamma_of_lambda, d, temperature_k)
    grey = np.exp(-gamma_b * d)
    return {float(dd): float(g / e - 1.0) for dd, g, e in zip(d, grey, exact, strict=True)}
