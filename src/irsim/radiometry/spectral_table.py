"""A two-column spectral table and the one band integral every source shares.

``SolarSpectrum`` (§5.4) and ``AirglowSpectrum`` (§5.5) differ in what their numbers mean -- one
is an absolute irradiance, the other a relative shape whose level is configuration -- but they
are integrated against a sensor's R(λ) in exactly the same way, and getting that integral
slightly different in two places is how two sources come to disagree about what a band is.

The convention is the band LUT's: **unnormalised**, ∫ R(λ) X(λ) dλ, so that a band irradiance and
a band radiance live in the same measure and a Lambertian surface's reflected radiance is E_B/π
with nothing left over. The photon form divides by hc/λ **inside** the integral -- a band average,
never hc/λ̄ outside it, which is a 19 % error for the SWIR band (M11.2).

docs/physics-model.md §5.4, §5.5, §3.2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import C_LIGHT, H_PLANCK
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["SpectralTable"]


@dataclass(frozen=True)
class SpectralTable:
    """X(λ) on a strictly increasing micrometre grid, with its provenance."""

    wavelength_um: NDArray[np.float64]
    values: NDArray[np.float64]
    source_path: str
    sha256: str

    @property
    def support_um(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])

    def integral(self) -> float:
        """∫X dλ over the whole file."""
        return float(np.trapezoid(self.values, self.wavelength_um))

    def _check_overlap(self, response: SpectralResponse) -> None:
        """Hook for a subclass that trusts only part of its own range."""

    def integration_grid(self, response: SpectralResponse) -> NDArray[np.float64]:
        """Both grids merged over the overlap, so neither file's structure is stepped over.

        A source file is dense and uniform; a response file has kinks (a band edge, a filter
        shoulder). Integrating on either alone loses the other's features -- and the response's
        cut-off is exactly where the integrand is largest in a reflective band.
        """
        lo_r, hi_r = response.support_um
        lo_s, hi_s = self.support_um
        lo, hi = max(lo_r, lo_s), min(hi_r, hi_s)
        if not hi > lo:
            raise ValueError(
                f"spectral response {lo_r}-{hi_r} um does not overlap {self.source_path} "
                f"({lo_s}-{hi_s} um)"
            )
        self._check_overlap(response)
        own = self.wavelength_um[(self.wavelength_um >= lo) & (self.wavelength_um <= hi)]
        theirs = response.wavelength_um[
            (response.wavelength_um >= lo) & (response.wavelength_um <= hi)
        ]
        return np.asarray(np.union1d(own, theirs), dtype=np.float64)

    def band_integral(self, response: SpectralResponse, photon: bool = False) -> float:
        """∫ R(λ) X(λ) dλ, optionally divided by the photon energy inside the integral."""
        grid = self.integration_grid(response)
        x = np.interp(grid, self.wavelength_um, self.values)
        weight = response.resampled(grid)
        if photon:
            weight = weight * (grid * 1e-6) / (H_PLANCK * C_LIGHT)
        return float(np.trapezoid(weight * x, grid))
