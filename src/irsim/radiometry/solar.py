"""Solar spectral irradiance on disk, and its band integrals (§5.4).

Two files live under ``data/spectra/solar/`` and both are **modelled, not measured** (ADR 0064):
an extraterrestrial spectrum (a 5778 K Planck normalised to the solar constant) and a clear-sky
direct-normal one at AM1.5. They are named after their models rather than after ASTM E-490 and
G-173, which they are not. Swapping in the real tables changes nothing here: both are just
(wavelength µm, irradiance W m⁻² µm⁻¹) tables and this module only integrates them.

Band integrals follow the same convention the band LUT uses (``irsim.radiometry.band_integration``)
-- **unnormalised**, E_B = ∫ R(λ) E(λ) dλ -- so that E_B and L_B live in the same measure and a
Lambertian surface's reflected radiance is E_B/π with no leftover normalisation. The photon form
divides by hc/λ **inside** the integral -- a band average, never hc/λ̄ outside it, which for the
SWIR band is a 19 % error (M11.2).

The trusted range is a hard edge, not a note. The effective-temperature model is good to a few
percent above ~0.7 µm and badly wrong in the ultraviolet, so a spectral response that reaches
below :data:`TRUSTED_MIN_UM` is refused rather than quietly integrated over the region where the
model fails.

docs/physics-model.md §5.2, §5.4; ADR 0064
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.spectral_table import SpectralTable

__all__ = [
    "TRUSTED_MIN_UM",
    "SolarSpectrum",
    "load_solar_spectrum",
    "solar_data_dir",
    "TOA_FILE",
    "AM15_DIRECT_FILE",
]

#: Below this wavelength the 5778 K effective-temperature model departs from the real solar
#: spectrum by tens of percent (it puts ~12 % of the constant below 0.4 µm against a real ~8 %).
#: Every band in ``irsim.config.bands`` starts above it; a visible band would need E-490.
TRUSTED_MIN_UM = 0.70

TOA_FILE = "spectra/solar/toa_planck_5778k.csv"
AM15_DIRECT_FILE = "spectra/solar/direct_normal_am1p5.csv"


def solar_data_dir(data_dir: str | os.PathLike[str] | None = None) -> pathlib.Path:
    from irsim.config.loader import resolve_data_dir

    return resolve_data_dir(data_dir) / "spectra" / "solar"


@dataclass(frozen=True)
class SolarSpectrum(SpectralTable):
    """E(λ) in W m⁻² µm⁻¹ on a strictly increasing micrometre grid."""

    def _check_overlap(self, response: Any) -> None:
        lo_r, _ = response.support_um
        if lo_r < TRUSTED_MIN_UM:
            raise ValueError(
                f"spectral response reaches {lo_r} um, below the {TRUSTED_MIN_UM} um this solar "
                f"model is trusted to (ADR 0064): the effective-temperature shape is tens of "
                f"percent wrong in the ultraviolet. Supply a measured ASTM E-490 table instead."
            )

    @property
    def irradiance_w_m2_um(self) -> NDArray[np.float64]:
        """The table under its physical name."""
        return self.values

    def total_w_m2(self) -> float:
        """∫E dλ over the whole file: the solar constant for a TOA spectrum."""
        return self.integral()

    def band_irradiance(self, response: Any) -> float:
        """E_B = ∫ R(λ) E(λ) dλ, W m⁻². Same unnormalised convention as the band LUT."""
        return self.band_integral(response, photon=False)

    def band_photon_irradiance(self, response: Any) -> float:
        """E_B,q = ∫ R(λ) E(λ) λ/(hc) dλ, photons s⁻¹ m⁻². The band average, not hc/λ̄."""
        return self.band_integral(response, photon=True)

    def band(self, response: Any, quantity: str) -> float:
        """Dispatch on the LUT's quantity tag so a caller never picks the unit by hand."""
        if quantity in ("lb", "dlb_dt"):
            return self.band_irradiance(response)
        if quantity in ("lb_q", "dlb_q_dt"):
            return self.band_photon_irradiance(response)
        raise ValueError(f"unknown quantity {quantity!r}")


def load_solar_spectrum(path: str | os.PathLike[str]) -> SolarSpectrum:
    """Parse ``lambda_um,irradiance`` rows; ``#`` comments and one header line are skipped."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"solar spectrum {p} does not exist")
    text = p.read_text(encoding="utf-8")
    lam: list[float] = []
    val: list[float] = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [c.strip() for c in stripped.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"{p} line {i}: expected 'lambda_um,irradiance', got {stripped!r}")
        try:
            lam.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError:
            if not lam:  # the column header
                continue
            raise ValueError(f"{p} line {i}: non-numeric entry {stripped!r}") from None
    if len(lam) < 2:
        raise ValueError(f"{p}: at least two rows are needed")
    wl = np.asarray(lam, dtype=np.float64)
    e = np.asarray(val, dtype=np.float64)
    if np.any(np.diff(wl) <= 0.0):
        raise ValueError(f"{p}: wavelengths must be strictly increasing")
    if np.any(e < 0.0):
        raise ValueError(f"{p}: spectral irradiance must be non-negative")
    return SolarSpectrum(
        wavelength_um=wl,
        values=e,
        source_path=str(p),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
