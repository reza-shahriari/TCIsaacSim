"""Property spectra on disk: λ (µm), value in [0, 1] -- ε(λ), ρ(λ) or τ(λ) tables (§12.3).

Same text format as the spectral-response files (``#`` comments, ``lambda_um,value`` rows,
increasing wavelength) but a *property* is a fraction, not a normalised response: it is never
rescaled to a peak of one. Files live under ``data/spectra/materials/`` (ADR 0040).
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.band_average import Spectrum, tabulated

__all__ = ["PropertySpectrum", "load_property_spectrum"]


@dataclass(frozen=True)
class PropertySpectrum:
    wavelength_um: NDArray[np.float64]
    values: NDArray[np.float64]
    path: pathlib.Path

    @property
    def as_callable(self) -> Spectrum:
        return tabulated(self.wavelength_um, self.values)

    @property
    def support_um(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])


def load_property_spectrum(path: str | os.PathLike[str]) -> PropertySpectrum:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"property spectrum {p} does not exist")
    lam: list[float] = []
    val: list[float] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = [c.strip() for c in text.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"{p} line {i}: expected 'lambda_um,value', got {text!r}")
        try:
            lam.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"{p} line {i}: non-numeric entry {text!r}") from exc
    if len(lam) < 2:
        raise ValueError(f"{p}: at least two rows are needed")
    wl = np.asarray(lam, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)
    if np.any(np.diff(wl) <= 0.0):
        raise ValueError(f"{p}: wavelengths must be strictly increasing")
    if np.any(v < 0.0) or np.any(v > 1.0):
        raise ValueError(f"{p}: a property spectrum lies in [0, 1] (got {v.min()}..{v.max()})")
    if wl[0] <= 0.0:
        raise ValueError(f"{p}: wavelengths must be positive micrometres")
    return PropertySpectrum(wavelength_um=wl, values=v, path=p.resolve())
