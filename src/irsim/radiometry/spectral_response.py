"""Spectral response file contract and loader.

R(λ) is the *shape* of a camera's relative spectral response -- optics transmission, filter and
detector response folded together -- normalised to **peak 1.0**. It is dimensionless and carries no
quantum efficiency: QE is ``fpa.quantum_efficiency`` in the sensor config and is applied exactly
once, in the detector model (ADR 0009). A file whose peak is not 1 is refused rather than
renormalised, because an area-normalised or per-nanometre file silently rescales every radiance
downstream and nothing else would catch it.

File format (``data/spectra/responses/<name>.csv``)::

    # provenance lines: source, date, ESTIMATED or MEASURED, notes
    wavelength_um,response        <- optional header line (any non-numeric line)
    7.00,0.000
    7.01,0.001
    ...

Two numeric columns; wavelength strictly increasing in **micrometres** (0.1-100); response in
[0, 1] with max == 1 ± 1e-6; no NaN. Outside the file's support the response is zero.

docs/physics-model.md §2 (R(λ)), §3.2 (b), §12.2
"""

from __future__ import annotations

import hashlib
import os
import pathlib
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SpectralResponse",
    "load_spectral_response",
    "PEAK_TOLERANCE",
    "RESAMPLE_DL_UM",
    "RESPONSE_LAMBDA_MIN_UM",
    "RESPONSE_LAMBDA_MAX_UM",
]

PEAK_TOLERANCE = 1e-6
RESAMPLE_DL_UM = 0.01  # §3.2 (b): "Simpson on a 0.01 µm grid is ample"
RESPONSE_LAMBDA_MIN_UM = 0.1
RESPONSE_LAMBDA_MAX_UM = 100.0

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpectralResponse:
    """A validated R(λ): peak-1 shape on a strictly increasing micrometre grid."""

    wavelength_um: FloatArray
    response: FloatArray
    source_path: str
    sha256: str
    provenance: tuple[str, ...] = field(default_factory=tuple)

    @property
    def support_um(self) -> tuple[float, float]:
        """First and last wavelength of the file. R is zero outside."""
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])

    def resampled(self, grid_um: FloatArray) -> FloatArray:
        """R on an arbitrary grid by linear interpolation; zero outside the file's support."""
        grid = np.asarray(grid_um, dtype=np.float64)
        return np.asarray(
            np.interp(grid, self.wavelength_um, self.response, left=0.0, right=0.0),
            dtype=np.float64,
        )

    def integral_um(self) -> float:
        """∫R dλ on the native grid (µm). For a top-hat this is its width."""
        return float(np.trapezoid(self.response, self.wavelength_um))

    def half_power_points_um(self) -> tuple[float, float]:
        """Outermost 50 % crossings (linear interpolation): the conventional band edges."""
        return self._crossings(0.5)

    def support(self, threshold: float = 1e-3) -> tuple[float, float]:
        """Wavelength span where R exceeds ``threshold``."""
        return self._crossings(threshold)

    def _crossings(self, level: float) -> tuple[float, float]:
        above = self.response >= level
        idx = np.flatnonzero(above)
        if idx.size == 0:
            raise ValueError(f"response never reaches {level}")
        lam, r = self.wavelength_um, self.response
        i0, i1 = int(idx[0]), int(idx[-1])
        lo = (
            lam[i0]
            if i0 == 0
            else float(np.interp(level, [r[i0 - 1], r[i0]], [lam[i0 - 1], lam[i0]]))
        )
        hi = (
            lam[i1]
            if i1 == lam.size - 1
            else float(np.interp(level, [r[i1 + 1], r[i1]], [lam[i1 + 1], lam[i1]]))
        )
        return float(lo), float(hi)


def _parse(text: str, name: str) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    provenance: list[str] = []
    rows: list[tuple[float, float]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            provenance.append(line.lstrip("#").strip())
            continue
        parts = [p.strip() for p in line.replace(";", ",").replace("\t", ",").split(",")]
        parts = [p for p in parts if p]
        try:
            values = [float(p) for p in parts]
        except ValueError:
            if rows:
                raise ValueError(f"{name}:{lineno}: non-numeric line after data began") from None
            continue  # column header
        if len(values) != 2:
            raise ValueError(f"{name}:{lineno}: expected two columns (wavelength_um, response)")
        rows.append((values[0], values[1]))
    if len(rows) < 2:
        raise ValueError(f"{name}: a spectral response needs at least two samples")
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], tuple(provenance)


def _validate(lam: FloatArray, r: FloatArray, name: str) -> None:
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(r))):
        raise ValueError(f"{name}: NaN or inf in the response table")
    if np.any(np.diff(lam) <= 0):
        raise ValueError(f"{name}: wavelength must be strictly increasing")
    if lam[0] < RESPONSE_LAMBDA_MIN_UM or lam[-1] > RESPONSE_LAMBDA_MAX_UM:
        raise ValueError(
            f"{name}: wavelengths span [{lam[0]}, {lam[-1]}], outside "
            f"[{RESPONSE_LAMBDA_MIN_UM}, {RESPONSE_LAMBDA_MAX_UM}] um -- the file must be in "
            "MICROMETRES (7500-13500 looks like nanometres)"
        )
    if np.any(r < 0.0) or np.any(r > 1.0 + PEAK_TOLERANCE):
        raise ValueError(f"{name}: response must lie in [0, 1], got [{r.min()}, {r.max()}]")
    peak = float(r.max())
    if abs(peak - 1.0) > PEAK_TOLERANCE:
        raise ValueError(
            f"{name}: peak response is {peak:.6g}, not 1.0 -- R(lambda) is a peak-normalised "
            "shape (an area-normalised or QE-scaled file is refused, never rescaled; QE lives in "
            "fpa.quantum_efficiency, ADR 0009)"
        )


def load_spectral_response(path: str | os.PathLike[str]) -> SpectralResponse:
    """Read and validate a response CSV. See the module docstring for the contract."""
    p = pathlib.Path(path)
    raw = p.read_bytes()
    lam, r, provenance = _parse(raw.decode("utf-8"), p.name)
    _validate(lam, r, p.name)
    r = np.minimum(r, 1.0)  # a peak of 1 + 1e-7 from rounding is clamped, not propagated
    return SpectralResponse(
        wavelength_um=lam,
        response=r,
        source_path=str(p.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        provenance=provenance,
    )
