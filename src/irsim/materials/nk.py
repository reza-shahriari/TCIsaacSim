"""Complex refractive index tables and the band-effective directional emissivity built on them.

A ``data/nk/<name>.csv`` file carries n(λ) and k(λ) for one material. Feeding them through M7.4's
branch-safe Fresnel gives ε(λ, θ); Planck-weighting that over a camera's spectral response through
the one sanctioned route (:func:`irsim.radiometry.band_average.band_average`, ADR 0010) gives the
scalar ε_B(θ) the pipeline actually uses.

**Why the band average is not a formality here.** Water's optical constants swing hard across the
LWIR: over a Boson's 7.5-13.5 µm window, n runs 1.28 → 1.16 and k runs 0.033 → 0.35, an order of
magnitude. Evaluating Fresnel at a single "representative" 10 µm and calling it the band value is
the kind of shortcut that looks harmless and is not — the error grows with angle, which is exactly
where a sea surface lives.

**Interpolation.** n and k are interpolated **separately** and linearly. Interpolating a derived
quantity instead (ε, or R) would be interpolating a nonlinear function of the data, and the result
would depend on the table's sampling rather than on the material. Extrapolation raises: a table
that stops at 15.6 µm has nothing to say about 20 µm, and quietly holding the last value there
would put a fabricated number into a radiometric result. The one place end-holding is allowed is
inside a band average whose response support has already been checked to lie within the table
(:meth:`NKTable.emissivity_spectrum`), where it only ever affects the padding sample of a
quadrature grid that rounds one step past the response's end, where R is zero.

docs/physics-model.md §4.2, §12.3; roadmap M7.5, MM.1; ADR 0041 (data provenance), ADR 0079
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.loader import resolve_data_dir
from irsim.materials.fresnel import fresnel_reflectance
from irsim.radiometry.band_average import T_REF_K, Spectrum, WeightingForm, band_average
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "NK_SUBDIR",
    "NKTable",
    "load_nk_table",
    "fresnel_from_table",
    "band_directional_emissivity",
]

#: Where n/k tables live under the data root.
NK_SUBDIR = "nk"


@dataclass(frozen=True)
class NKTable:
    """n(λ) and k(λ) for one material, with the provenance the file declared."""

    name: str
    wavelength_um: NDArray[np.float64]
    n: NDArray[np.float64]
    k: NDArray[np.float64]
    source: str
    path: pathlib.Path

    @property
    def support_um(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])

    def at(self, wavelength_um: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(n, k) at one or more wavelengths. Raises outside the table -- never extrapolates."""
        lam = np.asarray(wavelength_um, dtype=np.float64)
        lo, hi = self.support_um
        if np.any(lam < lo) or np.any(lam > hi):
            bad = lam[(lam < lo) | (lam > hi)]
            raise ValueError(
                f"{self.name}: wavelength {bad.ravel()[0]:.4g} µm is outside the table's "
                f"[{lo:.4g}, {hi:.4g}] µm; n/k are not extrapolated"
            )
        return (
            np.asarray(np.interp(lam, self.wavelength_um, self.n), dtype=np.float64),
            np.asarray(np.interp(lam, self.wavelength_um, self.k), dtype=np.float64),
        )

    def emissivity_spectrum(self, cos_theta: float) -> Spectrum:
        """λ → ε(λ, θ) as a callable for :func:`band_average`, end-held (module docstring)."""

        def spectrum(grid_um: NDArray[np.float64]) -> NDArray[np.float64]:
            lam = np.clip(np.asarray(grid_um, dtype=np.float64), *self.support_um)
            n = np.interp(lam, self.wavelength_um, self.n)
            k = np.interp(lam, self.wavelength_um, self.k)
            return np.asarray(1.0 - fresnel_reflectance(n, k, cos_theta)[0], dtype=np.float64)

        return spectrum


def load_nk_table(name: str, data_dir: str | os.PathLike[str] | None = None) -> NKTable:
    """Read ``<data root>/nk/<name>.csv``, or a path if one is given.

    The file must declare provenance in a ``# source:`` comment. That is a hard requirement, not a
    lint: an optical-constants table with no source is a set of numbers nobody can check, and every
    radiometric result computed from it inherits that.
    """
    path = pathlib.Path(name)
    if path.suffix != ".csv":
        path = resolve_data_dir(data_dir) / NK_SUBDIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"n/k table {path} does not exist")

    source_lines: list[str] = []
    lam: list[float] = []
    n_vals: list[float] = []
    k_vals: list[float] = []
    in_source = False
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if text.startswith("#"):
            body = text.lstrip("#").strip()
            if body.lower().startswith("source:"):
                in_source = True
                source_lines.append(body[len("source:") :].strip())
            elif in_source and body:
                source_lines.append(body)
            elif not body:
                in_source = False
            continue
        if not text or text.lower().startswith("wavelength"):
            continue
        parts = [c.strip() for c in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{path} line {i}: expected 'lambda_um,n,k', got {text!r}")
        try:
            values = [float(c) for c in parts]
        except ValueError as exc:
            raise ValueError(f"{path} line {i}: not three numbers: {text!r}") from exc
        lam.append(values[0])
        n_vals.append(values[1])
        k_vals.append(values[2])

    if not source_lines:
        raise ValueError(
            f"{path} has no '# source:' header. An n/k table without provenance is unusable "
            "(ADR 0041): every radiometric result computed from it would be uncheckable."
        )
    if len(lam) < 2:
        raise ValueError(f"{path} needs at least two rows, got {len(lam)}")

    wavelength = np.asarray(lam, dtype=np.float64)
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError(f"{path}: wavelength must strictly increase")
    n_arr = np.asarray(n_vals, dtype=np.float64)
    k_arr = np.asarray(k_vals, dtype=np.float64)
    if np.any(n_arr <= 0.0) or np.any(k_arr < 0.0):
        raise ValueError(f"{path}: needs n > 0 and k >= 0 (a passive medium)")

    return NKTable(
        name=path.stem,
        wavelength_um=wavelength,
        n=n_arr,
        k=k_arr,
        source=" ".join(source_lines),
        path=path,
    )


def fresnel_from_table(
    table: NKTable, wavelength_um: Any, cos_theta: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """M7.4's Fresnel evaluated at tabulated (n, k): ``(R, R⊥, R∥)``."""
    n, k = table.at(wavelength_um)
    return fresnel_reflectance(n, k, cos_theta)


def band_directional_emissivity(
    table: NKTable,
    response: SpectralResponse,
    cos_theta: Any,
    t_ref_k: float = T_REF_K,
    form: WeightingForm = "energy",
) -> NDArray[np.float64]:
    """ε_B(θ): Planck-weighted band average of the Fresnel ε(λ, θ) over ``response``.

    The response's support must lie inside the table's, and that is checked rather than assumed --
    a response reaching past the data would otherwise be silently averaging held end values.
    """
    lo, hi = table.support_um
    r_lo, r_hi = response.support_um
    if r_lo < lo or r_hi > hi:
        raise ValueError(
            f"spectral response spans [{r_lo:.4g}, {r_hi:.4g}] µm but the {table.name} n/k table "
            f"only covers [{lo:.4g}, {hi:.4g}] µm; extend the table rather than extrapolating it"
        )
    shape = np.shape(cos_theta)
    mu = np.atleast_1d(np.asarray(cos_theta, dtype=np.float64)).ravel()
    out = np.array(
        [band_average(response, table.emissivity_spectrum(float(c)), t_ref_k, form) for c in mu],
        dtype=np.float64,
    )
    return np.asarray(out.reshape(shape) if shape else out[0], dtype=np.float64)
