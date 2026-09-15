"""Material library: load ``configs/materials/*.yaml``, derive the third optical quantity (§4.1).

For a material and a band the library returns the band-effective triple (ε_B, ρ_B, τ_B) with
ε_B + ρ_B + τ_B = 1 by construction (ADR 0040, ADR 0046): the authored quantity comes from the
scalar-per-band table or from the Planck-weighted band average of the spectral file (ADR 0010,
the only sanctioned reduction); τ_B is the authored per-band transmittance (0 when absent); the
remaining quantity is derived. An authored value that leaves no room (ε_B + τ_B > 1) is refused
here, after the band average, so a spectral file cannot smuggle a closure violation through.

Band averages need a spectral response: pass the sensor's (``SpectralResponse``) or let the
library use the nominal top-hat of a registry band (``NOMINAL_RANGES_UM``). A band the registry
does not know (``lwir_wide``) is fine for scalar tables and needs an explicit response for
spectral ones.

docs/physics-model.md §4.1, §4.4, §12.3, CLAUDE.md #4
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import yaml

from irsim.config.bands import NOMINAL_RANGES_UM
from irsim.config.loader import file_sha256, resolve_data_dir
from irsim.config.materials import FresnelAngular, MaterialConfig, MaterialSpec
from irsim.materials.spectra import PropertySpectrum, load_property_spectrum
from irsim.radiometry.band_average import WeightingForm
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = [
    "MATERIAL_DIR",
    "CLOSURE_TOL",
    "BandProperties",
    "Material",
    "MaterialLibrary",
    "nominal_response",
]

MATERIAL_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "materials"
CLOSURE_TOL = 1e-6


@dataclass(frozen=True)
class BandProperties:
    """ε + ρ + τ = 1 on band-effective values; ``authored`` says which one came from the file."""

    emissivity: float
    reflectance: float
    transmittance: float
    authored: Literal["emissivity", "reflectance"]

    def __post_init__(self) -> None:
        for name in ("emissivity", "reflectance", "transmittance"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} = {v} outside [0, 1]")
        closure = self.emissivity + self.reflectance + self.transmittance
        if abs(closure - 1.0) > CLOSURE_TOL:
            raise ValueError(f"Kirchhoff closure ε + ρ + τ = {closure} != 1 (CLAUDE.md #4)")


def nominal_response(band: str) -> SpectralResponse:
    """A unit top-hat over the registry's nominal range for a standard band id."""
    if band not in NOMINAL_RANGES_UM:
        raise KeyError(
            f"band {band!r} has no nominal range; pass the sensor's SpectralResponse explicitly"
        )
    lo, hi = NOMINAL_RANGES_UM[band]
    return SpectralResponse(
        wavelength_um=np.array([lo, hi]),
        response=np.array([1.0, 1.0]),
        source_path=f"<nominal top-hat {band}>",
        sha256="",
    )


@dataclass(frozen=True)
class Material:
    spec: MaterialSpec
    path: pathlib.Path
    spectrum: PropertySpectrum | None  # the authored spectral property, if any
    n_k_path: pathlib.Path | None

    @property
    def name(self) -> str:
        return self.spec.name

    def band_properties(
        self,
        band: str,
        response: SpectralResponse | None = None,
        form: WeightingForm = "energy",
        t_ref_k: float = 300.0,
    ) -> BandProperties:
        """(ε_B, ρ_B, τ_B), closed to 1e-6; raises when the authored values leave no room."""
        optical = self.spec.optical
        tau = optical.transmittance(band)
        if self.spectrum is not None:
            resp = response if response is not None else nominal_response(band)
            # threshold 0: the whole response file must be covered, not just where R is
            # appreciable. A library material is reused across cameras, so the stricter rule is
            # the right one here -- and it is the *same* rule, in one place (M7.3).
            if not self.spectrum.covers(resp, threshold=0.0):
                s_lo, s_hi = self.spectrum.support_um
                lo, hi = resp.support_um
                raise ValueError(
                    f"{self.name}: spectral file covers {s_lo}-{s_hi} um, band {band!r} needs "
                    f"{lo}-{hi} um (extend the table rather than extrapolate)"
                )
            authored_value = float(self.spectrum.band_effective(resp, t_ref_k, form, threshold=0.0))
        else:
            table = optical.emissivity_per_band or optical.reflectance_per_band or {}
            if band not in table:
                raise KeyError(f"{self.name}: no {optical.authored} authored for band {band!r}")
            authored_value = float(table[band])
        if authored_value + tau > 1.0 + CLOSURE_TOL:
            raise ValueError(
                f"{self.name} band {band!r}: {optical.authored} {authored_value:.4f} + "
                f"transmittance {tau:.4f} > 1 violates Kirchhoff closure (CLAUDE.md #4)"
            )
        derived = max(0.0, 1.0 - authored_value - tau)
        if optical.authored == "emissivity":
            return BandProperties(authored_value, derived, tau, "emissivity")
        return BandProperties(derived, authored_value, tau, "reflectance")

    @property
    def declared_bands(self) -> frozenset[str]:
        return self.spec.optical.declared_bands

    def content_hash(self) -> str:
        """SHA-256 of the validated spec with data file paths replaced by their content hashes."""
        dump: dict[str, Any] = self.spec.model_dump(mode="json")
        opt = dump["optical"]
        for key in ("spectral_emissivity", "spectral_reflectance"):
            if opt.get(key) is not None:
                assert self.spectrum is not None
                opt[key] = file_sha256(self.spectrum.path)
        if self.n_k_path is not None:
            opt["angular_model"]["n_k_file"] = file_sha256(self.n_k_path)
        canonical = json.dumps(dump, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve(raw: str, data_dir: pathlib.Path, what: str) -> pathlib.Path:
    p = pathlib.Path(raw).expanduser()
    if not p.is_absolute():
        p = data_dir / p
    p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{what} = {raw!r} resolves to {p}, which does not exist")
    return p


def load_material(
    path: str | os.PathLike[str], data_dir: str | os.PathLike[str] | None = None
) -> Material:
    p = pathlib.Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "materials" in raw:
        raise ValueError(
            f"{p}: the 'materials:' wrapper is not the file layout -- one YAML per material with a "
            "top-level 'material:' block (ADR 0040)"
        )
    spec = MaterialConfig.model_validate(raw).material
    if p.parent.resolve() == MATERIAL_DIR.resolve() and spec.name != p.stem:
        raise ValueError(f"material name {spec.name!r} does not match file name {p.stem!r}")
    root = resolve_data_dir(data_dir)
    spectrum = None
    if spec.optical.spectral_file is not None:
        spectrum = load_property_spectrum(
            _resolve(spec.optical.spectral_file, root, f"{spec.name}.optical.spectral_*")
        )
    n_k = None
    if isinstance(spec.optical.angular_model, FresnelAngular):
        n_k = _resolve(
            spec.optical.angular_model.n_k_file, root, f"{spec.name}.angular_model.n_k_file"
        )
    return Material(spec=spec, path=p.resolve(), spectrum=spectrum, n_k_path=n_k)


class MaterialLibrary(Mapping[str, Material]):
    """All materials of a directory, by name."""

    def __init__(self, materials: Mapping[str, Material]) -> None:
        self._materials = dict(materials)

    @classmethod
    def load(
        cls,
        material_dir: str | os.PathLike[str] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
    ) -> MaterialLibrary:
        root = pathlib.Path(material_dir) if material_dir is not None else MATERIAL_DIR
        materials = {}
        for path in sorted(root.glob("*.yaml")):
            if path.stem == "mapping":
                continue
            m = load_material(path, data_dir)
            if m.name in materials:
                raise ValueError(f"duplicate material name {m.name!r}")
            materials[m.name] = m
        if not materials:
            raise FileNotFoundError(f"no material YAML files under {root}")
        return cls(materials)

    def __getitem__(self, name: str) -> Material:
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(f"unknown material {name!r}; have {sorted(self._materials)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._materials)

    def __len__(self) -> int:
        return len(self._materials)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._materials)

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self._materials):
            h.update(name.encode())
            h.update(self._materials[name].content_hash().encode())
        return h.hexdigest()
