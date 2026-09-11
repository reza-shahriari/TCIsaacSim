"""Materials: spectral model, Kirchhoff closure, Fresnel, per-band tables.

docs/physics-model.md §4, §12.3
"""

from irsim.materials.library import (
    CLOSURE_TOL,
    MATERIAL_DIR,
    BandProperties,
    Material,
    MaterialLibrary,
    load_material,
    nominal_response,
)
from irsim.materials.spectra import PropertySpectrum, load_property_spectrum
from irsim.materials.table import UNMAPPED_MATERIAL_ID, MaterialTable

__all__ = [
    "CLOSURE_TOL",
    "MATERIAL_DIR",
    "BandProperties",
    "Material",
    "MaterialLibrary",
    "load_material",
    "nominal_response",
    "PropertySpectrum",
    "load_property_spectrum",
    "UNMAPPED_MATERIAL_ID",
    "MaterialTable",
]
