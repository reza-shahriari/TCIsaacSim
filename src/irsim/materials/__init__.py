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
from irsim.materials.mapping import (
    AuditReport,
    MappingRules,
    MaterialResolver,
    PrimRecord,
    Resolution,
    audit,
    load_mapping_rules,
)
from irsim.materials.spectra import PropertySpectrum, load_property_spectrum
from irsim.materials.table import (
    UNMAPPED_MATERIAL_ID,
    UNMAPPED_NAME,
    MaterialTable,
    StaleMaterialTableError,
)

__all__ = [
    "CLOSURE_TOL",
    "MATERIAL_DIR",
    "BandProperties",
    "Material",
    "MaterialLibrary",
    "load_material",
    "nominal_response",
    "AuditReport",
    "MappingRules",
    "MaterialResolver",
    "PrimRecord",
    "Resolution",
    "audit",
    "load_mapping_rules",
    "PropertySpectrum",
    "load_property_spectrum",
    "UNMAPPED_MATERIAL_ID",
    "UNMAPPED_NAME",
    "MaterialTable",
    "StaleMaterialTableError",
]
