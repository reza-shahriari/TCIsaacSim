"""Validation: Tier 2 benches (SITF, NETD, 3-D noise, MTF) and the NumPy-only metric functions.

docs/physics-model.md §15
"""

from irsim.validation.aerial import (
    AerialTarget,
    apparent_target_radiance,
    background_radiance,
    target_contrast,
    target_leaving_radiance,
    zero_contrast_elevation,
)
from irsim.validation.bench import SitfResult, measured_netd_k, sitf
from irsim.validation.mtf import SlantEdgeResult, slant_edge_mtf
from irsim.validation.noise import (
    Decomposition3D,
    SpatialPSD,
    compare_psd,
    decompose_3d,
    estimate_floors,
    spatial_psd,
    temporal_psd,
)

__all__ = [
    "AerialTarget",
    "apparent_target_radiance",
    "background_radiance",
    "target_contrast",
    "target_leaving_radiance",
    "zero_contrast_elevation",
    "SitfResult",
    "SlantEdgeResult",
    "slant_edge_mtf",
    "measured_netd_k",
    "sitf",
    "Decomposition3D",
    "SpatialPSD",
    "compare_psd",
    "decompose_3d",
    "estimate_floors",
    "spatial_psd",
    "temporal_psd",
]
