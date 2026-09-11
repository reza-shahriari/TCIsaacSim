"""Validation: Tier 2 benches (SITF, NETD, 3-D noise, MTF) and the NumPy-only metric functions.

docs/physics-model.md §15
"""

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
