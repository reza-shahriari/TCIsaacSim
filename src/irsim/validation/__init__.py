"""Validation: Tier 2 benches (SITF, NETD, 3-D noise, MTF) and the NumPy-only metric functions.

docs/physics-model.md §15
"""

from irsim.validation.bench import SitfResult, sitf

__all__ = ["SitfResult", "sitf"]
