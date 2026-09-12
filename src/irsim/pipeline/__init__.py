"""The engine-free CPU reference pipeline: G-buffer → radiance → focal plane → DN → outputs.

This NumPy implementation is the **oracle** (ADR 0018). The Warp and SPG kernels are fast paths
and are tested against it; nothing in here may depend on an engine. Stages follow
docs/physics-model.md §13.4: 1 band radiance, 2 atmosphere, 3 optics, 4 detector, 5 noise, 6 ISP.

docs/physics-model.md §13.4, §13.6, §16.4
"""

from irsim.pipeline.core import PipelineConfig, PipelineState, Stage
from irsim.pipeline.frame import Outputs, run_frame
from irsim.pipeline.point_target import PointTarget, inject_point_targets
from irsim.pipeline.radiance import band_radiance_stage

__all__ = [
    "PointTarget",
    "inject_point_targets",
    "PipelineConfig",
    "PipelineState",
    "Stage",
    "band_radiance_stage",
    "Outputs",
    "run_frame",
]
