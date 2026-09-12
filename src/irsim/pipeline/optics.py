"""Stage 3 on the plane dict — the adaptor `irsim.optics.stage.apply_optics` is reached through.

Stages 1 and 2 are `(planes, config, state) -> planes` functions (`irsim.pipeline.radiance`,
`irsim.pipeline.atmosphere`); stage 3 was only ever called positionally from `run_frame`, which
left the Warp twin (M10.5) with no oracle of the same shape to be compared against. This module
supplies one. It computes nothing: the physics is `apply_optics` -- PSF at the k× pitch, box-mean
downsample to the detector grid, × π τ_opt/(4F²+1) · cos⁴θ · A_d, + Φ_self, in that fixed order
(ADR 0020) -- and the only thing added here is where the housing radiance comes from, which is
the same lookup `run_frame` does: the LUT at ``state.housing_temp_k``, in the pipeline's quantity.

``radiance`` arrives on the supersampled grid and ``flux`` leaves on the detector grid, so this is
the one stage that changes the shape of the planes.

docs/physics-model.md §8.1-§8.3, §13.4 stage 3
"""

from __future__ import annotations

import numpy as np

from irsim.optics.stage import apply_optics
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes

__all__ = ["housing_band_radiance", "optics_stage", "OpticsStage"]


def housing_band_radiance(config: PipelineConfig, state: PipelineState) -> float:
    """L_B(T_housing) now, in the pipeline's quantity — the level Φ_self is built on (§8.2)."""
    return float(config.lut.lookup(state.housing_temp_k, config.quantity)[()])


def optics_stage(planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
    """Stage-3 entry point: supersampled ``radiance`` → ``flux`` on the detector grid."""
    return {
        "flux": apply_optics(
            np.asarray(planes["radiance"]),
            config.sensor.sensor,
            housing_band_radiance(config, state),
            supersample=config.supersample,
            psf=config.psf,
        )
    }


class OpticsStage:
    name = "optics"

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return optics_stage(planes, config, state)
