"""Noise: seeded streams, the NVESD 3-D synthesiser, FPN drift, bad pixels, the noise stage.

docs/physics-model.md §10
"""

from irsim.noise.defects import (
    CLUSTER_RADIUS_PX,
    BadPixelMap,
    DefectKind,
    DefectState,
    advance_state,
    apply_defects,
    generate_map,
)
from irsim.noise.drift import DRIFTING_COMPONENTS, FpnDrift, drift_rng, ou_step
from irsim.noise.seeding import (
    NoiseStream,
    field_normal,
    hash_normal,
    hash_u64,
    hash_uniform,
    noise_rng,
    sensor_rng,
    stream_key,
)
from irsim.noise.stage import NoiseStage, measure_from_uniform_scene
from irsim.noise.three_d import FixedPattern, Sigmas7, synthesize_frame

__all__ = [
    "BadPixelMap",
    "DefectKind",
    "DefectState",
    "CLUSTER_RADIUS_PX",
    "generate_map",
    "advance_state",
    "apply_defects",
    "DRIFTING_COMPONENTS",
    "FpnDrift",
    "drift_rng",
    "ou_step",
    "NoiseStream",
    "field_normal",
    "hash_normal",
    "hash_u64",
    "hash_uniform",
    "noise_rng",
    "sensor_rng",
    "stream_key",
    "FixedPattern",
    "Sigmas7",
    "synthesize_frame",
    "NoiseStage",
    "measure_from_uniform_scene",
]
