"""Optics: aperture factor, natural vignetting, self-emission, MTF cascade.

docs/physics-model.md §8
"""

from irsim.optics.aperture import aperture_factor, cone_half_angle, fpa_irradiance, pixel_power
from irsim.optics.sampling import box_downsample, box_transfer, required_render_size
from irsim.optics.self_emission import (
    OpticalElement,
    self_emission_power,
    stack_self_radiance,
    stack_transmittance,
)
from irsim.optics.stage import apply_optics, invert_optics, optics_field
from irsim.optics.vignetting import cos4_at_radius, cos4_field, field_angle_map

__all__ = [
    "aperture_factor",
    "cone_half_angle",
    "fpa_irradiance",
    "pixel_power",
    "cos4_at_radius",
    "cos4_field",
    "field_angle_map",
    "OpticalElement",
    "self_emission_power",
    "stack_self_radiance",
    "stack_transmittance",
    "box_downsample",
    "box_transfer",
    "required_render_size",
    "apply_optics",
    "invert_optics",
    "optics_field",
]
