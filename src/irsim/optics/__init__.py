"""Optics: aperture factor, natural vignetting, self-emission, MTF cascade.

docs/physics-model.md §8
"""

from irsim.optics.aperture import aperture_factor, cone_half_angle, fpa_irradiance, pixel_power
from irsim.optics.vignetting import cos4_at_radius, cos4_field, field_angle_map

__all__ = [
    "aperture_factor",
    "cone_half_angle",
    "fpa_irradiance",
    "pixel_power",
    "cos4_at_radius",
    "cos4_field",
    "field_angle_map",
]
