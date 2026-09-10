"""Optics: aperture factor, natural vignetting, self-emission, MTF cascade.

docs/physics-model.md §8
"""

from irsim.optics.aperture import aperture_factor, cone_half_angle, fpa_irradiance, pixel_power

__all__ = ["aperture_factor", "cone_half_angle", "fpa_irradiance", "pixel_power"]
