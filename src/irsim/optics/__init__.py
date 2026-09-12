"""Optics: aperture factor, natural vignetting, self-emission, MTF cascade.

docs/physics-model.md §8
"""

from irsim.optics.aperture import aperture_factor, cone_half_angle, fpa_irradiance, pixel_power
from irsim.optics.housing import HousingTemperature, HousingTempMode
from irsim.optics.mtf import (
    cutoff_frequency_cyc_per_mm,
    mtf_detector,
    mtf_diffraction,
    mtf_gaussian,
    mtf_motion,
    mtf_system,
    nyquist_frequency_cyc_per_mm,
)
from irsim.optics.projection import (
    Intrinsics,
    distort_normalised,
    opencv_pinhole_coeffs,
    project,
    project_usd,
    undistort_normalised,
    usd_camera_to_opencv,
)
from irsim.optics.psf import apply_psf, optical_psf
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
    "HousingTemperature",
    "HousingTempMode",
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
    "cutoff_frequency_cyc_per_mm",
    "nyquist_frequency_cyc_per_mm",
    "mtf_diffraction",
    "mtf_detector",
    "mtf_motion",
    "mtf_gaussian",
    "mtf_system",
    "optical_psf",
    "apply_psf",
    "Intrinsics",
    "distort_normalised",
    "undistort_normalised",
    "opencv_pinhole_coeffs",
    "project",
    "project_usd",
    "usd_camera_to_opencv",
]
