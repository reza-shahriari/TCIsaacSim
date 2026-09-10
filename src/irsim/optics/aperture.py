"""The aperture factor π/(4F² + 1), FPA irradiance and pixel power -- defined here and nowhere else.

For an extended Lambertian source of in-band radiance L_B seen through a circular aperture of
working f-number F, the irradiance on the focal plane is

    E = π L_B τ_opt / (4F² + 1) · cos⁴θ            (docs/physics-model.md §8.1, §2)

The ``+1`` form is exact: E = π L sin²θ_max with tan θ_max = 1/(2F). The paraxial form 1/(4F²) is
20 % wrong at F/1.0, routine for uncooled LWIR, and is the most common radiometric bug in home-grown
IR simulators (§2 convention warning). CLAUDE.md non-negotiable #5: this factor is written exactly
once, in this file; ``tests/unit/test_aperture_guard.py`` walks the AST of every other module in
``src/irsim`` and ``src/irsim_isaac`` to make sure it stays that way.

All functions are quantity-agnostic: feed energy radiance (W m⁻² sr⁻¹) and get W; feed photon
radiance (photons s⁻¹ m⁻² sr⁻¹) and get photons s⁻¹. float16 is refused; float32 input gives
float32 output (non-negotiable #2).

docs/physics-model.md §2, §8.1
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["aperture_factor", "cone_half_angle", "fpa_irradiance", "pixel_power"]

FloatArray = NDArray[np.floating]


def _check_f_number(f_number: float) -> None:
    if not f_number > 0.0:
        raise ValueError(f"f_number must be positive, got {f_number}")


def _radiance(x: object) -> FloatArray:
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError("radiance is float16; use float32 or better (CLAUDE.md non-negotiable #2)")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def cone_half_angle(f_number: float) -> float:
    """Half-angle of the cone the pixel sees through the aperture: θ_max = atan(1 / (2F)), rad."""
    _check_f_number(f_number)
    return math.atan(1.0 / (2.0 * f_number))


def aperture_factor(f_number: float) -> float:
    """Ω_eff = π / (4F² + 1) = π sin²θ_max, the projected solid angle of the aperture (sr).

    §8.1, §2. The single definition in the codebase.
    """
    _check_f_number(f_number)
    return math.pi / (4.0 * f_number * f_number + 1.0)


def fpa_irradiance(
    radiance: object, f_number: float, tau_opt: float, cos4: object = 1.0
) -> FloatArray:
    """E_FPA = Ω_eff τ_opt cos⁴θ · L_B. Units follow the radiance (W m⁻² or photons s⁻¹ m⁻²)."""
    if not 0.0 < tau_opt <= 1.0:
        raise ValueError(f"tau_opt must be in (0, 1], got {tau_opt}")
    lb = _radiance(radiance)
    factor = np.asarray(aperture_factor(f_number) * tau_opt, dtype=lb.dtype)
    c4 = _radiance(cos4).astype(lb.dtype, copy=False)
    return np.asarray(lb * factor * c4, dtype=lb.dtype)


def pixel_power(
    radiance: object, f_number: float, tau_opt: float, active_area_m2: float, cos4: object = 1.0
) -> FloatArray:
    """Φ = E_FPA · A_d: in-band power (W) or photon rate (s⁻¹) collected by one pixel (§2)."""
    if not active_area_m2 > 0.0:
        raise ValueError(f"active_area_m2 must be positive, got {active_area_m2}")
    e = fpa_irradiance(radiance, f_number, tau_opt, cos4)
    return np.asarray(e * np.asarray(active_area_m2, dtype=e.dtype), dtype=e.dtype)
