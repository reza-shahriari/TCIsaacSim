"""Level A angular reflectance: branch-safe complex Fresnel from n + ik (§4.2, roadmap M7.4).

Emissivity falls toward grazing incidence for dielectrics. That is why a smooth curved object at
one uniform temperature shows a cooler rim in LWIR, and -- the reason this module exists now --
why a flat sea does not read as one temperature: at 1 degree depression water is reflecting the
cold sky far more than it is emitting, and at 60 degrees it is almost a blackbody. A constant
emissivity cannot produce either effect, and §4.2 Level C explicitly refuses water.

The A/B reparameterisation is not cosmetic. With Z = n~^2 - sin^2(theta), a naive complex square
root picks either branch; passivity requires the transmitted field to decay with depth, and only
one branch does that. The wrong one produces reflectance discontinuities across wavelength,
instability near grazing incidence, and reflectances above 1. Writing sqrt(Z) = A + jB with

    A = sqrt((|Z| + Re Z) / 2),    B = sgn(Im Z) sqrt((|Z| - Re Z) / 2)

selects the admissible branch by construction [R1]. In the infrared this matters because k is
large for many materials -- it is not a corner case.

``sgn(0) := 0``, which is NumPy's ``np.sign`` and is also the right physics here. Im(Z) = 0 happens
for a real index (k = 0). Where Z > 0 the second radicand is 0 and the sign is irrelevant; where
Z < 0 (a rare-to-dense path past the critical angle) it forces A = B = 0, and both r_perp and
r_par then have unit modulus -- R = 1 exactly, which is total internal reflection. The
conventional branch B = sqrt(-Z) gives unit modulus too, so the choice is unobservable in R.

For an absorbing medium there is **no sharp TIR knee at all**: R stays below 1 at every incidence
angle, and the same validation work confirms it [R1]. ``tests/unit/test_fresnel.py`` pins that,
because a model that grows a knee has picked the wrong branch somewhere.

Back faces enter through |cos theta|: a facet whose normal points away from the camera is the same
surface seen from the other side, and the reflectance is even in cos theta.

docs/physics-model.md §4.2 Level A [R1]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["fresnel_reflectance", "directional_emissivity_fresnel"]


def fresnel_reflectance(
    n: Any, k: Any, cos_theta: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Unpolarised, perpendicular and parallel reflectance of n + ik at incidence ``cos_theta``.

    Returns ``(R, R_perp, R_par)`` with ``R = (R_perp + R_par) / 2``, each broadcast over the three
    inputs -- so a spectral (n, k) pair against a G-buffer of cosines costs one call.

    ``cos_theta`` is taken in absolute value (back faces, see the module docstring). ``n`` must be
    positive and ``k`` non-negative: a negative k is a gain medium, which this is not, and it would
    silently flip the branch B selects.
    """
    n_arr = np.asarray(n, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    mu = np.abs(np.asarray(cos_theta, dtype=np.float64))

    if np.any(n_arr <= 0.0):
        raise ValueError("refractive index n must be positive")
    if np.any(k_arr < 0.0):
        raise ValueError("extinction coefficient k must be non-negative (a passive medium)")
    if np.any(mu > 1.0 + 1e-12):
        raise ValueError("cos_theta must lie in [-1, 1]")
    mu = np.clip(mu, 0.0, 1.0)

    # n~^2 = (n^2 - k^2) + j(2nk), then Z = n~^2 - sin^2(theta). Kept as real/imaginary pairs
    # rather than a complex dtype so the branch below is visibly the §4.2 formula.
    n2_re = n_arr * n_arr - k_arr * k_arr
    n2_im = 2.0 * n_arr * k_arr
    z_re = n2_re - (1.0 - mu * mu)
    z_im = n2_im
    z_abs = np.hypot(z_re, z_im)

    # The branch. np.maximum guards the radicands against -1e-17 from cancellation, which would
    # otherwise make a perfectly ordinary real index return NaN.
    A = np.sqrt(np.maximum(0.5 * (z_abs + z_re), 0.0))
    B = np.sign(z_im) * np.sqrt(np.maximum(0.5 * (z_abs - z_re), 0.0))

    # r_perp = (mu - (A + jB)) / (mu + (A + jB)); the modulus squared is written out so the
    # numerator and denominator are each a real sum of squares.
    num_perp = (mu - A) ** 2 + B * B
    den_perp = (mu + A) ** 2 + B * B
    r_perp = num_perp / den_perp

    # r_par = ((A + jB) - n~^2 mu) / ((A + jB) + n~^2 mu)
    p_re = n2_re * mu
    p_im = n2_im * mu
    num_par = (A - p_re) ** 2 + (B - p_im) ** 2
    den_par = (A + p_re) ** 2 + (B + p_im) ** 2
    r_par = num_par / den_par

    r_unpol = 0.5 * (r_perp + r_par)
    return (
        np.asarray(r_unpol, dtype=np.float64),
        np.asarray(r_perp, dtype=np.float64),
        np.asarray(r_par, dtype=np.float64),
    )


def directional_emissivity_fresnel(n: Any, k: Any, cos_theta: Any) -> NDArray[np.float64]:
    """ε(θ) = 1 − R(θ) for an opaque medium: Kirchhoff plus τ = 0 (§4.1, §4.2).

    Opacity is the assumption that makes this one line. A semi-transparent slab needs ρ from the
    two interfaces and the internal path, not this function -- see :mod:`irsim.materials.surface`.
    """
    r_unpol, _, _ = fresnel_reflectance(n, k, cos_theta)
    return np.asarray(1.0 - r_unpol, dtype=np.float64)
