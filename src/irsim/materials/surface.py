"""The surface radiance leaving one pixel: emission, reflection and transmission (§4.4, §4.1).

    L = ε L_B(T_s) + ρ L_env + τ L_behind,        ε + ρ + τ = 1

This is the three-term form of §4.4 and the scalar reference implementation the pipeline stage
is tested against. The point of carrying τ explicitly is the band transition: a windshield is
opaque in LWIR (τ ≈ 0) and shows its own temperature, and transparent in SWIR (τ ≈ 0.7) where it
shows whatever is behind it -- the driver. A model that folded τ into (1 − ε) would render both
bands identically and look perfectly plausible doing it.

``L_behind`` is what a second ray through the surface returns. Until the renderer supplies a
second-hit AOV it defaults to ``L_env`` (ADR 0046), which makes this form reduce **exactly** to
the two-term ε L_B + (1 − ε) L_env of M7.13.

docs/physics-model.md §4.1, §4.4, §12.3; ADR 0046
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["CLOSURE_TOL", "surface_radiance"]

CLOSURE_TOL = 1e-6  # CLAUDE.md #4: eps + rho + tau = 1 to 1e-6


def surface_radiance(
    emissivity: Any,
    reflectance: Any,
    transmittance: Any,
    l_body: Any,
    l_env: Any,
    l_behind: Any | None = None,
    closure_tol: float = CLOSURE_TOL,
) -> NDArray[np.float64]:
    """ε L_B(T_s) + ρ L_env + τ L_behind, with per-pixel closure enforced (§4.4).

    Raises when ε + ρ + τ departs from 1 by more than ``closure_tol`` anywhere -- the guard that
    makes a hand-authored material triple fail loudly instead of quietly not conserving energy.
    """
    eps = np.asarray(emissivity, dtype=np.float64)
    rho = np.asarray(reflectance, dtype=np.float64)
    tau = np.asarray(transmittance, dtype=np.float64)
    closure = eps + rho + tau
    if np.any(np.abs(closure - 1.0) > closure_tol):
        worst = float(closure.flat[int(np.argmax(np.abs(closure - 1.0)))])
        raise ValueError(
            f"Kirchhoff closure ε + ρ + τ = {worst:.6f} != 1 (tolerance {closure_tol:g}); "
            "author one property and derive the rest (CLAUDE.md #4)"
        )
    for name, value in (("emissivity", eps), ("reflectance", rho), ("transmittance", tau)):
        if np.any(value < -closure_tol) or np.any(value > 1.0 + closure_tol):
            raise ValueError(f"{name} outside [0, 1]")
    body = np.asarray(l_body, dtype=np.float64)
    env = np.asarray(l_env, dtype=np.float64)
    behind = env if l_behind is None else np.asarray(l_behind, dtype=np.float64)
    return np.asarray(eps * body + rho * env + tau * behind, dtype=np.float64)
