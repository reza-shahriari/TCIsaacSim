"""One entry point for ε(θ), dispatching on the material's declared angular level (§4.2, §13.3).

Three levels, and the choice is the **material's**, not the caller's:

* **A — Fresnel** from a checked-in n/k table. Exact for a smooth interface, and the only level
  that can represent a metal, whose ε *rises* with angle.
* **B — empirical** ε₀[1 − a(1 − cos θ)^p], baked once per class against Level A (M7.6). Two
  instructions in a shader.
* **C — constant.** §4.2 permits it "only for rough, high-emissivity surfaces (ε > 0.93) within
  ±50° of normal", and lists vehicle bodies, glass and water as violating it.

**Level C is refused below ε_B = 0.93 rather than warned about**, because the error it hides is
large and one-sided: a glass panel at 60° has ε about 0.12 below its normal value, so a constant ε
renders the limb of every windscreen and every water surface too warm, everywhere, in a way that
looks like a plausible scene. §4.2 states the bound; this is it, enforced.

The G-buffer carries ``normal_dot_view`` as float32, and this returns float32 — not because the
precision is needed for ε itself, but because a silent widening to float64 in the middle of the
stage-1 chain is how a float32 discipline stops being one (non-negotiable #2).

docs/physics-model.md §4.2, §13.3, §13.5; ADR 0042
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.materials import ConstantAngular, EmpiricalAngular, FresnelAngular
from irsim.materials.angular import emissivity_empirical
from irsim.radiometry.band_average import T_REF_K, WeightingForm
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["LEVEL_C_MIN_EPSILON", "directional_emissivity", "angular_level"]

#: §4.2's own bound on Level C. Below this a constant ε is not an approximation, it is a
#: different material at every angle off normal.
LEVEL_C_MIN_EPSILON = 0.93


def angular_level(material: Any) -> str:
    """``"A"``, ``"B"`` or ``"C"`` for a material, from its declared ``angular_model``."""
    model = material.spec.optical.angular_model
    if isinstance(model, FresnelAngular):
        return "A"
    if isinstance(model, EmpiricalAngular):
        return "B"
    if isinstance(model, ConstantAngular):
        return "C"
    raise TypeError(f"unknown angular model {type(model).__name__}")


def directional_emissivity(
    material: Any,
    band: str,
    cos_theta: Any,
    response: SpectralResponse | None = None,
    t_ref_k: float = T_REF_K,
    form: WeightingForm = "energy",
    data_dir: Any = None,
) -> NDArray[np.float32]:
    """ε_B(θ) for one material in one band, at the level the material declares. float32 in, out.

    ``cos_theta`` is taken in absolute value at every level: two sides of a thin panel have the
    same emissivity, and which way its normal was authored is not physics. The result is finite
    at cos θ = 0 — Level A goes to ε → 0 there and Level B to ε₀(1 − a), both of which are
    numbers; a formulation that divided by cos θ would not be.
    """
    c = np.asarray(cos_theta)
    if c.dtype == np.float16:
        raise TypeError("cos_theta is float16 (non-negotiable #2)")
    c64 = np.abs(np.asarray(c, dtype=np.float64))
    if np.any(c64 > 1.0 + 1e-6):
        raise ValueError("cos_theta must lie in [-1, 1]")
    c64 = np.clip(c64, 0.0, 1.0)

    level = angular_level(material)
    epsilon_b = float(material.band_properties(band, response, form, t_ref_k).emissivity)

    if level == "C":
        if epsilon_b < LEVEL_C_MIN_EPSILON:
            raise ValueError(
                f"{material.name}: angular_model 'constant' needs ε_B >= {LEVEL_C_MIN_EPSILON} "
                f"(§4.2), and ε_B({band!r}) is {epsilon_b:.4f}. A constant ε here renders every "
                f"limb too warm, in a way that looks like a plausible scene -- author "
                f"'empirical' (M7.6 bakes the parameters) or 'fresnel'."
            )
        return np.full(c64.shape, epsilon_b, dtype=np.float32)

    if level == "B":
        model = material.spec.optical.angular_model
        return np.asarray(emissivity_empirical(epsilon_b, model.a, model.p, c64), dtype=np.float32)

    from irsim.materials.nk import band_directional_emissivity, load_nk_table

    if response is None:
        raise ValueError(
            f"{material.name}: Level A needs the camera's own R(λ) -- ε_B(θ) is a band average "
            "of a spectral Fresnel curve, so it cannot be formed without a band"
        )
    table = load_nk_table(material.spec.optical.angular_model.n_k_file, data_dir)
    return np.asarray(
        band_directional_emissivity(table, response, c64, t_ref_k, form), dtype=np.float32
    )
