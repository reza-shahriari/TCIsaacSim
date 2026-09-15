"""Level B: the empirical angular emissivity falloff, and the fit that produces it (§4.2).

    ε(θ) = ε₀ [1 − a (1 − cos θ)^p],     a ≈ 0.15–0.35, p ≈ 4–6

Two instructions in a shader, and close enough to Fresnel over the angles a camera actually sees.
§4.2's instruction is "fit (a, p) once per material class against Level A and bake it"; this
module is that fit.

**The fit refuses rather than returns, and it refuses on *shape* before it refuses on error.**
Fresnel emissivity falls with angle for a dielectric and **rises** for a metal — bare aluminium's
ε goes from 0.019 at normal to 0.030 at 70°. Level B cannot represent a rise at any (a, p) with
a ≥ 0, so fitting aluminium returns a = 0 and a flat curve with the wrong sign of slope.

A residual test alone does **not** catch that, and this is the trap the step is about: aluminium's
ε is so small that a completely wrong shape has an RMS residual of **0.0034** — comfortably inside
the 0.02 the roadmap asks for. An absolute error bar is meaningless on a quantity that never
exceeds 0.03. So the monotonic direction is checked first, on a relative scale, and a rising curve
raises whatever its residual is; the caller is expected to author ``angular_model: fresnel``.

The fitting range stops at 70° on purpose (§4.2's own bound). Beyond it Fresnel turns over towards
ε → 0 at grazing, the two forms diverge fast, and including that region would drag the fit away
from the angles that matter to buy accuracy where a pixel is a sliver.

docs/physics-model.md §4.2 Level B, §13.5; ADR 0042
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FIT_MAX_ANGLE_DEG",
    "MAX_RMS_RESIDUAL",
    "RISE_FRACTION",
    "CANDIDATE_P",
    "AngularFit",
    "emissivity_empirical",
    "fit_empirical",
    "fit_from_nk",
    "fit_band_level_b",
    "bake_angular_models",
    "FIT_ANGLES_DEG",
]

#: §4.2's own bound on Level B. Beyond it Fresnel turns over towards zero at grazing and the two
#: forms diverge; fitting there buys accuracy where a pixel is a sliver, at the cost of the angles
#: a camera spends its time looking at.
FIT_MAX_ANGLE_DEG = 70.0

#: The fit is a *class* model, not a curve tracer. Above this the Level B form is not describing
#: the material and returning its best effort would be worse than refusing.
MAX_RMS_RESIDUAL = 0.02

#: A rise of more than this fraction of ε₀ across the fitted range means the material is a metal
#: and Level B is the wrong model. Relative, because the absolute size of the rise is tiny for
#: exactly the materials that do it -- which is why a residual test alone lets aluminium through.
RISE_FRACTION = 0.02
#: A floor under the relative test, so a material with ε₀ near zero cannot trip it on noise.
RISE_FLOOR = 1e-4

#: §4.2 gives p ≈ 4–6. Searched exhaustively rather than optimised: three integers, and `a` is
#: linear given p, so each candidate is one least-squares solve and the global optimum over the
#: set is exact. A gradient search on two parameters would be slower and could find a local one.
CANDIDATE_P: tuple[float, ...] = (4.0, 5.0, 6.0)


@dataclass(frozen=True)
class AngularFit:
    """The baked Level B parameters for one material in one band, with the fit's own error."""

    epsilon_0: float
    a: float
    p: float
    rms_residual: float
    max_residual: float
    n_samples: int

    @property
    def is_flat(self) -> bool:
        """``a == 0``: a rough dielectric the model says has no angular falloff at all."""
        return self.a == 0.0


def emissivity_empirical(epsilon_0: Any, a: Any, p: Any, cos_theta: Any) -> NDArray[np.float64]:
    """ε(θ) = ε₀ [1 − a (1 − cos θ)^p], clipped into [0, 1].

    ``cos_theta`` may be negative — a back-facing normal — and is taken in absolute value, the
    same convention :func:`irsim.materials.fresnel.fresnel_reflectance` uses. Two sides of a thin
    panel have the same emissivity; which way its normal was authored is not physics.
    """
    e0 = np.asarray(epsilon_0, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    if np.any(a_arr < 0.0):
        raise ValueError("a must be non-negative: Level B cannot represent a rising ε(θ)")
    if np.any(p_arr <= 0.0):
        raise ValueError("p must be positive")
    c = np.abs(np.asarray(cos_theta, dtype=np.float64))
    if np.any(c > 1.0 + 1e-9):
        raise ValueError("cos_theta must lie in [-1, 1]")
    c = np.clip(c, 0.0, 1.0)
    return np.asarray(np.clip(e0 * (1.0 - a_arr * (1.0 - c) ** p_arr), 0.0, 1.0))


def fit_empirical(
    cos_theta: Any,
    epsilon: Any,
    name: str = "<material>",
    max_rms: float = MAX_RMS_RESIDUAL,
    candidate_p: tuple[float, ...] = CANDIDATE_P,
) -> AngularFit:
    """Fit (a, p) to a Level A curve over 0–70°, with ε₀ taken at normal incidence.

    ε₀ is **not** fitted. It is ε(0), which Level A gives exactly and which every consumer of the
    material already uses as its band value; letting the fit move it would make the normal-incidence
    emissivity depend on how well the angular model happened to fit, which is backwards.

    Given p, the model is linear in ``a`` — ε = ε₀ − ε₀ a (1−cos θ)^p — so each candidate p is one
    least-squares solve and the best over the (small, integer) candidate set is the global optimum.
    ``a`` is clamped at zero: a negative ``a`` is a rising ε(θ), which is a metal, which Level B
    must refuse rather than fit.
    """
    c = np.asarray(cos_theta, dtype=np.float64)
    e = np.asarray(epsilon, dtype=np.float64)
    if c.shape != e.shape or c.ndim != 1 or c.size < 3:
        raise ValueError("cos_theta and epsilon must be matching 1-D arrays of at least 3 samples")
    cutoff = math.cos(math.radians(FIT_MAX_ANGLE_DEG))
    inside = c >= cutoff - 1e-12
    if int(inside.sum()) < 3:
        raise ValueError(
            f"{name}: fewer than three samples inside 0-{FIT_MAX_ANGLE_DEG:g} deg; Level B is "
            "fitted over the angles a camera actually sees, not over the whole hemisphere"
        )
    c, e = c[inside], e[inside]
    order = np.argsort(-c)
    c, e = c[order], e[order]
    epsilon_0 = (
        float(e[0]) if float(c[0]) >= 1.0 - 1e-9 else float(np.interp(1.0, c[::-1], e[::-1]))
    )
    if not epsilon_0 > 0.0:
        raise ValueError(f"{name}: normal-incidence emissivity is not positive")

    third = max(1, c.size // 3)
    head, tail = float(e[:third].mean()), float(e[-third:].mean())
    if tail > head + max(RISE_FLOOR, RISE_FRACTION * epsilon_0):
        raise ValueError(
            f"{name}: ε rises with angle ({head:.4f} near normal to {tail:.4f} near "
            f"{FIT_MAX_ANGLE_DEG:g} deg), which is a metal. Level B has no (a, p) that can do "
            f"that -- a best-effort fit would make the limb darker when it should be brighter. "
            f"Author `angular_model: fresnel` instead. Note that the RMS residual would NOT have "
            f"caught this: for a material whose ε never exceeds a few percent, a completely wrong "
            f"shape still fits to well inside {max_rms:g}."
        )

    best: AngularFit | None = None
    for p in candidate_p:
        basis = epsilon_0 * (1.0 - c) ** p
        denom = float(np.dot(basis, basis))
        a = 0.0 if denom <= 0.0 else max(0.0, float(np.dot(basis, epsilon_0 - e) / denom))
        residual = e - emissivity_empirical(epsilon_0, a, p, c)
        rms = float(np.sqrt(np.mean(residual**2)))
        candidate = AngularFit(
            epsilon_0=epsilon_0,
            a=a,
            p=float(p),
            rms_residual=rms,
            max_residual=float(np.max(np.abs(residual))),
            n_samples=int(c.size),
        )
        if best is None or candidate.rms_residual < best.rms_residual:
            best = candidate
    assert best is not None

    if best.rms_residual > max_rms:
        raise ValueError(
            f"{name}: Level B fit residual {best.rms_residual:.4f} exceeds {max_rms:g} "
            f"(best p = {best.p:g}, a = {best.a:.4f})"
        )
    return best


#: The angles the fit samples, 0-70° at 2°. Uniform in *angle* rather than in cos θ: the Level B
#: form is written in cos θ, so a cos-uniform grid would pile samples where the curve is flat and
#: starve the shoulder, which is the part the fit is actually trying to place.
FIT_ANGLES_DEG = np.arange(0.0, FIT_MAX_ANGLE_DEG + 1e-9, 2.0)


def fit_from_nk(
    n: Any, k: Any, name: str = "<material>", angles_deg: Any = None, **kwargs: Any
) -> AngularFit:
    """Fit Level B against Level A: sample ``fresnel_reflectance`` and regress (a, p) on it.

    This is §4.2's instruction taken literally -- "fit (a, p) once per material class against
    Level A and bake it" -- so the Level A model is the oracle and Level B is a compression of it,
    not an independent description.
    """
    from irsim.materials.fresnel import directional_emissivity_fresnel

    angles = FIT_ANGLES_DEG if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
    cos_theta = np.cos(np.deg2rad(angles))
    return fit_empirical(cos_theta, directional_emissivity_fresnel(n, k, cos_theta), name, **kwargs)


def fit_band_level_b(
    table: Any,
    response: Any,
    name: str = "<material>",
    angles_deg: Any = None,
    **kwargs: Any,
) -> AngularFit:
    """Fit Level B against the **band-averaged** Level A curve, which is what §4.2 asks for.

    Fitting at a single representative wavelength and then comparing against a band average is a
    real error, not a rounding one: water over the Boson band is 0.0038 below its 10 µm Fresnel
    value at normal, and the gap grows with angle -- 0.0275 at 70°, which is past the 0.02 the
    acceptance criterion allows. The band is part of the material's angular behaviour, so the fit
    has to see it.
    """
    from irsim.materials.nk import band_directional_emissivity

    angles = FIT_ANGLES_DEG if angles_deg is None else np.asarray(angles_deg, dtype=np.float64)
    cos_theta = np.cos(np.deg2rad(angles))
    return fit_empirical(
        cos_theta, band_directional_emissivity(table, response, cos_theta), name, **kwargs
    )


def bake_angular_models(
    curves: dict[str, tuple[Any, Any]], strict: bool = True
) -> tuple[dict[str, AngularFit], dict[str, str]]:
    """Fit every named (cos θ, ε) curve; return the fits and, separately, the refusals.

    Refusals are **returned, not swallowed**. A baking pass over a library will legitimately hit
    metals, and the caller needs to know which materials came back without a Level B model so it
    can author Level A for them -- a silently missing entry becomes a silently constant ε, which
    is the failure Level B exists to fix. ``strict=False`` is for that survey; ``strict=True``
    raises on the first refusal, which is what a build step wants.
    """
    fits: dict[str, AngularFit] = {}
    refused: dict[str, str] = {}
    for material, (cos_theta, epsilon) in curves.items():
        try:
            fits[material] = fit_empirical(cos_theta, epsilon, material)
        except ValueError as exc:
            if strict:
                raise
            refused[material] = str(exc)
    return fits, refused
