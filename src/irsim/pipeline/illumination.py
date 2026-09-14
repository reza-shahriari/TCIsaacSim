"""The illumination bundle stage 1 reflects: one units tag, three incident terms, one gate.

§5.2 describes "one shading core, two illumination paths" and puts the choice behind a
*compile-time* flag. This module makes it a runtime function of ``band.regime`` alone (ADR 0063),
so that adding a band stays a YAML change -- the whole point of §12.2 and of M11.1's guard.

**Everything in the bundle is an incident band radiance**, in the band and the quantity the LUT
is tabulated in. That is what lets one multiplication by ρ serve all three terms and keeps the
kernel from knowing which is which:

* ``l_env`` -- the thermal environment the surface sees, V_s L_sky,eff + (1 − V_s) L_ground
  (§5.3 a). Already a radiance.
* ``l_sun`` -- the **equivalent isotropic incident radiance** of the direct beam,
  E_B τ_sun cos θ_s S / π (§5.4). §5.4 writes the reflected result as (ρ_B/π) E_B τ_sun cos θ_s S;
  dividing the irradiance by π here and letting the kernel apply ρ once gives exactly that, and
  makes the Lambertian identity ρ = 1 → E_B/π an identity of the code rather than of the algebra.
* ``l_night`` -- the reflected night sources (airglow, moon), isotropic, so already a radiance
  (§5.5).

Two rules, both of which are easy to get wrong in the direction that still produces a picture:

**The units tag is checked, not assumed.** ``lb`` and ``lb_q`` differ by ~1e19 at these
wavelengths. A solar spectrum integrated in W m⁻² handed to a photon-unit kernel does not raise,
does not produce NaN and does not look obviously wrong after AGC -- it produces a scene that is
uniformly, invisibly mis-scaled. So the bundle carries its quantity and the kernel refuses a
mismatch.

**Self-emission is never culled** (ADR 0063). ``enabled_illumination_terms`` omits
``self_emission`` for a reflective band, and that is a statement about a 300 K scene, not about
the band: SWIR self-emission rises by more than 10⁶ between 300 K and 900 K (M11.1), so a jet
exhaust, a flare or a brake disc glows in SWIR at night with no sun and no airglow at all. The
regime gates ``solar`` and ``night``; ε L_B(T) is always evaluated.

docs/physics-model.md §5.2, §5.3, §5.4, §5.5, §12.1; ADR 0063
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.bands import IlluminationTerm, enabled_illumination_terms
from irsim.config.sensor import Regime
from irsim.radiometry.lut import QUANTITIES, Quantity

__all__ = ["SOURCE_TERMS", "Illumination", "illumination_from_planes"]

#: The bundle's two *gated* terms, in the order they are summed. ``l_env`` is not among them:
#: the thermal environment is not an illumination source that a band may switch off, it is what
#: every surface in every band sees, and in SWIR it is simply small.
SOURCE_TERMS: tuple[tuple[str, IlluminationTerm], ...] = (
    ("l_sun", "solar"),
    ("l_night", "night"),
)


def _as_incident(value: Any, what: str) -> NDArray[np.float64] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == np.float16:
        raise TypeError(f"{what} is float16 (non-negotiable #2)")
    out = np.asarray(arr, dtype=np.float64)
    if np.any(out < 0.0):
        raise ValueError(f"{what} must be a non-negative incident radiance")
    return out


@dataclass(frozen=True)
class Illumination:
    """Incident band radiances to be reflected, already gated by the band's regime.

    Build with :meth:`for_regime` -- that is the single place the regime is consulted, so no
    kernel downstream has to, and a bundle in hand is a bundle that has already been gated.
    """

    quantity: Quantity
    l_env: NDArray[np.float64] | None = None
    l_sun: NDArray[np.float64] | None = None
    l_night: NDArray[np.float64] | None = None
    #: What ``enabled_illumination_terms`` returned for this band's regime, kept for provenance
    #: and for the tests that assert the gate did something. ``self_emission`` may appear here
    #: and means nothing to the kernel: emission is unconditional.
    terms: frozenset[IlluminationTerm] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}, got {self.quantity!r}")
        shapes = {np.shape(v) for v in self._live() if np.ndim(v) > 0}
        if len(shapes) > 1:
            raise ValueError(f"illumination terms disagree on shape: {sorted(shapes)}")

    def _live(self) -> list[NDArray[np.float64]]:
        return [v for v in (self.l_env, self.l_sun, self.l_night) if v is not None]

    @classmethod
    def for_regime(
        cls,
        regime: Regime,
        quantity: Quantity,
        *,
        l_env: Any = None,
        l_sun: Any = None,
        l_night: Any = None,
    ) -> Illumination:
        """Apply §5.2's regime gate once, at construction.

        A term the regime does not enable is dropped to ``None`` -- not zeroed -- so that an
        emissive band with a solar term supplied takes the *same* code path as one with none and
        the two frames come out bit-identical rather than merely close.
        """
        terms = enabled_illumination_terms(regime)
        kept: dict[str, NDArray[np.float64] | None] = {
            "l_env": _as_incident(l_env, "l_env"),
            "l_sun": None,
            "l_night": None,
        }
        for name, term in SOURCE_TERMS:
            value = {"l_sun": l_sun, "l_night": l_night}[name]
            if value is not None and term in terms:
                kept[name] = _as_incident(value, name)
        return cls(quantity=quantity, terms=terms, **kept)

    @property
    def dropped_terms(self) -> frozenset[IlluminationTerm]:
        """Source terms this band's regime switches off. Empty for ``mixed``."""
        return frozenset(term for _, term in SOURCE_TERMS if term not in self.terms)

    def total_incident(self) -> NDArray[np.float64] | None:
        """Σ of the live terms, or ``None`` when nothing illuminates the surface.

        ``None`` rather than a zero array on purpose: it is what lets stage 1 fall back to the
        emission-only branch exactly, instead of adding ρ·0 and rounding differently.
        """
        live = self._live()
        if not live:
            return None
        total = live[0]
        for term in live[1:]:
            total = total + term
        return np.asarray(total, dtype=np.float64)

    def require_quantity(self, quantity: Quantity, where: str) -> None:
        """Refuse an energy-unit bundle in a photon-unit kernel (and the reverse)."""
        if self.quantity != quantity:
            raise ValueError(
                f"{where}: illumination is in {self.quantity!r} but the kernel runs on "
                f"{quantity!r}. These differ by ~1e19 at infrared wavelengths and the mistake "
                f"is invisible after AGC -- convert the source, do not relabel it."
            )


def illumination_from_planes(
    regime: Regime,
    quantity: Quantity,
    planes: dict[str, NDArray[Any]],
    l_env: Any = None,
) -> Illumination:
    """Bundle the ``l_sun`` / ``l_night`` planes a scene supplies with the computed ``l_env``.

    The two source planes are optional, exactly as ``motion_px`` and ``radiance_behind`` are: a
    scene that has not asked for illumination costs nothing and renders identically. M11.3 and
    M11.4 write them.
    """
    return Illumination.for_regime(
        regime,
        quantity,
        l_env=l_env,
        l_sun=planes.get("l_sun"),
        l_night=planes.get("l_night"),
    )
