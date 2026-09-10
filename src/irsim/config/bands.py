"""Band registry: canonical band ids, nominal ranges and per-band defaults.

"Bands are data, not code" (CLAUDE.md): a kernel must never switch on a band name. This module
is the one place that knows the four canonical bands of docs/physics-model.md §12.1 and it exposes
them as *data* -- a wavelength range, a default regime, a default detector model -- plus two
derivations:

* :func:`band_id_for` classifies a configured (λ_min, λ_max) by maximal fractional overlap with
  the nominal ranges, so a config may omit ``band.id`` or state it and have it checked.
* :func:`enabled_illumination_terms` encodes §5.2's "one shading core, two illumination paths"
  as a function of the *regime*, not the band: emissive → self-emission; reflective → solar and
  night sources; mixed → all three. §5.2 speaks of a compile-time flag; here it is a runtime
  switch on ``regime`` so adding a band touches YAML only (§12.2, spec issue S17).

Nominal ranges follow the §12.1 table. §5.2's table differs slightly (SWIR 1.0–1.7 "ext. 2.5",
LWIR 8–12 "(7.5–13.5)"); §12.1 is taken as canonical because it is the configuration section.

docs/physics-model.md §12.1, §12.2, §5.2
"""

from __future__ import annotations

from typing import Literal, get_args

__all__ = [
    "BandId",
    "BAND_IDS",
    "NOMINAL_RANGES_UM",
    "DEFAULT_REGIME",
    "DEFAULT_DETECTOR_MODEL",
    "MIN_OVERLAP_FRACTION",
    "band_id_for",
    "overlap_fraction",
    "enabled_illumination_terms",
]

BandId = Literal["nir", "swir", "mwir", "lwir"]
BAND_IDS: tuple[BandId, ...] = get_args(BandId)

# §12.1 "Range (µm)" row.
NOMINAL_RANGES_UM: dict[BandId, tuple[float, float]] = {
    "nir": (0.75, 1.0),
    "swir": (0.9, 1.7),
    "mwir": (3.0, 5.0),
    "lwir": (7.5, 13.5),
}
# §12.1 "Illumination needed" / "Self-emission" rows -> §12.2 regime.
DEFAULT_REGIME: dict[BandId, Literal["emissive", "reflective", "mixed"]] = {
    "nir": "reflective",
    "swir": "reflective",
    "mwir": "mixed",
    "lwir": "emissive",
}
# §12.1 "Detector model" row.
DEFAULT_DETECTOR_MODEL: dict[BandId, Literal["photon", "bolometer"]] = {
    "nir": "photon",
    "swir": "photon",
    "mwir": "photon",
    "lwir": "bolometer",
}

# A configured band must overlap some nominal band by at least this fraction of its own width.
MIN_OVERLAP_FRACTION = 0.5

IlluminationTerm = Literal["self_emission", "solar", "night"]


def overlap_fraction(lambda_min_um: float, lambda_max_um: float, band: BandId) -> float:
    """Fraction of the configured band's width that lies inside ``band``'s nominal range."""
    if lambda_max_um <= lambda_min_um:
        raise ValueError("lambda_max_um must exceed lambda_min_um")
    lo, hi = NOMINAL_RANGES_UM[band]
    overlap = max(0.0, min(hi, lambda_max_um) - max(lo, lambda_min_um))
    return overlap / (lambda_max_um - lambda_min_um)


def band_id_for(lambda_min_um: float, lambda_max_um: float) -> BandId:
    """Classify a configured band by maximal overlap with the §12.1 nominal ranges.

    Raises ``ValueError`` when no nominal band covers at least half of the configured width
    (e.g. 1.8–2.6 µm falls between SWIR and MWIR): such a band needs an explicit new entry here,
    which is the intended friction -- a fifth band is a registry change, not a kernel change.
    """
    best: BandId | None = None
    best_frac = 0.0
    for band in BAND_IDS:
        frac = overlap_fraction(lambda_min_um, lambda_max_um, band)
        if frac > best_frac:
            best, best_frac = band, frac
    if best is None or best_frac < MIN_OVERLAP_FRACTION:
        raise ValueError(
            f"band [{lambda_min_um}, {lambda_max_um}] um overlaps no canonical band by >= "
            f"{MIN_OVERLAP_FRACTION:.0%} (best {best_frac:.0%}); nominal ranges are "
            f"{NOMINAL_RANGES_UM}. Add the band to irsim.config.bands if it is real."
        )
    return best


def enabled_illumination_terms(
    regime: Literal["emissive", "reflective", "mixed"],
) -> frozenset[IlluminationTerm]:
    """§5.2: which illumination paths a kernel evaluates for a given regime.

    ``self_emission`` is ε·B(T); ``solar`` is reflected sun (and glint); ``night`` is the
    reflected night sources (moon, airglow, artificial). Kernels take this set; they never
    inspect the band id.
    """
    if regime == "emissive":
        return frozenset({"self_emission"})
    if regime == "reflective":
        return frozenset({"solar", "night"})
    if regime == "mixed":
        return frozenset({"self_emission", "solar", "night"})
    raise ValueError(f"unknown regime {regime!r}")
