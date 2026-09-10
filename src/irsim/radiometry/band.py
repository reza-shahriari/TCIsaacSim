"""A band: the nominal edges and regime from the sensor config plus its measured R(λ).

This is the object band integration (M1.4) and the LUT builder (M1.6) take. Bands are data:
constructing one touches YAML and a CSV, never a kernel (docs/physics-model.md §12).

Precedence between config edges and the CSV (ADR 0009): the **CSV is the response**; the config
edges are the nominal band used for classification (``band.id``), for top-hat cross-checks and
for reporting. They must agree: the response's half-power points may differ from the config
edges by at most ``EDGE_TOLERANCE_UM``, else the wrong file is attached and loading fails.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from irsim.config.bands import BandId
from irsim.config.sensor import BandSpec
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

__all__ = ["Band", "EDGE_TOLERANCE_UM"]

EDGE_TOLERANCE_UM = 0.5


@dataclass(frozen=True)
class Band:
    id: BandId
    lambda_min_um: float
    lambda_max_um: float
    regime: Literal["emissive", "reflective", "mixed"]
    response: SpectralResponse

    @classmethod
    def from_spec(cls, spec: BandSpec, response_path: str | os.PathLike[str] | None = None) -> Band:
        """Build from a validated ``BandSpec``; the response path defaults to the spec's
        (already resolved by the config loader)."""
        response = load_spectral_response(response_path or spec.spectral_response)
        lo, hi = response.half_power_points_um()
        if (
            abs(lo - spec.lambda_min_um) > EDGE_TOLERANCE_UM
            or abs(hi - spec.lambda_max_um) > EDGE_TOLERANCE_UM
        ):
            raise ValueError(
                f"response half-power points [{lo:.3f}, {hi:.3f}] um disagree with the configured "
                f"band [{spec.lambda_min_um}, {spec.lambda_max_um}] um by more than "
                f"{EDGE_TOLERANCE_UM} um: wrong file for this band? ({response.source_path})"
            )
        return cls(
            id=spec.band_id,
            lambda_min_um=spec.lambda_min_um,
            lambda_max_um=spec.lambda_max_um,
            regime=spec.regime,
            response=response,
        )
