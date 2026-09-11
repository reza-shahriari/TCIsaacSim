"""The uniform detector interface (ADR 0026): pre-quantisation signal in DN, DN, and σ.

The detector owns the **per-pixel temporal** noise -- Poisson shot (photon FPAs) and the
scene-independent Gaussian σ_TVH -- and adds it in its physical signal space before converting
to DN units. The **correlated** 3-D terms (V, H, VH, TV, TH, T) are the noise stage's (M4.9),
built from the same σ_TVH by the configured ratios. Quantisation is last (spec issue S10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["DetectorFrame", "Detector"]


@dataclass(frozen=True)
class DetectorFrame:
    """One detector output: un-quantised signal (DN units, float32, noise included), DN (uint16),
    and the per-pixel temporal σ in DN units (float32) the noise stage scales its ratios from."""

    signal_dn: NDArray[np.float32]
    dn: NDArray[np.uint16]
    sigma_dn: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.signal_dn.dtype != np.float32 or self.sigma_dn.dtype != np.float32:
            raise TypeError("signal_dn and sigma_dn must be float32")
        if self.dn.dtype != np.uint16:
            raise TypeError("dn must be uint16")
        if not (self.signal_dn.shape == self.dn.shape == self.sigma_dn.shape):
            raise ValueError("signal_dn, dn and sigma_dn must share a shape")


class Detector(Protocol):
    def response(
        self, flux: NDArray[np.floating], frame_index: int, sensor_seed: int
    ) -> DetectorFrame: ...

    def noiseless_signal_dn(self, flux: NDArray[np.floating]) -> NDArray[np.float32]: ...
