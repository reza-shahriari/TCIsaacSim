"""Band radiance look-up table: the fast path, float32, with the kernel's interpolation contract.

A ``BandLUT`` tabulates four quantities on a uniform temperature grid (docs/physics-model.md
§3.2 b, §13.5): ``lb`` (band radiance, W m^-2 sr^-1), ``lb_q`` (band photon radiance,
photons s^-1 m^-2 sr^-1), ``dlb_dt`` and ``dlb_q_dt`` (their thermal derivatives, §3.4/§9.4).
Tables are built in float64 by the Simpson oracle (:mod:`irsim.radiometry.band_integration`)
and cast to float32 **once**; the grid is 200-1000 K at 0.05 K, 16001 entries (ADR 0011), the
constants shared with the temperature encoding in :mod:`irsim.radiometry.constants`.

:meth:`BandLUT.lookup` reproduces the CUDA kernel of §13.5 exactly, in float32::

    u  = (T - T0) / (T1 - T0) * (N - 1)
    u  = clamp(u, 0, N - 1.001)
    i0 = int(u);  fr = u - i0
    L  = lut[i0] * (1 - fr) + lut[i0 + 1] * fr

so that the CPU reference and the GPU path can be compared to float32 rounding. Temperatures
outside [T0, T1] are **clamped** (never extrapolated, never NaN); :meth:`out_of_range` reports
which inputs were, for the caller to flag (spec issue S14).

float16 is refused on input (CLAUDE.md non-negotiable #2); output is float32.

docs/physics-model.md §3.2 (b), §3.3, §13.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.band_integration import (
    band_photon_radiance,
    band_radiance,
    d_band_photon_radiance_dT,
    d_band_radiance_dT,
)
from irsim.radiometry.constants import LUT_N, LUT_T_MAX_K, LUT_T_MIN_K
from irsim.radiometry.spectral_response import SpectralResponse

__all__ = ["BandLUT", "Quantity", "QUANTITIES", "KERNEL_CLAMP_MARGIN"]

Quantity = Literal["lb", "lb_q", "dlb_dt", "dlb_q_dt"]
QUANTITIES: tuple[Quantity, ...] = ("lb", "lb_q", "dlb_dt", "dlb_q_dt")
KERNEL_CLAMP_MARGIN = 1.001  # §13.5: u <= N - 1.001 so that i0 + 1 stays in range
_BUILD_CHUNK = 1024  # temperatures per Simpson batch (bounds memory at ~16001 x grid floats)

Float32Array = NDArray[np.float32]


def _require_not_fp16(x: object, what: str) -> NDArray[np.floating]:
    arr = np.asarray(x)
    if arr.dtype == np.float16:
        raise TypeError(f"{what} is float16 (0.25 K spacing at 300 K); use float32 or better")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


@dataclass(frozen=True)
class BandLUT:
    """Four float32 tables on a uniform grid ``t0_k + i * dt_k``, i = 0..n-1."""

    t0_k: float
    t1_k: float
    n: int
    lb: Float32Array
    lb_q: Float32Array
    dlb_dt: Float32Array
    dlb_q_dt: Float32Array
    band_hash: str = ""

    def __post_init__(self) -> None:
        for name in QUANTITIES:
            table = getattr(self, name)
            if table.dtype != np.float32 or table.shape != (self.n,):
                raise TypeError(
                    f"{name} must be float32 of shape ({self.n},), got {table.dtype} {table.shape}"
                )
        if not (self.t1_k > self.t0_k and self.n >= 2):
            raise ValueError("LUT needs t1 > t0 and n >= 2")

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        response: SpectralResponse,
        t0_k: float = LUT_T_MIN_K,
        t1_k: float = LUT_T_MAX_K,
        n: int = LUT_N,
        band_hash: str = "",
    ) -> BandLUT:
        """Tabulate by Simpson quadrature in float64, cast to float32 once."""
        temps = np.linspace(t0_k, t1_k, n, dtype=np.float64)
        parts: dict[str, list[NDArray[np.float64]]] = {q: [] for q in QUANTITIES}
        fns = {
            "lb": band_radiance,
            "lb_q": band_photon_radiance,
            "dlb_dt": d_band_radiance_dT,
            "dlb_q_dt": d_band_photon_radiance_dT,
        }
        for start in range(0, n, _BUILD_CHUNK):
            chunk = temps[start : start + _BUILD_CHUNK]
            for q, fn in fns.items():
                parts[q].append(fn(response, chunk))
        tables = {q: np.concatenate(parts[q]).astype(np.float32) for q in QUANTITIES}
        return cls(t0_k=float(t0_k), t1_k=float(t1_k), n=int(n), band_hash=band_hash, **tables)

    # -- grid ------------------------------------------------------------------------------

    @property
    def dt_k(self) -> float:
        return (self.t1_k - self.t0_k) / (self.n - 1)

    @property
    def temperatures_k(self) -> NDArray[np.float64]:
        """Grid temperatures in float64 (for tests and sidecars; the kernel never needs them)."""
        return np.linspace(self.t0_k, self.t1_k, self.n, dtype=np.float64)

    def table(self, quantity: Quantity) -> Float32Array:
        if quantity not in QUANTITIES:
            raise ValueError(f"quantity must be one of {QUANTITIES}, got {quantity!r}")
        return np.asarray(getattr(self, quantity), dtype=np.float32)

    # -- forward lookup (the §13.5 contract) --------------------------------------------

    def out_of_range(self, temperature_k: object) -> NDArray[np.bool_]:
        """True where a temperature would be clamped by :meth:`lookup`."""
        t = _require_not_fp16(temperature_k, "temperature_k")
        return np.asarray((t < self.t0_k) | (t > self.t1_k))

    def lookup(self, temperature_k: object, quantity: Quantity = "lb") -> Float32Array:
        """Linear interpolation in float32, exactly as the kernel does it; clamps out of range."""
        table = self.table(quantity)
        t = _require_not_fp16(temperature_k, "temperature_k").astype(np.float32)
        span = np.float32(self.t1_k) - np.float32(self.t0_k)
        u = (t - np.float32(self.t0_k)) / span * np.float32(self.n - 1)
        u = np.minimum(
            np.maximum(u, np.float32(0.0)), np.float32(self.n) - np.float32(KERNEL_CLAMP_MARGIN)
        )
        i0 = u.astype(np.int64)
        fr = (u - i0.astype(np.float32)).astype(np.float32)
        out = table[i0] * (np.float32(1.0) - fr) + table[i0 + 1] * fr
        return np.asarray(out, dtype=np.float32)
