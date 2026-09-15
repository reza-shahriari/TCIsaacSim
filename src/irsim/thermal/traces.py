"""Heat traces and thermal shadows: the ghosts a scene leaves behind (§6.6 last row).

§6.6 singles this out: *"That last row is worth implementing. Thermal shadows and residual heat
traces are a signature phenomenon of the band and they routinely confuse detectors trained only on
synthetic data that lacks them."* A car pulls away and leaves a cold rectangle where it stood and
two warm stripes where its tyres were. Both decay over minutes. Neither is predicted by any surface
energy balance a scene can afford to run.

**Traces are overlays, not conserved quantities** (ADR 0039). A trace is a ΔT added to whatever the
thermal field says, with its own amplitude and its own decay constant, and it does **not** take
that heat from anywhere. Making them conserved would require the ground solver to know that a car
was parked there — which is exactly the state a scripted overlay exists to avoid carrying. The cost
is stated rather than hidden: a scene with many overlapping traces is adding energy that no balance
accounts for, so amplitudes are per-trace and bounded by §6.6's own ranges.

Superposition is **linear and exact**: traces add, and a cell no footprint touches is bit-identical
to the field beneath it. That matters more than it sounds — a layer that blended or clamped would
make an untouched road pixel depend on whether a car drove past somewhere else in the frame.

docs/physics-model.md §6.6; ADR 0039
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TYRE_TRACE_K",
    "BODY_SHADOW_K",
    "TYRE_TAU_S",
    "SHADOW_TAU_S",
    "GroundGrid",
    "Footprint",
    "HeatTraceLayer",
    "tyre_trace_from_trajectory",
    "body_shadow_from_pose",
]

#: §6.6's own ranges, as midpoints. ESTIMATED.
TYRE_TRACE_K = 6.0  # "+3 … +10 K"
BODY_SHADOW_K = -5.5  # "−3 … −8 K"
TYRE_TAU_S = 420.0  # "5-20 min", toward the short end: a contact patch is small
SHADOW_TAU_S = 720.0  # "5-20 min"


@dataclass(frozen=True)
class GroundGrid:
    """A regular 2-D patch of ground in metres, with its origin at cell (0, 0)'s corner."""

    width: int
    height: int
    cell_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        if self.cell_m <= 0.0:
            raise ValueError("cell_m must be positive")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def cell_of(self, x_m: Any, y_m: Any) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """(row, column) of the cells containing these world points, unclipped."""
        col = np.floor((np.asarray(x_m, dtype=np.float64) - self.origin_x_m) / self.cell_m)
        row = np.floor((np.asarray(y_m, dtype=np.float64) - self.origin_y_m) / self.cell_m)
        return np.asarray(row, dtype=np.intp), np.asarray(col, dtype=np.intp)


@dataclass(frozen=True)
class Footprint:
    """One deposited trace: where, how strong, when, and how fast it forgets."""

    mask: NDArray[np.bool_]
    amplitude_k: float
    deposited_t_s: float
    tau_s: float

    def __post_init__(self) -> None:
        if self.tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        if self.mask.dtype != np.bool_:
            raise TypeError("a footprint mask must be boolean")

    def delta_t_at(self, t_s: float) -> float:
        """ΔT₀ e^{−(t − t₀)/τ}; zero before it was deposited."""
        if t_s < self.deposited_t_s:
            return 0.0
        return float(self.amplitude_k * math.exp(-(t_s - self.deposited_t_s) / self.tau_s))


@dataclass
class HeatTraceLayer:
    """A stack of decaying footprints over one ground grid."""

    grid: GroundGrid
    footprints: list[Footprint] = field(default_factory=list)

    def deposit(self, footprint: Footprint) -> None:
        if footprint.mask.shape != self.grid.shape:
            raise ValueError(
                f"footprint mask {footprint.mask.shape} does not match grid {self.grid.shape}"
            )
        self.footprints.append(footprint)

    def delta_t(self, t_s: float) -> NDArray[np.float64]:
        """Σ ΔT_i(t) over the footprints covering each cell. Exactly zero where none do."""
        total = np.zeros(self.grid.shape, dtype=np.float64)
        for footprint in self.footprints:
            value = footprint.delta_t_at(t_s)
            if value != 0.0:
                total[footprint.mask] += value
        return total

    def apply(self, temperature_k: Any, t_s: float) -> NDArray[np.float64]:
        """Add the layer to a ground field. Untouched cells come back **bit-identical**."""
        base = np.asarray(temperature_k, dtype=np.float64)
        if base.shape != self.grid.shape:
            raise ValueError(f"field {base.shape} does not match grid {self.grid.shape}")
        return np.asarray(base + self.delta_t(t_s))


def _segment_mask(grid: GroundGrid, start: Any, end: Any, width_m: float) -> NDArray[np.bool_]:
    """Cells whose centres lie within ``width_m``/2 of the segment — an exact distance test.

    Rasterised by distance rather than by walking the line, so the result does not depend on the
    trajectory's sample spacing: a path sampled at 1 Hz and the same path at 100 Hz stripe the
    same cells.
    """
    if width_m <= 0.0:
        raise ValueError("width_m must be positive")
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    x = grid.origin_x_m + (cols + 0.5) * grid.cell_m
    y = grid.origin_y_m + (rows + 0.5) * grid.cell_m
    ab = b - a
    length2 = float(ab @ ab)
    if length2 == 0.0:
        distance = np.hypot(x - a[0], y - a[1])
    else:
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / length2, 0.0, 1.0)
        distance = np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))
    return np.asarray(distance <= 0.5 * width_m)


def tyre_trace_from_trajectory(
    grid: GroundGrid,
    path_xy_m: Any,
    track_width_m: float = 1.55,
    tyre_width_m: float = 0.20,
    amplitude_k: float = TYRE_TRACE_K,
    tau_s: float = TYRE_TAU_S,
    deposited_t_s: float = 0.0,
) -> list[Footprint]:
    """Two warm stripes, one per wheel track, along a path of world (x, y) points.

    Returns **two** footprints rather than one, because the pair is the identifying cue: a single
    wide stripe is a skid, two parallel ones at a known gauge are a vehicle, and a detector that
    has only ever seen the first will not learn the second.
    """
    path = np.asarray(path_xy_m, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("path_xy_m must be (N >= 2, 2) world points")
    footprints: list[Footprint] = []
    for side in (-0.5, 0.5):
        mask = np.zeros(grid.shape, dtype=bool)
        for start, end in zip(path[:-1], path[1:], strict=True):
            direction = end - start
            norm = float(np.hypot(*direction))
            if norm == 0.0:
                continue
            offset = np.array([-direction[1], direction[0]]) / norm * side * track_width_m
            mask |= _segment_mask(grid, start + offset, end + offset, tyre_width_m)
        footprints.append(Footprint(mask, amplitude_k, deposited_t_s, tau_s))
    return footprints


def body_shadow_from_pose(
    grid: GroundGrid,
    centre_xy_m: Any,
    length_m: float,
    width_m: float,
    heading_rad: float = 0.0,
    amplitude_k: float = BODY_SHADOW_K,
    tau_s: float = SHADOW_TAU_S,
    deposited_t_s: float = 0.0,
) -> Footprint:
    """The cold rectangle a departed vehicle leaves: the ground it shaded, still catching up."""
    if length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("footprint dimensions must be positive")
    centre = np.asarray(centre_xy_m, dtype=np.float64)
    rows, cols = np.mgrid[0 : grid.height, 0 : grid.width]
    x = grid.origin_x_m + (cols + 0.5) * grid.cell_m - centre[0]
    y = grid.origin_y_m + (rows + 0.5) * grid.cell_m - centre[1]
    cos_h, sin_h = math.cos(-heading_rad), math.sin(-heading_rad)
    along = x * cos_h - y * sin_h
    across = x * sin_h + y * cos_h
    mask = (np.abs(along) <= 0.5 * length_m) & (np.abs(across) <= 0.5 * width_m)
    return Footprint(np.asarray(mask), amplitude_k, deposited_t_s, tau_s)
