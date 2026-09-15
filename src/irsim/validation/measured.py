"""Where a measured Tier 2 bench file lives, and what a comparison against it means (M12.4).

docs/physics-model.md §15 Tier 2; ADR 0003.

**No IR camera is available to this project.** Every Tier 2 number the repository reports is a
self-consistency check: the simulated SITF is linear in L_B because the code makes it so, and the
simulated NETD matches its own anchor because that is what an anchor is. None of that is evidence
about a real camera, and the README says so.

This module is the shape of the hole. It fixes **where** a measured file goes, **what columns it
has**, and **what a comparison against it is allowed to conclude** -- so that the day a camera
arrives, the benches already know how to read it and nobody has to decide the tolerances while
looking at the data. The procedure that produces them is in
`docs/validation/tier2-bench-protocol.md`.

The layout is one directory per quantity and one file per camera, named by the camera's config
`name`:

    data/validation/sitf/<camera>.csv        T_blackbody_K, DN
    data/validation/netd/<camera>.csv        T_blackbody_K, NETD_mK
    data/validation/noise3d/<camera>.csv     component, sigma_DN      (the NVESD seven)
    data/validation/mtf/<camera>.csv         frequency_cyc_per_mm, MTF

Every comparison here is **shape-only against a fitted gain and offset** where the quantity has an
arbitrary scale (SITF), and absolute where it does not (NETD in mK, MTF as a ratio). That split is
the point: ADR 0019 makes the absolute DN of a simulated SITF a range choice, so requiring it to
match a bench would be requiring the simulator to guess the bench's gain.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MEASURED_QUANTITIES",
    "Comparison",
    "measured_path",
    "load_measured_pairs",
    "compare_shape",
    "compare_absolute",
]

Quantity = Literal["sitf", "netd", "noise3d", "mtf"]

#: quantity -> (subdirectory, the two column headings, whether the scale is arbitrary).
MEASURED_QUANTITIES: dict[str, tuple[str, tuple[str, str], bool]] = {
    "sitf": ("sitf", ("T_blackbody_K", "DN"), True),
    "netd": ("netd", ("T_blackbody_K", "NETD_mK"), False),
    "noise3d": ("noise3d", ("component", "sigma_DN"), True),
    "mtf": ("mtf", ("frequency_cyc_per_mm", "MTF"), False),
}


def measured_path(
    quantity: str, camera: str, data_dir: str | os.PathLike[str] | None = None
) -> pathlib.Path:
    """``<data>/validation/<quantity>/<camera>.csv``. Never creates anything."""
    if quantity not in MEASURED_QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}; known: {sorted(MEASURED_QUANTITIES)}")
    from irsim.config.loader import resolve_data_dir

    return (
        resolve_data_dir(data_dir)
        / "validation"
        / MEASURED_QUANTITIES[quantity][0]
        / f"{camera}.csv"
    )


def load_measured_pairs(
    path: str | os.PathLike[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two numeric columns from a ``#``-commented CSV, with one optional header line.

    Refuses an empty or single-row file rather than returning something a fit would accept: a
    bench table with one point cannot constrain a shape, and a silent pass there would be the
    worst possible outcome for a file whose whole purpose is to be the ground truth.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no measured file at {p}")
    rows: list[tuple[float, float]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [c.strip() for c in stripped.replace(";", ",").split(",")]
        if len(parts) < 2:
            raise ValueError(f"{p}: expected two columns, got {stripped!r}")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            if not rows:  # a header line
                continue
            raise ValueError(f"{p}: non-numeric row {stripped!r}") from None
    if len(rows) < 2:
        raise ValueError(f"{p}: at least two rows are needed to compare a shape")
    array = np.asarray(rows, dtype=np.float64)
    return array[:, 0], array[:, 1]


@dataclass(frozen=True)
class Comparison:
    """What a bench comparison concluded, with the fit it needed to get there."""

    n: int
    rms: float
    span: float
    gain: float
    offset: float
    tolerance: float

    @property
    def relative_rms(self) -> float:
        return self.rms / self.span if self.span > 0.0 else float("inf")

    @property
    def passed(self) -> bool:
        return self.relative_rms <= self.tolerance

    def describe(self) -> str:
        fit = (
            f" after a gain {self.gain:.4g} and offset {self.offset:.4g} fit"
            if (self.gain, self.offset) != (1.0, 0.0)
            else ""
        )
        return (
            f"{self.n} points: rms {self.rms:.4g} = {self.relative_rms:.2%} of the "
            f"{self.span:.4g} span{fit}; tolerance {self.tolerance:.2%}"
        )


def compare_shape(simulated: Any, measured: Any, *, tolerance: float = 0.01) -> Comparison:
    """Compare after fitting a **gain and offset**: for a quantity whose scale is a range choice.

    The residual is reported relative to the measured **span**, not in absolute units, because
    the units are exactly what the fit removed.
    """
    sim = np.asarray(simulated, dtype=np.float64).ravel()
    meas = np.asarray(measured, dtype=np.float64).ravel()
    if sim.size != meas.size or sim.size < 2:
        raise ValueError("need at least two paired points to fit a shape")
    gain, offset = np.polyfit(sim, meas, 1)
    residual = meas - (gain * sim + offset)
    return Comparison(
        n=int(sim.size),
        rms=float(np.sqrt(np.mean(residual**2))),
        span=float(np.ptp(meas)),
        gain=float(gain),
        offset=float(offset),
        tolerance=float(tolerance),
    )


def compare_absolute(simulated: Any, measured: Any, *, tolerance: float = 0.15) -> Comparison:
    """Compare with **no fit**: for a quantity in physical units a simulator must actually predict.

    NETD in millikelvin and MTF as a ratio are the two. A gain fit here would let a model be wrong
    by any factor and still pass, which is the failure this function exists to make impossible.

    The residual is relative to the measured **magnitude**, since nothing was fitted away.
    """
    sim = np.asarray(simulated, dtype=np.float64).ravel()
    meas = np.asarray(measured, dtype=np.float64).ravel()
    if sim.size != meas.size or sim.size < 1:
        raise ValueError("need at least one paired point")
    residual = meas - sim
    # The denominator is the measured **magnitude**, not its peak-to-peak range. NETD at two
    # setpoints spans 13 mK around a level of 47 mK, and normalising by the span would call a 7 %
    # error a 27 % one -- reading a tolerance on the *difference between setpoints* when what was
    # asked for is a tolerance on the value. The shape comparison normalises by the span because
    # there the level is exactly what the fit removed; here nothing was removed.
    scale = float(np.mean(np.abs(meas)))
    return Comparison(
        n=int(sim.size),
        rms=float(np.sqrt(np.mean(residual**2))),
        span=scale if scale > 0.0 else 1.0,
        gain=1.0,
        offset=0.0,
        tolerance=float(tolerance),
    )
