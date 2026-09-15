"""Scenarios drawn from the reference set's own measured distributions (M12.1, §15 T4).

A Tier 4 comparison is only as good as the match between what was rendered and what was filmed. If
the synthetic clips are all clear sky at 100 m and the real ones are half cloud at 200 m, the
difference the report measures is the difference in *scenario*, and the physics is never tested.

This module draws the scenario -- sky, cloud, range, target size, boresight -- from the bands ME.5
measured on the published set, rather than from numbers someone typed. Where ME.5 could not measure
something, the parameter is **ESTIMATED and says so in its own field**, because the distinction
between "matched to the data" and "chosen to look right" is the whole value of the exercise.

**What could not be matched, and why it matters.** ME.5 found the reference set's annotation boxes
are MATLAB MCOS objects with no Python reader, so the *target* distributions -- size, SCR, range --
were never measured. Those three are ESTIMATED here from the publication's stated flight
regulations (a 200 m ceiling) and from the classes it names. Any Tier 4 statistic that depends on a
target is therefore comparing a rendered distribution against an assumed one, and
:class:`ScenarioSampler` marks it.

docs/physics-model.md §15 T4; ADR 0068
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "target_pixels",
    "elevation_for_range",
    "Provenance",
    "ScenarioParameter",
    "Scenario",
    "ScenarioSampler",
    "HALMSTAD_PRIORS",
]

#: Where a number came from. A report that cannot tell these apart is not a validation.
Provenance = Literal["measured", "stated", "estimated"]


@dataclass(frozen=True)
class ScenarioParameter:
    """One sampled quantity, its range, and where the range came from."""

    name: str
    low: float
    high: float
    provenance: Provenance
    source: str
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.high >= self.low:
            raise ValueError(f"{self.name}: high must not be below low")
        if not self.source.strip():
            raise ValueError(f"{self.name}: every parameter must say where its range came from")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


#: The Halmstad set's priors. `measured` means ME.5 measured it on the clips; `stated` means the
#: publication says it; `estimated` means neither, and the value is a judgement.
HALMSTAD_PRIORS: tuple[ScenarioParameter, ...] = (
    ScenarioParameter(
        "cloud_fraction", 0.0, 1.0, "estimated",
        "not measurable on this set: the sky-region statistics need a labelled sky and a visible "
        "horizon, and ME.5 refused to invent either. Uniform is the honest uninformative choice",
    ),
    ScenarioParameter(
        "boresight_elevation_deg", 5.0, 35.0, "estimated",
        "a tripod-mounted monocular tracking a drone; no pose metadata is published", "deg",
    ),
    ScenarioParameter(
        "range_m", 30.0, 200.0, "stated",
        "the publication states a 200 m maximum sensor-to-target range under the flight "
        "regulations in force; the lower end is a judgement", "m",
    ),
    ScenarioParameter(
        "target_size_m", 0.3, 2.0, "stated",
        "from the four classes the set names (drone, bird, airplane, helicopter); the exact "
        "distribution is not published", "m",
    ),
    ScenarioParameter(
        "target_delta_t_k", 2.0, 25.0, "estimated",
        "a small multirotor's motors and pack against ambient; no radiometry is published and the "
        "clips are 8-bit post-conversion, so none can be recovered", "K",
    ),
)  # fmt: skip


@dataclass(frozen=True)
class Scenario:
    """One drawn scenario, with the provenance of every value it was drawn from."""

    index: int
    values: dict[str, float]
    provenance: dict[str, Provenance]
    seed: int

    @property
    def measured_fraction(self) -> float:
        """Share of this scenario's parameters that ME.5 actually measured.

        Reported rather than hidden: on the Halmstad set it is **zero**, because the statistics
        that would have constrained sky and target came out unmeasurable, and a reader should see
        that before reading a Tier 4 number derived from these scenarios.
        """
        if not self.provenance:
            return 0.0
        measured = sum(1 for p in self.provenance.values() if p == "measured")
        return measured / len(self.provenance)

    def __getitem__(self, key: str) -> float:
        return self.values[key]


@dataclass
class ScenarioSampler:
    """Draws scenarios reproducibly from a set of priors."""

    priors: tuple[ScenarioParameter, ...] = HALMSTAD_PRIORS
    seed: int = 20260915
    _drawn: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        names = [p.name for p in self.priors]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        if not self.priors:
            raise ValueError("a sampler with no priors draws nothing")

    def draw(self, n: int) -> list[Scenario]:
        """``n`` scenarios. Each gets **its own seeded generator**, derived from the sampler's seed
        and its index, so scenario 7 is the same whether you asked for ten or a thousand -- which
        is what makes a failing case reproducible without re-running the whole set."""
        if n <= 0:
            raise ValueError("n must be positive")
        out = []
        for index in range(self._drawn, self._drawn + n):
            rng = np.random.default_rng([self.seed, index])
            out.append(
                Scenario(
                    index=index,
                    values={p.name: p.draw(rng) for p in self.priors},
                    provenance={p.name: p.provenance for p in self.priors},
                    seed=self.seed,
                )
            )
        self._drawn += n
        return out

    def summary(self) -> dict[str, Any]:
        """What a report prints beside its numbers: how much of this was matched to data."""
        counts: dict[str, int] = {"measured": 0, "stated": 0, "estimated": 0}
        for p in self.priors:
            counts[p.provenance] += 1
        return {
            "n_parameters": len(self.priors),
            "provenance_counts": counts,
            "parameters": [
                {
                    "name": p.name,
                    "low": p.low,
                    "high": p.high,
                    "unit": p.unit,
                    "provenance": p.provenance,
                    "source": p.source,
                }
                for p in self.priors
            ],
        }


def target_pixels(size_m: float, range_m: float, focal_mm: float, pitch_um: float) -> float:
    """f·s/(R·p) -- the extent a target of this size subtends, in native pixels."""
    if min(size_m, range_m, focal_mm, pitch_um) <= 0.0:
        raise ValueError("every argument must be positive")
    return float(focal_mm * 1e-3 * size_m / (range_m * pitch_um * 1e-6))


def elevation_for_range(range_m: float, altitude_m: float) -> float:
    """Boresight elevation, in degrees, for a target at this slant range and height."""
    if range_m <= 0.0:
        raise ValueError("range must be positive")
    return float(math.degrees(math.asin(min(1.0, max(-1.0, altitude_m / range_m)))))
