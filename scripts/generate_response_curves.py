#!/usr/bin/env python3
"""Generate the modelled detector spectral-response curves (roadmap M11.1, M11.5).

    python scripts/generate_response_curves.py --out data/spectra/responses

Writes one peak-normalised R(lambda) per detector family. Each file is ESTIMATED -- a shape built
from the physics of the part, not a measurement of a specific unit -- and says so in its own
header, like data/spectra/responses/boson_vox.csv does.

Keeping them in a script rather than hand-authoring them means the *reasoning* is in one place:
where a cut-on comes from (a substrate or a cold filter), where a cut-off comes from (the alloy's
band gap), and how wide the edges are. Rerun and diff to see that nothing drifted.
"""

from __future__ import annotations

import argparse
import math
import pathlib

CURVES: dict[str, dict[str, object]] = {}


def _raised_cosine_up(lam: float, lo: float, hi: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * (lam - lo) / (hi - lo)))


def _raised_cosine_down(lam: float, lo: float, hi: float) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * (lam - lo) / (hi - lo)))


# --- InGaAs (SWIR, M11.1) ---------------------------------------------------------------------

INGAAS = dict(
    edges=(0.80, 1.80),
    step=0.005,
    decimals=3,
    header="""\
# Generic 640x512 InGaAs SWIR camera, relative spectral response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Optics + detector folded together.
#
# Shape (docs/physics-model.md §12.1 SWIR row, roadmap M11.1):
#   * cut-on: raised cosine over 0.85-0.95 um. Back-illuminated lattice-matched In0.53Ga0.47As
#     is read through its InP substrate, whose own gap (1.35 eV, 0.92 um) sets the short-wave
#     edge; the 0.1 um width is the absorption edge broadened by the AR coat and window.
#   * plateau: 0.95 -> 1.00 linearly over 0.95-1.60 um. Quantum efficiency creeps up as the
#     absorption depth grows towards the band edge; datasheets show a few percent of tilt, not a
#     flat top.
#   * cut-off: raised cosine over 1.60-1.72 um, i.e. half response at 1.66 um. The alloy's gap is
#     0.75 eV -> 1.653 um; vendors quote "1.7 um cutoff" at the 10 % point, which this shape puts
#     at 1.70 um.
# Expected error vs a measured curve: the plateau level is a normalisation, so it cancels in a
# peak-normalised R(lambda); the cut-off placement is good to ~0.02 um, the cut-on to ~0.05 um.
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-15 (roadmap M11.1).
wavelength_um,response
""",
)


def ingaas(lam: float) -> float:
    if lam <= 0.85 or lam >= 1.72:
        return 0.0
    if lam < 0.95:
        return _raised_cosine_up(lam, 0.85, 0.95) * 0.95
    if lam < 1.60:
        return 0.95 + 0.05 * (lam - 0.95) / (1.60 - 0.95)
    return _raised_cosine_down(lam, 1.60, 1.72)


# --- InSb (MWIR, M11.5) -----------------------------------------------------------------------

INSB = dict(
    edges=(2.50, 6.00),
    step=0.005,
    decimals=3,
    header="""\
# Generic 640x512 InSb MWIR camera at 77 K, relative spectral response R(lambda), peak-normalised.
# STATUS: ESTIMATED -- not a measurement of any specific unit. Cold filter + detector + optics
# folded together.
#
# Shape (docs/physics-model.md §12.1 MWIR row, roadmap M11.5):
#   * cut-on: raised cosine over 2.90-3.10 um. InSb itself responds from about 1 um; the short-
#     wave edge of a 3-5 um camera is the COLD FILTER, not the detector. That is why it is
#     steeper than the InGaAs substrate edge -- an interference filter at 77 K, not an
#     absorption edge.
#   * plateau: 0.92 -> 1.00 linearly over 3.10-4.90 um, the usual mild rise towards the band edge.
#   * cut-off: raised cosine over 4.90-5.25 um. Two mechanisms coincide here and the narrower one
#     wins: the cold filter's long-wave edge sits just inside InSb's own gap (0.23 eV at 77 K ->
#     5.39 um), which is why a warmer detector's cut-off creeps out and a filtered camera's does
#     not.
# NOTE: this is a plain 3-5 um shape. Many MWIR cameras ship a "blue" filter (about 3.6-4.2 um)
# to sit outside the 4.2-4.4 um CO2 absorption; that is a different file, not a different model,
# which is the point of keeping bands in data.
# Columns: wavelength in MICROMETRES, dimensionless response in [0, 1]. Grid 0.005 um.
# Authored 2026-09-15 (roadmap M11.5).
wavelength_um,response
""",
)


def insb(lam: float) -> float:
    if lam <= 2.90 or lam >= 5.25:
        return 0.0
    if lam < 3.10:
        return _raised_cosine_up(lam, 2.90, 3.10) * 0.92
    if lam < 4.90:
        return 0.92 + 0.08 * (lam - 3.10) / (4.90 - 3.10)
    return _raised_cosine_down(lam, 4.90, 5.25)


CURVES["ingaas"] = {**INGAAS, "fn": ingaas}
CURVES["insb"] = {**INSB, "fn": insb}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/spectra/responses")
    parser.add_argument("--only", action="append", choices=sorted(CURVES))
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(CURVES):
        spec = CURVES[name]
        lo, hi = spec["edges"]  # type: ignore[misc]
        step = float(spec["step"])  # type: ignore[arg-type]
        fn = spec["fn"]  # type: ignore[assignment]
        decimals = int(spec["decimals"])  # type: ignore[arg-type]
        n = int(round((hi - lo) / step)) + 1
        rows = []
        peak = 0.0
        for i in range(n):
            lam = lo + i * step
            value = float(fn(lam))  # type: ignore[operator]
            peak = max(peak, value)
            rows.append(f"{lam:.{decimals}f},{value:.6f}")
        path = out / f"{name}.csv"
        path.write_text(str(spec["header"]) + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"wrote {path} ({n} rows), peak {peak:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
