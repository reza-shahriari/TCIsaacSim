#!/usr/bin/env python3
"""Sky apparent temperature vs elevation from the layered atmosphere against the R13 anchors.

    python scripts/validate_sky_r13.py

R13 (Sensors 21:7067, Tucson, clear, low humidity, FLIR T1020 7.5-14 um and TELOPS M1k 2.2-5.5 um):
the LWIR sky reaches -40 C at 15 deg elevation while the MWIR sky reads above +10 C. Prints the
model's T_app(theta) per band for the calibration weather (ADR 0071).
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path


def main() -> int:
    import numpy as np

    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.atmosphere.library import load_atmosphere_preset
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.thermal import WeatherSample, WeatherSeries

    tmp = Path(tempfile.mkdtemp())

    def tophat(lo: float, hi: float):  # type: ignore[no-untyped-def]
        p = tmp / f"tophat_{lo}_{hi}.csv"
        p.write_text(f"# exact top-hat\n{lo},1.0\n{hi},1.0\n")
        return load_spectral_response(p)

    bands = {"lwir": (7.5, 14.0), "mwir": (2.2, 5.5)}
    responses = {b: tophat(*edges) for b, edges in bands.items()}
    luts = {b: BandLUT.build(r) for b, r in responses.items()}
    weather = WeatherSeries.constant(
        WeatherSample(288.15, 0.20, 2.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, luts, responses)
    print(
        "R13 anchors: LWIR -40 C at 15 deg, MWIR > +10 C "
        "(T_air 288.15 K, RH 0.20, V 23 km, US Standard; thermal-only model)"
    )
    for band in bands:
        es = atm.exponential_sum(band, 0.0)
        print(f"{band}: classes {es.names}")
        print(f"  weights {np.round(es.weights, 3)}")
        print(f"  gamma_0 {es.gamma_0}")
        print(f"  gamma_aer {es.gamma_aerosol:.2e}")
        row = []
        for deg in (0.5, 2, 5, 10, 15, 30, 45, 60, 90):
            t = atm.apparent_sky_temperature_k(band, 0.0, math.radians(deg))
            row.append(f"{deg:4g}deg {t - 273.15:+6.1f}C")
        print("  " + "  ".join(row))
        print(f"  tau(200 m, horizontal) = {float(atm.transmittance(band, 0.0, 200.0)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
