#!/usr/bin/env python3
"""Quantify the grey-band (single gamma_B) atmosphere approximation error (ADR 0048, M8.7).

    python scripts/validate_atmosphere_band_average.py

For two synthetic spectral extinction curves -- a two-level LWIR gamma(lambda) over 8-12 um and
a MWIR gamma(lambda) with an opaque CO2 notch at 4.2-4.4 um -- compute the exact Planck-weighted
band transmittance tau_B(d) by quadrature (irsim.atmosphere.spectral), fit the grey gamma_B over
0-300 m as the L2 model does, and print the relative error of exp(-gamma_B d) at 500 and 1000 m,
plus the curve of growth gamma_eff(d). The numbers go into ADR 0048 and the README limitation.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    from irsim.atmosphere.spectral import effective_gamma, fit_grey_gamma, grey_fit_error
    from irsim.radiometry.spectral_response import load_spectral_response

    tmp = Path(tempfile.mkdtemp())

    def tophat(lo: float, hi: float):  # type: ignore[no-untyped-def]
        p = tmp / f"tophat_{lo}_{hi}.csv"
        p.write_text(f"# exact top-hat\n{lo},1.0\n{hi},1.0\n")
        return load_spectral_response(p)

    cases = {
        "LWIR two-level (8-10 um: 2e-4/m, 10-12 um: 1.2e-3/m)": (
            tophat(8.0, 12.0),
            lambda lam: np.where(lam < 10.0, 2e-4, 1.2e-3),
        ),
        "MWIR with opaque CO2 notch 4.2-4.4 um (3e-4/m elsewhere)": (
            tophat(3.0, 5.0),
            lambda lam: np.where((lam >= 4.2) & (lam <= 4.4), 5.0, 3e-4),
        ),
    }
    for name, (resp, gamma) in cases.items():
        g_b = fit_grey_gamma(resp, gamma)
        err = grey_fit_error(resp, gamma)
        d = np.array([50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0])
        g_eff = effective_gamma(resp, gamma, d)
        print(name)
        print(f"  grey gamma_B fitted over 0-300 m: {g_b:.4e} /m")
        print(
            "  gamma_eff(d) [/m]: "
            + ", ".join(f"{dd:.0f} m: {g:.3e}" for dd, g in zip(d, g_eff, strict=True))
        )
        for dd, e in err.items():
            print(f"  grey fit relative error at {dd:.0f} m: {e * 100:+.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
