#!/usr/bin/env python3
"""Generate band radiance LUTs from sensor configs and spectral response data.

Not implemented yet. Planned behaviour (docs/physics-model.md §3.2 (b)):

  For each sensor config:
    1. Load R(lambda) from data/spectra/, assert normalised to peak 1.0.
    2. Integrate Planck (energy and photon forms) and dL/dT against R(lambda)
       by Simpson quadrature on a 0.01 um grid, over T in [200, 1000] K at
       0.05 K spacing.
    3. Write float32 arrays to data/lut/<sensor>_{Lb,Lb_q,dLb_dT}.npy
    4. Write a sidecar <sensor>_lut.json with the SHA-256 of the config and
       spectral file, so a stale LUT is detectable rather than silently wrong.

Cross-check every generated table against band_radiance_tophat() for a rectangular
response; disagreement above 0.1% means one of the two paths has a bug.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    print(
        "ERROR: not implemented -- see docs/physics-model.md §3.2 and the ir-radiometry skill.",
        file=sys.stderr,
    )
    sys.exit(1)
