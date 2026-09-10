#!/usr/bin/env python3
"""Generate band radiance LUT bundles from sensor configs and spectral response data.

    python scripts/generate_luts.py --configs configs/sensors --out data/lut --data data

For every sensor YAML (docs/physics-model.md §12.2) found in ``--configs`` (a directory or
explicit files):

  1. load and validate the config; resolve and validate its R(λ) file against the data root
     (peak-1 shape in micrometres -- anything else is refused before anything is written);
  2. tabulate Lb, Lb_q, dLb/dT, dLb_q/dT over 200-1000 K at 0.05 K by Simpson quadrature
     (§3.2 b), float32 (ADR 0011);
  3. cross-check the table against the closed-form top-hat over the config's nominal band at
     300 K and 600 K and report the difference in percent and in millikelvin -- for a soft-edged
     response a few percent is expected; a factor is a bug;
  4. write ``<key>_*.npy``, ``<key>_*.f32`` and ``<key>_lut.json`` (ADR 0012) and print a summary.

Adding a band touches only a YAML, a CSV and this command (§12.2). Exit status is non-zero, and
nothing is written for that sensor, if any input is invalid. Bundles are gitignored: rerun after
changing a response file or a band block; the loader refuses stale bundles anyway.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

CHECK_TEMPERATURES_K = (300.0, 600.0)


def _find_configs(items: Sequence[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for item in items:
        p = pathlib.Path(item)
        if p.is_dir():
            out += sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        else:
            out.append(p)
    return out


def generate_one(
    config_path: pathlib.Path, out_dir: pathlib.Path, data_dir: pathlib.Path | None
) -> str:
    """Build and write one bundle; return a one-paragraph summary. Raises on any invalid input."""
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.band_integration import d_band_radiance_dT
    from irsim.radiometry.lut_files import build_band_lut_for_config
    from irsim.radiometry.planck import band_radiance_tophat
    from irsim.radiometry.spectral_response import load_spectral_response

    cfg = load_sensor_config(config_path, data_dir)
    band = cfg.sensor.band
    lut, paths = build_band_lut_for_config(cfg, out_dir, data_dir)
    response = load_spectral_response(band.spectral_response)

    lines = [
        f"{cfg.sensor.name}: band {band.band_id} {band.lambda_min_um}-{band.lambda_max_um} um, "
        f"{lut.n} entries {lut.t0_k:.0f}-{lut.t1_k:.0f} K @ {lut.dt_k:.3f} K, float32",
        f"  response: {pathlib.Path(band.spectral_response).name} sha256 {response.sha256[:12]}… "
        f"half-power {response.half_power_points_um()[0]:.2f}-"
        f"{response.half_power_points_um()[1]:.2f} um",
    ]
    for t in CHECK_TEMPERATURES_K:
        lb = float(lut.lookup(t)[()])
        tophat = band_radiance_tophat(band.lambda_min_um, band.lambda_max_um, t)
        slope = float(d_band_radiance_dT(response, t)[0])
        lines.append(
            f"  Lb({t:.0f} K) = {lb:.4f} W/m2/sr; top-hat closed form {tophat:.4f} "
            f"({(lb / tophat - 1) * 100:+.2f} %, {(lb - tophat) / slope * 1e3:+.1f} mK equivalent)"
        )
    lines.append(f"  wrote {paths.sidecar} (+ 4 .npy, 4 .f32), band_hash {lut.band_hash[:16]}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs", nargs="+", required=True, help="sensor YAML files or directories"
    )
    parser.add_argument("--out", required=True, help="output directory for the bundles (data/lut)")
    parser.add_argument(
        "--data",
        default=None,
        help="data root for relative response paths (default: <repo>/data or $IRSIM_DATA_DIR)",
    )
    args = parser.parse_args(argv)

    configs = _find_configs(args.configs)
    if not configs:
        print(f"error: no sensor configs found in {args.configs}", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(args.out)
    data_dir = pathlib.Path(args.data) if args.data else None
    failures = 0
    for path in configs:
        try:
            print(generate_one(path, out_dir, data_dir))
        except Exception as exc:  # noqa: BLE001 -- report every failing sensor, then exit non-zero
            failures += 1
            print(f"error: {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"{len(configs) - failures}/{len(configs)} bundle(s) written to {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
