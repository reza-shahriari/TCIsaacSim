#!/usr/bin/env python3
"""Roadmap M10.19: render the phase-1 aerial demo stage through `IrCamera` and export the frames.

This is the end of the M10 chain as a single command: a USD stage with small targets against sky,
the Boson configuration, the scene's one WeatherSeries feeding the atmosphere, the sky model and
the target solvers, the whole sensor chain, and four outputs per frame on disk in physical units
(M10.10a). It is the thing to run when the question is "what does the camera see".

There is no sky dome in the stage. Background rays take T_sky(theta) from the sky model at their
own elevation, or T_ground below the horizon, because every colour AOV on this build is float16
and would quantise the sky to ~100 mK against a 50 mK NETD (ADR 0014, ADR 0060).

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_aerial_demo.py --frames 8 --out outputs/aerial_demo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/aerial_demo")
parser.add_argument("--frames", type=int, default=4)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/sky_target_clear_day.yaml"))
parser.add_argument("--tilt-deg", type=float, default=8.0)
parser.add_argument("--t0", type=float, default=0.0, help="scene time of the first frame, seconds")
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument(
    "--no-point-targets",
    action="store_true",
    help="render sub-pixel targets as geometry instead of injecting them (ADR 0071 ablation)",
)
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
boot_s = time.time() - t_boot


def main() -> int:
    import numpy as np

    from irsim.config.loader import band_hash, config_hash, load_sensor_config
    from irsim.io import write_frame
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config
    from irsim.scene import Scene
    from irsim_isaac.aerial_demo import analytic_targets, build_aerial_demo, describe
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor = load_sensor_config(args.sensor)
    spec = sensor.sensor
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    scene = Scene.from_file(args.scene, {spec.band.band_id: lut})

    demo = build_aerial_demo(camera_tilt_deg=args.tilt_deg)
    if demo.errors:
        print(f"stage errors: {demo.errors}", file=sys.stderr)

    # Below one native pixel the renderer samples geometry and gets a phase-dependent fraction of
    # the flux (ADR 0071), so those targets are hidden and injected analytically instead (MS.6).
    # `analytic_targets` hides them in the same call that produces their specs, so the two paths
    # cannot both claim a target.
    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    analytic = [] if args.no_point_targets else analytic_targets(demo, ifov_mrad)

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)

    pipeline = PipelineConfig.from_sensor(
        sensor,
        table,
        lut,
        sky=scene.sky_models[spec.band.band_id],
        atmosphere=scene.layered,
    )
    if not args.no_chain:
        from irsim.pipeline.sensor_chain import attach_sensor_chain

        # The chain's housing node reads ambient, so it takes the scene's own WeatherSeries --
        # the same object the atmosphere, the sky and the target solvers hold (CLAUDE.md #6).
        pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    rows = describe(demo, ifov_mrad)
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, IFOV {ifov_mrad:.3f} mrad, "
        f"{spec.optics.supersample_factor}x supersampled, boresight "
        f"{demo.boresight_elevation_deg():.1f} deg above the horizon"
    )
    injected = {t.name for t in analytic}
    for row in rows:
        flag = "  ANALYTIC (MS.6)" if row["name"] in injected else ""
        if row["subpixel"] and row["name"] not in injected:
            flag = "  SUB-PIXEL, rendered anyway (--no-point-targets)"
        print(
            f"  {row['name']:12s} {row['range_m']:7.0f} m  {row['mrad']:6.3f} mrad  "
            f"{row['pixels']:6.2f} px  el {row['elevation_deg']:5.1f} deg  "
            f"{row['material']}{flag}"
        )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        analytic_targets=analytic,
        camera_path=demo.camera_path,
        strict_materials=False,
    ).open(settle_frames=args.settle)

    written = []
    t_render = time.time()
    for index in range(args.frames):
        outputs = camera.get_outputs()
        record = write_frame(
            out_dir,
            outputs,
            frame_index=index,
            t_s=scene.t0_s + camera.t_rel_s,
            start_utc=scene.weather.start_utc,
            config_hash=config_hash(sensor),
            band_hash=band_hash(sensor),
            quantity=pipeline.quantity,
            float_format=args.float_format,
            extra_metadata={
                "scene": pathlib.Path(args.scene).name,
                "camera_tilt_deg": demo.camera_tilt_deg,
                "targets": rows,
                "analytic_targets": sorted(injected),
            },
        )
        written.append(record)
        if outputs.apparent_t is not None:
            t_app = np.asarray(outputs.apparent_t)
            print(
                f"  frame {index}: T_app {t_app.min():.2f} .. {t_app.max():.2f} K "
                f"(median {np.median(t_app):.2f})"
            )
    render_s = time.time() - t_render
    camera.close()

    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": len(written),
        "sensor": spec.name,
        "resolution": [spec.fpa.width, spec.fpa.height],
        "supersample": spec.optics.supersample_factor,
        "ifov_mrad": round(ifov_mrad, 4),
        "targets": rows,
        "analytic": sorted(injected),
        "files": [str(p.name) for r in written for p in r.files.values()],
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n{len(written)} frames in {render_s:.1f} s -> {out_dir}\nwrote {path}")
    return 0


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a demo crash should still report, not hang
    import traceback

    traceback.print_exc()
    print(f"render failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    app.close(exit_code=status if isinstance(status, int) else 1)
