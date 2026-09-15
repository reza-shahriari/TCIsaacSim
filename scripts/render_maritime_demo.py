#!/usr/bin/env python3
"""Roadmap MM.7: render the maritime demo stage — vessels on open water — and export the frames.

The maritime counterpart of `render_aerial_demo.py`, and it is the same command in the same shape
for the same reason: a scene YAML in, physical-unit frames out, one run.

What differs is what is below the horizon. The aerial stage has one scalar `T_ground` down there
and never looks at it. Here the sea *is* the picture, and it is **computed, not rendered**: every
pixel whose ray goes below the horizon takes `SeaModel.apparent_temperature_k` at its own
depression angle (ADR 0078), because at 3 km a Boson pixel spans 2.6 m and contains thousands of
independent wave facets — the detector integrates a slope *distribution*, which no tessellation
delivers.

The water mesh in the stage is therefore for the **visible** frame only. It occludes, it sets the
horizon and it carries the Earth's curvature, but its infrared temperature comes from the analytic
profile, not from a material and a solver node. `--no-water` proves it: the infrared frame is
unchanged and the visible one loses its surface.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_maritime_demo.py --frames 4 --rgb --out outputs/maritime_demo
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/maritime_demo")
parser.add_argument("--frames", type=int, default=4)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/vessel_transit_clear_day.yaml"))
parser.add_argument(
    "--tilt-deg", type=float, default=-5.0, help="camera tilt; negative looks DOWN at the sea"
)
parser.add_argument("--camera-height-m", type=float, default=20.0, help="eye height above the sea")
parser.add_argument("--t0", type=float, default=0.0, help="scene time of the first frame, seconds")
parser.add_argument("--float-format", default="npy", choices=("npy", "exr"))
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--rgb", action="store_true", help="also capture the companion visible frame")
parser.add_argument(
    "--heading-deg", type=float, default=120.0, help="compass bearing the camera looks along"
)
parser.add_argument("--cloud-seed", type=int, default=None, help="structured cloud (ADR 0076)")
parser.add_argument(
    "--wind-dir-deg", type=float, default=35.0, help="wave train direction for the visible surface"
)
parser.add_argument(
    "--no-water",
    action="store_true",
    help="author no water geometry (ADR 0078 ablation: the IR frame is identical)",
)
parser.add_argument("--water-rings", type=int, default=420)
parser.add_argument("--water-sectors", type=int, default=448)
parser.add_argument(
    "--water-half-angle-deg",
    type=float,
    default=26.0,
    help="half-width of the water wedge; must cover the horizontal field with margin",
)
parser.add_argument("--no-dome", action="store_true", help="untextured grey dome (ADR 0073)")
parser.add_argument("--no-flat-field", action="store_true", help="skip the flat-field correction")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
boot_s = time.time() - t_boot


def main() -> int:
    import numpy as np

    from irsim.atmosphere.sea import SeaModel, slant_range_m
    from irsim.config.loader import band_hash, config_hash, load_sensor_config
    from irsim.io import write_frame
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.nk import load_nk_table
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene
    from irsim_isaac.maritime_demo import build_maritime_demo, describe
    from irsim_isaac.pipeline.illumination_isaac import SceneIllumination
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.visible_sky import dome_spec_from_scene

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sensor = load_sensor_config(args.sensor)
    spec = sensor.sensor
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    scene = Scene.from_file(args.scene, {spec.band.band_id: lut})

    environment = scene.environment
    if environment is None or environment.ground.mode != "sea":
        print(
            f"scene {args.scene} does not use a sea environment preset "
            f"(ground.mode must be 'sea'); nothing below the horizon would be water",
            file=sys.stderr,
        )
        return 2
    bulk_sst_k = float(environment.ground.bulk_sst_k or 0.0)

    # One SkyModel, one WeatherSeries, and the sea reflects that sky (CLAUDE.md #6, ADR 0078).
    sky = scene.sky_models[spec.band.band_id]
    sea = SeaModel(
        sky,
        load_nk_table("water"),
        load_spectral_response(str(spec.band.spectral_response)),
        bulk_sst_k=bulk_sst_k,
        camera_height_m=args.camera_height_m,
    )
    horizon_deg = math.degrees(sea.horizon_rad)
    wind_m_s = float(scene.weather.at(scene.t0_s).wind_speed_m_s)
    horizon_km = float(slant_range_m(args.camera_height_m, sea.horizon_rad)) / 1e3
    tilt_deg = math.degrees(sea.tilt_sigma(scene.t0_s))
    print(
        f"\nsea: bulk SST {bulk_sst_k:.2f} K, camera {args.camera_height_m:.0f} m up, "
        f"horizon {horizon_deg:.4f} deg down at {horizon_km:.1f} km; "
        f"wind {wind_m_s:.1f} m/s -> rms facet tilt {tilt_deg:.2f} deg"
    )
    probe = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 45.0, 90.0])
    probe = probe[np.radians(probe) >= sea.horizon_rad]
    t_probe = sea.apparent_temperature_k(scene.t0_s, np.radians(probe))
    print("  sea profile (depression deg -> T_app K):")
    for d, t in zip(probe, np.atleast_1d(t_probe), strict=False):
        rng = float(slant_range_m(args.camera_height_m, math.radians(d)))
        print(f"    {d:6.2f} -> {float(t):7.2f}   (range {rng:8.0f} m)")

    dome = None
    if not args.no_dome:
        dome = dome_spec_from_scene(
            scene, heading_deg=args.heading_deg, camera_height_m=args.camera_height_m
        )
        # Below the horizon the dome paints "terrain". Whatever the water mesh does not cover is
        # sea, so it gets a sea albedo rather than the default soil green.
        dome = type(dome)(**{**dome.__dict__, "ground_albedo": (0.020, 0.035, 0.050)})
        print(
            f"\nenvironment dome: sun at {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg:.1f} deg azimuth "
            f"({dome.sun_azimuth_deg - args.heading_deg:+.1f} deg from boresight); "
            f"turbidity {dome.turbidity:.2f} at {dome.visibility_m / 1e3:.0f} km visibility"
        )

    demo = build_maritime_demo(
        camera_tilt_deg=args.tilt_deg,
        camera_height_m=args.camera_height_m,
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
        water=not args.no_water,
        water_half_angle_deg=args.water_half_angle_deg,
        # The wave train is scaled by the SAME wind Cox-Munk reads, so the picture and the
        # radiometry cannot disagree about how hard it is blowing (CLAUDE.md #6).
        wind_speed_m_s=wind_m_s,
        water_rings=args.water_rings,
        water_sectors=args.water_sectors,
        wind_dir_deg=args.wind_dir_deg,
    )
    if demo.errors:
        print(f"stage errors: {demo.errors}", file=sys.stderr)

    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
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
        sky=sky,
        atmosphere=scene.layered,
        flat_field_enabled=not args.no_flat_field,
    )
    if not args.no_chain:
        from irsim.pipeline.sensor_chain import attach_sensor_chain

        pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    rows = describe(demo, ifov_mrad)
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, IFOV {ifov_mrad:.3f} mrad, "
        f"boresight {demo.camera_tilt_deg:+.1f} deg, horizon "
        f"{demo.horizon_elevation_deg():+.2f} deg from boresight"
    )
    for row in rows:
        print(
            f"  {row['name']:11s} {row['range_m']:7.0f} m  {row['length_m']:5.0f} m  "
            f"{row['mrad']:7.3f} mrad  {row['pixels']:6.1f} px  "
            f"waterline {row['depression_deg']:6.3f} deg down  {row['material']}"
        )

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=demo.prim_to_target,
        resolutions=resolutions,
        camera_path=demo.camera_path,
        capture_rgb=args.rgb,
        illumination=SceneIllumination.for_camera(
            sensor, scene, pipeline.quantity, heading_deg=args.heading_deg
        ),
        heading_deg=args.heading_deg,
        strict_materials=False,
        cloud_seed=args.cloud_seed,
        sea=sea,
        background_prim_paths=demo.water_paths,
    ).open(settle_frames=args.settle)

    written = []
    t_render = time.time()
    for index in range(args.frames):
        outputs = camera.get_outputs()
        extra = {}
        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            extra["rgb"] = camera.last_frame.rgb
        elif args.rgb and camera.rgb_problem:
            print(f"  RGB not captured: {camera.rgb_problem}", file=sys.stderr)
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
            extra_planes=extra,
            extra_metadata={
                "scene": pathlib.Path(args.scene).name,
                "camera_tilt_deg": demo.camera_tilt_deg,
                "camera_height_m": demo.camera_height_m,
                "camera_heading_deg": args.heading_deg,
                "horizon_depression_deg": round(horizon_deg, 5),
                "bulk_sst_k": bulk_sst_k,
                "water_geometry": bool(demo.water_paths),
                "vessels": rows,
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
        "ifov_mrad": round(ifov_mrad, 4),
        "sea": {
            "bulk_sst_k": bulk_sst_k,
            "camera_height_m": args.camera_height_m,
            "horizon_depression_deg": round(horizon_deg, 5),
            "wind_m_s": round(wind_m_s, 3),
            "rms_facet_tilt_deg": round(tilt_deg, 3),
            "profile_deg": [round(float(d), 3) for d in probe],
            "profile_k": [round(float(t), 3) for t in np.atleast_1d(t_probe)],
            "water_geometry": bool(demo.water_paths),
        },
        "vessels": rows,
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
