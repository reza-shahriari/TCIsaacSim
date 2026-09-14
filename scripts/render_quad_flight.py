#!/usr/bin/env python3
"""Roadmap M10.20: film a heavy-lift quadrotor flying a mission, in LWIR (ADR 0074).

One resolved multirotor at 20 m -- 105 px across the span, 6 px per motor bell -- against sky,
through the whole camera model, while ADR 0072's heat sources take the motors from ambient to
+45 K and back down over a 30-minute flight. The four hot bells, the warm speed controllers and
the warm battery are separate objects in the image, which is what a thermal sensor actually keys
on in a drone.

**This is a time-lapse, and that is a physics decision, not a shortcut.** The node law is
T = T_air(t) + dT_max u(t)^2 -- a steady-state relation with no thermal time constant (ADR 0072).
It is only defensible where the throttle moves slowly against a real motor's minutes-scale
response, so the flight is authored over 1800 s and the camera takes one frame every few seconds
of it. Every stage is told the truth about the interval, so the FFC fires on its real schedule and
the temporal noise decorrelates between frames exactly as it would in a real time-lapse. Running
the same profile at 60 Hz would look smoother and would be a lie about what metal can do.

Run with the Isaac interpreter (docs/decisions/0002), after `make luts`:
    python.sh scripts/render_quad_flight.py --frames 300 --rgb
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/quad_flight")
parser.add_argument("--frames", type=int, default=300)
parser.add_argument("--sensor", default=str(REPO / "configs/sensors/flir_boson_640_lwir.yaml"))
parser.add_argument("--scene", default=str(REPO / "configs/scenes/quad_flight_clear_noon.yaml"))
parser.add_argument(
    "--interval-s",
    type=float,
    default=6.0,
    help="scene seconds per captured frame. frames x interval must fit inside the flight profile",
)
parser.add_argument("--fps", type=float, default=30.0, help="playback rate of the encoded video")
parser.add_argument("--range-m", type=float, default=20.0)
parser.add_argument("--tilt-deg", type=float, default=15.0)
parser.add_argument("--heading-deg", type=float, default=110.0)
parser.add_argument("--settle", type=int, default=16)
parser.add_argument("--rt-subframes", type=int, default=8)
parser.add_argument("--rgb", action="store_true", help="also film the companion visible frame")
parser.add_argument("--no-chain", action="store_true", help="ideal camera: no M9 sensor chain")
parser.add_argument("--no-dome", action="store_true", help="untextured grey dome (ADR 0073)")
parser.add_argument(
    "--span-c",
    type=float,
    nargs=2,
    default=None,
    metavar=("LO", "HI"),
    help="fixed apparent-temperature span of the main video, Celsius. Default: air +/- 50",
)
parser.add_argument("--palette", default="ironbow", help="palette for the fixed-span video")
parser.add_argument("--no-overlay", action="store_true", help="no burnt-in readout")
parser.add_argument("--keep-frames", action="store_true", help="keep the PNG sequence")
args = parser.parse_args()

t_boot = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
boot_s = time.time() - t_boot


def throttle_at(scene: object, t_rel_s: float) -> float:
    """The flight profile's throttle at a scene-relative time, read back off the scene config."""
    import numpy as np

    for target in scene.spec.targets:  # type: ignore[attr-defined]
        if target.solver == "heat_source" and target.throttle_s:
            return float(np.interp(t_rel_s, target.throttle_s, target.throttle))
    return 0.0


def decorate(
    image: Any,
    caption: str,
    t_rel_s: float,
    throttle: float,
    temps: dict,
    span_k: tuple,
    interval_s: float,
    bare: bool,
) -> Any:
    """One frame with the readout burnt in: clock, throttle, the node temperatures, the mapping.

    The caption names which mapping the picture is under, because the two videos are the same
    scene and differ only in that -- and telling them apart afterwards from the image alone is
    exactly the mistake the pair exists to prevent.
    """
    import numpy as np

    from irsim_eval.video import annotate, temperature_bar

    if bare:
        return np.ascontiguousarray(np.asarray(image)[..., :3])
    minutes, seconds = divmod(int(t_rel_s), 60)
    out = annotate(
        image,
        [
            f"T+{minutes:02d}:{seconds:02d}   throttle {throttle * 100:3.0f}%",
            f"air {temps['airframe'] - 273.15:5.1f}C   motor {temps['motor'] - 273.15:5.1f}C",
            f"1 frame / {interval_s:g} s time-lapse   {caption}",
        ],
    )
    out = temperature_bar(
        out,
        {
            "motor  ": temps["motor"],
            "esc    ": temps["esc"],
            "battery": temps["battery"],
            "air    ": temps["airframe"],
        },
        span_k,
    )
    return np.ascontiguousarray(out[..., :3])


def main() -> int:
    import numpy as np

    from irsim.config.loader import band_hash, config_hash, load_sensor_config
    from irsim.io.png import write_png
    from irsim.isp.palette import palette_table, quantise_display
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable
    from irsim.pipeline.core import PipelineConfig
    from irsim.radiometry.lut_files import load_band_lut_for_config
    from irsim.scene import Scene
    from irsim_eval.video import encode_mp4, ffmpeg_available
    from irsim_isaac.pipeline.ir_camera import IrCamera
    from irsim_isaac.pipeline.materials_usd import prim_records
    from irsim_isaac.quad_flight import build_quad_flight, tracking_pose
    from irsim_isaac.visible_sky import dome_spec_from_scene

    out_dir = pathlib.Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    sensor = load_sensor_config(args.sensor)
    spec = sensor.sensor
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    scene = Scene.from_file(args.scene, {spec.band.band_id: lut})

    span_s = args.frames * args.interval_s
    print(
        f"\n{spec.name}: {spec.fpa.width}x{spec.fpa.height}, "
        f"{args.frames} frames every {args.interval_s:g} s = {span_s:.0f} s of flight, "
        f"played at {args.fps:g} fps -> {args.frames / args.fps:.1f} s of video"
    )

    dome = None
    if not args.no_dome:
        dome = dome_spec_from_scene(scene, heading_deg=args.heading_deg)
        print(
            f"environment: sun {dome.sun_elevation_deg:.1f} deg elevation, "
            f"{dome.sun_azimuth_deg - args.heading_deg:+.1f} deg off boresight, "
            f"turbidity {dome.turbidity:.2f}"
        )
    stage = build_quad_flight(
        camera_tilt_deg=args.tilt_deg,
        range_m=args.range_m,
        dome=dome,
        dome_texture_path=out_dir / "env_dome.exr",
    )
    if stage.errors:
        print(f"stage errors: {stage.errors}", file=sys.stderr)

    ifov_mrad = 1e3 * spec.fpa.pitch_um * 1e-3 / spec.optics.focal_length_mm
    across = stage.pixels_across(ifov_mrad)
    print(
        f"at {stage.range_m:.0f} m: span "
        f"{1e3 * stage.spec.span_m / stage.range_m / ifov_mrad:.0f} px, "
        f"motor {across['motor_0']:.1f} px, esc {across['esc_0']:.1f} px, "
        f"battery {across['battery']:.1f} px"
    )
    missing = set(stage.thermal_nodes()) - set(scene.targets)
    if missing:
        print(f"scene has no solver for: {sorted(missing)}", file=sys.stderr)
        return 1

    table = MaterialTable.from_library(MaterialLibrary.load(), spec.band.band_id)
    resolver = MaterialResolver(load_mapping_rules(), list(table.names))
    resolutions = resolver.resolve_all(prim_records(root="/World/Targets"))
    unresolved = [r.path for r in resolutions if not r.mapped]
    if unresolved:
        print(f"unmapped prims: {unresolved}", file=sys.stderr)

    pipeline = PipelineConfig.from_sensor(
        sensor, table, lut, sky=scene.sky_models[spec.band.band_id], atmosphere=scene.layered
    )
    if not args.no_chain:
        from irsim.pipeline.sensor_chain import attach_sensor_chain

        pipeline = attach_sensor_chain(pipeline, scene.weather, t0_s=scene.t0_s)

    camera = IrCamera(
        sensor,
        scene,
        pipeline=pipeline,
        prim_to_target=stage.prim_to_target,
        resolutions=resolutions,
        camera_path=stage.camera_path,
        capture_rgb=args.rgb,
        strict_materials=False,
        # One capture every `interval_s` of scene time: a time-lapse camera, with every stage
        # told the truth about the gap (ADR 0074).
        frame_period_s=args.interval_s,
    ).open(settle_frames=args.settle)

    import omni.usd
    from pxr import Gf, UsdGeom

    usd_stage = omni.usd.get_context().get_stage()
    quad = UsdGeom.Xformable(usd_stage.GetPrimAtPath(stage.quad_path))
    translate_op, rotate_op = quad.GetOrderedXformOps()[:2]

    # **The main video is a fixed span, not the camera's AGC, and that is the whole point.**
    # Both of §11.3's AGC modes rescale themselves from the current frame's own histogram, so a
    # target that warms 40 K over a flight comes out looking identical in every frame -- the gain
    # follows the target and cancels exactly the change the video exists to show. Worse, plateau
    # equalisation allocates display codes by population, and an aircraft covering under 1 % of
    # the frame gets almost none of them: the whole airframe saturates to flat white and the
    # motors are indistinguishable from the arms. A real operator switches to a manual span for
    # this, and so do we: one stated mapping from apparent temperature to colour, held for every
    # frame, so two frames are comparable. The camera's own AGC output is filmed alongside it.
    air_c = scene.weather_at(0.0).t_air_k - 273.15
    # A symmetric +/-50 K window on ambient. The floor has to sit well below air temperature or
    # the carbon airframe -- which forced convection holds *at* air temperature -- maps to the
    # bottom of the palette and vanishes into the sky, leaving four hot dots and no aircraft. At
    # +/-50 K the clear sky (about 45 K below ambient at these elevations) still reads as sky, the
    # airframe sits mid-palette where its shape is legible, and the motors have the whole upper
    # half to climb through. Every number here is a display choice and none of it is a
    # measurement: the gauge and the sidecar carry the temperatures.
    span_c = tuple(args.span_c) if args.span_c else (air_c - 50.0, air_c + 50.0)
    if span_c[1] <= span_c[0]:
        print("--span-c must be increasing", file=sys.stderr)
        return 1
    span_k = (span_c[0] + 273.15, span_c[1] + 273.15)
    palette = palette_table(args.palette)
    print(
        f"display: fixed span {span_c[0]:.1f} to {span_c[1]:.1f} C ({args.palette}); "
        f"the camera's own {spec.isp.agc} output is filmed alongside as agc_*"
    )
    history: list[dict[str, float]] = []
    t_render = time.time()
    for index in range(args.frames):
        t_rel = camera.t_rel_s
        throttle = throttle_at(scene, t_rel)
        translate, rotate = tracking_pose(
            t_rel, throttle, range_m=stage.range_m, camera_tilt_deg=stage.camera_tilt_deg
        )
        translate_op.Set(Gf.Vec3d(*translate))
        rotate_op.Set(Gf.Vec3f(*rotate))

        outputs = camera.get_outputs(rt_subframes=args.rt_subframes)
        temps = camera.bridge.temperatures()
        t_app = np.asarray(outputs.apparent_t)
        history.append(
            {
                "frame": index,
                "t_rel_s": round(t_rel, 3),
                "throttle": round(throttle, 4),
                **{f"{k}_k": round(v, 3) for k, v in temps.items()},
                "t_app_min_k": round(float(t_app.min()), 3),
                "t_app_max_k": round(float(t_app.max()), 3),
            }
        )

        # Fixed span -> palette. `quantise_display` is the ISP's own rounding, so the mapping is
        # the one the display branch uses and not a second, subtly different one.
        scaled = (t_app - span_k[0]) / (span_k[1] - span_k[0])
        spanned = palette[quantise_display(scaled)]

        write_png(
            frames_dir / f"ir_{index:05d}.png",
            decorate(
                spanned,
                f"span {span_c[0]:.0f}-{span_c[1]:.0f}C",
                t_rel,
                throttle,
                temps,
                span_k,
                args.interval_s,
                args.no_overlay,
            ),
        )
        write_png(
            frames_dir / f"agc_{index:05d}.png",
            decorate(
                np.asarray(outputs.display8),
                f"camera {spec.isp.agc}",
                t_rel,
                throttle,
                temps,
                span_k,
                args.interval_s,
                args.no_overlay,
            ),
        )

        if args.rgb and camera.last_frame is not None and camera.last_frame.rgb is not None:
            write_png(
                frames_dir / f"rgb_{index:05d}.png",
                np.ascontiguousarray(np.asarray(camera.last_frame.rgb)[..., :3]),
            )
        if index % 25 == 0 or index == args.frames - 1:
            print(
                f"  frame {index:4d}  T+{t_rel:7.1f}s  u={throttle:4.2f}  "
                f"motor {temps['motor'] - 273.15:5.1f}C  "
                f"T_app {t_app.min():.1f}..{t_app.max():.1f} K"
            )
    render_s = time.time() - t_render
    camera.close()

    videos = {}
    if ffmpeg_available():
        videos["ir"] = str(
            encode_mp4(str(frames_dir / "ir_*.png"), out_dir / "quad_flight_ir.mp4", fps=args.fps)
        )
        videos["agc"] = str(
            encode_mp4(str(frames_dir / "agc_*.png"), out_dir / "quad_flight_agc.mp4", fps=args.fps)
        )
        if args.rgb and any(frames_dir.glob("rgb_*.png")):
            videos["rgb"] = str(
                encode_mp4(
                    str(frames_dir / "rgb_*.png"), out_dir / "quad_flight_rgb.mp4", fps=args.fps
                )
            )
        if not args.keep_frames:
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
    else:
        print("ffmpeg not found: keeping the PNG sequence", file=sys.stderr)

    peak = max(history, key=lambda row: row["motor_k"])
    summary = {
        "boot_s": round(boot_s, 2),
        "render_s": round(render_s, 2),
        "frames": args.frames,
        "interval_s": args.interval_s,
        "flight_span_s": span_s,
        "fps": args.fps,
        "sensor": spec.name,
        "scene": pathlib.Path(args.scene).name,
        "config_hash": config_hash(sensor),
        "band_hash": band_hash(sensor),
        "range_m": stage.range_m,
        "ifov_mrad": round(ifov_mrad, 4),
        "pixels_across": {k: round(v, 2) for k, v in across.items()},
        "motor_peak_k": peak["motor_k"],
        "motor_peak_at_s": peak["t_rel_s"],
        "motor_swing_k": round(peak["motor_k"] - min(r["motor_k"] for r in history), 3),
        "display_span_c": list(span_c),
        "palette": args.palette,
        "videos": videos,
        "history": history,
    }
    path = out_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(
        f"\n{args.frames} frames in {render_s:.0f} s. motor peaked at "
        f"{peak['motor_k'] - 273.15:.1f} C at T+{peak['t_rel_s']:.0f} s, "
        f"swing {summary['motor_swing_k']:.1f} K\n{videos}\nwrote {path}"
    )
    return 0


try:
    status = main()
except Exception as exc:  # noqa: BLE001 - a demo crash should still report, not hang
    import traceback

    traceback.print_exc()
    print(f"render failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    app.close(exit_code=status if isinstance(status, int) else 1)
