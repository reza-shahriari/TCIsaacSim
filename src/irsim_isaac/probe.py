"""Isaac Sim gate-spike probes (roadmap M2.1 / M2.2; ADR 0014).

Everything here runs *inside* a live ``SimulationApp``; every engine import is inside a function so
that ``import irsim_isaac.probe`` works on a machine without Isaac Sim (CLAUDE.md non-negotiable #1,
enforced by ``tests/unit/test_layering.py``).

The scene is the §13.3 experiment: a grid of emissive quads whose OmniPBR ``emissive_color``
carries ``c = (T - T_ref) / T_span`` (``irsim.radiometry.encoding``), no lights, read back through
every AOV that could plausibly transport temperature, in the real-time and path-traced render
modes. The report records, per AOV, the dtype the annotator returns, the per-quad readback, the
decoded temperature error against the authored ramp, and whether the values are float16-quantised
(§3.2 precision trap). Two *control* quads are authored through Kit's own material commands so a
silent failure of the hand-authored materials is distinguishable from a renderer limitation.

docs/physics-model.md §13.1, §13.3, §13.6.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from irsim.radiometry.constants import T_ENCODE_REF_K, T_ENCODE_SPAN_K
from irsim.radiometry.encoding import decode_temperature, encode_temperature

__all__ = [
    "AOV_CANDIDATES",
    "EXTENSIONS_OF_INTEREST",
    "RampScene",
    "build_ramp_scene",
    "probe_environment",
    "probe_render_mode",
    "ramp_temperatures",
    "save_report",
    "summarize",
]

EXTENSIONS_OF_INTEREST: tuple[str, ...] = (
    "omni.rtx.spg",
    "isaacsim.sensors.experimental.rtx",
    "isaacsim.sensors.camera",
    "isaacsim.sensors.rtx",
    "omni.replicator.core",
    "omni.syntheticdata",
    "omni.warp.core",
    "omni.sensors.nv.camera",
    "omni.hydra.rtx",
    "omni.usd.schema.omni_lens_distortion",
    "omni.usd.schema.omni_sensors",
)

# AOV annotators to try per render mode. Names are Replicator's AOV-backed annotator names
# (omni.replicator.core annotators_default.py); the dtype recorded there is what the annotator
# *returns*, which is the question — a float32 export of a float16 render var is still float16.
AOV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "rt": (
        "GroundTruthEmission",
        "GroundTruthEmissionAndForegroundMask",
        "EmissionAndForegroundMask",
        "HdrColor",
        "LdrColor",
        "GroundTruthDiffuseAlbedo",
        "DiffuseAlbedo",
        "SmoothNormal",
        "BumpNormal",
        "AmbientOcclusion",
        "GroundTruthAmbientOcclusion",
        "DistanceToCameraSD",
        "DistanceToImagePlaneSD",
        "Motion2d",
    ),
    "pt": (
        "PtSelfIllumination",
        "PtDirectIllumation",
        "PtGlobalIllumination",
        "PtBackground",
        "HdrColor",
        "LdrColor",
        "PtWorldNormal",
        "PtZDepth",
        "DistanceToCameraSD",
        "DistanceToImagePlaneSD",
    ),
}

_SETTINGS_OF_INTEREST: tuple[str, ...] = (
    "/rtx/rendermode",
    "/rtx/spg/enabled",
    "/rtx/pathtracing/spp",
    "/rtx/pathtracing/totalSpp",
    "/rtx/pathtracing/maxBounces",
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
    "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/directLighting/sampledLighting/enabled",
)

_DISTANCE_AOVS = ("DistanceToCameraSD", "DistanceToImagePlaneSD", "PtZDepth")
_ALBEDO_AOVS = ("DiffuseAlbedo", "GroundTruthDiffuseAlbedo")
_NORMAL_AOVS = ("SmoothNormal", "BumpNormal", "PtWorldNormal")


def _try(fn: Any, errors: dict[str, str], key: str) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — a probe records failures, it does not raise
        errors[key] = f"{type(exc).__name__}: {exc}"
        return None


def probe_environment() -> dict[str, Any]:
    """M2.1: versions, extensions, settings, Warp-inside-Kit, sensor API, annotator registry."""
    import carb
    import omni.kit.app

    app = omni.kit.app.get_app()
    mgr = app.get_extension_manager()
    errors: dict[str, str] = {}
    report: dict[str, Any] = {"errors": errors}
    report["kit_version"] = _try(app.get_kit_version, errors, "kit_version")

    def _isaac_version() -> Any:
        from isaacsim.core.version import get_version

        return list(get_version())

    report["isaac_version"] = _try(_isaac_version, errors, "isaac_version")

    exts: dict[str, Any] = {name: {"present": False} for name in EXTENSIONS_OF_INTEREST}
    for ext in mgr.get_extensions():
        name = ext.get("name")
        if name in exts:
            exts[name] = {
                "present": True,
                "version": ext.get("version"),
                "enabled": bool(ext.get("enabled")),
                "path": ext.get("path"),
            }
    report["extensions"] = exts

    settings = carb.settings.get_settings()
    report["settings"] = {key: settings.get(key) for key in _SETTINGS_OF_INTEREST}

    def _warp() -> dict[str, Any]:
        import warp as wp

        wp.init()
        return {
            "version": wp.__version__,
            "file": wp.__file__,
            "cuda_device_count": wp.get_cuda_device_count(),
            "cuda_devices": [str(d) for d in wp.get_cuda_devices()],
        }

    report["warp"] = _try(_warp, errors, "warp")

    def _sensor_api() -> dict[str, bool]:
        import isaacsim.sensors.experimental.rtx as rtx

        names = (
            "RtxCamera",
            "CameraSensor",
            "TiledCameraSensor",
            "SPGNode",
            "SingleViewDepthCameraSensor",
        )
        return {n: hasattr(rtx, n) for n in names}

    report["sensor_api"] = _try(_sensor_api, errors, "sensor_api")

    def _deprecated_imported() -> list[str]:
        import sys

        prefixes = ("isaacsim.sensors.camera", "isaacsim.sensors.rtx")
        return sorted(m for m in sys.modules if m.startswith(prefixes))

    report["deprecated_modules_imported"] = _try(_deprecated_imported, errors, "deprecated")

    def _annotators() -> list[str]:
        import omni.replicator.core as rep

        return sorted(rep.AnnotatorRegistry.get_registered_annotators())

    report["annotators_registered"] = _try(_annotators, errors, "annotators")
    return report


# ----------------------------------------------------------------------------------------------
# Ramp scene
# ----------------------------------------------------------------------------------------------


def ramp_temperatures(n_ramp: int = 57) -> np.ndarray:
    """The authored temperature set: a 200–1000 K ramp plus the §13.3 spot checks.

    300.000 / 300.050 / 300.100 K resolve a 50 mK NETD; 600 K is c = 0.5 exactly (linearity check);
    999.2 K is c = 0.999 (top of range, no clamp); 1100 K is c = 1.125 (out of range — must not be
    silently clamped to 1000 K).
    """
    ramp = np.linspace(200.0, 1000.0, n_ramp)
    spots = np.array([300.0, 300.05, 300.1, 600.0, 999.2, 1100.0, 250.0])
    return np.asarray(np.concatenate([ramp, spots]), dtype=np.float64)


@dataclass
class RampScene:
    camera_path: str
    quad_temperatures: list[float]
    quad_centres: list[tuple[float, float]]  # (x, y) at z = -distance
    distance_m: float
    focal_length_mm: float
    aperture_mm: float
    quad_half_size_m: float
    diffuse_ids: list[int]
    controls: list[dict[str, Any]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def pixel_of(self, centre: tuple[float, float], width: int, height: int) -> tuple[int, int]:
        """Pinhole projection of a point on the quad plane: (column, row) at the resolution."""
        x, y = centre
        f_px = self.focal_length_mm / self.aperture_mm * width
        u = 0.5 * width + f_px * x / self.distance_m
        v = 0.5 * height - f_px * y / self.distance_m
        return int(round(u)), int(round(v))

    def ray_length(self, index: int) -> float:
        x, y = self.quad_centres[index]
        return math.sqrt(x * x + y * y + self.distance_m * self.distance_m)


def _pair_encode(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probe-local coarse/fine split for testing an fp16 channel pair (§13.3 fallback).

    coarse = float16(c) (exactly representable), fine = 0.5 + 1024·(c − coarse). Any fp16 channel
    that reproduces fp16-representable values exactly carries this pair losslessly; the library
    codec in ``irsim.radiometry.encoding`` is validated against the engine in M2.2 proper.
    """
    coarse = c.astype(np.float16).astype(np.float64)
    fine = 0.5 + 1024.0 * (c - coarse)
    return coarse, fine


def _pair_decode(coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
    return coarse + (fine - 0.5) / 1024.0


def _author_quad(stage: Any, path: str, centre: tuple[float, float], half: float, z: float) -> Any:
    from pxr import Gf, UsdGeom

    x, y = centre
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(
        [
            Gf.Vec3f(x - half, y - half, z),
            Gf.Vec3f(x + half, y - half, z),
            Gf.Vec3f(x + half, y + half, z),
            Gf.Vec3f(x - half, y + half, z),
        ]
    )
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetSubdivisionSchemeAttr().Set("none")
    return mesh


def _author_omnipbr_usdshade(
    stage: Any,
    mat_path: str,
    *,
    emissive: tuple[float, float, float] | None,
    emissive_intensity: float,
    diffuse: tuple[float, float, float],
) -> Any:
    """OmniPBR through UsdShade directly (the pattern omni.kit.material.library uses)."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(emissive is not None)
    shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(
        float(emissive_intensity)
    )
    if emissive is not None:
        emis = shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f)
        emis.Set(Gf.Vec3f(*emissive))
        emis.GetAttr().SetColorSpace("raw")
    diff = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
    diff.Set(Gf.Vec3f(*diffuse))
    diff.GetAttr().SetColorSpace("raw")
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _author_omnipbr_commands(
    stage: Any,
    mat_path: str,
    mesh_path: str,
    *,
    emissive: tuple[float, float, float] | None,
    emissive_intensity: float,
    diffuse: tuple[float, float, float],
) -> None:
    """OmniPBR through Kit's own commands (what the Create menu does) — the control path."""
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf

    omni.kit.commands.execute(
        "CreateMdlMaterialPrimCommand", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=mat_path
    )
    shader = stage.GetPrimAtPath(mat_path + "/Shader")
    omni.usd.create_material_input(
        shader, "enable_emission", emissive is not None, Sdf.ValueTypeNames.Bool
    )
    omni.usd.create_material_input(
        shader, "emissive_intensity", float(emissive_intensity), Sdf.ValueTypeNames.Float
    )
    if emissive is not None:
        omni.usd.create_material_input(
            shader, "emissive_color", Gf.Vec3f(*emissive), Sdf.ValueTypeNames.Color3f
        )
    omni.usd.create_material_input(
        shader, "diffuse_color_constant", Gf.Vec3f(*diffuse), Sdf.ValueTypeNames.Color3f
    )
    omni.kit.commands.execute("BindMaterialCommand", prim_path=mesh_path, material_path=mat_path)


def build_ramp_scene(
    temperatures_k: np.ndarray | None = None,
    *,
    grid: int = 9,
    distance_m: float = 2.0,
    focal_length_mm: float = 24.0,
    aperture_mm: float = 20.955,
    pitch_m: float = 0.19,
    quad_half_size_m: float = 0.08,
    emissive_intensity: float = 1.0,
    export_path: str | None = None,
) -> RampScene:
    """Author the ramp scene: camera at the origin looking -Z, grid² emissive quads, no lights.

    Each ramp quad's OmniPBR material: ``enable_emission``, ``emissive_intensity`` (the MDL default
    is 40), ``emissive_color = (c, coarse, fine)`` with colorSpace ``raw``, black diffuse except a
    per-quad ``diffuse_color_constant.r = id/255`` for the material-id transport check, rough and
    non-metallic so no quad reflects another. The two grid slots after the ramp hold the control
    quads (Kit-command-authored: emissive grey 0.5, and non-emissive red).
    """
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade

    temps = (
        ramp_temperatures()
        if temperatures_k is None
        else np.asarray(temperatures_k, dtype=np.float64)
    )
    if temps.size + 2 > grid * grid:
        raise ValueError(f"{temps.size} temperatures + 2 controls do not fit a {grid}x{grid} grid")
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Looks", "Scope")
    stage.DefinePrim("/World/Quads", "Xform")

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.GetFocalLengthAttr().Set(focal_length_mm)
    cam.GetHorizontalApertureAttr().Set(aperture_mm)
    cam.GetVerticalApertureAttr().Set(aperture_mm)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))

    c_plain = encode_temperature(temps.astype(np.float32)).astype(np.float64)
    coarse, fine = _pair_encode(c_plain)
    origin = -0.5 * pitch_m * (grid - 1)

    def centre_of(slot: int) -> tuple[float, float]:
        return (origin + pitch_m * (slot % grid), origin + pitch_m * (slot // grid))

    centres: list[tuple[float, float]] = []
    ids: list[int] = []
    errors: dict[str, str] = {}
    z = -distance_m
    for i in range(temps.size):
        centre = centre_of(i)
        centres.append(centre)
        ids.append(i + 1)  # 0 stays free for an UNMAPPED sentinel test later
        mesh = _author_quad(stage, f"/World/Quads/Quad_{i:02d}", centre, quad_half_size_m, z)
        material = _author_omnipbr_usdshade(
            stage,
            f"/World/Looks/Emissive_{i:02d}",
            emissive=(float(c_plain[i]), float(coarse[i]), float(fine[i])),
            emissive_intensity=emissive_intensity,
            diffuse=(ids[-1] / 255.0, 0.0, 0.0),
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    controls: list[dict[str, Any]] = []
    for label, slot, emissive, diffuse in (
        ("control_emissive_grey", temps.size, (0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),
        ("control_diffuse_red", temps.size + 1, None, (1.0, 0.0, 0.0)),
    ):
        centre = centre_of(slot)
        mesh_path = f"/World/Quads/{label}"
        _author_quad(stage, mesh_path, centre, quad_half_size_m, z)
        try:
            _author_omnipbr_commands(
                stage,
                f"/World/Looks/{label}",
                mesh_path,
                emissive=emissive,
                emissive_intensity=emissive_intensity,
                diffuse=diffuse,
            )
        except Exception as exc:  # noqa: BLE001
            errors[label] = f"{type(exc).__name__}: {exc}"
        controls.append(
            {"label": label, "centre": centre, "emissive": emissive, "diffuse": diffuse}
        )

    if export_path:
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
        stage.GetRootLayer().Export(export_path)

    return RampScene(
        camera_path="/World/Camera",
        quad_temperatures=[float(t) for t in temps],
        quad_centres=centres,
        distance_m=distance_m,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        quad_half_size_m=quad_half_size_m,
        diffuse_ids=ids,
        controls=controls,
        errors=errors,
    )


# ----------------------------------------------------------------------------------------------
# Readback
# ----------------------------------------------------------------------------------------------


def _as_array(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("data", data)
    if hasattr(data, "numpy"):
        data = data.numpy()
    try:
        return np.asarray(data)
    except Exception:  # noqa: BLE001
        return None


def _sample_points(
    arr: np.ndarray, scene: RampScene, centres: list[tuple[float, float]], window: int = 5
) -> np.ndarray:
    """Median over a small window at each projected centre; shape (n, channels)."""
    h, w = arr.shape[0], arr.shape[1]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    out = np.full((len(centres), ch), np.nan)
    r = window // 2
    for i, centre in enumerate(centres):
        u, v = scene.pixel_of(centre, w, h)
        if not (r <= u < w - r and r <= v < h - r):
            continue
        patch = arr[v - r : v + r + 1, u - r : u + r + 1].reshape(-1, ch).astype(np.float64)
        out[i] = np.median(patch, axis=0)
    return out


def _decode_analysis(vals: np.ndarray, scene: RampScene) -> dict[str, Any]:
    """Temperature-transport metrics for an AOV whose channel 0 should carry c = encode(T)."""
    temps = np.asarray(scene.quad_temperatures)
    in_range = temps <= 1000.0
    c0 = vals[:, 0]
    c_authored = encode_temperature(temps.astype(np.float32)).astype(np.float64)
    ok = np.isfinite(c0) & in_range
    if not ok.any():
        return {"usable": False}
    decoded = decode_temperature(np.clip(c0[ok], 0.0, 1.0).astype(np.float32)).astype(np.float64)
    err_k = decoded - temps[ok]
    c16 = c_authored[ok].astype(np.float16).astype(np.float64)
    fp16_exact = np.isclose(c0[ok], c16, rtol=0, atol=1e-9)
    fp32_exact = np.isclose(c0[ok], c_authored[ok], rtol=0, atol=1e-7)
    slope = float(np.polyfit(c_authored[ok], c0[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    result: dict[str, Any] = {
        "usable": True,
        "n_quads": int(ok.sum()),
        "slope_readback_vs_authored": slope,
        "max_abs_err_mk": float(np.max(np.abs(err_k)) * 1e3),
        "rms_err_mk": float(np.sqrt(np.mean(err_k**2)) * 1e3),
        "fraction_exactly_fp16_rounded": float(fp16_exact.mean()),
        "fraction_exactly_fp32": float(fp32_exact.mean()),
        "spot_checks_mk": {},
    }
    for spot in (300.0, 300.05, 300.1, 600.0, 999.2, 250.0):
        j = int(np.argmin(np.abs(temps - spot)))
        if np.isfinite(c0[j]):
            dec = decode_temperature(np.array([np.clip(c0[j], 0, 1)], dtype=np.float32))[0]
            result["spot_checks_mk"][str(spot)] = float((float(dec) - temps[j]) * 1e3)
    j = int(np.argmax(temps))  # the 1100 K quad
    result["out_of_range_readback_c"] = None if not np.isfinite(c0[j]) else float(c0[j])
    result["out_of_range_expected_c"] = float((1100.0 - T_ENCODE_REF_K) / T_ENCODE_SPAN_K)
    if vals.shape[1] >= 3:
        pair = _pair_decode(vals[:, 1], vals[:, 2])
        pair_ok = ok & np.isfinite(pair)
        if pair_ok.any():
            dec_pair = decode_temperature(np.clip(pair[pair_ok], 0.0, 1.0).astype(np.float32))
            err_pair = dec_pair.astype(np.float64) - temps[pair_ok]
            result["fp16_pair_max_abs_err_mk"] = float(np.max(np.abs(err_pair)) * 1e3)
    return result


def _distance_analysis(vals: np.ndarray, scene: RampScene) -> dict[str, Any]:
    d = vals[:, 0]
    ok = np.isfinite(d)
    if not ok.any():
        return {"usable": False}
    ray = np.array([scene.ray_length(i) for i in range(len(scene.quad_temperatures))])
    plane = np.full_like(ray, scene.distance_m)
    return {
        "usable": True,
        "mean_abs_err_vs_ray_length_m": float(np.mean(np.abs(d[ok] - ray[ok]))),
        "mean_abs_err_vs_plane_depth_m": float(np.mean(np.abs(d[ok] - plane[ok]))),
        "min": float(np.nanmin(d)),
        "max": float(np.nanmax(d)),
    }


def _albedo_analysis(vals: np.ndarray, scene: RampScene, dtype: str) -> dict[str, Any]:
    ids = np.asarray(scene.diffuse_ids, dtype=np.float64)
    r = vals[:, 0]
    ok = np.isfinite(r)
    if not ok.any():
        return {"usable": False}
    expected = ids / 255.0 if not dtype.startswith("uint") else ids
    return {
        "usable": True,
        "fraction_exact_id": float(np.mean(np.isclose(r[ok], expected[ok], rtol=0, atol=1e-6))),
        "max_abs_err": float(np.max(np.abs(r[ok] - expected[ok]))),
        "first_values": [float(v) for v in r[:6]],
    }


def probe_render_mode(
    scene: RampScene,
    mode: str,
    *,
    resolution: int = 256,
    settle_frames: int = 20,
    rt_subframes: int = 1,
    aovs: tuple[str, ...] | None = None,
    pt_max_bounces: int | None = None,
    disable_aa: bool = True,
    dump_dir: str | None = None,
) -> dict[str, Any]:
    """Render the ramp scene in one mode and read every candidate AOV back through its annotator."""
    import carb
    import omni.replicator.core as rep

    settings = carb.settings.get_settings()
    if mode.startswith("pt"):
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 32)
        settings.set("/rtx/pathtracing/totalSpp", 32)
        if pt_max_bounces is not None:
            settings.set("/rtx/pathtracing/maxBounces", int(pt_max_bounces))
        candidates = AOV_CANDIDATES["pt"]
    else:
        # Kit 110 names the real-time renderer "RealTimePathTracing"; "RaytracedLighting" is
        # silently ignored (run1, ADR 0014). Leave the build default and record it.
        candidates = AOV_CANDIDATES["rt"]
    settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
    if disable_aa:
        # DLSS/AA renders internally at reduced resolution (128² for a 256² product in run1) and
        # upscales; several AOVs then come back at the internal resolution. Off for measurements.
        settings.set("/rtx/post/aa/op", 0)
    if aovs is not None:
        candidates = aovs

    report: dict[str, Any] = {
        "mode": mode,
        "resolution": resolution,
        "settings": {key: settings.get(key) for key in _SETTINGS_OF_INTEREST},
        "aovs": {},
        "timing_s": {},
    }
    t0 = time.time()
    rp = rep.create.render_product(scene.camera_path, (resolution, resolution))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    for _ in range(settle_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)
    report["timing_s"]["settle"] = round(time.time() - t0, 2)
    control_centres = [tuple(c["centre"]) for c in scene.controls]

    for name in candidates:
        entry: dict[str, Any] = {}
        report["aovs"][name] = entry
        t1 = time.time()
        try:
            anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            anno.attach(rp_path)
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "attach_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            for _ in range(4):
                rep.orchestrator.step(rt_subframes=rt_subframes)
            arr = _as_array(anno.get_data())
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "read_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            arr = None
        try:
            anno.detach(rp_path)
        except Exception as exc:  # noqa: BLE001
            entry["detach_error"] = f"{type(exc).__name__}: {exc}"
        entry["timing_s"] = round(time.time() - t1, 2)
        if arr is None or arr.size == 0:
            entry.setdefault("status", "no_data")
            continue
        entry["status"] = "ok"
        entry["dtype"] = str(arr.dtype)
        entry["shape"] = list(arr.shape)
        with contextlib.suppress(TypeError, ValueError):
            entry["frame_min_max"] = [float(np.nanmin(arr)), float(np.nanmax(arr))]
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, f"{mode}_{name}.npy"), arr)
        if arr.ndim < 2 or arr.shape[0] != resolution:
            entry["note"] = "not an image of the render resolution; skipped quad analysis"
            continue
        vals = _sample_points(arr, scene, scene.quad_centres)
        entry["quad_values_ch0"] = [None if not np.isfinite(v) else float(v) for v in vals[:, 0]]
        if control_centres:
            cvals = _sample_points(arr, scene, control_centres)
            entry["control_values"] = {
                c["label"]: [None if not np.isfinite(v) else float(v) for v in cvals[k]]
                for k, c in enumerate(scene.controls)
            }
        if name in _DISTANCE_AOVS:
            entry["distance"] = _distance_analysis(vals, scene)
        elif name in _ALBEDO_AOVS:
            entry["albedo"] = _albedo_analysis(vals, scene, entry["dtype"])
        elif name in _NORMAL_AOVS:
            n = vals[:, : min(3, vals.shape[1])]
            entry["normal_mean"] = [float(x) for x in np.nanmean(n, axis=0)]
        else:
            entry["decode"] = _decode_analysis(vals, scene)
    report["timing_s"]["total"] = round(time.time() - t0, 2)
    return report


def summarize(report: dict[str, Any]) -> str:
    lines = [f"mode={report['mode']} res={report['resolution']} settings={report['settings']}"]
    for name, e in report["aovs"].items():
        s = f"  {name:38s} {e.get('status'):12s} {e.get('dtype', ''):8s}"
        s += f" {str(e.get('shape', '')):18s}"
        mm = e.get("frame_min_max")
        if mm:
            s += f" range=[{mm[0]:.4g},{mm[1]:.4g}]"
        d = e.get("decode")
        if d and d.get("usable"):
            spots = {k: round(v, 2) for k, v in d["spot_checks_mk"].items()}
            s += f" slope={d['slope_readback_vs_authored']:.4f} maxErr={d['max_abs_err_mk']:.3f} mK"
            s += f" fp16={d['fraction_exactly_fp16_rounded']:.2f}"
            s += f" fp32={d['fraction_exactly_fp32']:.2f}"
            s += f" spots={spots} oor={d['out_of_range_readback_c']}"
            if "fp16_pair_max_abs_err_mk" in d:
                s += f" pair={d['fp16_pair_max_abs_err_mk']:.3f} mK"
        if "distance" in e and e["distance"].get("usable"):
            dist = e["distance"]
            s += f" ray={dist['mean_abs_err_vs_ray_length_m']:.4f}"
            s += f" plane={dist['mean_abs_err_vs_plane_depth_m']:.4f}"
        if "albedo" in e and e["albedo"].get("usable"):
            alb = e["albedo"]
            s += f" idExact={alb['fraction_exact_id']:.2f} first={alb['first_values']}"
        if "normal_mean" in e:
            s += f" n={e['normal_mean']}"
        if "control_values" in e:
            ctl = {
                k[:12]: [None if x is None else round(x, 5) for x in v[:3]]
                for k, v in e["control_values"].items()
            }
            s += f" ctl={ctl}"
        if "error" in e:
            s += f" {e['error'][:120]}"
        lines.append(s)
    return "\n".join(lines)


def save_report(report: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return str(obj)


# ----------------------------------------------------------------------------------------------
# Id + geometry transport (the float32 alternative to temperature-in-emission)
# ----------------------------------------------------------------------------------------------

_ID_ANNOTATORS: tuple[str, ...] = (
    "instance_segmentation",
    "semantic_segmentation",
    "instance_id_segmentation",
)
_POSITION_AOVS: tuple[str, ...] = (
    "PtWorldPos",
    "Camera3dPositionSD",
    "PtWorldNormal",
    "DistanceToCameraSD",
)


def label_quads(scene: RampScene) -> dict[str, str]:
    """Apply a semantic label quad_NN to every ramp quad (semantic-id transport test)."""
    import omni.usd

    errors: dict[str, str] = {}
    stage = omni.usd.get_context().get_stage()
    for i in range(len(scene.quad_temperatures)):
        prim = stage.GetPrimAtPath(f"/World/Quads/Quad_{i:02d}")
        label = f"quad_{i:02d}"
        try:
            from isaacsim.core.experimental.utils.semantics import add_labels

            add_labels(prim, labels=[label], taxonomy="class")
        except Exception as exc:  # noqa: BLE001
            errors[f"{label}:add_labels"] = f"{type(exc).__name__}: {exc}"
            try:  # legacy schema, the pattern omni.replicator.core.utils uses
                from pxr import Semantics

                sem = Semantics.SemanticsAPI.Apply(prim, f"class_{label}")
                sem.CreateSemanticTypeAttr().Set("class")
                sem.CreateSemanticDataAttr().Set(label)
            except Exception as exc2:  # noqa: BLE001
                errors[f"{label}:legacy"] = f"{type(exc2).__name__}: {exc2}"
    return errors


def probe_id_transport(
    scene: RampScene,
    mode: str,
    *,
    resolution: int = 256,
    settle_frames: int = 20,
    rt_subframes: int = 1,
    pt_max_bounces: int = 1,
) -> dict[str, Any]:
    """Segmentation ids (exact integers?) and float32 positions at every quad centre."""
    import carb
    import omni.replicator.core as rep

    settings = carb.settings.get_settings()
    if mode.startswith("pt"):
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 32)
        settings.set("/rtx/pathtracing/totalSpp", 32)
        settings.set("/rtx/pathtracing/maxBounces", int(pt_max_bounces))
    settings.set("/rtx/post/aa/op", 0)
    report: dict[str, Any] = {"mode": mode, "resolution": resolution, "ids": {}, "positions": {}}
    rp = rep.create.render_product(scene.camera_path, (resolution, resolution))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    for _ in range(settle_frames):
        rep.orchestrator.step(rt_subframes=rt_subframes)
    n = len(scene.quad_temperatures)
    w = h = resolution

    for name in _ID_ANNOTATORS:
        entry: dict[str, Any] = {}
        report["ids"][name] = entry
        try:
            anno = rep.AnnotatorRegistry.get_annotator(
                name, init_params={"colorize": False}, device="cpu"
            )
            anno.attach(rp_path)
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=rt_subframes)
            raw = anno.get_data()
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        with contextlib.suppress(Exception):
            anno.detach(rp_path)
        data = raw.get("data") if isinstance(raw, dict) else raw
        info = raw.get("info", {}) if isinstance(raw, dict) else {}
        arr = _as_array(data)
        if arr is None or arr.ndim < 2:
            entry["status"] = "no_data"
            continue
        arr2 = arr if arr.ndim == 2 else arr[..., 0]
        entry["status"] = "ok"
        entry["dtype"] = str(arr.dtype)
        entry["shape"] = list(arr.shape)
        centre_ids = []
        for i in range(n):
            u, v = scene.pixel_of(scene.quad_centres[i], w, h)
            centre_ids.append(int(arr2[v, u]))
        # purity: the 5x5 window around each centre must contain a single id
        pure = 0
        for i in range(n):
            u, v = scene.pixel_of(scene.quad_centres[i], w, h)
            patch = arr2[v - 2 : v + 3, u - 2 : u + 3]
            pure += int(np.unique(patch).size == 1)
        # edge purity: along the row through quad centres, every pixel is either a quad id or bg
        background = int(np.bincount(arr2.ravel().astype(np.int64)).argmax())
        distinct = np.unique(arr2)
        entry["n_distinct_ids_in_frame"] = int(distinct.size)
        entry["background_id"] = background
        entry["centre_ids_unique"] = bool(len(set(centre_ids)) == n)
        entry["centre_ids_nonbackground"] = int(sum(1 for c in centre_ids if c != background))
        entry["centre_windows_pure"] = pure
        entry["first_centre_ids"] = centre_ids[:6]
        id_to_labels = info.get("idToLabels") if isinstance(info, dict) else None
        if id_to_labels:
            entry["id_to_labels_sample"] = {str(k): id_to_labels[k] for k in list(id_to_labels)[:4]}
            entry["n_labels"] = len(id_to_labels)

    for name in _POSITION_AOVS:
        entry = {}
        report["positions"][name] = entry
        try:
            anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            anno.attach(rp_path)
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=rt_subframes)
            arr = _as_array(anno.get_data())
        except Exception as exc:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            continue
        with contextlib.suppress(Exception):
            anno.detach(rp_path)
        if arr is None or arr.ndim < 2 or arr.shape[0] != resolution:
            entry["status"] = "no_data" if arr is None else "wrong_resolution"
            entry["shape"] = None if arr is None else list(arr.shape)
            continue
        entry["status"] = "ok"
        entry["dtype"] = str(arr.dtype)
        entry["shape"] = list(arr.shape)
        vals = _sample_points(arr, scene, scene.quad_centres)
        expected = np.array([[x, y, -scene.distance_m] for (x, y) in scene.quad_centres])
        if vals.shape[1] >= 3 and name in ("PtWorldPos", "Camera3dPositionSD"):
            err = vals[:, :3] - expected
            ok = np.isfinite(err).all(axis=1)
            if ok.any():
                entry["mean_abs_err_m_xyz"] = [float(v) for v in np.mean(np.abs(err[ok]), axis=0)]
                entry["max_abs_err_m"] = float(np.max(np.abs(err[ok])))
                entry["first_values"] = [
                    [round(float(x), 5) for x in vals[i, :3]] for i in range(3)
                ]
        elif name == "PtWorldNormal":
            entry["mean_normal"] = [float(v) for v in np.nanmean(vals[:, :3], axis=0)]
        else:
            entry["first_values"] = [round(float(v), 5) for v in vals[:3, 0]]
    return report
