"""Roadmap M2.3: omni.rtx.spg capability experiments (ADR 0014).

Throwaway shaders authored through ``RtxCamera.author_spg`` / ``SPGNode`` exactly as the shipped
``spg_grayscale.py`` example does, one camera + render product per experiment so a failing graph
cannot take the others down:

* **pass-through** — does an AOV survive an SPG kernel and a float32 annotator readback bit-exactly
  (``LdrColor`` uchar4, ``HdrColor`` half4, ``DistanceToCameraSD`` / ``DistanceToImagePlaneSD``
  R32F, ``Camera3dPositionSD`` / ``SmoothNormal`` RGBA32F);
* **state** — does a ``cuda.static`` device buffer mutated in-kernel survive to the next frame
  (the bolometer IIR and noise drift need this); does a file-scope Lua local; does a feedback edge
  (node output AOV wired back into its own input); what ``rtx.frameId`` does per orchestrator step;
* **LUT** — can a 16 001-entry float table reach the kernel: Lua literal, Lua loop, ``io.open``
  (each in its own file because the sandbox rejects a whole file on one forbidden token), the two
  Lua forms again with the instruction/memory limits raised, and a ``__device__`` array baked into
  the ``.cu`` source;
* **sensor** — does a ``CameraSensor`` render product on a new ``RtxCamera`` see the same
  geometry as a plain Replicator render product (a first run read all-inf depth from it).

Every engine import is inside a function (CLAUDE.md #1). docs/physics-model.md §13.2, §13.5, §13.6.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["run_spg_experiments", "save"]

LUT_N = 16001
LUT_SLOPE = 0.05
LUT_OFFSET = 0.25


def lut_value(i: int) -> float:
    return i * LUT_SLOPE + LUT_OFFSET


_KERNEL_HEAD = """
#define PIXEL_GUARD \\
    int x = blockIdx.x * blockDim.x + threadIdx.x; \\
    int y = blockIdx.y * blockDim.y + threadIdx.y; \\
    if (x >= width || y >= height) return;
"""

PASS_CU = (
    _KERNEL_HEAD
    + """
extern "C" __global__ void pass_u8(
    int width, int height, cudaTextureObject_t inTex, cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    uchar4 v = tex2D<uchar4>(inTex, x, y);
    surf2Dwrite<uchar4>(v, outSurf, x * sizeof(uchar4), y);
}
extern "C" __global__ void pass_f4(
    int width, int height, cudaTextureObject_t inTex, cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    float4 v = tex2D<float4>(inTex, x, y);
    surf2Dwrite<float4>(v, outSurf, x * sizeof(float4), y);
}
extern "C" __global__ void pass_f32(
    int width, int height, cudaTextureObject_t inTex, cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    float v = tex2D<float>(inTex, x, y);
    surf2Dwrite<float>(v, outSurf, x * sizeof(float), y);
}
"""
)

_LUA_LAUNCH = """
local function launch(w, h, args)
    return cuda.kernel({ args = args, block = { 32, 32 },
        grid = { math.ceil(w / 32), math.ceil(h / 32) } })
end
"""

PASS_LUA = (
    _LUA_LAUNCH
    + """
local function describe(tag, r)
    info("SPGPROBE " .. tag .. " shape=" .. tostring(r.shape[1]) .. "x"
        .. tostring(r.shape[2]) .. " dtype=" .. tostring(r.dtype and r.dtype.name)
        .. " buffer=" .. tostring(r.bufferType))
end
local function pass(inputs, outputs, tag, dtype)
    local r = inputs["In"]; describe(tag, r)
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, dtype)
    return launch(w, h, { cuda.int(w), cuda.int(h), cuda.TextureObject(r),
        cuda.SurfaceObject(outputs["Out"]) })
end
function pass_u8(inputs, outputs) return pass(inputs, outputs, "pass_u8", cuda.uchar4) end
function pass_f4(inputs, outputs) return pass(inputs, outputs, "pass_f4", cuda.float4) end
function pass_f32(inputs, outputs) return pass(inputs, outputs, "pass_f32", cuda.float) end
"""
)

STATE_CU = (
    _KERNEL_HEAD
    + """
extern "C" __global__ void counter(
    int width, int height, unsigned int* state, int frameId, int luaCalls,
    cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    unsigned int v;
    if (x == 0 && y == 0) { v = atomicAdd(state, 1u) + 1u; } else { v = *state; }
    float4 o = make_float4((float)v, (float)frameId, (float)luaCalls, 0.0f);
    surf2Dwrite<float4>(o, outSurf, x * sizeof(float4), y);
}
extern "C" __global__ void counter_empty(
    int width, int height, unsigned int* state, int frameId, int luaCalls,
    cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    unsigned int v;
    if (x == 0 && y == 0) { v = atomicAdd(state, 1u) + 1u; } else { v = *state; }
    float4 o = make_float4((float)v, (float)frameId, (float)luaCalls, 0.0f);
    surf2Dwrite<float4>(o, outSurf, x * sizeof(float4), y);
}
extern "C" __global__ void counter_zeros(
    int width, int height, unsigned int* state, int frameId, int luaCalls,
    cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    unsigned int v;
    if (x == 0 && y == 0) { v = atomicAdd(state, 1u) + 1u; } else { v = *state; }
    float4 o = make_float4((float)v, (float)frameId, (float)luaCalls, 0.0f);
    surf2Dwrite<float4>(o, outSurf, x * sizeof(float4), y);
}
extern "C" __global__ void feedback(
    int width, int height, cudaTextureObject_t prevTex, cudaSurfaceObject_t outSurf)
{
    PIXEL_GUARD
    float p = tex2D<float>(prevTex, 0, 0);
    surf2Dwrite<float>(p + 1.0f, outSurf, x * sizeof(float), y);
}
"""
)

STATE_LUA = (
    _LUA_LAUNCH
    + """
local lua_calls = 0
function counter(inputs, outputs)
    lua_calls = lua_calls + 1
    local r = inputs["In"]
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, cuda.float4)
    local state = cuda.static(function() return cuda.array({ 0 }, cuda.uint) end)
    local fid = -1
    if rtx ~= nil and rtx.frameId ~= nil then fid = rtx.frameId end
    return launch(w, h, { cuda.int(w), cuda.int(h), state, cuda.int(fid),
        cuda.int(lua_calls), cuda.SurfaceObject(outputs["Out"]) })
end
local function counter_with(inputs, outputs, state)
    lua_calls = lua_calls + 1
    local r = inputs["In"]
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, cuda.float4)
    local fid = -1
    if rtx ~= nil and rtx.frameId ~= nil then fid = rtx.frameId end
    return launch(w, h, { cuda.int(w), cuda.int(h), state, cuda.int(fid),
        cuda.int(lua_calls), cuda.SurfaceObject(outputs["Out"]) })
end
function counter_empty(inputs, outputs)
    local state = cuda.static(function() return cuda.empty({ 1 }, cuda.uint) end)
    return counter_with(inputs, outputs, state)
end
function counter_zeros(inputs, outputs)
    local state = cuda.static(function() return cuda.zeros({ 1 }, cuda.uint) end)
    return counter_with(inputs, outputs, state)
end
function feedback(inputs, outputs)
    local r = inputs["In"]
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, cuda.float)
    return launch(w, h, { cuda.int(w), cuda.int(h), cuda.TextureObject(inputs["Prev"]),
        cuda.SurfaceObject(outputs["Out"]) })
end
"""
)


def _lut_cu(fn: str) -> str:
    """The LUT probe kernel; SPG resolves the CUDA symbol by the node's ``sub_identifier``."""
    return (
        _KERNEL_HEAD
        + f"""
extern "C" __global__ void {fn}(
    int width, int height, const float* lut, int n, cudaSurfaceObject_t outSurf)
{{
    PIXEL_GUARD
    float v = (x < n) ? lut[x] : -1.0f;
    if (y == 0 && x == 0) v = (float)n;   // pixel (0,0) reports the table length
    surf2Dwrite<float>(v, outSurf, x * sizeof(float), y);
}}
"""
    )


_LUT_LUA_LAUNCH = (
    _LUA_LAUNCH
    + """
local function lut_launch(inputs, outputs, arr, n)
    local r = inputs["In"]
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, cuda.float)
    return launch(w, h, { cuda.int(w), cuda.int(h), arr, cuda.int(n),
        cuda.SurfaceObject(outputs["Out"]) })
end
"""
)


def _lut_lua_literal(values: list[float], fn: str) -> str:
    rows = [", ".join(f"{v:.6f}" for v in values[i : i + 10]) for i in range(0, len(values), 10)]
    table = ",\n  ".join(rows)
    return (
        _LUT_LUA_LAUNCH
        + f"""
local LUT_LITERAL = {{
  {table}
}}
function {fn}(inputs, outputs)
    local arr = cuda.static(function() return cuda.array(LUT_LITERAL, cuda.float) end)
    return lut_launch(inputs, outputs, arr, #LUT_LITERAL)
end
"""
    )


def _lut_lua_loop(fn: str, n: int = LUT_N) -> str:
    return (
        _LUT_LUA_LAUNCH
        + f"""
function {fn}(inputs, outputs)
    local n = {n}
    local arr = cuda.static(function(count)
        local t = {{}}
        for i = 1, count do t[i] = (i - 1) * {LUT_SLOPE} + {LUT_OFFSET} end
        return cuda.array(t, cuda.float)
    end, n)
    return lut_launch(inputs, outputs, arr, n)
end
"""
    )


def _lut_lua_io(fn: str, lut_path: str) -> str:
    return (
        _LUT_LUA_LAUNCH
        + f"""
function {fn}(inputs, outputs)
    local t = {{ -2.0, -2.0, -2.0 }}
    if io ~= nil then
        t[1] = 1.0
        local f = io.open("{lut_path}", "rb")
        if f ~= nil then
            t[2] = 1.0
            local data = f:read(4); f:close()
            if string.unpack ~= nil and data ~= nil then t[3] = string.unpack("<f", data) end
        end
    end
    local arr = cuda.static(function() return cuda.array(t, cuda.float) end)
    return lut_launch(inputs, outputs, arr, #t)
end
"""
    )


def _lut_cu_baked(values: list[float]) -> str:
    rows = [", ".join(f"{v:.6f}f" for v in values[i : i + 8]) for i in range(0, len(values), 8)]
    table = ",\n  ".join(rows)
    return (
        _KERNEL_HEAD
        + f"""
__device__ const float LUT_BAKED[{len(values)}] = {{
  {table}
}};
extern "C" __global__ void lut_baked(int width, int height, cudaSurfaceObject_t outSurf)
{{
    PIXEL_GUARD
    float v = (x < {len(values)}) ? LUT_BAKED[x] : -1.0f;
    if (y == 0 && x == 0) v = (float){len(values)};
    surf2Dwrite<float>(v, outSurf, x * sizeof(float), y);
}}
"""
    )


LUT_BAKED_LUA = (
    _LUA_LAUNCH
    + """
function lut_baked(inputs, outputs)
    local r = inputs["In"]
    local h, w = r.shape[1], r.shape[2]
    outputs["Out"] = cuda.image(w, h, cuda.float)
    return launch(w, h, { cuda.int(w), cuda.int(h), cuda.SurfaceObject(outputs["Out"]) })
end
"""
)


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _as_np(data: Any) -> NDArray[Any] | None:
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


def _plane(arr: NDArray[Any]) -> NDArray[Any]:
    """First channel of an (H, W[, C]) array as float32."""
    a = arr.astype(np.float32)
    return a.reshape(a.shape[0], a.shape[1], -1)[..., 0]


def _readback(rep: Any, name: str, rp_path: str, steps: int, rt_subframes: int) -> dict[str, Any]:
    """Attach an annotator for an AOV, step, and return the array plus a status entry."""
    entry: dict[str, Any] = {}
    arr = None
    try:
        anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
        anno.attach(rp_path)
        for _ in range(steps):
            rep.orchestrator.step(rt_subframes=rt_subframes)
        arr = _as_np(anno.get_data())
        with contextlib.suppress(Exception):
            anno.detach(rp_path)
    except Exception as exc:  # noqa: BLE001
        entry["status"] = "error"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry
    if arr is None or arr.size == 0:
        entry["status"] = "no_data"
        return entry
    entry["status"] = "ok"
    entry["dtype"] = str(arr.dtype)
    entry["shape"] = list(arr.shape)
    entry["_array"] = arr
    return entry


def _readback_pair(
    rep: Any, names: tuple[str, str], rp_path: str, steps: int, rt_subframes: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read two AOVs from the same rendered frame (colour AOVs vary frame to frame)."""
    entries: list[dict[str, Any]] = [{}, {}]
    annos: list[Any] = []
    try:
        for name in names:
            anno = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            anno.attach(rp_path)
            annos.append(anno)
        for _ in range(steps):
            rep.orchestrator.step(rt_subframes=rt_subframes)
        for entry, anno in zip(entries, annos, strict=True):
            arr = _as_np(anno.get_data())
            if arr is None or arr.size == 0:
                entry["status"] = "no_data"
            else:
                entry.update(status="ok", dtype=str(arr.dtype), shape=list(arr.shape), _array=arr)
    except Exception as exc:  # noqa: BLE001
        for entry in entries:
            entry.setdefault("status", "error")
            entry.setdefault("error", f"{type(exc).__name__}: {exc}")
    for anno in annos:
        with contextlib.suppress(Exception):
            anno.detach(rp_path)
    return entries[0], entries[1]


def _public(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "_array"}


def _compare(ref: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Bit-exactness of an SPG pass-through against the direct annotator read."""
    res: dict[str, Any] = {"ref": _public(ref), "out": _public(out)}
    if "_array" not in ref or "_array" not in out:
        return res
    a, b = ref["_array"], out["_array"]
    res["same_shape"] = bool(a.shape == b.shape)
    if a.shape != b.shape:
        return res
    if a.dtype == np.uint8:
        res["identical"] = bool(np.array_equal(a, b))
        res["max_abs_diff"] = float(np.max(np.abs(a.astype(int) - b.astype(int))))
        return res
    af, bf = a.astype(np.float32), b.astype(np.float32)
    finite = np.isfinite(af) & np.isfinite(bf)
    res["finite_fraction_ref"] = float(np.isfinite(af).mean())
    res["finite_pattern_identical"] = bool(np.array_equal(np.isfinite(af), np.isfinite(bf)))
    res["identical_where_finite"] = bool(np.array_equal(af[finite], bf[finite]))
    res["max_abs_diff_finite"] = (
        float(np.max(np.abs(af[finite] - bf[finite]))) if finite.any() else None
    )
    res["out_on_fp16_grid"] = bool(
        np.array_equal(bf[finite], bf[finite].astype(np.float16).astype(np.float32))
    )
    centre = _plane(a)[a.shape[0] // 2, a.shape[1] // 2 - 2 : a.shape[1] // 2 + 2]
    res["centre_ref"] = [float(v) for v in centre]
    return res


def _camera_diag(stage: Any, path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    out: dict[str, Any] = {
        "type": prim.GetTypeName(),
        "schemas": [str(s) for s in prim.GetAppliedSchemas()],
    }
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name.startswith(
            ("xformOp", "omni:sensor", "focal", "horizontalAp", "verticalAp", "clipping")
        ):
            with contextlib.suppress(Exception):
                out[name] = str(attr.Get())
    return out


ALL_EXPERIMENTS = (
    "pass_colour",
    "pass_geometry",
    "state",
    "state_empty",
    "state_zeros",
    "feedback",
    "lut_literal",
    "lut_loop",
    "lut_io",
    "lut_baked",
    "camera_sensor",
)


def run_spg_experiments(
    out_dir: str,
    *,
    experiments: tuple[str, ...] = ALL_EXPERIMENTS,
    lut_n: int = LUT_N,
    raise_limits: bool = False,
    resolution: tuple[int, int] = (256, 256),
    settle_frames: int = 20,
    rt_subframes: int = 1,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Run the selected SPG experiments on the current stage (needs the M2.2 ramp scene at /World).

    The report is written to ``save_path`` after every experiment: a Lua error inside
    ``omni.rtx.spg`` takes the whole Kit process down (measured on the 16 001-entry literal), so
    anything not saved before it is lost. ``raise_limits`` sets the sandbox instruction/memory
    limits high before any graph is authored.
    """
    import carb
    import isaacsim.core.experimental.utils.app as app_utils
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SPGNode
    from pxr import Gf, UsdGeom

    app_utils.enable_extension("omni.rtx.spg")
    settings = carb.settings.get_settings()
    settings.set("/rtx/spg/enabled", True)
    settings.set("/rtx/post/aa/op", 0)
    if raise_limits:
        settings.set("/rtx/spg/lua/instructionLimit", 10_000_000)
        settings.set("/rtx/spg/lua/memoryLimit", 268_435_456)
    report: dict[str, Any] = {
        "errors": {},
        "experiments": {},
        "timing_s": {},
        "lut_n": lut_n,
        "limits": {
            "instructionLimit": settings.get("/rtx/spg/lua/instructionLimit"),
            "memoryLimit": settings.get("/rtx/spg/lua/memoryLimit"),
            "raised": raise_limits,
        },
        "settings": {
            k: settings.get(k) for k in ("/rtx/rendermode", "/rtx/spg/enabled", "/rtx/post/aa/op")
        },
    }
    t0 = time.time()
    kdir = os.path.join(out_dir, "kernels")
    values = [lut_value(i) for i in range(lut_n)]
    lut_path = os.path.join(out_dir, "lut.f32")
    np.array(values, dtype="<f4").tofile(lut_path)
    pass_cu = _write(os.path.join(kdir, "PassKernel.cu"), PASS_CU)
    _write(os.path.join(kdir, "PassKernel.cu.lua"), PASS_LUA)
    state_cu = _write(os.path.join(kdir, "StateKernel.cu"), STATE_CU)
    _write(os.path.join(kdir, "StateKernel.cu.lua"), STATE_LUA)
    lut_files = {
        "lut_literal": _write(os.path.join(kdir, "LutLiteral.cu"), _lut_cu("lut_literal")),
        "lut_loop": _write(os.path.join(kdir, "LutLoop.cu"), _lut_cu("lut_loop")),
        "lut_io": _write(os.path.join(kdir, "LutIo.cu"), _lut_cu("lut_io")),
        "lut_baked": _write(os.path.join(kdir, "LutBaked.cu"), _lut_cu_baked(values)),
    }
    _write(os.path.join(kdir, "LutLiteral.cu.lua"), _lut_lua_literal(values, "lut_literal"))
    _write(os.path.join(kdir, "LutLoop.cu.lua"), _lut_lua_loop("lut_loop", lut_n))
    _write(os.path.join(kdir, "LutIo.cu.lua"), _lut_lua_io("lut_io", lut_path))
    _write(os.path.join(kdir, "LutBaked.cu.lua"), LUT_BAKED_LUA)

    stage = omni.usd.get_context().get_stage()
    base = UsdGeom.Camera(stage.GetPrimAtPath("/World/Camera"))
    focal = float(base.GetFocalLengthAttr().Get())
    aperture = float(base.GetHorizontalApertureAttr().Get())
    h, w = resolution

    def camera_for(path: str) -> Any:
        cam = RtxCamera(path)  # creates the prim with OmniSensorAPI, identity transform
        usd_cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
        usd_cam.GetFocalLengthAttr().Set(focal)
        usd_cam.GetHorizontalApertureAttr().Set(aperture)
        usd_cam.GetVerticalApertureAttr().Set(aperture)
        usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        return cam

    def product_for(path: str) -> tuple[Any, str]:
        cam = camera_for(path)
        rp = rep.create.render_product(path, (w, h))
        return cam, str(rp.path)

    def register(aov: str, dtype: Any, channels: int) -> None:
        rep.AnnotatorRegistry.register_annotator_from_aov(
            aov=aov, output_data_type=dtype, output_channels=channels, is_gpu_enabled=True
        )

    def settle() -> None:
        for _ in range(settle_frames):
            rep.orchestrator.step(rt_subframes=rt_subframes)

    def read(name: str, rp: str) -> dict[str, Any]:
        return _readback(rep, name, rp, 2, rt_subframes)

    def read_pair(ref: str, out: str, rp: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return _readback_pair(rep, (ref, out), rp, 2, rt_subframes)

    def checkpoint(name: str, start: float) -> None:
        report["timing_s"][name] = round(time.time() - start, 2)
        report["timing_s"]["total"] = round(time.time() - t0, 2)
        if save_path:
            save(report, save_path)

    def run_pass_colour(exp: dict[str, Any]) -> None:
        cam, rp = product_for("/World/CamA")
        cam.author_spg(
            [
                SPGNode(
                    "PassU8", pass_cu, sub_identifier="pass_u8", inputs=["In"], outputs=["Out"]
                ),
                SPGNode(
                    "PassHdr", pass_cu, sub_identifier="pass_f4", inputs=["In"], outputs=["Out"]
                ),
            ],
            connections=[
                ("LdrColor", "PassU8.inputs:In"),
                ("PassU8.outputs:Out", "OutU8"),
                ("HdrColor", "PassHdr.inputs:In"),
                ("PassHdr.outputs:Out", "OutHdr"),
            ],
            render_product=rp,
        )
        register("OutU8", np.uint8, 4)
        register("OutHdr", np.float32, 4)
        settle()
        exp["LdrColor"] = _compare(*read_pair("LdrColor", "OutU8", rp))
        exp["HdrColor"] = _compare(*read_pair("HdrColor", "OutHdr", rp))

    def run_pass_geometry(exp: dict[str, Any]) -> None:
        geometry = (
            ("DistanceToCameraSD", "pass_f32", np.float32, 1),
            ("DistanceToImagePlaneSD", "pass_f32", np.float32, 1),
            ("Camera3dPositionSD", "pass_f4", np.float32, 4),
            ("SmoothNormal", "pass_f4", np.float32, 4),
        )
        for k, (aov, sub, dtype, channels) in enumerate(geometry):
            try:
                cam, rp = product_for(f"/World/CamG{k}")
                out_name = f"OutGeom{k}"
                cam.author_spg(
                    SPGNode(
                        f"Pass{k}", pass_cu, sub_identifier=sub, inputs=["In"], outputs=["Out"]
                    ),
                    connections=[(aov, f"Pass{k}.inputs:In"), (f"Pass{k}.outputs:Out", out_name)],
                    render_product=rp,
                )
                register(out_name, dtype, channels)
                settle()
                exp[aov] = _compare(*read_pair(aov, out_name, rp))
            except Exception as exc:  # noqa: BLE001
                exp[aov] = {"error": f"{type(exc).__name__}: {exc}"}

    def run_state(exp: dict[str, Any], sub: str, cam_path: str) -> None:
        node = "Counter" + sub.removeprefix("counter").title()
        out_name = f"Out{node}"
        cam, rp = product_for(cam_path)
        cam.author_spg(
            SPGNode(node, state_cu, sub_identifier=sub, inputs=["In"], outputs=["Out"]),
            connections=[("LdrColor", f"{node}.inputs:In"), (f"{node}.outputs:Out", out_name)],
            render_product=rp,
        )
        register(out_name, np.float32, 4)
        settle()
        anno = rep.AnnotatorRegistry.get_annotator(out_name, device="cpu")
        anno.attach(rp)
        seq: list[list[float]] = []
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=rt_subframes)
            arr = _as_np(anno.get_data())
            if arr is None or arr.size == 0:
                seq.append([float("nan")] * 3)
                continue
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            seq.append([float(arr[0, 0, c]) for c in range(3)])
        with contextlib.suppress(Exception):
            anno.detach(rp)
        exp["sequence_static_frameid_luacalls"] = seq
        cols = [[s[c] for s in seq if np.isfinite(s[c])] for c in range(3)]
        deltas = [[b - a for a, b in zip(col[:-1], col[1:], strict=True)] for col in cols]
        exp["static_deltas"], exp["frame_id_deltas"], exp["lua_calls_deltas"] = deltas
        exp["static_persists"] = bool(cols[0] and all(d >= 1 for d in deltas[0]))
        exp["lua_upvalue_persists"] = bool(cols[2] and all(d >= 1 for d in deltas[2]))

    def run_feedback(exp: dict[str, Any]) -> None:
        cam, rp = product_for("/World/CamF")
        cam.author_spg(
            SPGNode(
                "Feedback",
                state_cu,
                sub_identifier="feedback",
                inputs=["In", "Prev"],
                outputs=["Out"],
            ),
            connections=[
                ("LdrColor", "Feedback.inputs:In"),
                ("OutFeedback", "Feedback.inputs:Prev"),
                ("Feedback.outputs:Out", "OutFeedback"),
            ],
            render_product=rp,
        )
        register("OutFeedback", np.float32, 1)
        settle()
        anno = rep.AnnotatorRegistry.get_annotator("OutFeedback", device="cpu")
        anno.attach(rp)
        fseq: list[float] = []
        for _ in range(6):
            rep.orchestrator.step(rt_subframes=rt_subframes)
            arr = _as_np(anno.get_data())
            fseq.append(float(_plane(arr)[0, 0]) if arr is not None and arr.size else float("nan"))
        with contextlib.suppress(Exception):
            anno.detach(rp)
        exp["sequence"] = fseq
        fin = [v for v in fseq if np.isfinite(v)]
        exp["persists"] = bool(
            len(fin) >= 2 and all(b > a for a, b in zip(fin[:-1], fin[1:], strict=True))
        )

    def run_lut(exp: dict[str, Any], tag: str, node: str, sub: str, cu: str, cam_path: str) -> None:
        cam, rp = product_for(cam_path)
        out_name = f"Out{node}"
        cam.author_spg(
            SPGNode(node, cu, sub_identifier=sub, inputs=["In"], outputs=["Out"]),
            connections=[("LdrColor", f"{node}.inputs:In"), (f"{node}.outputs:Out", out_name)],
            render_product=rp,
        )
        register(out_name, np.float32, 1)
        settle()
        out = read(out_name, rp)
        exp["out"] = _public(out)
        if "_array" not in out:
            return
        b = _plane(out["_array"])
        exp["reported_n"] = float(b[0, 0])
        exp["first_values"] = [float(v) for v in b[1, :6]]
        if tag != "lut_io":
            n = min(w, lut_n)
            expect = np.array([lut_value(i) for i in range(1, n)], dtype=np.float32)
            got = b[1, 1:n]
            exp["max_abs_err_first_columns"] = float(np.max(np.abs(got - expect)))
            exp["matches_numpy"] = bool(np.allclose(got, expect, rtol=0, atol=1e-6))

    def run_camera_sensor(exp: dict[str, Any]) -> None:
        cam = camera_for("/World/CamCS")
        exp["camera_after_wrap"] = _camera_diag(stage, "/World/CamCS")
        sensor = CameraSensor(cam, resolution=(h, w))
        rp_sensor = str(sensor.render_product.GetPath())
        exp["camera_after_sensor"] = _camera_diag(stage, "/World/CamCS")
        rp_prim = stage.GetPrimAtPath(rp_sensor)
        exp["render_product"] = {
            "path": rp_sensor,
            "camera_rel": [str(p) for p in rp_prim.GetRelationship("camera").GetTargets()],
            "resolution": str(rp_prim.GetAttribute("resolution").Get()),
        }
        rp_plain = str(rep.create.render_product("/World/CamCS", (w, h)).path)
        settle()
        for tag, rp in (("sensor_product", rp_sensor), ("plain_product", rp_plain)):
            e = read("DistanceToCameraSD", rp)
            d: dict[str, Any] = _public(e)
            if "_array" in e:
                plane = _plane(e["_array"])
                d["finite_fraction"] = float(np.isfinite(plane).mean())
                d["centre"] = [float(v) for v in plane[h // 2, w // 2 - 2 : w // 2 + 2]]
            exp[tag] = d
        exp["base_camera"] = _camera_diag(stage, "/World/Camera")

    state_cases = {
        "state": ("counter", "/World/CamS"),
        "state_empty": ("counter_empty", "/World/CamSE"),
        "state_zeros": ("counter_zeros", "/World/CamSZ"),
    }
    lut_cases = {
        "lut_literal": ("LutLiteral", "lut_literal", "/World/CamL0"),
        "lut_loop": ("LutLoop", "lut_loop", "/World/CamL1"),
        "lut_io": ("LutIo", "lut_io", "/World/CamL2"),
        "lut_baked": ("LutBaked", "lut_baked", "/World/CamL3"),
    }
    for name in experiments:
        t = time.time()
        exp: dict[str, Any] = {}
        report["experiments"][name] = exp
        try:
            if name == "pass_colour":
                run_pass_colour(exp)
            elif name == "pass_geometry":
                run_pass_geometry(exp)
            elif name in state_cases:
                run_state(exp, *state_cases[name])
            elif name == "feedback":
                run_feedback(exp)
            elif name in lut_cases:
                node, sub, cam_path = lut_cases[name]
                run_lut(exp, name, node, sub, lut_files[name], cam_path)
            elif name == "camera_sensor":
                run_camera_sensor(exp)
            else:
                raise ValueError(f"unknown experiment {name!r}")
        except Exception as exc:  # noqa: BLE001
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
        checkpoint(name, t)
    return report


def save(report: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
