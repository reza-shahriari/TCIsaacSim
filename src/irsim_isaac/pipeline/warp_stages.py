"""Roadmap M10.4: Warp stage 1 — band radiance on the device, op for op the CPU reference.

    L = ε₀[material] · L_B(T) + (1 − ε₀) · L_env,   ε₀ = 1 under the sky mask

`irsim.pipeline.radiance.band_radiance` is the oracle (ADR 0018); this module is the GPU path
that is *compared against it*, never a second definition of the physics. The kernel repeats
`BandLUT.lookup` exactly — float32 index, clamp to ``n − KERNEL_CLAMP_MARGIN``, linear
interpolation — so the only differences left are the compiler's fused multiply-adds and the
CPU's float64 blend, both far below the 1e-4 relative / 5 mK budget of the equivalence harness
(`tests/integration/test_kernels_vs_reference.py`, ir-sim-testing skill).

What stays on the host, deliberately: the reflected-environment plane ``l_env`` is built by
`irsim.pipeline.environment.environment_radiance` (a small tilt LUT over the sky-view factor plus
one ground radiance per frame) and uploaded; the emissivity table and the band LUT are uploaded
**once** per (table, device) and cached in :class:`DeviceTables`, so the LUT device pointer does
not change between frames (the M10.4 check that the table is not rebuilt).

Warp is the ``omni.warp.core`` Kit extension, so the import is guarded: the module imports on any
machine, the kernels exist only when Warp does, and every entry point raises a clear error
otherwise. Kit is not needed to get at it — `irsim_isaac.env.ensure_warp_on_path` puts the
extension on ``sys.path`` and both the ``cpu`` and ``cuda:0`` devices then work from a bare Isaac
Sim interpreter (ADR 0014 addendum), which is what makes the equivalence harness a seconds-long
check rather than a Kit boot. No ``from __future__ import annotations`` here: Warp reads the
kernel's real type annotations.

ADR 0061 (Warp first; SPG only for stages proven stateless). docs/physics-model.md §13.5, §13.6.
"""

# mypy: disable-error-code="valid-type,no-untyped-def,untyped-decorator"
# (Warp kernel signatures are runtime-evaluated type constructors, e.g. wp.array2d(dtype=...))
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.gbuffer import UNMAPPED_MATERIAL_ID
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes, require_fp32_or_better
from irsim.pipeline.environment import environment_radiance
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import KERNEL_CLAMP_MARGIN, BandLUT, Quantity
from irsim_isaac.env import ensure_warp_on_path

__all__ = [
    "DEFAULT_DEVICE",
    "DeviceTables",
    "EQUIVALENCE_STAGES",
    "WarpBandRadianceStage",
    "band_radiance_stage_warp",
    "band_radiance_warp",
    "device_tables",
    "has_warp_module",
    "launch_band_radiance",
    "tables_key",
    "validate_material_ids",
]

DEFAULT_DEVICE = "cuda:0"


def _import_warp() -> Any:
    """Import Warp if it is available, adding the Kit extension cache to ``sys.path`` first.

    Warp is the ``omni.warp.core`` extension; Kit is normally what puts it on the path, but the
    extension is a plain package and `ensure_warp_on_path` finds it, so this module's kernels
    also run from a bare Isaac Sim interpreter (ADR 0014 addendum).
    """
    ensure_warp_on_path()
    try:
        import warp
    except ImportError:
        return None
    return warp


wp: Any = _import_warp()


def has_warp_module() -> bool:
    """True when Warp imported (inside Kit); the unit gate sees False and still imports us."""
    return wp is not None


def _require() -> Any:
    if wp is None:
        raise RuntimeError(
            "irsim_isaac.pipeline.warp_stages needs NVIDIA Warp, which ships as the "
            "omni.warp.core Kit extension: run inside Isaac Sim (ADR 0014, 0061)."
        )
    return wp


if wp is not None:

    @wp.kernel
    def _band_radiance_kernel(
        temperature_k: wp.array2d(dtype=wp.float32),
        material_id: wp.array2d(dtype=wp.int32),
        sky_mask: wp.array2d(dtype=wp.uint8),
        l_env: wp.array2d(dtype=wp.float32),
        emissivity: wp.array(dtype=wp.float32),
        lut: wp.array(dtype=wp.float32),
        t0_k: wp.float32,
        span_k: wp.float32,
        n: wp.int32,
        margin: wp.float32,
        use_env: wp.int32,
        radiance: wp.array2d(dtype=wp.float32),
    ):
        i, j = wp.tid()
        t = temperature_k[i, j]
        # BandLUT.lookup, op for op: float32 index, clamp, linear interpolation (§13.5)
        u = (t - t0_k) / span_k * wp.float32(n - 1)
        u = wp.min(wp.max(u, wp.float32(0.0)), wp.float32(n) - margin)
        i0 = wp.int32(u)
        fr = u - wp.float32(i0)
        lb = lut[i0] * (wp.float32(1.0) - fr) + lut[i0 + 1] * fr
        eps = wp.float32(1.0)
        if sky_mask[i, j] == wp.uint8(0):
            eps = emissivity[material_id[i, j]]
        if use_env != 0:
            radiance[i, j] = eps * lb + (wp.float32(1.0) - eps) * l_env[i, j]
        else:
            radiance[i, j] = eps * lb


# ---- device-resident tables ------------------------------------------------------------------


def tables_key(
    lut: BandLUT, materials: MaterialTable, quantity: Quantity, device: str
) -> tuple[Any, ...]:
    """Cache key: the LUT and the emissivity column by content, so a rebuilt table re-uploads
    and an unchanged one never does (identity alone would miss an in-place rebuild)."""
    lut_digest = hashlib.sha1(np.ascontiguousarray(lut.table(quantity)).tobytes()).hexdigest()
    eps_digest = hashlib.sha1(np.ascontiguousarray(materials.emissivity).tobytes()).hexdigest()
    return (lut_digest, float(lut.t0_k), float(lut.t1_k), int(lut.n), eps_digest, device)


@dataclass(frozen=True)
class DeviceTables:
    """The band LUT and the ε₀ column on one device, uploaded once."""

    key: tuple[Any, ...]
    device: str
    lut: Any  # wp.array(float32), n entries
    emissivity: Any  # wp.array(float32), index = material id (NaN where undefined)
    t0_k: float
    t1_k: float
    n: int

    @property
    def lut_ptr(self) -> int:
        return int(self.lut.ptr)


_TABLES: dict[tuple[Any, ...], DeviceTables] = {}


def device_tables(
    lut: BandLUT,
    materials: MaterialTable,
    quantity: Quantity = "lb",
    device: str = DEFAULT_DEVICE,
) -> DeviceTables:
    """Upload (or fetch the cached) LUT and emissivity tables for ``device``."""
    warp = _require()
    key = tables_key(lut, materials, quantity, device)
    cached = _TABLES.get(key)
    if cached is not None:
        return cached
    table = np.ascontiguousarray(lut.table(quantity), dtype=np.float32)
    eps = np.ascontiguousarray(materials.emissivity, dtype=np.float32)
    tables = DeviceTables(
        key=key,
        device=device,
        lut=warp.array(table, dtype=warp.float32, device=device),
        emissivity=warp.array(eps, dtype=warp.float32, device=device),
        t0_k=float(lut.t0_k),
        t1_k=float(lut.t1_k),
        n=int(lut.n),
    )
    _TABLES[key] = tables
    return tables


# ---- host-side guards (the CPU reference raises on these; the kernel cannot) -----------------


def validate_material_ids(
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    sky_mask: NDArray[np.bool_] | None = None,
) -> None:
    """Refuse what `MaterialTable.emissivity_for` refuses, before anything is uploaded."""
    ids = np.asarray(material_id)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"material_id must be an integer plane, got {ids.dtype}")
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != ids.shape:
            raise ValueError("sky_mask must be a bool plane with the material_id shape")
        ids = ids[~sky]
    if ids.size == 0:
        return
    used = np.unique(ids)
    if used[0] == UNMAPPED_MATERIAL_ID:
        raise ValueError(
            "G-buffer contains material id 0 (UNMAPPED): an asset has no material mapping; "
            "fix the resolver rather than rendering a default emissivity"
        )
    if used[0] < 0 or used[-1] >= materials.emissivity.size:
        raise ValueError(
            f"material ids span {used[0]}..{used[-1]}, table has {materials.emissivity.size} rows"
        )
    if np.any(np.isnan(materials.emissivity[used])):
        bad = used[np.isnan(materials.emissivity[used])]
        raise ValueError(f"material ids {bad.tolist()} have no emissivity in this band")


# ---- the stage ------------------------------------------------------------------------------


def _as_device(warp: Any, value: Any, dtype: Any, device: str) -> Any:
    """A Warp array on ``device`` (uploaded from NumPy, or the caller's array unchanged)."""
    if isinstance(value, warp.array):
        return value
    return warp.array(np.ascontiguousarray(value), dtype=dtype, device=device)


def launch_band_radiance(
    temperature_k: Any,
    material_id: Any,
    sky_mask: Any,
    l_env: Any,
    tables: DeviceTables,
    radiance: Any,
) -> Any:
    """Launch the kernel on device arrays already resident on ``tables.device``.

    ``sky_mask`` and ``l_env`` may be ``None`` (no sky pixels / emission only); ``radiance`` is
    the float32 output array to fill. Returns it. This is the entry point the on-device chain
    (M10.9a) uses; :func:`band_radiance_warp` wraps it for NumPy callers.
    """
    warp = _require()
    h, w = temperature_k.shape
    device = tables.device
    if sky_mask is None:
        sky_mask = warp.zeros((h, w), dtype=warp.uint8, device=device)
    use_env = 0 if l_env is None else 1
    if l_env is None:
        l_env = warp.zeros((h, w), dtype=warp.float32, device=device)
    warp.launch(
        _band_radiance_kernel,
        dim=(h, w),
        inputs=[
            temperature_k,
            material_id,
            sky_mask,
            l_env,
            tables.emissivity,
            tables.lut,
            np.float32(tables.t0_k),
            np.float32(np.float32(tables.t1_k) - np.float32(tables.t0_k)),
            np.int32(tables.n),
            np.float32(KERNEL_CLAMP_MARGIN),
            np.int32(use_env),
        ],
        outputs=[radiance],
        device=device,
    )
    return radiance


def band_radiance_warp(
    temperature_k: NDArray[np.floating],
    material_id: NDArray[np.integer],
    materials: MaterialTable,
    lut: BandLUT,
    quantity: Quantity = "lb",
    sky_mask: NDArray[np.bool_] | None = None,
    l_env: NDArray[np.floating] | None = None,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the signature of `irsim.pipeline.radiance.band_radiance`."""
    warp = _require()
    t = require_fp32_or_better(np.asarray(temperature_k), "temperature_k").astype(np.float32)
    ids = np.asarray(material_id)
    if ids.shape != t.shape:
        raise ValueError(f"material_id shape {ids.shape} != temperature shape {t.shape}")
    validate_material_ids(ids, materials, sky_mask)
    env_np = None
    if l_env is not None:
        env_np = require_fp32_or_better(np.asarray(l_env), "l_env").astype(np.float32)
        if env_np.shape != t.shape:
            raise ValueError(f"l_env shape {env_np.shape} != temperature shape {t.shape}")
    tables = device_tables(lut, materials, quantity, device)
    out = warp.zeros(t.shape, dtype=warp.float32, device=device)
    launch_band_radiance(
        _as_device(warp, t, warp.float32, device),
        _as_device(warp, ids.astype(np.int32), warp.int32, device),
        None
        if sky_mask is None
        else _as_device(warp, np.asarray(sky_mask).astype(np.uint8), warp.uint8, device),
        None if env_np is None else _as_device(warp, env_np, warp.float32, device),
        tables,
        out,
    )
    return np.asarray(out.numpy(), dtype=np.float32)


def band_radiance_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-1 entry point on the plane dict, the GPU twin of `band_radiance_stage`."""
    l_env = None
    if config.sky is not None:
        l_env = environment_radiance(
            config.sky,
            config.lut,
            state.t_s,
            np.asarray(planes["sky_view_factor"]),
            config.quantity,
        )
    out = band_radiance_warp(
        planes["temperature_k"],
        planes["material_id"],
        config.materials,
        config.lut,
        config.quantity,
        sky_mask=planes.get("sky_mask"),
        l_env=l_env,
        device=device,
    )
    return {"radiance": out}


class WarpBandRadianceStage:
    """`irsim.pipeline.core.Stage` implementation running stage 1 on ``device``."""

    name = "band_radiance"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return band_radiance_stage_warp(planes, config, state, self.device)


#: (CPU oracle, GPU twin) per stage name — the equivalence harness parametrises over this, and
#: every later Warp stage registers here so it is compared the same way.
EQUIVALENCE_STAGES: dict[str, tuple[Any, Any]] = {
    "band_radiance": (band_radiance_stage, band_radiance_stage_warp),
}
