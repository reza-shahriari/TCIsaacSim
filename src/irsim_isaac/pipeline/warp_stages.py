"""Warp stages on the device, op for op the CPU reference. Stage 1 (M10.4), 2-3 (M10.5), 4 (M10.6).

    stage 1   L  = ε₀[material] · L_B(T) + (1 − ε₀) · L_env,   ε₀ = 1 under the sky mask
    stage 2   L' = τ(d) L + (1 − τ(d)) L_B(T_air),             τ = Σ_k w_k exp(−γ_k d)
    stage 3   Φ  = box_k(PSF ∗ L') · Ω_eff τ_opt cos⁴θ A_d + Φ_self,   Ω_eff = π/(4F² + 1)
    stage 4   S_n = S_{n−1} + (S_ideal(Φ) − S_{n−1}) α,                α = 1 − e^{−Δt/τ_th}

`irsim.pipeline` is the oracle (ADR 0018); this module is the GPU path
that is *compared against it*, never a second definition of the physics. The kernel repeats
`BandLUT.lookup` exactly — float32 index, clamp to ``n − KERNEL_CLAMP_MARGIN``, linear
interpolation — so the only differences left are the compiler's fused multiply-adds and the
CPU's float64 blend, both far below the 1e-4 relative / 5 mK budget of the equivalence harness
(`tests/integration/test_kernels_vs_reference.py`, ir-sim-testing skill).

What stays on the host, deliberately: the reflected-environment plane ``l_env`` is built by
`irsim.pipeline.environment.environment_radiance` (a small tilt LUT over the sky-view factor plus
one ground radiance per frame) and uploaded; the emissivity table and the band LUT are uploaded
**once** per (table, device) and cached in :class:`DeviceTables`, so the LUT device pointer does
not change between frames (the M10.4 check that the table is not rebuilt); and every per-frame
scalar of stages 2 and 3 — the exponential sum's (w_k, γ_k), L_B(T_air), the aperture factor,
Φ_self — is evaluated by the model that owns it (:class:`AtmosphereTerms`, :class:`OpticsTerms`).
The kernels scale and add; they never re-derive a coefficient, which is how non-negotiable #5
survives having a second implementation of the imaging chain.

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

from irsim.atmosphere.beer_lambert import transmittance
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.config.gbuffer import UNMAPPED_MATERIAL_ID
from irsim.config.sensor import SensorSpec
from irsim.detector.bolometer import PHOTON_SCALE_GUARD, MicrobolometerDetector
from irsim.detector.lowpass import alpha_for
from irsim.detector.params import BolometerParams
from irsim.detector.photon import PhotonDetector
from irsim.detector.quantise import dn_max_for_bits
from irsim.materials.table import MaterialTable
from irsim.optics.aperture import aperture_factor
from irsim.optics.self_emission import self_emission_power
from irsim.optics.stage import optics_field
from irsim.pipeline.atmosphere import atmosphere_stage
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes, require_fp32_or_better
from irsim.pipeline.detector import detector_stage
from irsim.pipeline.environment import environment_radiance
from irsim.pipeline.optics import housing_band_radiance, optics_stage
from irsim.pipeline.radiance import band_radiance_stage
from irsim.radiometry.lut import KERNEL_CLAMP_MARGIN, BandLUT, Quantity
from irsim_isaac.env import ensure_warp_on_path

__all__ = [
    "DEFAULT_DEVICE",
    "AtmosphereTerms",
    "DEVICE_STATE_KEY",
    "DetectorTerms",
    "DeviceTables",
    "EQUIVALENCE_OUTPUT",
    "EQUIVALENCE_STAGES",
    "OpticsTerms",
    "WarpAtmosphereStage",
    "WarpBandRadianceStage",
    "WarpDetectorStage",
    "WarpOpticsStage",
    "WarpPipelineState",
    "apply_atmosphere_warp",
    "apply_optics_warp",
    "atmosphere_stage_warp",
    "atmosphere_terms",
    "band_radiance_stage_warp",
    "band_radiance_warp",
    "detector_stage_warp",
    "detector_terms",
    "device_tables",
    "has_warp_module",
    "launch_atmosphere",
    "launch_band_radiance",
    "launch_detector",
    "launch_optics",
    "optics_stage_warp",
    "optics_terms",
    "quantise_warp",
    "tables_key",
    "validate_distance",
    "validate_flux",
    "validate_material_ids",
    "warp_pipeline_state",
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
    """True when Warp imported; the unit gate on a machine without it sees False and still
    imports this module, so the engine-free half of every stage stays testable."""
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

    @wp.kernel
    def _atmosphere_kernel(
        radiance: wp.array2d(dtype=wp.float32),
        distance_m: wp.array2d(dtype=wp.float32),
        sky_mask: wp.array2d(dtype=wp.uint8),
        weights: wp.array(dtype=wp.float32),
        gamma_per_m: wp.array(dtype=wp.float32),
        n_terms: wp.int32,
        l_air: wp.float32,
        tau_at_inf: wp.float32,
        tau_const: wp.float32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """L' = τ(d) L + (1 − τ(d)) L_B(T_air), τ = Σ_k w_k exp(−γ_k d)  (§7.1, MS.1 per term).

        Σ w_k = 1, so the per-term path radiance Σ_k w_k (1 − τ_k) L_air collapses to
        (1 − τ) L_air exactly -- the same expression serves the grey M8.1 path (one term), the
        layered horizontal path (2-5 terms) and the constant-τ L1 fallback, and no term-weighted
        sum has to be carried separately. ``tau_const`` < 0 means "compute τ from the distance".
        """
        i, j = wp.tid()
        l_in = radiance[i, j]
        if sky_mask[i, j] != wp.uint8(0):
            out[i, j] = l_in  # ADR 0050: a sky pixel already carries the whole column
            return
        tau = tau_const
        if tau_const < wp.float32(0.0):
            d = distance_m[i, j]
            if wp.isinf(d):
                tau = tau_at_inf  # the host decides: 0, or 1 for a grey γ of exactly 0
            else:
                tau = wp.float32(0.0)
                for k in range(n_terms):
                    tau += weights[k] * wp.exp(-gamma_per_m[k] * d)
        out[i, j] = tau * l_in + (wp.float32(1.0) - tau) * l_air

    @wp.kernel
    def _psf_kernel(
        image: wp.array2d(dtype=wp.float32),
        psf: wp.array2d(dtype=wp.float32),
        radius_y: wp.int32,
        radius_x: wp.int32,
        out: wp.array2d(dtype=wp.float32),
    ):
        """Direct convolution with edge replication, at the supersampled pitch (§8.3, ADR 0059).

        The CPU oracle convolves by FFT over an edge-padded image and crops; because the padding
        is at least the kernel radius the circular convolution never wraps into the crop, so it
        equals this direct sum with clamped indices — same result, no FFT to reproduce. The index
        is ``i + r − a``, the convolution flip: the PSF is radially symmetric so it makes no
        difference today, and it makes the kernel correct if an asymmetric one is ever passed.
        """
        i, j = wp.tid()
        h = image.shape[0]
        w = image.shape[1]
        acc = wp.float32(0.0)
        for a in range(2 * radius_y + 1):
            y = wp.clamp(i + radius_y - a, 0, h - 1)
            for b in range(2 * radius_x + 1):
                x = wp.clamp(j + radius_x - b, 0, w - 1)
                acc += psf[a, b] * image[y, x]
        out[i, j] = acc

    @wp.kernel
    def _optics_kernel(
        radiance_ss: wp.array2d(dtype=wp.float32),
        cos4: wp.array2d(dtype=wp.float32),
        factor: wp.float32,
        area_m2: wp.float32,
        phi_self: wp.float32,
        supersample: wp.int32,
        flux: wp.array2d(dtype=wp.float32),
    ):
        """Box-mean k× downsample, then Φ = E A_d + Φ_self on the detector grid (§8.1, §8.3).

        ``factor`` is Ω_eff τ_opt and ``phi_self`` is A_d Ω_eff (1 − τ_opt) L_B(T_housing), both
        computed on the host by `irsim.optics` — the aperture factor π/(4F² + 1) is written once,
        in `irsim.optics.aperture`, and `tests/unit/test_aperture_guard.py` walks the AST of this
        file to keep it that way (non-negotiable #5). The box is the full pitch; the fill factor
        is already in A_d, not in the box (ADR 0020).
        """
        i, j = wp.tid()
        acc = wp.float32(0.0)
        for a in range(supersample):
            for b in range(supersample):
                acc += radiance_ss[i * supersample + a, j * supersample + b]
        mean = acc / (wp.float32(supersample) * wp.float32(supersample))
        flux[i, j] = mean * factor * cos4[i, j] * area_m2 + phi_self

    @wp.kernel
    def _bolometer_signal_kernel(
        flux_w: wp.array2d(dtype=wp.float32),
        gain_dn_per_w: wp.float32,
        offset_w: wp.float32,
        signal_dn: wp.array2d(dtype=wp.float32),
    ):
        """S_ideal = gain · (Φ − offset), the ADR 0019 linear static transfer (§9.2).

        Written as (Φ − offset) · gain, not Φ·gain − offset·gain: Φ and the offset are both of
        order 1e-8 W and their difference spans the ADC, so the second form would subtract two
        large numbers to get a small one. In this form the float32 spacing at 1e-8 W is ~0.003 DN.
        """
        i, j = wp.tid()
        signal_dn[i, j] = (flux_w[i, j] - offset_w) * gain_dn_per_w

    @wp.kernel
    def _photon_signal_kernel(
        flux_q: wp.array2d(dtype=wp.float32),
        qe_t_int: wp.float32,
        offset_e: wp.float32,
        dn_per_electron: wp.float32,
        signal_dn: wp.array2d(dtype=wp.float32),
    ):
        """N_e = η t_int Φ_q + N_dark + N_bg, then S = N_e / N_well · 2^bits (§9.1).

        No lag: a cooled photon detector is memoryless on these timescales (ADR 0052), which is
        the Tier 3 check that LWIR smears and cooled MWIR does not.
        """
        i, j = wp.tid()
        signal_dn[i, j] = (flux_q[i, j] * qe_t_int + offset_e) * dn_per_electron

    @wp.kernel
    def _bolometer_lag_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        alpha: wp.float32,
        adopt: wp.int32,
        state: wp.array2d(dtype=wp.float32),
        out: wp.array2d(dtype=wp.float32),
    ):
        """S_n = S_{n−1} + (S_ideal − S_{n−1}) α, in place on the persistent state (§9.2).

        ``adopt`` is the first frame: the membrane starts settled, not at zero, so a sequence does
        not open with a frame-long ramp no real core shows. ``state`` is the device buffer
        :class:`WarpPipelineState` owns; it is updated in place and never leaves the device, and
        ``out`` carries the frame onward so the next stage has something to read that the next
        frame will not overwrite.
        """
        i, j = wp.tid()
        s = signal_dn[i, j]
        if adopt != 0:
            state[i, j] = s
        else:
            state[i, j] = state[i, j] + (s - state[i, j]) * alpha
        out[i, j] = state[i, j]

    @wp.kernel
    def _quantise_kernel(
        signal_dn: wp.array2d(dtype=wp.float32),
        dn_max: wp.int32,
        dn: wp.array2d(dtype=wp.uint16),
    ):
        """DN = clip(floor(S), 0, 2^bits − 1) (§2 𝒬, §9.1).

        Floor, not round: an ideal ADC compares against thresholds, which is what makes the
        quantisation error uniform on [0, 1) LSB with 0.29 LSB rms — the figure the Tier 2 SITF
        bench is written against (ADR 0019). Saturation clips and never wraps.
        """
        i, j = wp.tid()
        code = wp.int32(wp.floor(signal_dn[i, j]))
        dn[i, j] = wp.uint16(wp.clamp(code, 0, dn_max))


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


# ---- stage 2: atmosphere ---------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphereTerms:
    """What stage 2 needs for one frame, read off the CPU model on the host (M10.5).

    The exponential sum is a property of the band and the weather, not of the pixel, so it is
    evaluated once per frame by the model that owns it and only ``(w_k, γ_k)``, ``L_B(T_air)``
    and the two sentinels cross to the device. Nothing here re-derives physics: ``tau_at_inf``
    is obtained by *asking the oracle* for τ(∞) rather than restating its rule, which is how the
    grey model's "γ = 0 means a transparent infinite path" and the layered model's "an infinite
    horizontal path is opaque" both come out right without a branch here.

    ``gamma_per_m`` is the total surface extinction per term including aerosol; on a horizontal
    ray every column integral is just ``d``, so the two contributions add (docs/physics-model.md
    §7.1, `irsim.atmosphere.layered.ExponentialSum.optical_depths`). Per-pixel *slant* paths are
    MS.8's job and are not in this form yet.
    """

    weights: NDArray[np.float32]
    gamma_per_m: NDArray[np.float32]
    l_air: float
    tau_at_inf: float
    tau_const: float = -1.0  # < 0: compute τ from the distance; >= 0: the L1 constant-τ fallback

    def __post_init__(self) -> None:
        if self.weights.shape != self.gamma_per_m.shape or self.weights.ndim != 1:
            raise ValueError("weights and gamma_per_m must be one matching 1-D array each")
        if abs(float(self.weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"class weights sum to {float(self.weights.sum())}, not 1")
        if np.any(self.gamma_per_m < 0.0) or not np.all(np.isfinite(self.gamma_per_m)):
            raise ValueError("extinction coefficients must be finite and non-negative")

    @property
    def n_terms(self) -> int:
        return int(self.weights.size)

    def transmittance(self, distance_m: NDArray[np.floating]) -> NDArray[np.float64]:
        """τ(d) as the kernel computes it, on the host — the check that the terms were read off
        the model correctly is `tests/unit/test_warp_atmosphere_terms.py`, no GPU involved."""
        d = np.asarray(distance_m, dtype=np.float64)
        if self.tau_const >= 0.0:
            return np.full(d.shape, self.tau_const, dtype=np.float64)
        od = self.gamma_per_m.astype(np.float64)[:, None] * d.reshape(1, -1)
        tau = (self.weights.astype(np.float64)[:, None] * np.exp(-od)).sum(axis=0)
        return np.asarray(np.where(np.isinf(d.ravel()), self.tau_at_inf, tau), dtype=np.float64)


def atmosphere_terms(config: PipelineConfig, state: PipelineState) -> AtmosphereTerms | None:
    """Read stage 2's per-frame terms off whichever atmosphere the config holds; None = identity.

    Mirrors `irsim.pipeline.atmosphere.atmosphere_stage` branch for branch: no atmosphere is the
    identity stage, a `LayeredAtmosphere` gives the MS.1 exponential sum at the frame's weather
    time, and a grey `Atmosphere` gives the single M8.1 term (with ``tau_override`` as the L1
    fallback). Both take L_B(T_air) from the LUT the oracle uses — the atmosphere's own for the
    layered model, the pipeline's for the grey one — so an isothermal scene stays invariant
    against stage 1 on the device as well (§7.1).
    """
    atmosphere = config.atmosphere
    if atmosphere is None:
        return None
    band = config.sensor.sensor.band.band_id
    if isinstance(atmosphere, LayeredAtmosphere):
        es = atmosphere.exponential_sum(band, state.t_s)
        return AtmosphereTerms(
            weights=es.weights.astype(np.float32),
            gamma_per_m=(es.gamma_0 + es.gamma_aerosol).astype(np.float32),
            l_air=atmosphere.air_radiance(band, state.t_s, config.quantity),
            tau_at_inf=float(es.transmittance(np.inf, 0.0)[()]),
        )
    atm_state = atmosphere.state(state.t_s)
    gamma = float(atm_state.gamma_per_m[band])
    return AtmosphereTerms(
        weights=np.ones(1, dtype=np.float32),
        gamma_per_m=np.asarray([gamma], dtype=np.float32),
        l_air=float(config.lut.lookup(np.float64(atm_state.t_air_k), config.quantity)[()]),
        tau_at_inf=float(transmittance(np.inf, gamma)[()]),
        tau_const=-1.0 if config.tau_override is None else float(config.tau_override),
    )


def launch_atmosphere(
    radiance: Any, distance_m: Any, sky_mask: Any, terms: AtmosphereTerms, out: Any, device: str
) -> Any:
    """Launch stage 2 on device arrays already resident on ``device``; fills and returns ``out``.

    ``sky_mask`` may be None (no sky pixels). This is the entry point the on-device chain
    (M10.9a) uses, so the radiance never returns to the host between stages 1 and 2.
    """
    warp = _require()
    h, w = radiance.shape
    if sky_mask is None:
        sky_mask = warp.zeros((h, w), dtype=warp.uint8, device=device)
    warp.launch(
        _atmosphere_kernel,
        dim=(h, w),
        inputs=[
            radiance,
            distance_m,
            sky_mask,
            _as_device(warp, terms.weights, warp.float32, device),
            _as_device(warp, terms.gamma_per_m, warp.float32, device),
            np.int32(terms.n_terms),
            np.float32(terms.l_air),
            np.float32(terms.tau_at_inf),
            np.float32(terms.tau_const),
        ],
        outputs=[out],
        device=device,
    )
    return out


def validate_distance(
    distance_m: NDArray[np.floating], shape: tuple[int, ...], sky_mask: Any = None
) -> None:
    """Refuse what `irsim.atmosphere.beer_lambert.transmittance` refuses, before any upload.

    A negative or NaN range is a broken G-buffer in either model; the grey path raises on it and
    the layered path would quietly return nonsense, so the guard is applied to both.
    """
    d = np.asarray(distance_m)
    if d.shape != shape:
        raise ValueError(f"distance_m shape {d.shape} != radiance shape {shape}")
    if np.any(d < 0.0) or np.any(np.isnan(d)):
        raise ValueError("distance_m must be non-negative (inf allowed)")
    if sky_mask is not None:
        sky = np.asarray(sky_mask)
        if sky.dtype != np.bool_ or sky.shape != shape:
            raise ValueError("sky_mask must be a bool plane with the radiance shape")


def apply_atmosphere_warp(
    radiance: NDArray[np.floating],
    distance_m: NDArray[np.floating],
    terms: AtmosphereTerms,
    sky_mask: NDArray[np.bool_] | None = None,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the GPU twin of `apply_atmosphere_gbuffer` / `apply_layered_gbuffer`."""
    warp = _require()
    l_in = require_fp32_or_better(np.asarray(radiance), "radiance").astype(np.float32)
    validate_distance(distance_m, l_in.shape, sky_mask)
    out = warp.zeros(l_in.shape, dtype=warp.float32, device=device)
    launch_atmosphere(
        _as_device(warp, l_in, warp.float32, device),
        _as_device(warp, np.asarray(distance_m, dtype=np.float32), warp.float32, device),
        None
        if sky_mask is None
        else _as_device(warp, np.asarray(sky_mask).astype(np.uint8), warp.uint8, device),
        terms,
        out,
        device,
    )
    return np.asarray(out.numpy(), dtype=np.float32)


def atmosphere_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-2 entry point on the plane dict, the GPU twin of `atmosphere_stage`."""
    radiance = np.asarray(planes["radiance"])
    terms = atmosphere_terms(config, state)
    if terms is None:
        return {"radiance": radiance}
    return {
        "radiance": apply_atmosphere_warp(
            radiance,
            np.asarray(planes["distance_m"]),
            terms,
            sky_mask=planes.get("sky_mask"),
            device=device,
        )
    }


class WarpAtmosphereStage:
    """`irsim.pipeline.core.Stage` implementation running stage 2 on ``device``."""

    name = "atmosphere"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return atmosphere_stage_warp(planes, config, state, self.device)


# ---- stage 3: optics -------------------------------------------------------------------------


@dataclass(frozen=True)
class OpticsTerms:
    """What stage 3 needs for one frame, all of it computed by `irsim.optics` on the host (M10.5).

    Every radiometric scalar crosses to the device already evaluated. That is not an optimisation:
    ``factor`` is Ω_eff τ_opt with Ω_eff = π/(4F² + 1) from `irsim.optics.aperture.aperture_factor`,
    and non-negotiable #5 says that expression exists once in the codebase. A kernel that took the
    f-number and squared it would be a second definition, 20 % wrong at F/1.0 the day someone
    typed the paraxial form (§2, §8.1).

    ``cos4`` is the native-grid field from `irsim.optics.stage.optics_field` (ones when vignetting
    is disabled) and ``psf`` the kernel from `irsim.optics.psf.optical_psf` at the k× pitch, or
    None for no optical blur. The PSF is cast to float32 for the device while the oracle convolves
    with float64 taps; the taps sum to 1 to ~1e-7 either way, which is a DC gain error two orders
    below the equivalence budget.
    """

    factor: float  # Omega_eff * tau_opt
    area_m2: float
    phi_self: float
    supersample: int
    cos4: NDArray[np.float32]
    psf: NDArray[np.float32] | None

    def __post_init__(self) -> None:
        if self.supersample < 1:
            raise ValueError("supersample factor must be >= 1")
        if self.psf is not None and (
            self.psf.ndim != 2 or self.psf.shape[0] % 2 == 0 or self.psf.shape[1] % 2 == 0
        ):
            raise ValueError("psf must be 2-D with odd sides")


def optics_terms(
    sensor: SensorSpec,
    lb_housing: float,
    supersample: int | None = None,
    psf: NDArray[np.floating] | None = None,
) -> OpticsTerms:
    """Stage 3's scalars and fields from the sensor spec, exactly as `apply_optics` takes them."""
    k = sensor.optics.supersample_factor if supersample is None else supersample
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    a_d = sensor.detector_active_area_m2
    return OpticsTerms(
        factor=aperture_factor(f) * tau,
        area_m2=a_d,
        phi_self=self_emission_power(a_d, f, tau, lb_housing),
        supersample=k,
        cos4=np.ascontiguousarray(optics_field(sensor), dtype=np.float32),
        psf=None if psf is None else np.ascontiguousarray(psf, dtype=np.float32),
    )


def launch_optics(radiance_ss: Any, terms: OpticsTerms, flux: Any, device: str) -> Any:
    """Launch stage 3 on device arrays already resident on ``device``; fills and returns ``flux``.

    Allocates one intermediate only when there is a PSF, and the convolution is out-of-place
    because a pixel's neighbours must not have been overwritten before it is read.
    """
    warp = _require()
    h_ss, w_ss = radiance_ss.shape
    blurred = radiance_ss
    if terms.psf is not None:
        blurred = warp.zeros((h_ss, w_ss), dtype=warp.float32, device=device)
        warp.launch(
            _psf_kernel,
            dim=(h_ss, w_ss),
            inputs=[
                radiance_ss,
                _as_device(warp, terms.psf, warp.float32, device),
                np.int32(terms.psf.shape[0] // 2),
                np.int32(terms.psf.shape[1] // 2),
            ],
            outputs=[blurred],
            device=device,
        )
    warp.launch(
        _optics_kernel,
        dim=flux.shape,
        inputs=[
            blurred,
            _as_device(warp, terms.cos4, warp.float32, device),
            np.float32(terms.factor),
            np.float32(terms.area_m2),
            np.float32(terms.phi_self),
            np.int32(terms.supersample),
        ],
        outputs=[flux],
        device=device,
    )
    return flux


def apply_optics_warp(
    radiance_ss: NDArray[np.floating],
    terms: OpticsTerms,
    *,
    device: str = DEFAULT_DEVICE,
) -> NDArray[np.float32]:
    """NumPy in, NumPy out: the GPU twin of `irsim.optics.stage.apply_optics`."""
    warp = _require()
    l_ss = require_fp32_or_better(np.asarray(radiance_ss), "radiance").astype(np.float32)
    if l_ss.ndim != 2:
        raise ValueError("stage 3 works on one (H, W) supersampled plane")
    k = terms.supersample
    h_ss, w_ss = l_ss.shape
    if h_ss % k or w_ss % k:
        raise ValueError(f"shape {(h_ss, w_ss)} is not divisible by the supersample factor {k}")
    native = (h_ss // k, w_ss // k)
    if terms.cos4.shape != native:
        raise ValueError(f"cos4 field {terms.cos4.shape} != detector grid {native}")
    flux = warp.zeros(native, dtype=warp.float32, device=device)
    launch_optics(_as_device(warp, l_ss, warp.float32, device), terms, flux, device)
    return np.asarray(flux.numpy(), dtype=np.float32)


def optics_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-3 entry point on the plane dict, the GPU twin of `optics_stage`."""
    terms = optics_terms(
        config.sensor.sensor,
        housing_band_radiance(config, state),
        config.supersample,
        config.psf,
    )
    return {
        "flux": apply_optics_warp(np.asarray(planes["radiance"]), terms, device=device),
    }


class WarpOpticsStage:
    """`irsim.pipeline.core.Stage` implementation running stage 3 on ``device``."""

    name = "optics"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return optics_stage_warp(planes, config, state, self.device)


# ---- stage 4: detector, and the cross-frame state that goes with it ---------------------------

#: Where the device-side companion of a `PipelineState` lives in its ``buffers`` (ADR 0052).
DEVICE_STATE_KEY = "warp_device_state"


class WarpPipelineState:
    """The device-resident cross-frame buffers of one camera on one device (ADR 0052).

    Today that is the bolometer's membrane IIR; M10.7's fixed-pattern drift, bad-pixel map and
    NUC residual join it. The point of a single owner is the reset path: a format change, a cold
    start or a new camera must clear *all* of it together, and a buffer that lives wherever it
    was first needed gets forgotten by exactly one of those.

    The IIR buffer is allocated once and updated in place; ``iir_ptr`` exists so a test can show
    the pointer does not move between frames and the state is never round-tripped to the host.
    """

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device
        self._iir: Any = None
        self._iir_shape: tuple[int, int] | None = None

    def iir_state(self, shape: tuple[int, int]) -> tuple[Any, bool]:
        """The IIR buffer for ``shape`` and whether this frame must adopt its input.

        A shape change reallocates and re-adopts rather than raising, because the only way to get
        here with a new shape is a deliberate reconfiguration; the CPU filter raises instead, and
        the difference is deliberate -- see the note in :func:`detector_stage_warp`.
        """
        warp = _require()
        if self._iir is None or self._iir_shape != shape:
            self._iir = warp.zeros(shape, dtype=warp.float32, device=self.device)
            self._iir_shape = shape
            return self._iir, True
        return self._iir, False

    @property
    def iir_ptr(self) -> int | None:
        return None if self._iir is None else int(self._iir.ptr)

    def reset(self) -> None:
        """Cold start: drop every device buffer, so the next frame adopts its input again."""
        self._iir = None
        self._iir_shape = None


def warp_pipeline_state(state: PipelineState, device: str = DEFAULT_DEVICE) -> WarpPipelineState:
    """The device companion of ``state``, created on first use and kept in its ``buffers``."""
    existing = state.buffers.get(DEVICE_STATE_KEY)
    if isinstance(existing, WarpPipelineState) and existing.device == device:
        return existing
    fresh = WarpPipelineState(device)
    state.buffers[DEVICE_STATE_KEY] = fresh
    return fresh


@dataclass(frozen=True)
class DetectorTerms:
    """Stage 4's per-camera scalars, read off the detector the CPU path uses (M10.6).

    One dataclass for both FPA types because the kernel launch differs only in which scalars it
    carries; ``kind`` selects the path, from ``fpa.type``, never from a heuristic on the data.
    ``alpha`` is `irsim.detector.lowpass.alpha_for` at the configured frame interval -- the blend
    weight 1 − e^{−Δt/τ_th}, which is 0.811 at 60 Hz and τ_th = 10 ms, not the "0.6 frames" of
    §9.2 (spec issue S8).
    """

    kind: str
    dn_max: int
    gain_dn_per_w: float = 0.0
    offset_w: float = 0.0
    alpha: float = 0.0
    qe_t_int: float = 0.0
    offset_e: float = 0.0
    dn_per_electron: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("bolometer", "photon"):
            raise ValueError(f"unknown FPA type {self.kind!r}")
        if self.kind == "bolometer" and not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"IIR blend weight must lie in (0, 1], got {self.alpha}")


def detector_terms(config: PipelineConfig) -> DetectorTerms:
    """Read stage 4's scalars off the configured detector; the transfer stays the oracle's."""
    fpa = config.fpa
    if isinstance(fpa, BolometerParams):
        detector = config.detector
        if not isinstance(detector, MicrobolometerDetector):
            raise TypeError("a bolometer FPA needs a MicrobolometerDetector")
        return DetectorTerms(
            kind="bolometer",
            dn_max=fpa.dn_max,
            gain_dn_per_w=detector.transfer.gain_dn_per_w,
            offset_w=detector.transfer.offset_w,
            alpha=alpha_for(fpa.frame_dt_s, fpa.thermal_time_constant_s),
        )
    detector_q = config.detector
    if not isinstance(detector_q, PhotonDetector):
        raise TypeError("a photon FPA needs a PhotonDetector")
    budget = detector_q.budget
    return DetectorTerms(
        kind="photon",
        dn_max=fpa.dn_max,
        qe_t_int=fpa.quantum_efficiency * fpa.integration_time_s,
        offset_e=budget.dark_electrons + budget.background_electrons,
        dn_per_electron=2**fpa.bit_depth / fpa.well_capacity_e,
    )


def validate_flux(flux: NDArray[np.floating], kind: str) -> NDArray[np.float32]:
    """Refuse what the CPU detector refuses, before anything is uploaded (§9.1, §9.2)."""
    phi = require_fp32_or_better(np.asarray(flux), "flux")
    if np.any(phi < 0.0):
        raise ValueError("pixel power cannot be negative")
    if kind == "bolometer" and np.any(phi > PHOTON_SCALE_GUARD):
        raise ValueError(
            f"pixel power {float(phi.max()):.3e} exceeds {PHOTON_SCALE_GUARD} W: this looks like "
            "a photon rate, not watts -- the bolometer takes energy-form band radiance (§9.2)"
        )
    return phi.astype(np.float32)


def launch_detector(
    flux: Any,
    terms: DetectorTerms,
    device_state: WarpPipelineState,
    signal_dn: Any,
    lagged: Any | None = None,
) -> Any:
    """Launch stage 4 on device arrays; returns the plane the next stage should read.

    ``signal_dn`` receives the ideal transfer and, for a bolometer, ``lagged`` receives the
    membrane's output while the persistent state is updated in place. The state never leaves the
    device: the IIR reads and writes it on the GPU and only ``lagged`` is ever downloaded.
    """
    warp = _require()
    shape = tuple(flux.shape)
    device = device_state.device
    if terms.kind == "bolometer":
        warp.launch(
            _bolometer_signal_kernel,
            dim=shape,
            inputs=[flux, np.float32(terms.gain_dn_per_w), np.float32(terms.offset_w)],
            outputs=[signal_dn],
            device=device,
        )
        state, adopt = device_state.iir_state((int(shape[0]), int(shape[1])))
        out = signal_dn if lagged is None else lagged
        warp.launch(
            _bolometer_lag_kernel,
            dim=shape,
            inputs=[signal_dn, np.float32(terms.alpha), np.int32(1 if adopt else 0), state],
            outputs=[out],
            device=device,
        )
        return out
    warp.launch(
        _photon_signal_kernel,
        dim=shape,
        inputs=[
            flux,
            np.float32(terms.qe_t_int),
            np.float32(terms.offset_e),
            np.float32(terms.dn_per_electron),
        ],
        outputs=[signal_dn],
        device=device,
    )
    return signal_dn


def detector_stage_warp(
    planes: Planes, config: PipelineConfig, state: PipelineState, device: str = DEFAULT_DEVICE
) -> Planes:
    """Stage-4 entry point on the plane dict, the GPU twin of `detector_stage`.

    One deliberate difference from the oracle: a frame whose shape does not match the IIR state
    reallocates and re-adopts here, where `BolometerLowPass` raises. On the CPU a shape change
    mid-sequence is a caller bug worth stopping on; on the device the buffer is owned by
    :class:`WarpPipelineState` and a reconfiguration legitimately replaces it. Feed a sequence of
    one shape and the two paths are identical, which is what the equivalence harness checks.
    """
    warp = _require()
    terms = detector_terms(config)
    phi = validate_flux(np.asarray(planes["flux"]), terms.kind)
    device_state = warp_pipeline_state(state, device)
    signal = warp.zeros(phi.shape, dtype=warp.float32, device=device)
    lagged = (
        warp.zeros(phi.shape, dtype=warp.float32, device=device)
        if terms.kind == "bolometer"
        else None
    )
    out = launch_detector(
        _as_device(warp, phi, warp.float32, device), terms, device_state, signal, lagged
    )
    return {"signal_dn": np.asarray(out.numpy(), dtype=np.float32)}


def quantise_warp(
    signal_dn: NDArray[np.floating], bit_depth: int, *, device: str = DEFAULT_DEVICE
) -> NDArray[np.uint16]:
    """DN = clip(floor(S), 0, 2^bits − 1) on the device, the twin of `irsim.detector.quantise`."""
    warp = _require()
    s = require_fp32_or_better(np.asarray(signal_dn), "signal_dn").astype(np.float32)
    if not np.all(np.isfinite(s)):
        raise ValueError("signal contains NaN or inf")
    dn = warp.zeros(s.shape, dtype=warp.uint16, device=device)
    warp.launch(
        _quantise_kernel,
        dim=s.shape,
        inputs=[_as_device(warp, s, warp.float32, device), np.int32(dn_max_for_bits(bit_depth))],
        outputs=[dn],
        device=device,
    )
    return np.asarray(dn.numpy(), dtype=np.uint16)


class WarpDetectorStage:
    """`irsim.pipeline.core.Stage` implementation running stage 4 on ``device``."""

    name = "detector"

    def __init__(self, device: str = DEFAULT_DEVICE) -> None:
        self.device = device

    def __call__(self, planes: Planes, config: PipelineConfig, state: PipelineState) -> Planes:
        return detector_stage_warp(planes, config, state, self.device)


#: (CPU oracle, GPU twin) per stage name — the equivalence harness parametrises over this, and
#: every later Warp stage registers here so it is compared the same way. The plane each stage
#: is compared on is `EQUIVALENCE_OUTPUT`, because stage 3 leaves ``flux``, not ``radiance``.
EQUIVALENCE_STAGES: dict[str, tuple[Any, Any]] = {
    "band_radiance": (band_radiance_stage, band_radiance_stage_warp),
    "atmosphere": (atmosphere_stage, atmosphere_stage_warp),
    "optics": (optics_stage, optics_stage_warp),
    "detector": (detector_stage, detector_stage_warp),
}

#: Which plane each registered stage produces.
EQUIVALENCE_OUTPUT: dict[str, str] = {
    "band_radiance": "radiance",
    "atmosphere": "radiance",
    "optics": "flux",
    "detector": "signal_dn",
}
