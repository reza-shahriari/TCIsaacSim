"""Tier 2 radiometric benches run *inside* the simulator, the way a real camera is characterised.

SITF -- signal transfer function: the camera stares at an extended blackbody at each of several
temperatures and the mean DN is recorded (docs/physics-model.md §15 Tier 2). For an ideal camera
DN is linear in the band radiance L_B(T), not in T, and the radiometric branch returns the
blackbody temperature. These benches are self-consistency checks until a measured SITF for the
same camera exists (ADR 0003: no camera on hand; a CSV under ``data/validation/`` is picked up by
the test when present).

docs/physics-model.md §15 Tier 2, §16.4 step 3
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.pipeline.core import PipelineConfig, PipelineState
from irsim.pipeline.frame import run_frame
from irsim.radiometry.encoding import encode_temperature

__all__ = ["SitfResult", "sitf", "blackbody_gbuffer", "measured_netd_k"]


def blackbody_gbuffer(
    shape: tuple[int, int], temperature_k: float, material_id: int = 1
) -> dict[str, NDArray[np.floating] | NDArray[np.integer]]:
    """An extended, uniform blackbody filling the field: the SITF stimulus."""
    t = np.full(shape, temperature_k, dtype=np.float32)
    return {
        "temperature_k": t,
        "encoded_t": encode_temperature(t),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 1.0, np.float32),
        "material_id": np.full(shape, material_id, np.int32),
        "sky_view_factor": np.zeros(shape, np.float32),
    }


@dataclass(frozen=True)
class SitfResult:
    temperatures_k: NDArray[np.float64]
    band_radiance: NDArray[np.float64]  # L_B(T) from the LUT, the abscissa DN must be linear in
    dn_mean: NDArray[np.float64]  # mean DN over the frame(s) and the centre ROI
    dn_std: NDArray[np.float64]  # spatial+temporal DN std in the ROI (residual cos⁴ + noise)
    apparent_t_mean: NDArray[np.float64]  # mean of the float32 radiometric branch in the ROI
    apparent_t_std: NDArray[np.float64]  # its std: cos⁴ is divided out, so ~0 for the ideal chain
    signal_mean: NDArray[np.float64]  # un-quantised mean signal (DN units)
    roi_fraction: float

    def linear_fit_residual_lsb(self) -> tuple[float, float, float]:
        """Fit DN = a·L_B + b; return (a, b, rms residual in LSB)."""
        a, b = np.polyfit(self.band_radiance, self.dn_mean, 1)
        resid = self.dn_mean - (a * self.band_radiance + b)
        return float(a), float(b), float(np.sqrt(np.mean(resid**2)))


def sitf(
    config: PipelineConfig,
    temperatures_k: Sequence[float],
    frames: int = 1,
    housing_temp_k: float | None = None,
    material_id: int = 1,
    roi_fraction: float = 0.1,
) -> SitfResult:
    """Run ``frames`` frames of an extended blackbody at each temperature; return the SITF.

    Statistics are taken over a centred ROI of ``roi_fraction`` of each dimension (a real SITF
    is measured at the image centre so that vignetting does not enter the DN statistics).
    """
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if not 0.0 < roi_fraction <= 1.0:
        raise ValueError("roi_fraction must lie in (0, 1]")
    sensor = config.sensor.sensor
    h, w = sensor.fpa_shape
    k = config.supersample
    rh, rw = max(1, int(round(h * roi_fraction))), max(1, int(round(w * roi_fraction)))
    r0, c0 = (h - rh) // 2, (w - rw) // 2
    roi = (slice(r0, r0 + rh), slice(c0, c0 + rw))
    t_house = config.t_housing_cal_k if housing_temp_k is None else housing_temp_k
    temps = np.asarray(temperatures_k, dtype=np.float64)
    dn_mean, dn_std, t_app, t_app_std, sig = [], [], [], [], []
    for t in temps:
        planes = blackbody_gbuffer((h * k, w * k), float(t), material_id)
        state = PipelineState(housing_temp_k=t_house)
        dns, tapps, sigs = [], [], []
        for _ in range(frames):
            out = run_frame(planes, config, state)
            assert out.dn16 is not None and out.apparent_t is not None
            dns.append(out.dn16[roi].astype(np.float64))
            tapps.append(out.apparent_t[roi].astype(np.float64))
            sigs.append(out.signal_dn[roi].astype(np.float64))
        stack = np.stack(dns)
        dn_mean.append(float(stack.mean()))
        dn_std.append(float(stack.std()))
        t_stack = np.stack(tapps)
        t_app.append(float(t_stack.mean()))
        t_app_std.append(float(t_stack.std()))
        sig.append(float(np.stack(sigs).mean()))
    lb = config.lut.lookup(temps, config.quantity).astype(np.float64)
    return SitfResult(
        temperatures_k=temps,
        band_radiance=lb,
        dn_mean=np.asarray(dn_mean),
        dn_std=np.asarray(dn_std),
        apparent_t_mean=np.asarray(t_app),
        apparent_t_std=np.asarray(t_app_std),
        signal_mean=np.asarray(sig),
        roi_fraction=roi_fraction,
    )


def measured_netd_k(
    cube_lo: NDArray[np.floating], cube_hi: NDArray[np.floating], delta_t_k: float
) -> float:
    """Two-blackbody NETD (§15 Tier 2): ΔT · σ_temporal / ΔS with σ_temporal the rms over pixels
    of the per-pixel temporal standard deviation of the cold stack and ΔS the difference of the
    stack means. Gain-free: works on DN, signal units or electrons alike."""
    lo = np.asarray(cube_lo, dtype=np.float64)
    hi = np.asarray(cube_hi, dtype=np.float64)
    if lo.ndim != 3 or hi.ndim != 3 or lo.shape[0] < 2:
        raise ValueError("cubes must be (frames >= 2, H, W)")
    if delta_t_k <= 0.0:
        raise ValueError("delta_t_k must be positive")
    sigma_t = float(np.sqrt(np.mean(lo.var(axis=0, ddof=1))))
    delta_s = float(hi.mean() - lo.mean())
    if delta_s <= 0.0:
        raise ValueError("the hot stack must read higher than the cold stack")
    return delta_t_k * sigma_t / delta_s
