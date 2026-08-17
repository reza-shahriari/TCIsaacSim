"""Orchestrates SS1-SS11 into one per-frame render. Implements docs/thermal_camera_model.md SS12.

This is the function Phase 2's Isaac Sim writer prototype (and, later, the
SPG kernel) both exist to feed real per-pixel arrays into.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .emission import surface_leaving_radiance, band_radiance
from .atmosphere import apply_atmosphere
from .optics import focal_plane_irradiance
from .detector import linear_detector_signal
from .blur import apply_system_blur
from .noise import netd_noise_sigma, add_temporal_noise, add_fixed_pattern_noise
from .quantize import quantize
from .agc import linear_agc
from .colormap import apply_palette


@dataclass
class SensorParams:
    """Illustrative LWIR-bolometer-class defaults -- see docs/thermal_camera_model.md SS13.
    Not calibrated against a real sensor; that's Phase 7 (docs/isaacsim_checklist.md).
    """
    band_m: tuple[float, float] = (8e-6, 14e-6)
    t_env_k: float = 290.0
    t_atm_k: float = 290.0
    gamma_per_m: float = 1e-4       # ~0.1/km, illustrative clear-air LWIR
    f_number: float = 1.2
    tau_optics: float = 0.8
    k_resp: float = 1.0
    detector_offset: float = 0.0
    blur_sigma_px: float = 0.8      # illustrative; deriving this from diffraction_mtf/
                                     # detector_mtf properly is a natural follow-up, not done here
    netd_k: float = 0.04
    bits: int = 14
    fpn_gain_sigma: float = 0.02
    fpn_offset_sigma: float = 0.0   # in raw signal units, at this k_resp scale
    palette: str = "white_hot"
    seed: int | None = None


def _chain_signal(temperature_k, emissivity, range_m, p: SensorParams):
    """SS1-SS5 (emission through detector), no noise -- the deterministic part of the
    chain. Reused both by render_frame and by estimate_d_signal_d_t.
    """
    l_point = surface_leaving_radiance(temperature_k, emissivity, p.t_env_k, p.band_m)
    l_atm = band_radiance(p.t_atm_k, p.band_m)
    l_received = apply_atmosphere(l_point, p.gamma_per_m, range_m, l_atm)
    irradiance = focal_plane_irradiance(l_received, p.f_number, p.tau_optics)
    return linear_detector_signal(irradiance, p.k_resp, p.detector_offset)


def estimate_d_signal_d_t(p: SensorParams, t_ref_k: float = 300.0, delta_k: float = 0.5) -> float:
    """Numerically estimate dSignal/dT at a reference scene temperature, for SS8's
    NETD -> noise-sigma conversion. Evaluated at emissivity=1, range=0 (i.e. ignoring
    atmosphere for this reference derivative -- a deliberate simplification).
    """
    s_plus = _chain_signal(t_ref_k + delta_k, 1.0, 0.0, p)
    s_minus = _chain_signal(t_ref_k - delta_k, 1.0, 0.0, p)
    return float((s_plus - s_minus) / (2 * delta_k))


def render_frame(temperature_k: np.ndarray, emissivity: np.ndarray, range_m: np.ndarray,
                  params: SensorParams | None = None) -> np.ndarray:
    """Full per-pixel chain: emission -> atmosphere -> optics -> detector -> blur ->
    noise -> quantize -> AGC -> colormap. Returns an (H, W, 3) uint8 image.

    Args:
        temperature_k, emissivity, range_m: (H, W) arrays -- the point-wise scene
            fields this whole project exists to support (SS0).
        params: sensor/scene parameters; SensorParams() defaults if omitted.
    """
    p = params or SensorParams()
    rng = np.random.default_rng(p.seed)

    signal = _chain_signal(temperature_k, emissivity, range_m, p)
    signal = apply_system_blur(signal, p.blur_sigma_px)

    sigma = netd_noise_sigma(p.netd_k, estimate_d_signal_d_t(p))
    signal = add_temporal_noise(signal, sigma, rng=rng)
    signal = add_fixed_pattern_noise(signal, p.fpn_gain_sigma, p.fpn_offset_sigma, rng=rng)

    v_min, v_max = float(np.min(signal)), float(np.max(signal))
    if v_max <= v_min:
        v_max = v_min + 1e-9
    dn_raw = quantize(signal, p.bits, v_min, v_max)

    dn8 = linear_agc(dn_raw)
    return apply_palette(dn8, p.palette)
