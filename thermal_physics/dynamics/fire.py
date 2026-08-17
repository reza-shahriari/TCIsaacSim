"""Fire -- a volumetric, non-opaque emitter, not a surface at a temperature.

Implements docs/thermal_object_dynamics.md SS5.
"""
from __future__ import annotations
import numpy as np

from ..emission import planck_radiance


def flicker_signal(t_s: float, base_freq_hz: float = 15.0, seed: int = 0) -> float:
    """Noise-modulated oscillation in [0, 1], mimicking buoyant-flame "puffing"
    (~10-20 Hz, SS5). Deterministic given (t_s, base_freq_hz, seed): built from
    smooth 1D value noise (interpolated between hashed grid points) combined
    with a base sinusoid, rather than pure noise or a pure sine wave.
    """
    grid_dt = 1.0 / (4 * base_freq_hz)
    idx = t_s / grid_dt
    i0 = int(np.floor(idx))
    frac = idx - i0

    def hashed_noise(i: int) -> float:
        rng = np.random.default_rng(abs(hash((seed, i))) % (2**32))
        return rng.uniform(-1.0, 1.0)

    n0, n1 = hashed_noise(i0), hashed_noise(i0 + 1)
    smooth_frac = frac * frac * (3 - 2 * frac)  # smoothstep
    noise = n0 + (n1 - n0) * smooth_frac

    oscillation = np.sin(2 * np.pi * base_freq_hz * t_s)
    signal = 0.6 * oscillation + 0.4 * noise
    return float(0.5 * (signal + 1.0))


def volumetric_emission(t_flame_k: float, kappa_soot: float, path_length_m: float,
                         wavelength_m: float) -> float:
    """L_flame = L_bb(T_flame) * (1 - exp(-kappa_soot * path_length)). SS5.

    Same structural form as thermal_physics.atmosphere.apply_atmosphere -- a
    flame is a small, hot, local "atmosphere" that emits and attenuates over
    a short path.
    """
    l_bb = planck_radiance(wavelength_m, t_flame_k)
    return l_bb * (1 - np.exp(-kappa_soot * path_length_m))
