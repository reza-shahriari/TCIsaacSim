"""Fire -- a volumetric, non-opaque emitter, not a surface at a temperature.

Implements docs/thermal_object_dynamics.md SS5.
"""
from __future__ import annotations
import numpy as np


def flicker_signal(t_s: float, base_freq_hz: float = 15.0, seed: int | None = None) -> float:
    """Noise-driven buoyant-flame "puffing" oscillation, ~10-20 Hz. SS5.

    Unit test target (docs/isaacsim_checklist.md 5.4): plotting several
    seconds of output should show oscillation in the expected frequency
    range, not a smooth/static line.
    """
    raise NotImplementedError("TODO: implement flicker noise + modulation, SS5")


def volumetric_emission(t_flame_k: float, kappa_soot: float, path_length_m: float,
                         wavelength_m: float) -> float:
    """L_flame = L_bb(T_flame) * (1 - exp(-kappa_soot * path_length)). SS5.

    Same structural form as thermal_physics.atmosphere.apply_atmosphere --
    a flame is a small, hot, local "atmosphere" that emits and attenuates
    over a short path.
    """
    raise NotImplementedError("TODO: implement volumetric flame emission, SS5")
