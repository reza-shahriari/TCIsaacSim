"""Night illumination sources: airglow, and a first-order moon (§5.5).

Two sources, and they behave differently in ways that matter more than their levels:

* **Airglow** is chemiluminescence in the upper atmosphere -- H + O₃ → OH* + O₂ at ~87 km, plus
  O₂(a¹Δ) at 1.27 µm -- so it is a **hemisphere source with no direction**. It casts no shadow,
  does not care where the sun is, and cloud attenuates it without cutting it off, because cloud
  scatters rather than absorbs (§5.5 says this explicitly). Its spectrum lands between 1 and 2 µm,
  which is why a SWIR camera works on a moonless night and a visible one does not.
* **The moon** is reflected sunlight, so it has the sun's shape (reddened, which is deferred), a
  direction, a phase and a horizon.

The **level** of airglow is configuration -- §5.5's 3.5-39 nW cm⁻², default ~10 -- and the shape
file says how it is distributed. The level is defined over the shape file's own support (ADR
0065), so a band takes its fraction: 66 % of it reaches a 0.9-1.7 µm InGaAs camera and 89 % an
extended one to 2.5 µm, because the strongest OH sequence sits at 1.4-2.0 µm and a standard
1.7 µm cut-off throws away much of it.

The full moon's irradiance is not authored: it follows from the visual magnitudes, m_sun = −26.74
and m_moon = −12.74, so the flux ratio is 10^(14/2.5) = 3.98e5 and E_full = 3.42e-3 W m⁻². The
phase law is Lane & Irvine's, Δm(α) = 0.026|α| + 4e-9 α⁴, which puts a quarter moon at 0.091 of
full -- the familiar "a quarter moon is a tenth of a full moon, not a half".

docs/physics-model.md §5.5; ADR 0065
"""

from __future__ import annotations

import hashlib
import math
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SOLAR_CONSTANT_W_M2
from irsim.radiometry.spectral_table import SpectralTable

__all__ = [
    "AIRGLOW_SHAPE_FILE",
    "SUN_V_MAGNITUDE",
    "FULL_MOON_V_MAGNITUDE",
    "FULL_MOON_IRRADIANCE_W_M2",
    "CLOUD_FLOOR",
    "AirglowSpectrum",
    "load_airglow_spectrum",
    "lunar_phase_factor",
    "phase_angle_deg",
    "cloud_attenuation",
    "airglow_variation",
]

AIRGLOW_SHAPE_FILE = "spectra/airglow_oh_meinel.csv"

#: Visual magnitudes at mean distance. The only two numbers the lunar level rests on.
SUN_V_MAGNITUDE = -26.74
FULL_MOON_V_MAGNITUDE = -12.74
#: E_full = S / 10^((m_moon − m_sun)/(−2.5)). 3.42e-3 W m⁻², i.e. the sun over 3.98e5.
FULL_MOON_IRRADIANCE_W_M2 = SOLAR_CONSTANT_W_M2 / 10.0 ** (
    (SUN_V_MAGNITUDE - FULL_MOON_V_MAGNITUDE) / -2.5
)

#: What solid overcast still returns. §5.5: "cloud attenuates it, but it does not have a
#: directional shadow" -- cloud scatters airglow rather than absorbing it, and a night sky under
#: thick cloud is dimmer, never black. ESTIMATED (ADR 0065); it is the floor, not a measurement.
CLOUD_FLOOR = 0.10


@dataclass(frozen=True)
class AirglowSpectrum(SpectralTable):
    """A **relative** shape s(λ). The scale of the file is arbitrary; the level is config."""

    def band_fraction(self, response: Any, photon: bool = False) -> float:
        """∫R s dλ / ∫s dλ -- the share of the configured level this band sees.

        In the photon form the numerator carries λ/(hc) and the denominator does not, so the
        result is photons per watt of configured level: the two are not interchangeable and the
        caller picks one through :meth:`band_irradiance` or :meth:`band_photon_irradiance`.
        """
        return self.band_integral(response, photon=photon) / self.integral()

    def band_irradiance(self, response: Any, level_w_m2: float) -> float:
        """E_B = level × ∫R s dλ / ∫s dλ, W m⁻²."""
        if level_w_m2 < 0.0:
            raise ValueError("airglow level must be non-negative")
        return level_w_m2 * self.band_fraction(response, photon=False)

    def band_photon_irradiance(self, response: Any, level_w_m2: float) -> float:
        """E_B,q, photons s⁻¹ m⁻², with hc/λ taken **inside** the integral."""
        if level_w_m2 < 0.0:
            raise ValueError("airglow level must be non-negative")
        return level_w_m2 * self.band_fraction(response, photon=True)

    def band(self, response: Any, level_w_m2: float, quantity: str) -> float:
        if quantity in ("lb", "dlb_dt"):
            return self.band_irradiance(response, level_w_m2)
        if quantity in ("lb_q", "dlb_q_dt"):
            return self.band_photon_irradiance(response, level_w_m2)
        raise ValueError(f"unknown quantity {quantity!r}")


def load_airglow_spectrum(path: str | os.PathLike[str]) -> AirglowSpectrum:
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"airglow shape {p} does not exist")
    text = p.read_text(encoding="utf-8")
    lam: list[float] = []
    val: list[float] = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [c.strip() for c in stripped.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"{p} line {i}: expected 'lambda_um,value', got {stripped!r}")
        try:
            lam.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError:
            if not lam:  # the column header
                continue
            raise ValueError(f"{p} line {i}: non-numeric entry {stripped!r}") from None
    if len(lam) < 2:
        raise ValueError(f"{p}: at least two rows are needed")
    wl = np.asarray(lam, dtype=np.float64)
    s = np.asarray(val, dtype=np.float64)
    if np.any(np.diff(wl) <= 0.0):
        raise ValueError(f"{p}: wavelengths must be strictly increasing")
    if np.any(s < 0.0):
        raise ValueError(f"{p}: a spectral shape must be non-negative")
    if not float(np.trapezoid(s, wl)) > 0.0:
        raise ValueError(f"{p}: shape integrates to zero")
    return AirglowSpectrum(
        wavelength_um=wl,
        values=s,
        source_path=str(p),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def phase_angle_deg(phase_fraction: Any) -> NDArray[np.float64]:
    """Illuminated fraction f (0 new, 1 full) → sun–moon–observer phase angle α in degrees."""
    f = np.asarray(phase_fraction, dtype=np.float64)
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("phase_fraction must lie in [0, 1] (0 new, 1 full)")
    return np.asarray(np.degrees(np.arccos(np.clip(2.0 * f - 1.0, -1.0, 1.0))))


def lunar_phase_factor(phase_fraction: Any) -> NDArray[np.float64]:
    """Lane & Irvine: Φ = 10^(−0.4 (0.026|α| + 4e-9 α⁴)), 1 at full moon.

    Strongly non-linear on purpose. Half the disc lit is **not** half the light: the quarter moon
    comes out at 0.091 of full, because the terminator is all shadowed relief and the full moon
    gets a coherent backscatter surge. A model that used the illuminated fraction directly would
    make quarter-moon scenes five times too bright.
    """
    alpha = phase_angle_deg(phase_fraction)
    delta_mag = 0.026 * np.abs(alpha) + 4.0e-9 * alpha**4
    return np.asarray(10.0 ** (-0.4 * delta_mag))


def cloud_attenuation(cloud_fraction: Any, k_cloud: float) -> NDArray[np.float64]:
    """1 − k_cloud · cloud · (1 − CLOUD_FLOOR): attenuates, never extinguishes (§5.5)."""
    cloud = np.asarray(cloud_fraction, dtype=np.float64)
    if np.any((cloud < 0.0) | (cloud > 1.0)):
        raise ValueError("cloud_fraction must lie in [0, 1]")
    if not 0.0 <= k_cloud <= 1.0:
        raise ValueError("k_cloud must lie in [0, 1]")
    return np.asarray(1.0 - k_cloud * cloud * (1.0 - CLOUD_FLOOR))


#: Periods of the three components of the airglow's slow variation (§5.5 "slow temporal
#: variation"). Chosen an order of magnitude apart so the sum does not repeat within a scene.
VARIATION_PERIODS_S = (1800.0, 5400.0, 17000.0)
VARIATION_AMPLITUDE = 0.30


def airglow_variation(
    t_s: Any, seed: int = 0, amplitude: float = VARIATION_AMPLITUDE
) -> NDArray[np.float64]:
    """A deterministic multiplier around 1, never negative and never zero.

    Three sinusoids with seeded phases, so the same seed gives the same night and a scene that
    reruns is reproducible. Not noise: airglow drifts over tens of minutes, it does not flicker,
    and a per-frame random draw would look like sensor noise rather than like the sky.
    """
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("amplitude must lie in [0, 1)")
    t = np.asarray(t_s, dtype=np.float64)
    phases = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=len(VARIATION_PERIODS_S))
    total = np.zeros_like(t)
    for period, phase in zip(VARIATION_PERIODS_S, phases, strict=True):
        total = total + np.cos(2.0 * math.pi * t / period + phase)
    return np.asarray(1.0 + amplitude * total / len(VARIATION_PERIODS_S))
