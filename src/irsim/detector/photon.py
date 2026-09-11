"""Ideal photon-detector transfer: band photon radiance → photoelectrons → DN.

docs/physics-model.md §9.1:

    N_e = η · A_d · t_int · π τ_opt / (4F² + 1) · cos⁴θ · L_{q,B}  +  N_dark
    DN  = min(2^bits − 1, ⌊ N_e / N_well · 2^bits ⌋)

η is ``quantum_efficiency``, applied **here and nowhere else** (R(λ) is a peak-1 shape, ADR 0009).
The aperture factor is imported from :mod:`irsim.optics` (non-negotiable #5). Input is the
band-integrated *photon* radiance at the sensor (photons s⁻¹ m⁻² sr⁻¹); a plausibility guard refuses
energy-form radiance (~1e20× smaller). Dark current and noise are M4; ``dark_electrons`` is a
hook that defaults to zero.

docs/physics-model.md §9.1, §8.1, §2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.detector.netd import NoiseBudget
from irsim.detector.params import PhotonParams
from irsim.detector.quantise import quantise
from irsim.detector.response import DetectorFrame
from irsim.noise.seeding import NoiseStream, field_normal, noise_rng, stream_key
from irsim.optics.aperture import fpa_irradiance

__all__ = [
    "ENERGY_SCALE_GUARD",
    "photoelectrons",
    "electrons_to_signal_dn",
    "electrons_to_dn",
    "PhotonDetector",
]

# Band photon radiance of any scene above ~150 K in any IR band exceeds 1e12 ph/s/m2/sr; energy
# radiance is < 1e5 W/m2/sr. A value below this is not a photon radiance.
ENERGY_SCALE_GUARD = 1e8


def _photon_radiance(lb_q: object) -> NDArray[np.float64]:
    x = np.asarray(lb_q)
    if x.dtype == np.float16:
        raise TypeError("photon radiance is float16; use float32 or better")
    x = x.astype(np.float64)
    if np.any(x < 0.0):
        raise ValueError("photon radiance cannot be negative")
    if np.any((x > 0.0) & (x < ENERGY_SCALE_GUARD)):
        raise ValueError(
            f"photon radiance {float(x[x > 0].min()):.3e} is below {ENERGY_SCALE_GUARD:.0e}: this "
            "looks like energy-form radiance (W m^-2 sr^-1); the photon detector takes Lb_q (§9.1)"
        )
    return x


def photoelectrons(
    lb_q: object,
    params: PhotonParams,
    f_number: float,
    tau_opt: float,
    cos4: object = 1.0,
    dark_electrons: object = 0.0,
) -> NDArray[np.float64]:
    """N_e per integration time (float64): η A_d t_int Ω_eff τ cos⁴ L_q + N_dark."""
    rate = fpa_irradiance(_photon_radiance(lb_q), f_number, tau_opt, cos4)  # photons s^-1 m^-2
    n_e = params.quantum_efficiency * params.active_area_m2 * params.integration_time_s * rate
    return np.asarray(n_e + np.asarray(dark_electrons, dtype=np.float64), dtype=np.float64)


def electrons_to_signal_dn(n_e: object, params: PhotonParams) -> NDArray[np.float32]:
    """Un-quantised signal N_e / N_well · 2^bits in DN units (float32); noise is added here."""
    x = np.asarray(n_e)
    if x.dtype == np.float16:
        raise TypeError("electron count is float16")
    return np.asarray(
        x.astype(np.float64) / params.well_capacity_e * 2**params.bit_depth, dtype=np.float32
    )


def electrons_to_dn(n_e: object, params: PhotonParams) -> NDArray[np.uint16]:
    """DN = min(2^bits − 1, ⌊N_e / N_well · 2^bits⌋), clipped at 0, uint16 (§9.1)."""
    return quantise(electrons_to_signal_dn(n_e, params), params.bit_depth)


@dataclass(frozen=True)
class PhotonDetector:
    """§9.1 photon detector with its noise: Poisson(signal + dark + background) + Gaussian read
    noise in **electron space**, then DN (ADR 0026; non-negotiable #3).

    ``budget`` is the anchored :class:`NoiseBudget` (σ_gaussian = read noise in e⁻, dark and
    background electrons per integration). Shot noise uses the sequential PCG64 stream (no
    counter-based Poisson yet, ADR 0022); read noise uses the per-pixel hash stream TVH.
    """

    params: PhotonParams
    budget: NoiseBudget

    def __post_init__(self) -> None:
        if self.budget.kind != "photon":
            raise ValueError("PhotonDetector needs a photon NoiseBudget")

    def electrons(self, flux_photons_per_s: NDArray[np.floating]) -> NDArray[np.float64]:
        """Mean electrons per integration from the photon rate on the pixel (η applied here)."""
        phi = np.asarray(flux_photons_per_s)
        if phi.dtype == np.float16:
            raise TypeError("photon flux is float16 (non-negotiable #2)")
        if np.any(phi < 0.0):
            raise ValueError("photon flux cannot be negative")
        n_e = (
            self.params.quantum_efficiency * self.params.integration_time_s * phi.astype(np.float64)
        )
        return n_e + self.budget.dark_electrons + self.budget.background_electrons

    def noiseless_signal_dn(self, flux: NDArray[np.floating]) -> NDArray[np.float32]:
        return electrons_to_signal_dn(self.electrons(flux), self.params)

    def sigma_electrons(self, flux: NDArray[np.floating]) -> NDArray[np.float64]:
        """σ_total per pixel in e⁻: √(N_e + N_dark + N_bg + σ_read²)."""
        return np.sqrt(self.electrons(flux) + self.budget.sigma_gaussian**2)

    def response(
        self, flux: NDArray[np.floating], frame_index: int, sensor_seed: int
    ) -> DetectorFrame:
        mean_e = self.electrons(flux)
        shot = (
            noise_rng(sensor_seed, frame_index, NoiseStream.SHOT).poisson(mean_e).astype(np.float64)
        )
        read = self.budget.sigma_gaussian * field_normal(
            stream_key(sensor_seed, frame_index, NoiseStream.TVH), mean_e.shape
        ).astype(np.float64)
        electrons = np.maximum(shot + read, 0.0)
        signal = electrons_to_signal_dn(electrons, self.params)
        sigma_dn = electrons_to_signal_dn(self.sigma_electrons(flux), self.params)
        return DetectorFrame(
            signal_dn=signal, dn=quantise(signal, self.params.bit_depth), sigma_dn=sigma_dn
        )
