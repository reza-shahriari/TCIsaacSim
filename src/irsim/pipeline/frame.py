"""``run_frame``: the whole CPU reference chain for one frame (docs/physics-model.md §13.4).

    stage 1  band radiance     ε₀ L_B(T) on the k× G-buffer          (irsim.pipeline.radiance)
    stage 2  atmosphere        τL + (1−τ)L_B(T_air) on the k× grid   (irsim.pipeline.atmosphere)
    stage 3  optics            PSF, box ↓k, aperture·cos⁴·A_d, +Φ_self (irsim.optics.stage)
    stage 4  detector          Φ → signal in DN with per-pixel noise    (irsim.detector, ADR 0026)
    stage 5  noise             + correlated 3-D components              (irsim.noise.stage)
    ADC      quantise          floor + clip → uint16                    (irsim.detector.quantise)
    stage 6  ISP               radiometric branch → radiance, T_app     (irsim.isp.radiometric)
             display branch    AGC → gamma → DDE → polarity → palette  (irsim.isp.display)

Outputs follow §12.2 ``outputs``: ``radiance`` (float32, scene band radiance at the native grid),
``apparent_t`` (float32 K, from the float32 signal route), ``dn16`` (uint16), ``display8``
(RGBA8 through the isp block, ADR 0031). A flag set to ``false`` yields ``None`` -- never zeros.
The bolometer path uses the energy-form LUT table; a photon FPA runs the same chain on the
photon table with N_e = η t_int Φ_q (ADR 0021).

docs/physics-model.md §13.4, §12.2 outputs, §16.4 step 3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.detector.params import BolometerParams, PhotonParams
from irsim.detector.quantise import quantise
from irsim.isp.display import run_display_branch
from irsim.isp.radiometric import apparent_temperature
from irsim.optics.stage import apply_optics, invert_optics
from irsim.pipeline.atmosphere import apply_atmosphere_gbuffer
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes
from irsim.pipeline.radiance import band_radiance

__all__ = ["Outputs", "run_frame"]


@dataclass(frozen=True)
class Outputs:
    radiance: NDArray[np.float32] | None
    apparent_t: NDArray[np.float32] | None
    dn16: NDArray[np.uint16] | None
    display8: NDArray[np.uint8] | None  # (H, W, 4) RGBA8
    signal_dn: NDArray[np.float32]  # un-quantised stage-4 signal, always kept for benches
    flux: NDArray[np.float32]  # stage-3 pixel power (W or photons/s), always kept
    isp_hash: str | None = None  # config hash of the isp block that produced display8


def _detector_signal(
    flux: NDArray[np.float32], config: PipelineConfig, state: PipelineState
) -> NDArray[np.float32]:
    """Stage 4 + 5: detector response (per-pixel noise) then the correlated 3-D noise; the
    ideal chain when noise is disabled. Seeded by the sensor's own frame index (ADR 0022)."""
    if not config.noise_enabled:
        return config.detector.noiseless_signal_dn(flux)
    frame = config.detector.response(flux, state.frame_index, config.sensor_seed)
    return config.noise.apply(frame.signal_dn, frame.sigma_dn, state.frame_index)


def _scene_radiance_from_signal(
    signal: NDArray[np.float32], config: PipelineConfig, lb_housing_cal: float
) -> NDArray[np.float32]:
    fpa = config.fpa
    if isinstance(fpa, BolometerParams):
        assert config.calibration is not None
        return config.calibration.radiance_from_signal(signal)
    assert isinstance(fpa, PhotonParams)
    n_e = signal.astype(np.float64) / 2**fpa.bit_depth * fpa.well_capacity_e
    phi_q = n_e / (fpa.quantum_efficiency * fpa.integration_time_s)
    return invert_optics(phi_q, config.sensor.sensor, lb_housing_cal)


def run_frame(planes: Planes, config: PipelineConfig, state: PipelineState) -> Outputs:
    """One frame through stages 1-6 (stage 2 is the identity without an Atmosphere).
    Advances ``state.frame_index``."""
    sensor = config.sensor.sensor
    lut = config.lut
    q = config.quantity
    h, w = sensor.fpa_shape
    k = config.supersample
    t = np.asarray(planes["temperature_k"])
    if t.shape != (h * k, w * k):
        raise ValueError(
            f"G-buffer {t.shape} is not the {k}x supersampled detector grid {(h * k, w * k)}; "
            "set optics.supersample_factor to match the render"
        )

    # stage 1 (k× grid)
    radiance_ss = band_radiance(
        t, planes["material_id"], config.materials, lut, q, sky_mask=planes.get("sky_mask")
    )
    # stage 2 (k× grid): per-ray atmosphere; sky pixels pass through (ADR 0050)
    if config.atmosphere is not None:
        atm_state = config.atmosphere.state(state.t_s)
        l_air = float(lut.lookup(np.float64(atm_state.t_air_k), q)[()])
        radiance_ss = apply_atmosphere_gbuffer(
            radiance_ss,
            planes["distance_m"],
            atm_state.gamma_per_m[sensor.band.band_id],
            l_air,
            sky_mask=planes.get("sky_mask"),
            tau_override=config.tau_override,
        )
    # stage 3
    lb_housing_now = float(lut.lookup(state.housing_temp_k, q)[()])
    flux = apply_optics(radiance_ss, sensor, lb_housing_now, supersample=k, psf=config.psf)
    # stages 4-5 (detector noise, correlated noise); ADC
    signal = _detector_signal(flux, config, state)
    dn16 = quantise(signal, sensor.fpa.bit_depth)
    # stage 6: radiometric branch inverts with the *calibration* housing level (ADR 0021)
    lb_housing_cal = float(lut.lookup(config.t_housing_cal_k, q)[()])
    outputs = sensor.outputs
    radiance = apparent_t = None
    if outputs.radiance_linear or outputs.apparent_temperature:
        scene = _scene_radiance_from_signal(signal, config, lb_housing_cal)
        if outputs.radiance_linear:
            radiance = scene
        if outputs.apparent_temperature:
            apparent_t = apparent_temperature(scene, lut, q)
    display8 = None
    isp_hash = None
    if outputs.display_8:
        display = run_display_branch(dn16, sensor.isp, sensor.fpa.bit_depth)
        display8, isp_hash = display.display8, display.isp_hash
    state.advance()
    return Outputs(
        radiance=radiance,
        apparent_t=apparent_t,
        dn16=dn16 if outputs.dn_16 else None,
        display8=display8,
        signal_dn=signal,
        flux=flux,
        isp_hash=isp_hash,
    )
