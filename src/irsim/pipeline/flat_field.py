"""Calibrate the camera's own flat-field correction from two synthetic blackbodies (§11.2).

docs/physics-model.md §11.1, §11.2; ADR 0021 (the DN level convention); roadmap M9.12.

`irsim.isp.TwoPointNuc` has existed since M5 and nothing applied it, which left the **picture**
carrying every fixed spatial structure the optics and the detector put into the DN plane -- most
visibly cos⁴ vignetting, 21 % at the corner of the Boson's 14 mm lens. The radiometric branch never
showed it, because `invert_optics` divides cos⁴ out per pixel analytically, so apparent temperature
came out flat across a row while the 8-bit image had dark corners. No real camera looks like that:
every one of them flat-fields before the AGC, which is exactly why no public thermal clip shows
vignetting.

**The correction belongs to the display branch alone**, and that is a decision rather than a
convenience. §11.1 forks the two branches after the ADC and the radiometric branch already removes
the optics analytically -- an *ideal* two-point correction done in closed form instead of from two
frames. Applying a measured correction to it as well would divide cos⁴ out twice. So `dn16` stays
the raw ADC plane, the radiometric outputs are untouched, and the flat field is what the camera's
own ISP does to the picture it shows you.

The coefficients come from the real forward chain at two blackbody temperatures with the noise off
-- the same thing a factory calibration does with two real blackbodies -- rather than from the
analytic cos⁴ field. That matters: a correction built from the formula would remove exactly the
term the formula describes and nothing else, and would therefore be silent about any other fixed
structure the chain grows later. Built from frames, it removes whatever is actually there.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from irsim.isp.nuc import TwoPointNuc
from irsim.optics.stage import apply_optics
from irsim.pipeline.core import PipelineConfig

__all__ = ["uniform_signal_dn", "calibrate_flat_field"]


def uniform_signal_dn(config: PipelineConfig, temperature_k: float) -> NDArray[np.float32]:
    """The noiseless DN plane a uniform blackbody at ``temperature_k`` produces.

    A blackbody fills the field, so ε = 1 everywhere and the scene radiance is one number; what
    makes the resulting plane non-uniform is entirely the camera -- cos⁴, the optical PSF at the
    edges, the self-emission pedestal, the detector's transfer. That is precisely what a
    flat-field correction is for.
    """
    sensor = config.sensor.sensor
    k = config.supersample
    rows, cols = sensor.fpa_shape
    radiance = float(config.lut.lookup(np.float64(temperature_k), config.quantity)[()])
    plane = np.full((rows * k, cols * k), radiance, dtype=np.float64)
    lb_housing = float(config.lut.lookup(config.t_housing_cal_k, config.quantity)[()])
    flux = apply_optics(plane, sensor, lb_housing, supersample=k, psf=config.psf)
    return np.asarray(config.detector.noiseless_signal_dn(flux), dtype=np.float32)


def calibrate_flat_field(
    config: PipelineConfig,
    t_low_k: float | None = None,
    t_high_k: float | None = None,
) -> TwoPointNuc:
    """Two-point coefficients for this camera, from two synthetic blackbody frames.

    Defaults to the ends of the radiometric range the ADC spans (ADR 0021), which is what the two
    blackbodies of a bench calibration would sit at. The pedestal is restored, so the corrected
    plane occupies the same DN range as the raw one and the AGC downstream sees the units it was
    written for -- without it a flat-fielded frame would read zero on a cold scene and the
    histogram would move for reasons that have nothing to do with the scene.
    """
    from irsim.pipeline.core import RADIOMETRIC_RANGE_K

    low = RADIOMETRIC_RANGE_K[0] if t_low_k is None else float(t_low_k)
    high = RADIOMETRIC_RANGE_K[1] if t_high_k is None else float(t_high_k)
    if not high > low:
        raise ValueError(f"the hot blackbody must be hotter: {high} K is not above {low} K")
    return TwoPointNuc.calibrate(
        uniform_signal_dn(config, low), uniform_signal_dn(config, high), restore_pedestal=True
    )
