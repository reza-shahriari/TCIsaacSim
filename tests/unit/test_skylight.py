"""M11.10: scattered sunlight in the sky, and the black-sky bug it fixes.

`SkyModel` is a *thermal* sky -- the atmospheric column's own emission -- which is right for LWIR
and essentially zero at 0.9 um. Rendering a NIR scene with it produced a black sky under a brightly
sunlit target, which is backwards: in the near infrared the daytime sky is the brightest thing in
the frame and an aircraft is usually a dark silhouette against it. These tests pin the size of the
term in each band, the reason the emissive one is left alone, and the unit arithmetic that makes
the isotropic form reproduce the weather's own DHI.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.atmosphere.skylight import (
    RAYLEIGH_EXPONENT,
    DiffuseSkylight,
    diffuse_shape,
    skylight_for_sensor,
)
from irsim.config.loader import load_sensor_config
from irsim.radiometry.solar import load_solar_spectrum
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
GROUND = DATA / "spectra" / "solar" / "direct_normal_am1p5.csv"
SENSORS = {
    "nir": "example_nir_si_1280",
    "swir": "example_swir_ingaas_640",
    "mwir": "example_mwir_insb_640",
    "lwir": "flir_boson_640_lwir",
}


def _sensor(band: str):  # type: ignore[no-untyped-def]
    return load_sensor_config(REPO / "configs" / "sensors" / f"{SENSORS[band]}.yaml")


def _quantity(cfg) -> str:  # type: ignore[no-untyped-def]
    return cfg.sensor.quantity


def test_the_isotropic_form_reproduces_the_measured_dhi() -> None:
    """The arithmetic that justifies isotropy: ∫ L cosθ dΩ over the hemisphere is L·π exactly.

    So ``L = f_B · DHI / π`` puts back exactly the diffuse irradiance the weather file measured.
    That is the whole case for an isotropic sky here -- it is the one angular distribution
    guaranteed right in the integral, and the integral is what was measured.
    """
    cfg = _sensor("nir")
    sky = DiffuseSkylight.for_sensor(cfg, "lb_q", data_dir=DATA)
    dhi = 120.0
    radiance = float(sky.radiance(dhi))
    # Integrate L cos(theta) over the hemisphere numerically and recover f_B · DHI.
    theta = np.linspace(0.0, math.pi / 2.0, 2001)
    irradiance = (
        radiance * 2.0 * math.pi * float(np.trapezoid(np.cos(theta) * np.sin(theta), theta))
    )
    assert irradiance == pytest.approx(radiance * math.pi, rel=1e-6)
    assert radiance * math.pi == pytest.approx(sky.per_dhi * math.pi * dhi, rel=1e-12)


def test_the_diffuse_spectrum_is_bluer_than_the_beam_that_made_it() -> None:
    """Rayleigh goes as λ⁻⁴, so skylight is far bluer than the direct beam -- which is exactly why
    a near-infrared band gets a much smaller share of DHI than of DNI, and why weighting the
    direct spectrum instead would over-light a NIR sky."""
    spectrum = load_solar_spectrum(GROUND)
    shape = diffuse_shape(spectrum)
    assert shape.sha256 == spectrum.sha256, "the shape must stay traceable to its source file"
    response = load_spectral_response(_sensor("nir").sensor.band.spectral_response)
    direct_share = spectrum.band_irradiance(response) / float(
        np.trapezoid(spectrum.irradiance_w_m2_um, spectrum.wavelength_um)
    )
    diffuse_share = shape.band_irradiance(response) / float(
        np.trapezoid(shape.irradiance_w_m2_um, shape.wavelength_um)
    )
    assert diffuse_share < direct_share, (diffuse_share, direct_share)
    assert RAYLEIGH_EXPONENT == 4.0


def test_the_term_matters_in_the_reflective_bands_and_not_in_lwir() -> None:
    """The measurement that decides where this model is load-bearing.

    Against a clear-sky LWIR column at ~50 W m⁻² sr⁻¹ the scattered term is eight orders below,
    which is why an emissive band gets ``None`` and keeps rendering exactly as it did.
    """
    dhi = 120.0
    levels = {}
    for band in SENSORS:
        cfg = _sensor(band)
        sky = DiffuseSkylight.for_sensor(cfg, _quantity(cfg), data_dir=DATA)
        levels[band] = float(sky.radiance(dhi))
    assert levels["lwir"] < 1e-5, levels["lwir"]  # against ~50 W/m2/sr of column emission
    assert levels["nir"] > 1e18 and levels["swir"] > 1e18
    # Photon units, so NIR and SWIR are comparable to each other and both dwarf MWIR's share.
    assert levels["nir"] > 100.0 * levels["mwir"]


def test_an_emissive_band_gets_no_skylight_at_all() -> None:
    """`None`, not a small number: it is what keeps every LWIR golden bit-identical."""
    assert skylight_for_sensor(_sensor("lwir"), "lb", DATA) is None
    for band in ("nir", "swir", "mwir"):
        cfg = _sensor(band)
        assert skylight_for_sensor(cfg, _quantity(cfg), DATA) is not None


def test_a_mismatched_quantity_is_refused_rather_than_scaled() -> None:
    """`lb` and `lb_q` differ by ~1e19. A mismatch would render a plausible sky at a wrong
    brightness that the AGC would then normalise away, so the sky model refuses it."""
    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.atmosphere.library import load_atmosphere_preset
    from irsim.atmosphere.sky import SkyModel
    from irsim.config.environment import load_environment_preset
    from irsim.radiometry.lut import BandLUT
    from irsim.thermal.weather_io import load_weather_csv

    cfg = _sensor("nir")
    lut = BandLUT.build(load_spectral_response(cfg.sensor.band.spectral_response), n=1001)
    weather = load_weather_csv(DATA / "weather" / "clear_midlat_summer_48h.csv")
    layered = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"nir": lut})
    environment = load_environment_preset("clear_dry")
    wrong = DiffuseSkylight.for_sensor(cfg, "lb", data_dir=DATA)
    with pytest.raises(ValueError, match="1e19"):
        SkyModel(layered, environment, "nir", lut, "lb_q", skylight=wrong)


def test_the_sky_model_is_brighter_with_the_term_than_without_it() -> None:
    """End to end on the real SkyModel: the term reaches `clear_radiance`, and therefore reaches
    both the sky background *and* the reflected environment term, from one place."""
    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.atmosphere.library import load_atmosphere_preset
    from irsim.atmosphere.sky import SkyModel
    from irsim.config.environment import load_environment_preset
    from irsim.radiometry.lut import BandLUT
    from irsim.thermal.weather_io import load_weather_csv

    cfg = _sensor("nir")
    lut = BandLUT.build(load_spectral_response(cfg.sensor.band.spectral_response), n=1001)
    weather = load_weather_csv(DATA / "weather" / "clear_midlat_summer_48h.csv")
    layered = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), weather, {"nir": lut})
    environment = load_environment_preset("clear_dry")
    sky = DiffuseSkylight.for_sensor(cfg, "lb_q", data_dir=DATA)
    noon = 12 * 3600.0

    thermal_only = SkyModel(layered, environment, "nir", lut, "lb_q")
    with_scatter = SkyModel(layered, environment, "nir", lut, "lb_q", skylight=sky)
    elevation = np.radians([5.0, 30.0, 90.0])
    dim = thermal_only.clear_radiance(noon, elevation)
    bright = with_scatter.clear_radiance(noon, elevation)
    assert np.all(bright > dim)
    # Isotropic: the *difference* is the same at every elevation, which is the model's own claim.
    delta = bright - dim
    assert np.allclose(delta, delta[0], rtol=1e-12)
    # And it dominates: the thermal sky is nothing in this band.
    assert float(dim.max()) < 1e-6 * float(bright.min())
