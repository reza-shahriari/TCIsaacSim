"""Semi-transparent surfaces: the second ray and per-pixel closure (M7.15).

§4.4's headline capability: a windshield is opaque in LWIR and shows its own temperature, and
transparent in SWIR where it shows the driver behind it. That only works if τ is carried
separately from ρ -- fold it into (1 − ε) and both bands render identically, plausibly, and
wrongly. These tests pin the mixing coefficients, the closure guard, and the fact that turning
the feature on changes nothing for opaque scenes.

docs/physics-model.md §4.4, §4.1, §12.3; ADR 0046
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.gbuffer import GBuffer
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialLibrary, MaterialTable, surface_radiance
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.radiance import band_radiance
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

# (eps, rho, tau) triples from the roadmap: opaque glass, a SWIR windshield, and a nearly
# transparent window.
TRIPLES = [(0.88, 0.12, 0.0), (0.1, 0.2, 0.7), (0.02, 0.03, 0.95)]


@pytest.mark.parametrize("eps, rho, tau", TRIPLES)
def test_isothermal_enclosure_holds_with_transmission(
    tophat_lwir_lut: BandLUT, eps: float, rho: float, tau: float
) -> None:
    """Surface, environment and whatever is behind it all at one temperature: the pixel reads
    that temperature within 1 mK, whatever the split between emission, reflection and
    transmission. This is the identity that fails the moment the three terms stop closing."""
    t_s = 295.0
    l_b = float(tophat_lwir_lut.lookup(np.float64(t_s))[()])
    out = float(surface_radiance(eps, rho, tau, l_b, l_b, l_b))
    reading = float(tophat_lwir_lut.apparent_temperature(np.asarray(out))[()])
    assert abs(reading - t_s) * 1e3 < 1.0, (eps, rho, tau, reading)


def test_closure_violation_is_refused() -> None:
    with pytest.raises(ValueError, match="closure"):
        surface_radiance(0.88, 0.13, 0.0, 50.0, 40.0)  # sums to 1.01
    with pytest.raises(ValueError, match="closure"):
        surface_radiance(0.1, 0.2, 0.8, 50.0, 40.0)  # sums to 1.10
    with pytest.raises(ValueError, match="closure"):
        surface_radiance(
            np.array([0.9, 0.5]), np.array([0.1, 0.1]), np.array([0.0, 0.0]), 50.0, 40.0
        )
    # and a legitimate triple is not refused for float32 packing noise
    eps = np.float32(0.9)
    surface_radiance(eps, np.float32(1.0) - eps, np.float32(0.0), 50.0, 40.0)


def test_glass_shows_itself_in_lwir_and_the_driver_in_swir() -> None:
    """dL/dL_behind is exactly τ, so for the committed windshield it is 0 in LWIR and 0.70 in
    SWIR: the same material, the same code path, opposite behaviour because the band differs."""
    glass = MaterialLibrary.load()["glass_windshield"]

    def sensitivity(band: str) -> float:
        props = glass.band_properties(band)
        args = (props.emissivity, props.reflectance, props.transmittance, 50.0, 30.0)
        a = float(surface_radiance(*args, 10.0))
        b = float(surface_radiance(*args, 20.0))
        return (b - a) / 10.0

    assert sensitivity("lwir") == pytest.approx(0.0, abs=1e-12)
    assert sensitivity("swir") == pytest.approx(0.70, abs=1e-6)
    assert sensitivity("nir") == pytest.approx(0.77, abs=1e-6)
    assert sensitivity("mwir") == pytest.approx(0.02, abs=1e-6)
    # asphalt is opaque in every band: nothing behind it can ever reach the sensor
    asphalt = MaterialLibrary.load()["asphalt_dry"]
    for band in ("nir", "swir", "mwir", "lwir"):
        assert asphalt.band_properties(band).transmittance == 0.0


def _table() -> MaterialTable:
    """ids 1..3 carry the roadmap triples; id 0 stays the UNMAPPED sentinel."""
    eps = np.array([np.nan, 0.88, 0.1, 0.02], dtype=np.float32)
    tau = np.array([np.nan, 0.0, 0.7, 0.95], dtype=np.float32)
    return MaterialTable(emissivity=eps, transmittance=tau, band_id="lwir")


def test_properties_for_derives_reflectance_and_handles_the_sky() -> None:
    table = _table()
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    eps, rho, tau = table.properties_for(ids)
    assert eps.dtype == rho.dtype == tau.dtype == np.float32
    np.testing.assert_allclose(eps + rho + tau, 1.0, atol=1e-6)
    np.testing.assert_allclose(tau[0], [0.0, 0.7, 0.95], atol=1e-7)
    np.testing.assert_allclose(rho[0], [0.12, 0.2, 0.03], atol=1e-6)
    # a table with no transmittance column is opaque, and rho is still derived
    plain = MaterialTable.from_mapping({1: 0.9, 2: 0.5})
    e2, r2, t2 = plain.properties_for(np.array([[1, 2]], dtype=np.int32))
    assert np.all(t2 == 0.0)
    np.testing.assert_allclose(r2[0], [0.1, 0.5], atol=1e-7)
    # under the sky mask a pixel is blackbody-equivalent whatever its id says
    sky = np.array([[True, False, False]])
    eps_s, rho_s, tau_s = table.properties_for(np.zeros((1, 3), dtype=np.int32) + ids, sky)
    assert eps_s[0, 0] == 1.0 and rho_s[0, 0] == 0.0 and tau_s[0, 0] == 0.0


def _planes(
    shape: tuple[int, int], ids: np.ndarray, behind_k: float | None
) -> dict[str, np.ndarray]:
    planes = {
        "temperature_k": np.full(shape, 300.0, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": ids,
        "sky_view_factor": np.full(shape, 0.5, np.float32),
    }
    if behind_k is not None:
        planes["radiance_behind"] = np.full(shape, behind_k, np.float32)
    return planes


def test_second_ray_plane_rides_the_existing_contract() -> None:
    """`radiance_behind` needs no contract change: the G-buffer already admits any plane whose
    name starts with "radiance", and treats it as precision-critical, so float16 is refused."""
    shape = (2, 3)
    ids = np.ones(shape, dtype=np.int32)
    g = GBuffer.from_dict(_planes(shape, ids, behind_k=25.0))
    assert "radiance_behind" in g.extra and g.extra["radiance_behind"].dtype == np.float32
    bad = _planes(shape, ids, behind_k=25.0)
    bad["radiance_behind"] = bad["radiance_behind"].astype(np.float16)
    with pytest.raises(TypeError, match="float16"):
        GBuffer.from_dict(bad)


def test_stage_one_mixes_the_second_ray_and_is_a_no_op_when_absent(
    tophat_lwir_lut: BandLUT,
) -> None:
    shape = (2, 3)
    table = _table()
    ids = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)
    t = np.full(shape, 300.0, np.float32)
    env = np.full(shape, 30.0, np.float32)
    lb = float(tophat_lwir_lut.lookup(np.float64(300.0))[()])

    without = band_radiance(t, ids, table, tophat_lwir_lut, l_env=env)
    expected_two_term = np.array(
        [[e * lb + (1.0 - e) * 30.0 for e in (0.88, 0.1, 0.02)]] * 2, dtype=np.float32
    )
    np.testing.assert_allclose(without, expected_two_term, rtol=1e-6)

    behind = np.full(shape, 80.0, np.float32)
    with_behind = band_radiance(t, ids, table, tophat_lwir_lut, l_env=env, l_behind=behind)
    eps, rho, tau = table.properties_for(ids)
    np.testing.assert_allclose(
        with_behind, surface_radiance(eps, rho, tau, lb, env, behind), rtol=1e-6
    )
    # the opaque material is untouched by what is behind it; the transparent ones are not
    assert with_behind[0, 0] == pytest.approx(without[0, 0], rel=1e-7)
    assert with_behind[0, 1] > without[0, 1] and with_behind[0, 2] > without[0, 2]
    # L_behind == L_env reproduces the two-term form exactly, which is why opaque scenes and
    # every existing golden are unaffected by this feature
    same = band_radiance(t, ids, table, tophat_lwir_lut, l_env=env, l_behind=env)
    np.testing.assert_array_equal(same, without)


def test_through_run_frame_a_transparent_pixel_follows_what_is_behind_it(
    tophat_lwir_lut: BandLUT, aerial_sky
) -> None:  # type: ignore[no-untyped-def]
    """Plumbing: the plane reaches stage 1 through run_frame, and only the τ > 0 pixels move."""
    shape = (8, 16)
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=16, height=8)
    d["sensor"]["optics"]["supersample_factor"] = 1
    sensor_cfg = SensorConfig.model_validate(d)
    ids = np.ones(shape, dtype=np.int32)
    ids[:, 8:] = 3  # tau = 0.95
    cfg = PipelineConfig.from_sensor(
        sensor_cfg,
        _table(),
        lut=tophat_lwir_lut,
        noise_enabled=False,
        psf_enabled=False,
        sky=aerial_sky,
    )
    # A state each: this is a steady-state radiometric identity, and since M9.13 the membrane
    # IIR is on `run_frame`'s path, so reusing one state would read the second frame 81 % of the
    # way through its settle (measured 65.50 against the 80.75 this asserts -- exactly alpha).
    cold = run_frame(
        _planes(shape, ids, behind_k=5.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
    )
    hot = run_frame(
        _planes(shape, ids, behind_k=90.0), cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k)
    )
    assert cold.radiance is not None and hot.radiance is not None
    opaque_delta = float(np.abs(hot.radiance[:, :8] - cold.radiance[:, :8]).max())
    clear_delta = float((hot.radiance[:, 8:] - cold.radiance[:, 8:]).mean())
    assert opaque_delta < 1e-4, "an opaque material cannot see through itself"
    assert clear_delta == pytest.approx(0.95 * (90.0 - 5.0), rel=1e-3)

    # and a second-ray plane with no environment model is an error, not a silent no-op
    no_sky = PipelineConfig.from_sensor(
        sensor_cfg, _table(), lut=tophat_lwir_lut, noise_enabled=False, psf_enabled=False
    )
    with pytest.raises(ValueError, match="needs an environment model"):
        run_frame(_planes(shape, ids, behind_k=5.0), no_sky, PipelineState())
