"""Stage 1, emission-only band radiance (M3.6, ADR 0018): closed-form blackbody, grey-body
apparent temperature through the LUT inverse, per-pixel material lookup, and the dtype/sentinel
guards."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.gbuffer import GBuffer
from irsim.materials import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, band_radiance_stage
from irsim.pipeline.radiance import BandRadianceStage, band_radiance
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.planck import band_radiance_tophat


def _config(lut: BandLUT, materials: MaterialTable) -> PipelineConfig:
    import pathlib

    import yaml

    from irsim.config.sensor import SensorConfig

    repo = pathlib.Path(__file__).resolve().parents[2]
    sensor = SensorConfig.model_validate(
        yaml.safe_load((repo / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
    )
    return PipelineConfig.from_sensor(sensor, materials, lut=lut)


def test_blackbody_ramp_matches_closed_form_under_5mK(
    tophat_lwir_lut: BandLUT, gbuffer_ramp: dict[str, np.ndarray]
) -> None:
    """ε = 1 over 250-450 K vs band_radiance_tophat (independent series), in mK via dLb/dT.
    A stray π, aperture factor or cos⁴ is a >20 % error here."""
    cfg = _config(tophat_lwir_lut, MaterialTable.constant(1.0))
    out = band_radiance_stage(gbuffer_ramp, cfg, PipelineState())["radiance"]
    assert out.dtype == np.float32 and out.shape == gbuffer_ramp["temperature_k"].shape
    row = gbuffer_ramp["temperature_k"][0].astype(np.float64)
    expected = np.array([band_radiance_tophat(7.5, 13.5, float(t)) for t in row])
    slope = tophat_lwir_lut.lookup(row, "dlb_dt").astype(np.float64)
    err_mk = np.abs(out[0].astype(np.float64) - expected) / slope * 1e3
    assert err_mk.max() < 5.0, f"{err_mk.max():.3f} mK"
    assert np.all(np.isfinite(out)) and np.all(np.diff(out[0]) > 0)


def test_grey_body_apparent_temperature_is_below_kinetic(tophat_lwir_lut: BandLUT) -> None:
    """ε = 0.9 at 300 K: T_app = Lb⁻¹(0.9 Lb(300)) = 293.51 K, strictly below 300 K, within 5 mK."""
    t = np.full((8, 8), 300.0, dtype=np.float32)
    ids = np.ones((8, 8), dtype=np.int32)
    out = band_radiance(t, ids, MaterialTable.constant(0.9), tophat_lwir_lut)
    t_app = tophat_lwir_lut.apparent_temperature(out)
    expected = tophat_lwir_lut.apparent_temperature(
        np.float32(0.9 * band_radiance_tophat(7.5, 13.5, 300.0))
    )
    assert abs(float(t_app[0, 0]) - float(expected)) * 1e3 < 5.0
    assert float(t_app[0, 0]) == pytest.approx(293.51, abs=0.05)
    assert 300.0 - float(t_app[0, 0]) > 1.0, "kinetic temperature must not leak through"


def test_two_material_split_is_pixelwise(
    boson_lut: BandLUT, gbuffer_two_material: dict[str, np.ndarray]
) -> None:
    materials = MaterialTable.from_mapping({1: 0.95, 2: 0.5})
    out = band_radiance_stage(gbuffer_two_material, _config(boson_lut, materials), PipelineState())[
        "radiance"
    ]
    left, right = out[:, :32].astype(np.float64), out[:, 32:].astype(np.float64)
    assert np.allclose(left / right, 0.95 / 0.5, rtol=1e-6)
    assert np.all(left == left[0, 0]) and np.all(right == right[0, 0])


def test_float16_rejected_and_unmapped_id_refused(boson_lut: BandLUT) -> None:
    ids = np.ones((2, 2), dtype=np.int32)
    with pytest.raises(TypeError, match="float16"):
        band_radiance(
            np.full((2, 2), 300.0, dtype=np.float16), ids, MaterialTable.constant(1.0), boson_lut
        )
    with pytest.raises(ValueError, match="UNMAPPED"):
        band_radiance(
            np.full((2, 2), 300.0, dtype=np.float32),
            np.zeros((2, 2), np.int32),
            MaterialTable.constant(1.0),
            boson_lut,
        )
    with pytest.raises(ValueError, match="no emissivity"):
        band_radiance(
            np.full((2, 2), 300.0, dtype=np.float32),
            np.full((2, 2), 3, np.int32),
            MaterialTable.from_mapping({1: 0.9, 4: 0.8}),
            boson_lut,
        )


def test_stage_works_on_supersampled_and_validated_gbuffers(
    boson_lut: BandLUT,
    gbuffer_step_edge: dict[str, np.ndarray],
    gbuffer_sphere: dict[str, np.ndarray],
) -> None:
    cfg = _config(boson_lut, MaterialTable.from_mapping({1: 0.9, 2: 0.2}))
    for planes in (gbuffer_step_edge, gbuffer_sphere):
        GBuffer.from_dict(planes)  # contract holds
        out = BandRadianceStage()(planes, cfg, PipelineState())["radiance"]
        assert out.shape == planes["temperature_k"].shape and out.dtype == np.float32
    edge = BandRadianceStage()(gbuffer_step_edge, cfg, PipelineState())["radiance"]
    assert len(np.unique(edge)) == 2, "an ideal edge in T and material gives exactly two radiances"


def test_material_table_rules() -> None:
    with pytest.raises(ValueError, match="UNMAPPED"):
        MaterialTable.from_mapping({0: 0.9})
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        MaterialTable.from_mapping({1: 1.2})
    t = MaterialTable.from_mapping({1: 0.9, 3: 0.5}, band_id="lwir")
    assert t.emissivity.dtype == np.float32 and np.isnan(t.emissivity[2])
    with pytest.raises(TypeError, match="integer"):
        t.emissivity_for(np.array([1.0]))


def test_sky_pixels_are_blackbody_equivalent_and_skip_the_material_check(
    tophat_lwir_lut: BandLUT,
) -> None:
    """Under sky_mask the temperature is the apparent sky temperature: L = L_B(T_sky) exactly
    (ε = 1), the renderer's background id 0 is not an UNMAPPED error there, and the same id 0
    *outside* the mask still is. Without the mask nothing changes for geometry pixels."""
    t = np.full((4, 6), 300.0, dtype=np.float32)
    t[:2] = 240.0  # cold apparent sky in the top half
    ids = np.ones((4, 6), dtype=np.int32)
    ids[:2] = 0  # renderer background id under the sky
    sky = np.zeros((4, 6), dtype=bool)
    sky[:2] = True
    materials = MaterialTable.constant(0.9)
    out = band_radiance(t, ids, materials, tophat_lwir_lut, sky_mask=sky)
    lb240 = float(tophat_lwir_lut.lookup(240.0)[()])
    lb300 = float(tophat_lwir_lut.lookup(300.0)[()])
    np.testing.assert_allclose(out[:2], lb240, rtol=1e-6)
    np.testing.assert_allclose(out[2:], 0.9 * lb300, rtol=1e-6)
    with pytest.raises(ValueError, match="UNMAPPED"):
        band_radiance(t, ids, materials, tophat_lwir_lut)
    ids[3, 5] = 0  # an unmapped asset outside the sky is still an error
    with pytest.raises(ValueError, match="UNMAPPED"):
        band_radiance(t, ids, materials, tophat_lwir_lut, sky_mask=sky)
    with pytest.raises(ValueError, match="sky_mask"):
        band_radiance(t, ids, materials, tophat_lwir_lut, sky_mask=sky.astype(np.uint8))
