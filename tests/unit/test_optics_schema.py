"""Optics/FPA schema extensions (M3.4, ADR 0017): derived quantities against the spec's worked
numbers, and the physical consistency rules the new optional fields carry."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.sensor import SCHEMA_VERSION, SensorConfig

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _with(dotted: str, value: Any) -> dict[str, Any]:
    d = copy.deepcopy(BOSON)
    node: Any = d
    *path, last = dotted.split(".")
    for key in path:
        node = node.setdefault(key, {})
    node[last] = value
    return d


def test_boson_derived_quantities_match_spec_worked_numbers() -> None:
    s = SensorConfig.model_validate(BOSON).sensor
    assert s.detector_active_area_m2 == pytest.approx(1.296e-10, rel=1e-15)
    assert s.active_width_um == pytest.approx(11.384, abs=1e-3)
    assert s.nyquist_cyc_per_mm == pytest.approx(41.667, abs=1e-3)
    assert s.reference_wavelength_um == 10.5  # band centre by default
    assert s.cutoff_cyc_per_mm == pytest.approx(1000.0 / (10.5 * 1.0), rel=1e-12)
    ten = SensorConfig.model_validate(
        _with("sensor.optics.mtf.reference_wavelength_um", 10.0)
    ).sensor
    assert ten.cutoff_cyc_per_mm == pytest.approx(100.0, rel=1e-12)  # §8.3 worked case
    assert ten.hfov_deg == pytest.approx(30.68, abs=0.01)


def test_new_fields_default_and_schema_version() -> None:
    cfg = SensorConfig.model_validate(BOSON)
    assert cfg.schema_version == SCHEMA_VERSION == 4
    o = cfg.sensor.optics
    assert o.supersample_factor == 4 and o.housing_temp_k is None and o.vignetting_map is None
    assert o.mtf.aberration_sigma_um == 0.0 and o.mtf.apply_motion_mtf is False
    assert o.housing_self_heating_k == 0.0


def test_fixed_housing_mode_requires_temperature() -> None:
    with pytest.raises(ValidationError, match="housing_temp_k"):
        SensorConfig.model_validate(_with("sensor.optics.housing_temp_mode", "fixed"))
    d = _with("sensor.optics.housing_temp_mode", "fixed")
    d["sensor"]["optics"]["housing_temp_k"] = 305.0
    assert SensorConfig.model_validate(d).sensor.optics.housing_temp_k == 305.0


def test_cos4_forbidden_with_non_rectilinear_models() -> None:
    d = _with("sensor.optics.distortion", {"model": "kannala_brandt", "coeffs": [0.0] * 4})
    with pytest.raises(ValidationError, match="rectilinear"):
        SensorConfig.model_validate(d)
    d["sensor"]["optics"]["vignetting_cos4"] = False
    assert SensorConfig.model_validate(d).sensor.optics.distortion.model == "kannala_brandt"


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("sensor.optics.supersample_factor", 0),
        ("sensor.optics.supersample_factor", 9),
        ("sensor.optics.housing_tau_s", 0.0),
        ("sensor.optics.housing_self_heating_k", -1.0),
        ("sensor.optics.mtf.aberration_sigma_um", -0.1),
        ("sensor.optics.mtf.reference_wavelength_um", 0.0),
        ("sensor.optics.mtf.unknown_field", 1.0),
    ],
)
def test_invalid_new_fields_rejected(dotted: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_with(dotted, value))
