"""Band registry (§12.1, §5.2): known answers for the four bands, classification, and the
radiometric facts the table encodes -- self-emission negligible in SWIR, dominant in LWIR, MWIR
straddling the crossover. A µm/nm mix-up or a wrong table entry breaks these by orders of
magnitude."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.bands import (
    BAND_IDS,
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_REGIME,
    NOMINAL_RANGES_UM,
    band_id_for,
    enabled_illumination_terms,
    overlap_fraction,
)
from irsim.config.sensor import SensorConfig
from irsim.radiometry.planck import band_radiance_tophat

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_DICT = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def test_nominal_ranges_classify_to_themselves() -> None:
    assert BAND_IDS == ("nir", "swir", "mwir", "lwir")
    for band, (lo, hi) in NOMINAL_RANGES_UM.items():
        assert band_id_for(lo, hi) == band
        assert overlap_fraction(lo, hi, band) == 1.0
    assert band_id_for(7.5, 13.5) == "lwir"
    assert band_id_for(8.0, 12.0) == "lwir"  # §5.2's narrower LWIR
    assert band_id_for(3.0, 5.0) == "mwir"
    assert band_id_for(0.9, 1.7) == "swir"
    assert band_id_for(0.75, 1.0) == "nir"
    with pytest.raises(ValueError, match="overlaps no canonical band"):
        band_id_for(1.8, 2.6)
    with pytest.raises(ValueError):
        band_id_for(12.0, 8.0)


def test_defaults_match_spec_12_1() -> None:
    assert DEFAULT_REGIME == {
        "nir": "reflective",
        "swir": "reflective",
        "mwir": "mixed",
        "lwir": "emissive",
    }
    assert DEFAULT_DETECTOR_MODEL == {
        "nir": "photon",
        "swir": "photon",
        "mwir": "photon",
        "lwir": "bolometer",
    }


def test_illumination_terms_follow_regime_not_band() -> None:
    assert enabled_illumination_terms("emissive") == {"self_emission"}
    assert enabled_illumination_terms("reflective") == {"solar", "night"}
    assert enabled_illumination_terms("mixed") == {"self_emission", "solar", "night"}
    with pytest.raises(ValueError):
        enabled_illumination_terms("visible")  # type: ignore[arg-type]


def test_self_emission_negligible_in_swir_dominant_in_lwir() -> None:
    """§12.1: 'negligible' vs 'dominant' at 300 K, via the closed-form top-hat radiance."""
    lb = {b: band_radiance_tophat(*NOMINAL_RANGES_UM[b], 300.0) for b in BAND_IDS}
    assert lb["swir"] / lb["lwir"] < 1e-6
    assert lb["nir"] / lb["lwir"] < 1e-9


def test_mwir_straddles_the_crossover() -> None:
    """§5.2 'both': MWIR self-emission at 300 K is a few percent of LWIR (expected ~0.035)."""
    ratio = band_radiance_tophat(3.0, 5.0, 300.0) / band_radiance_tophat(7.5, 13.5, 300.0)
    assert 1e-2 < ratio < 1e-1, ratio


def test_boson_resolves_lwir_and_explicit_id_is_checked() -> None:
    cfg = SensorConfig.model_validate(BOSON_DICT)
    assert cfg.sensor.band.id is None and cfg.sensor.band.band_id == "lwir"
    explicit = copy.deepcopy(BOSON_DICT)
    explicit["sensor"]["band"]["id"] = "lwir"
    assert SensorConfig.model_validate(explicit).sensor.band.band_id == "lwir"
    wrong: dict[str, Any] = copy.deepcopy(BOSON_DICT)
    wrong["sensor"]["band"]["id"] = "mwir"
    with pytest.raises(ValidationError, match="contradicts"):
        SensorConfig.model_validate(wrong)
    orphan = copy.deepcopy(BOSON_DICT)
    orphan["sensor"]["band"].update(lambda_min_um=1.8, lambda_max_um=2.6, regime="reflective")
    with pytest.raises(ValidationError, match="canonical band"):
        SensorConfig.model_validate(orphan)
