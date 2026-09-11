"""Noise and NUC schema (M4.1): variance closure of the 3-D ratios, the σ_TVH anchor, and the
bad-pixel bounds."""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from irsim.config.sensor import RATIO_ORDER, SensorConfig

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


def test_boson_variance_closure() -> None:
    """σ_total/σ_TVH = √(1 + 0.3² + 0.15² + 0.08² + 0.05² + 0.05² + 0.02²) = 1.0604 (§10.2).
    Summing the ratios linearly or dropping one fails this."""
    n = SensorConfig.model_validate(BOSON).sensor.noise
    expected = math.sqrt(1 + 0.09 + 0.0225 + 0.0064 + 0.0025 + 0.0025 + 0.0004)
    assert n.total_over_tvh() == pytest.approx(expected, rel=1e-12)
    assert n.total_over_tvh() == pytest.approx(1.0603, abs=1e-4)  # roadmap quotes 1.0604 (rounded)
    assert RATIO_ORDER == ("t", "v", "h", "tv", "th", "vh", "tvh")
    assert n.sigma_ratios() == (0.02, 0.08, 0.15, 0.05, 0.05, 0.30, 1.0)


def test_tvh_must_be_unity() -> None:
    with pytest.raises(ValidationError, match="exactly 1.0"):
        SensorConfig.model_validate(_with("sensor.noise.ratios_3d.tvh", 0.9))


def test_bad_pixel_fraction_bounds_and_type_mix() -> None:
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_with("sensor.noise.bad_pixel_fraction", 0.02))
    ok = SensorConfig.model_validate(_with("sensor.noise.bad_pixel_fraction", 0.0015)).sensor.noise
    mix = ok.bad_pixel_type_mix
    assert mix.dead + mix.hot + mix.flickering + mix.blinking == pytest.approx(1.0)
    with pytest.raises(ValidationError, match="sum to 1"):
        SensorConfig.model_validate(
            _with(
                "sensor.noise.bad_pixel_type_mix",
                {"dead": 0.5, "hot": 0.5, "flickering": 0.5, "blinking": 0.0},
            )
        )
    custom = SensorConfig.model_validate(
        _with(
            "sensor.noise.bad_pixel_type_mix",
            {"dead": 1.0, "hot": 0.0, "flickering": 0.0, "blinking": 0.0},
        )
    ).sensor.noise.bad_pixel_type_mix
    assert custom.dead == 1.0


def test_netd_reference_f_number() -> None:
    n = SensorConfig.model_validate(BOSON).sensor.noise
    assert n.netd_ref_f_number is None
    assert (
        SensorConfig.model_validate(
            _with("sensor.noise.netd_ref_f_number", 1.0)
        ).sensor.noise.netd_ref_f_number
        == 1.0
    )
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_with("sensor.noise.netd_ref_f_number", 0.0))


def test_nuc_block_loads_and_bounds() -> None:
    nuc = SensorConfig.model_validate(BOSON).sensor.nuc
    assert nuc.mode == "shuttered" and nuc.ffc_interval_s == 180 and nuc.ffc_freeze_ms == 700
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_with("sensor.nuc.mode", "manual"))
    with pytest.raises(ValidationError):
        SensorConfig.model_validate(_with("sensor.nuc.ffc_interval_s", 0))
