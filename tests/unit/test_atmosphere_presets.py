"""Atmosphere presets (M8.3): every preset x band reproduces its §7.2 row at the conditions it
was fitted at, the schema refuses weather, and the library is consistent."""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from irsim.atmosphere.extinction import regime_for_visibility, transmittance_per_band
from irsim.atmosphere.library import (
    PRESET_DIR,
    available_presets,
    load_atmosphere_preset,
    preset_hash,
)
from irsim.config.atmosphere import ATMOSPHERE_BAND_KEYS, AtmosphereConfig

# §7.2 table: tau at 200 m, horizontal, ground level (docs/physics-model.md).
ROWS: dict[str, dict[str, tuple[float, float]]] = {
    "clear": {
        "lwir": (0.90, 0.96),
        "mwir": (0.92, 0.97),
        "swir": (0.93, 0.98),
        "visible": (0.95, 0.98),
    },
    "humid": {
        "lwir": (0.72, 0.85),
        "mwir": (0.85, 0.93),
        "swir": (0.88, 0.95),
        "visible": (0.93, 0.97),
    },
    "haze": {
        "lwir": (0.85, 0.92),
        "mwir": (0.82, 0.90),
        "swir": (0.70, 0.85),
        "visible": (0.45, 0.65),
    },
    "fog200": {
        "lwir": (0.35, 0.60),
        "mwir": (0.25, 0.50),
        "swir": (0.10, 0.30),
        "visible": (0.02, 0.10),
    },
    "fog50": {
        "lwir": (0.02, 0.10),
        "mwir": (0.01, 0.06),
        "swir": (0.0, 0.01),
        "visible": (0.0, 1e-4),
    },
}
# Conditions each preset's provenance was checked at (T_air K, RH fraction, visibility m, row).
# They live here, not in the preset: a preset carries no weather (CLAUDE.md #6).
FIT: dict[str, tuple[float, float, float, str]] = {
    "us_standard_clear": (288.15, 0.46, 23000.0, "clear"),
    "midlat_summer_humid": (303.15, 0.80, 23000.0, "humid"),
    "midlat_winter_dry": (272.2, 0.76, 23000.0, "clear"),
    "tropical": (300.0, 0.74, 23000.0, "humid"),
    "haze": (288.15, 0.46, 1500.0, "haze"),
    "fog_light_200m": (283.15, 1.0, 200.0, "fog200"),
    "fog_dense_50m": (283.15, 1.0, 50.0, "fog50"),
}


def test_library_has_the_seven_presets() -> None:
    assert available_presets() == tuple(sorted(FIT))


@pytest.mark.parametrize("name", sorted(FIT))
def test_preset_reproduces_its_spec_row_at_200m(name: str) -> None:
    t_air, rh, vis, row = FIT[name]
    preset = load_atmosphere_preset(name)
    tau = transmittance_per_band(preset, t_air, rh, vis, 200.0)
    for band, (lo, hi) in ROWS[row].items():
        assert lo <= tau[band] <= hi, (
            f"{name}/{band}: tau(200 m) = {tau[band]:.4f} not in {row} row {lo}-{hi}"
        )
    assert all(0.0 < v <= 1.0 for v in tau.values())
    # aerosol extinction falls with wavelength (small particles steeply, droplets flatter):
    # the ratios are strictly ordered. (tau itself is not ordered in humid air: the 0.94 um
    # water band puts NIR below the visible, and SWIR water bands put SWIR below NIR.)
    r = {b: preset.bands[b].aerosol_ratio_to_visible for b in preset.bands}
    assert r["visible"] > r["nir"] > r["swir"] > r["mwir"] > r["lwir"] > 0.0, r
    # the regime tag agrees with what the fit visibility implies (WMO fog < 1 km)
    expected = "droplet" if regime_for_visibility(vis) == "droplet" else "aerosol"
    assert (preset.aerosol_regime == "droplet") == (expected == "droplet")


def test_identical_band_key_set_and_shared_molecular_physics() -> None:
    presets = [load_atmosphere_preset(n) for n in FIT]
    keys = {frozenset(p.bands) for p in presets}
    assert keys == {ATMOSPHERE_BAND_KEYS}
    for band in ATMOSPHERE_BAND_KEYS:
        g0 = {p.bands[band].gamma0_per_m for p in presets}
        beta = {p.bands[band].beta_per_m_per_g_m3 for p in presets}
        assert len(g0) == 1 and len(beta) == 1, f"{band}: molecular coefficients differ"
    assert all(p.bands["visible"].aerosol_ratio_to_visible == 1.0 for p in presets)
    assert all(p.valid_range_m == 500.0 and p.provenance.status == "ESTIMATED" for p in presets)


def _raw(name: str = "us_standard_clear") -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text())


@pytest.mark.parametrize(
    "path, key",
    [
        (("atmosphere",), "air_temperature_k"),
        (("atmosphere",), "visibility_m"),
        (("atmosphere", "provenance"), "relative_humidity"),
        (("atmosphere", "bands", "lwir"), "rh"),
        ((), "weather_file"),
    ],
)
def test_weather_like_keys_raise_anywhere(path: tuple[str, ...], key: str) -> None:
    raw = _raw()
    node = raw
    for p in path:
        node = node[p]
    node[key] = 1.0
    with pytest.raises(ValueError, match="WeatherSeries"):
        AtmosphereConfig.model_validate(raw)


def test_schema_guards() -> None:
    raw = _raw()
    bad = copy.deepcopy(raw)
    del bad["atmosphere"]["bands"]["nir"]
    with pytest.raises(ValueError, match="missing \\['nir'\\]"):
        AtmosphereConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["atmosphere"]["bands"]["thz"] = bad["atmosphere"]["bands"]["lwir"]
    with pytest.raises(ValueError, match="unknown \\['thz'\\]"):
        AtmosphereConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["atmosphere"]["bands"]["visible"]["aerosol_ratio_to_visible"] = 0.9
    with pytest.raises(ValueError, match="Koschmieder"):
        AtmosphereConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["schema_version"] = 99  # 2 is current (M8.8 added the solar block)
    with pytest.raises(ValueError, match="schema_version"):
        AtmosphereConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["atmosphere"]["bands"]["lwir"]["gamma0_per_m"] = -1e-5
    with pytest.raises(ValueError):
        AtmosphereConfig.model_validate(bad)


def test_library_loading_rules(tmp_path: pathlib.Path) -> None:
    by_name = load_atmosphere_preset("haze")
    by_path = load_atmosphere_preset(PRESET_DIR / "haze.yaml")
    assert by_name == by_path and preset_hash(by_name) == preset_hash(by_path)
    with pytest.raises(FileNotFoundError, match="available"):
        load_atmosphere_preset("mars")
    raw = _raw("haze")
    (tmp_path / "not_haze.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not match"):
        load_atmosphere_preset("not_haze", preset_dir=tmp_path)
    raw["atmosphere"]["bands"]["lwir"]["aerosol_ratio_to_visible"] = 0.11
    (tmp_path / "haze.yaml").write_text(yaml.safe_dump(raw))
    assert preset_hash(load_atmosphere_preset("haze", preset_dir=tmp_path)) != preset_hash(by_name)
