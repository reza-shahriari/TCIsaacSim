"""Environment / illumination presets: sky depression, ground mode, solar, night terms (§5.3–§5.5).

    schema_version: 1
    environment:
      name: clear_dry
      regime: clear | humid | overcast
      sky:    {delta_t_clear_k: {lwir: 62, mwir: 15, swir: 0, nir: 0}, q: 0.75}
      ground: {mode: air | solver | fixed, fixed_temperature_k: null}
      solar:  {enabled: true, glint_model: specular | lambertian}
      night:  {airglow_irradiance_nw_cm2: 10   # or airglow_irradiance_w_m2 -- exactly one
               airglow_shape_file: null, k_cloud: 0.5, moon: {enabled: true, phase_fraction: 0.5}}

A preset is an *illumination model* setting, not weather: the cloud fraction, humidity and
temperature that modulate these terms come from the one ``WeatherSeries`` (CLAUDE.md #6), so
weather-like keys are refused here exactly as in the atmosphere presets. Ranges follow §5.3
(ΔT_clear 55–70 K at zenith for a clear dry LWIR sky, 5–10 K under thick overcast, q 0.5–1.0)
and §5.5 (airglow 3.5–39 nW cm⁻², default ~10). The airglow level carries its unit in the key
and is exposed in SI (10 nW cm⁻² = 1.0e-4 W m⁻²).

docs/physics-model.md §5.3(a), §5.4, §5.5, §12.2; ADR 0017 (schema-gap policy)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from irsim.config.atmosphere import _find_weather_keys

__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "ENVIRONMENT_DIR",
    "Regime",
    "DELTA_T_LWIR_RANGE_K",
    "AIRGLOW_RANGE_W_M2",
    "NW_CM2_TO_W_M2",
    "SkySpec",
    "GroundSpec",
    "SolarSpec",
    "MoonSpec",
    "NightSpec",
    "EnvironmentSpec",
    "EnvironmentConfig",
    "load_environment_preset",
    "available_environments",
]

ENVIRONMENT_SCHEMA_VERSION = 1
ENVIRONMENT_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "environments"
Regime = Literal["clear", "humid", "overcast"]
# §5.3: zenith clear-sky depression in the LWIR window by regime (kelvin)
DELTA_T_LWIR_RANGE_K: dict[str, tuple[float, float]] = {
    "clear": (55.0, 70.0),
    "humid": (15.0, 55.0),
    "overcast": (0.0, 15.0),
}
# §5.5: reported airglow irradiance 3.5-39 nW cm^-2
NW_CM2_TO_W_M2 = 1e-9 / 1e-4  # 1 nW cm^-2 = 1e-5 W m^-2
AIRGLOW_RANGE_W_M2 = (3.5 * NW_CM2_TO_W_M2, 39.0 * NW_CM2_TO_W_M2)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SkySpec(_Frozen):
    """T_sky(θ) = T_air − ΔT_clear (1 − cloud) cos^q θ_zen  (§5.3 a), ΔT per band."""

    delta_t_clear_k: dict[str, float]
    q: float = Field(ge=0.5, le=1.0)

    @field_validator("delta_t_clear_k")
    @classmethod
    def _bands(cls, v: dict[str, float]) -> dict[str, float]:
        if "lwir" not in v:
            raise ValueError("delta_t_clear_k must include the lwir band (§5.3 values are LWIR)")
        for band, dt in v.items():
            if not 0.0 <= dt <= 80.0:
                raise ValueError(f"delta_t_clear_k[{band!r}] = {dt} outside [0, 80] K")
        return v


class GroundSpec(_Frozen):
    mode: Literal["air", "solver", "fixed"]
    fixed_temperature_k: float | None = Field(default=None, gt=150.0, lt=400.0)

    @model_validator(mode="after")
    def _fixed_needs_value(self) -> GroundSpec:
        if self.mode == "fixed" and self.fixed_temperature_k is None:
            raise ValueError("ground.mode 'fixed' needs fixed_temperature_k")
        if self.mode != "fixed" and self.fixed_temperature_k is not None:
            raise ValueError("fixed_temperature_k only applies to ground.mode 'fixed'")
        return self


class SolarSpec(_Frozen):
    enabled: bool = True
    glint_model: Literal["specular", "lambertian"] = "specular"  # §5.4: specular lobe


class MoonSpec(_Frozen):
    enabled: bool = True
    phase_fraction: float = Field(default=0.5, ge=0.0, le=1.0)  # 0 new, 1 full


class NightSpec(_Frozen):
    airglow_irradiance_w_m2: float | None = None
    airglow_irradiance_nw_cm2: float | None = None
    airglow_shape_file: str | None = None  # spectra/airglow_*.csv (M11); None = flat
    k_cloud: float = Field(default=0.5, ge=0.0, le=1.0)
    moon: MoonSpec = MoonSpec()

    @model_validator(mode="after")
    def _one_airglow_key(self) -> NightSpec:
        given = [
            k
            for k in ("airglow_irradiance_w_m2", "airglow_irradiance_nw_cm2")
            if getattr(self, k) is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "give exactly one of airglow_irradiance_w_m2 / airglow_irradiance_nw_cm2"
            )
        lo, hi = AIRGLOW_RANGE_W_M2
        e = self.airglow_w_m2
        if not lo <= e <= hi:
            raise ValueError(
                f"airglow irradiance {e:.3e} W/m^2 outside the §5.5 range "
                f"[{lo:.2e}, {hi:.2e}] W/m^2 (3.5-39 nW/cm^2)"
            )
        return self

    @property
    def airglow_w_m2(self) -> float:
        if self.airglow_irradiance_w_m2 is not None:
            return float(self.airglow_irradiance_w_m2)
        assert self.airglow_irradiance_nw_cm2 is not None
        return float(self.airglow_irradiance_nw_cm2) * NW_CM2_TO_W_M2


class EnvironmentSpec(_Frozen):
    name: str = Field(min_length=1)
    description: str = ""
    regime: Regime
    sky: SkySpec
    ground: GroundSpec
    solar: SolarSpec = SolarSpec()
    night: NightSpec

    @model_validator(mode="after")
    def _delta_t_matches_regime(self) -> EnvironmentSpec:
        lo, hi = DELTA_T_LWIR_RANGE_K[self.regime]
        dt = self.sky.delta_t_clear_k["lwir"]
        if not lo <= dt <= hi:
            raise ValueError(
                f"sky.delta_t_clear_k.lwir = {dt} K outside the §5.3 range [{lo}, {hi}] K for a "
                f"{self.regime} sky"
            )
        return self


class EnvironmentConfig(_Frozen):
    schema_version: int
    environment: EnvironmentSpec

    @model_validator(mode="before")
    @classmethod
    def _no_weather(cls, data: Any) -> Any:
        if isinstance(data, dict):
            hits = _find_weather_keys(data)
            if hits:
                raise ValueError(
                    f"weather-like keys {hits} are not allowed in an environment preset: cloud, "
                    "humidity and temperature come from the WeatherSeries (CLAUDE.md #6)"
                )
        return data

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: int) -> int:
        if v != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(f"environment schema_version {v} != {ENVIRONMENT_SCHEMA_VERSION}")
        return v


def available_environments(preset_dir: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    root = pathlib.Path(preset_dir) if preset_dir is not None else ENVIRONMENT_DIR
    return tuple(sorted(p.stem for p in root.glob("*.yaml")))


def load_environment_preset(
    name_or_path: str | os.PathLike[str], preset_dir: str | os.PathLike[str] | None = None
) -> EnvironmentSpec:
    root = pathlib.Path(preset_dir) if preset_dir is not None else ENVIRONMENT_DIR
    path = pathlib.Path(name_or_path)
    if path.suffix != ".yaml":
        path = root / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"environment preset {name_or_path!r} not found; "
            f"available: {available_environments(root)}"
        )
    spec = EnvironmentConfig.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    ).environment
    if path.parent == root and spec.name != path.stem:
        raise ValueError(f"environment name {spec.name!r} does not match file name {path.stem!r}")
    return spec
