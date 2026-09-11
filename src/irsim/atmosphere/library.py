"""The atmosphere preset library: ``configs/atmospheres/<name>.yaml`` → :class:`AtmospherePreset`.

docs/physics-model.md §7.2 step 3 ("store a small table indexed by atmosphere"), §12.2
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import yaml

from irsim.config.atmosphere import AtmosphereConfig, AtmospherePreset

__all__ = ["PRESET_DIR", "available_presets", "load_atmosphere_preset", "preset_hash"]

PRESET_DIR = pathlib.Path(__file__).resolve().parents[3] / "configs" / "atmospheres"


def available_presets(preset_dir: str | os.PathLike[str] | None = None) -> tuple[str, ...]:
    root = pathlib.Path(preset_dir) if preset_dir is not None else PRESET_DIR
    return tuple(sorted(p.stem for p in root.glob("*.yaml")))


def load_atmosphere_preset(
    name_or_path: str | os.PathLike[str], preset_dir: str | os.PathLike[str] | None = None
) -> AtmospherePreset:
    """Load by preset name (``"haze"``) from the library, or by path to a YAML file."""
    root = pathlib.Path(preset_dir) if preset_dir is not None else PRESET_DIR
    path = pathlib.Path(name_or_path)
    if path.suffix != ".yaml":
        path = root / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"atmosphere preset {name_or_path!r} not found (looked at {path}); "
            f"available: {available_presets(root)}"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    preset = AtmosphereConfig.model_validate(raw).atmosphere
    if path.parent == root and preset.name != path.stem:
        raise ValueError(f"preset name {preset.name!r} does not match file name {path.stem!r}")
    return preset


def preset_hash(preset: AtmospherePreset) -> str:
    """SHA-256 of the canonical JSON of the validated preset (keys sorted, floats as repr)."""
    canonical = json.dumps(preset.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
