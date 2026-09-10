"""YAML loading, data-root path resolution and canonical hashing for sensor configs.

Two hashes key everything that is derived from a config (docs/physics-model.md §3.2 b; the
ir-radiometry and ir-sim-testing skills):

* ``config_hash`` -- SHA-256 of the canonical JSON of the *validated* model, with every data
  file path replaced by the SHA-256 of that file's bytes. Comments, key order, whitespace and
  ``60`` vs ``60.0`` do not change it; every numeric or enumerated leaf does. Goldens and
  pipeline outputs are keyed on this.
* ``band_hash`` -- the same over the ``band`` block and the spectral-response bytes only. This
  is what keys a band LUT, so the three NETD grades of one detector (§16.1) share one LUT.

Data files referenced by a config (``band.spectral_response``; later material n/k and
emissivity files) are relative to a data root: the ``data_dir`` argument, else
``$IRSIM_DATA_DIR``, else ``<repo>/data``. The canonical layout is ``data/spectra/responses/``
for R(λ) files (spec issue T4, fixed in §12.2 by M0.10). Loading stores the resolved absolute
path on the model and fails by name if the file is missing.

docs/physics-model.md §12.2, §3.2 (b)
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

import yaml

from irsim.config.sensor import SensorConfig

__all__ = [
    "DEFAULT_DATA_DIR",
    "DATA_PATH_FIELDS",
    "resolve_data_dir",
    "load_sensor_config",
    "dump_sensor_config",
    "config_hash",
    "band_hash",
    "file_sha256",
]

DEFAULT_DATA_DIR = pathlib.Path(__file__).resolve().parents[3] / "data"
# Dotted paths (under ``sensor``) of fields that name data files. Extend here when material or
# atmosphere configs add file references; the hashes pick them up automatically.
DATA_PATH_FIELDS: tuple[str, ...] = ("band.spectral_response",)


def resolve_data_dir(data_dir: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Argument, else ``$IRSIM_DATA_DIR``, else ``<repo>/data``."""
    if data_dir is not None:
        return pathlib.Path(data_dir).expanduser().resolve()
    env = os.environ.get("IRSIM_DATA_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return DEFAULT_DATA_DIR


def _get(d: dict[str, Any], dotted: str) -> Any:
    node: Any = d
    for key in dotted.split("."):
        node = node[key]
    return node


def _set(d: dict[str, Any], dotted: str, value: Any) -> None:
    *path, last = dotted.split(".")
    node = d
    for key in path:
        node = node[key]
    node[last] = value


def _resolve_paths(sensor: dict[str, Any], data_dir: pathlib.Path) -> None:
    for field in DATA_PATH_FIELDS:
        raw = _get(sensor, field)
        path = pathlib.Path(raw).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"sensor.{field} = {raw!r} resolves to {path}, which does not exist "
                f"(data root {data_dir}; override with data_dir= or $IRSIM_DATA_DIR)"
            )
        _set(sensor, field, str(path))


def load_sensor_config(
    path: str | os.PathLike[str], data_dir: str | os.PathLike[str] | None = None
) -> SensorConfig:
    """Read a §12.2 YAML file, resolve its data paths against the data root, validate."""
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict) or "sensor" not in raw:
        raise ValueError(f"{path}: a sensor config needs a top-level 'sensor:' block")
    if not isinstance(raw["sensor"], dict):
        raise ValueError(f"{path}: 'sensor:' must be a mapping")
    _resolve_paths(raw["sensor"], resolve_data_dir(data_dir))
    return SensorConfig.model_validate(raw)


def dump_sensor_config(config: SensorConfig) -> str:
    """YAML text that :func:`load_sensor_config` reads back to an equal model."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)


def file_sha256(path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _dump_with_file_hashes(
    config: SensorConfig, data_dir: str | os.PathLike[str] | None
) -> dict[str, Any]:
    """Model dump with each data-file path replaced by {"sha256": <content hash>}.

    A model built from a dictionary (not the loader) may still hold a relative path; it is
    resolved against the data root here so hashing does not depend on how the model was made.
    """
    dumped = config.model_dump(mode="json")
    sensor = dumped["sensor"]
    root = resolve_data_dir(data_dir)
    for field in DATA_PATH_FIELDS:
        raw = _get(sensor, field)
        path = pathlib.Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"sensor.{field} = {raw!r}: {path} does not exist")
        _set(sensor, field, {"sha256": file_sha256(path)})
    return dumped


def config_hash(config: SensorConfig, data_dir: str | os.PathLike[str] | None = None) -> str:
    """SHA-256 over the whole validated config; data files by content, not by path."""
    return hashlib.sha256(_canonical(_dump_with_file_hashes(config, data_dir)).encode()).hexdigest()


def band_hash(config: SensorConfig, data_dir: str | os.PathLike[str] | None = None) -> str:
    """SHA-256 over ``sensor.band`` and the spectral-response bytes only -- the LUT key."""
    dumped = _dump_with_file_hashes(config, data_dir)
    return hashlib.sha256(_canonical(dumped["sensor"]["band"]).encode()).hexdigest()
