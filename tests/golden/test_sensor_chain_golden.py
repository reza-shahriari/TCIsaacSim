"""Golden frame for the assembled M9 sensor chain (M9.8).

The radiometric goldens describe the *ideal* camera (``chain=None``), which is what they should
describe: they exist to pin the physics, and a defect map or a drifting pattern would make a
millikelvin claim untestable. This one pins the other thing -- that the chain, once assembled,
keeps producing the same image. It is the regression lock on the wiring, not on the radiometry.

Regenerate deliberately with ``make golden-update`` and say in the commit body which mechanism
changed and why. ADR 0004.
"""

from __future__ import annotations

import copy
import hashlib
import pathlib

import numpy as np
import yaml

from irsim.config.loader import config_hash
from irsim.config.sensor import SensorConfig
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SHAPE = (48, 64)
SEED = 20260912
FPS = 6.0


def _scene() -> dict[str, np.ndarray]:
    """A ramp with a warm patch: smooth content for the gain residual, an edge for the optics."""
    rows, cols = SHAPE
    ramp = np.linspace(280.0, 320.0, cols, dtype=np.float32)
    t = np.tile(ramp, (rows, 1))
    t[rows // 3 : 2 * rows // 3, cols // 3 : 2 * cols // 3] += 40.0
    return {
        "temperature_k": t,
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }


def _config(lut: BandLUT) -> tuple[PipelineConfig, str]:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(
        width=SHAPE[1],
        height=SHAPE[0],
        frame_rate_hz=FPS,
        fpa_temp_mode="coupled",
        fpa_tau_s=20.0,
        fpa_self_heating_k=2.0,
    )
    d["sensor"]["optics"].update(supersample_factor=1)
    sensor = SensorConfig.model_validate(d)
    base = PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=lut, sensor_seed=SEED
    )
    # The stored reference depends on the camera, the array size, the chain's own parameters and
    # numpy's generators, so all of them go into the key: a change to any one is a STALE golden
    # to be regenerated deliberately, not a silent value drift.
    key = hashlib.sha256(
        (
            config_hash(sensor)
            + f"|{SHAPE[0]}x{SHAPE[1]}|chain=m9.8|seed={SEED}|numpy="
            + np.__version__
        ).encode()
    ).hexdigest()
    return attach_sensor_chain(base, ambient_provider=lambda t: 293.15 + 0.05 * t), key


def test_sensor_chain_golden(golden, tophat_lwir_lut: BandLUT) -> None:  # type: ignore[no-untyped-def]
    """Sixty seconds of drift, then one frame through the whole chain."""
    config, key = _config(tophat_lwir_lut)
    state = PipelineState()
    planes = _scene()
    out = None
    for _ in range(int(round(60.0 * FPS))):
        state.t_s = state.frame_index / FPS
        out = run_frame(planes, config, state)
    assert out is not None

    golden.check("boson_sensor_chain_dn16", out.dn16, config_hash=key, atol=2.0, units="DN")
    golden.check(
        "boson_sensor_chain_apparent_t",
        out.apparent_t,
        config_hash=key,
        atol=0.02,
        units="K",
    )
