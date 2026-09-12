"""Roadmap M10.4: the CPU-vs-GPU equivalence harness, parametrised over (stage, fixture, device).

Every Warp stage registers its (CPU oracle, GPU twin) pair in
`irsim_isaac.pipeline.warp_stages.EQUIVALENCE_STAGES`; this file runs both on the same synthetic
G-buffers from tests/conftest.py and demands agreement within the ir-sim-testing budget for a
GPU kernel against its CPU reference: ≤ 1e-4 relative, and ≤ 5 mK when the radiance error is
expressed through dL_B/dT at the pixel's temperature (one tenth of the tightest NETD modelled).
The CPU reference is the oracle (ADR 0018); a disagreement is a kernel bug until proven otherwise.

Nothing here renders, and nothing here needs Kit: `irsim_isaac.env.ensure_warp_on_path` puts the
`omni.warp.core` extension on `sys.path`, so the whole file runs from a bare Isaac Sim interpreter
in seconds (ADR 0014 addendum). `cuda:0` is the production device; Warp's `cpu` device runs the
same kernel source through the C++ backend and is included as a second, compiler-independent
check of the arithmetic.

    make test-all                      # with everything else
    $PYTHON -m pytest tests/integration/test_kernels_vs_reference.py -m gpu
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ("gbuffer_ramp", "gbuffer_sphere", "gbuffer_two_material")
DEVICES = ("cuda:0", "cpu")
REL_TOL = 1e-4
MK_TOL = 5.0
pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def warp() -> Any:
    from irsim_isaac.env import ensure_warp_on_path

    ensure_warp_on_path()
    import warp as wp

    wp.init()
    if not any(d.is_cuda for d in wp.get_devices()):
        pytest.skip("no CUDA device for the cuda:0 half of the harness")
    return wp


@pytest.fixture(scope="module")
def config(tophat_lwir_lut: Any) -> Any:
    import yaml

    from irsim.config.sensor import SensorConfig
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    sensor = SensorConfig.model_validate(
        yaml.safe_load((REPO / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    )
    return PipelineConfig.from_sensor(
        sensor, MaterialTable.from_mapping({1: 0.95, 2: 0.6}), lut=tophat_lwir_lut
    )


def _error_mk(cpu: np.ndarray, gpu: np.ndarray, t: np.ndarray, lut: Any) -> np.ndarray:
    slope = lut.lookup(t, "dlb_dt").astype(np.float64)
    return np.abs(gpu.astype(np.float64) - cpu.astype(np.float64)) / slope * 1e3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fixture", FIXTURES)
@pytest.mark.parametrize("stage", ["band_radiance"])
def test_stage_matches_cpu_reference(
    warp: Any, config: Any, request: pytest.FixtureRequest, stage: str, fixture: str, device: str
) -> None:
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import EQUIVALENCE_STAGES

    planes = request.getfixturevalue(fixture)
    cpu_stage, gpu_stage = EQUIVALENCE_STAGES[stage]
    ref = cpu_stage(planes, config, PipelineState())["radiance"]
    got = gpu_stage(planes, config, PipelineState(), device)["radiance"]
    assert got.dtype == np.float32 and got.shape == ref.shape
    rel = np.max(np.abs(got - ref) / np.abs(ref))
    mk = np.max(_error_mk(ref, got, planes["temperature_k"], config.lut))
    assert rel <= REL_TOL, f"{stage}/{fixture}/{device}: {rel:.2e} relative"
    assert mk <= MK_TOL, f"{stage}/{fixture}/{device}: {mk:.3f} mK"


def test_sky_mask_and_environment_term_match(
    warp: Any, config: Any, gbuffer_ramp: dict[str, np.ndarray]
) -> None:
    """ε = 1 under the mask, ids ignored there, and the reflected term blended like the CPU."""
    from irsim.pipeline.radiance import band_radiance
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = gbuffer_ramp["temperature_k"]
    ids = gbuffer_ramp["material_id"].copy()
    sky = np.zeros(t.shape, dtype=bool)
    sky[:, :32] = True
    ids[sky] = 0  # the renderer's background id: an error outside the mask, ignored under it
    l_env = (config.lut.lookup(np.full(t.shape, 280.0, np.float32)) * 0.7).astype(np.float32)
    ref = band_radiance(t, ids, config.materials, config.lut, sky_mask=sky, l_env=l_env)
    got = band_radiance_warp(t, ids, config.materials, config.lut, sky_mask=sky, l_env=l_env)
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= REL_TOL
    assert np.array_equal(got[sky], got[sky])  # finite everywhere under the mask
    assert np.all(np.isfinite(got))


def test_isothermal_enclosure_identity_on_gpu(
    warp: Any, config: Any, gbuffer_two_material: dict[str, np.ndarray]
) -> None:
    """Kirchhoff on the device: with L_env = L_B(T) the radiance is L_B(T) for every ε (1e-5)."""
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = gbuffer_two_material["temperature_k"]
    lb = config.lut.lookup(t)
    got = band_radiance_warp(
        t, gbuffer_two_material["material_id"], config.materials, config.lut, l_env=lb
    )
    assert np.max(np.abs(got - lb) / lb) <= 1e-5


def test_lut_device_pointer_is_stable_across_frames(
    warp: Any, config: Any, gbuffer_ramp: dict[str, np.ndarray]
) -> None:
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import band_radiance_stage_warp, device_tables

    tables = device_tables(config.lut, config.materials, config.quantity)
    ptr = tables.lut_ptr
    state = PipelineState()
    for _ in range(10):
        band_radiance_stage_warp(gbuffer_ramp, config, state)
        state.advance()
    again = device_tables(config.lut, config.materials, config.quantity)
    assert again is tables and again.lut_ptr == ptr


def test_gpu_refuses_what_the_oracle_refuses(warp: Any, config: Any) -> None:
    from irsim_isaac.pipeline.warp_stages import band_radiance_warp

    t = np.full((4, 4), 300.0, np.float32)
    with pytest.raises(ValueError, match="UNMAPPED"):
        band_radiance_warp(t, np.zeros((4, 4), np.int32), config.materials, config.lut)
    with pytest.raises(TypeError):
        band_radiance_warp(
            t.astype(np.float16), np.ones((4, 4), np.int32), config.materials, config.lut
        )
