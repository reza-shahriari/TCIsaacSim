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
#: (stage, fixture) pairs. Stages 1-2 are element-wise and run on any G-buffer; stage 3 consumes
#: the k-x supersampled grid of a specific detector format, so it takes the 4x step edge and the
#: sensor built to match it -- which is also the shape the PSF-then-box check wants (M10.5).
STAGE_FIXTURES = tuple(
    [("band_radiance", f) for f in FIXTURES]
    + [("atmosphere", f) for f in FIXTURES]
    + [("optics", "gbuffer_step_edge")]
)
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


def _sensor(**fpa: Any) -> Any:
    import yaml

    from irsim.config.sensor import SensorConfig

    raw = yaml.safe_load((REPO / "configs/sensors/flir_boson_640_lwir.yaml").read_text())
    raw["sensor"]["fpa"].update(fpa)
    return SensorConfig.model_validate(raw)


def _weather() -> Any:
    from irsim.thermal import WeatherSample, WeatherSeries

    return WeatherSeries.constant(
        WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


@pytest.fixture(scope="module")
def config(tophat_lwir_lut: Any) -> Any:
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(), MaterialTable.from_mapping({1: 0.95, 2: 0.6}), lut=tophat_lwir_lut
    )


@pytest.fixture(scope="module")
def config_layered(tophat_lwir_lut: Any) -> Any:
    """Stage 2 with the MS.1 exponential sum -- three terms in LWIR, so the kernel's per-term
    loop is actually exercised rather than degenerating to the grey single-term case."""
    from irsim.atmosphere import load_atmosphere_preset
    from irsim.atmosphere.layered import LayeredAtmosphere
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )
    return PipelineConfig.from_sensor(
        _sensor(),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
        atmosphere=atmosphere,
    )


@pytest.fixture(scope="module")
def config_grey(tophat_lwir_lut: Any) -> Any:
    from irsim.atmosphere import Atmosphere, load_atmosphere_preset
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
        atmosphere=Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather()),
    )


@pytest.fixture(scope="module")
def config_edge(tophat_lwir_lut: Any) -> Any:
    """A 256x256 detector at 4x: the format `gbuffer_step_edge` renders (1024x1024)."""
    from irsim.materials import MaterialTable
    from irsim.pipeline import PipelineConfig

    return PipelineConfig.from_sensor(
        _sensor(width=256, height=256),
        MaterialTable.from_mapping({1: 0.95, 2: 0.6}),
        lut=tophat_lwir_lut,
    )


STAGE_CONFIG = {"band_radiance": "config", "atmosphere": "config_layered", "optics": "config_edge"}


def _input_planes(stage: str, planes: dict[str, np.ndarray], config: Any) -> dict[str, np.ndarray]:
    """The planes a stage consumes: everything before it, run on the CPU oracle."""
    from irsim.pipeline import PipelineState
    from irsim.pipeline.atmosphere import atmosphere_stage
    from irsim.pipeline.radiance import band_radiance_stage

    out = dict(planes)
    if stage == "band_radiance":
        return out
    out.update(band_radiance_stage(out, config, PipelineState()))
    if stage == "atmosphere":
        return out
    out.update(atmosphere_stage(out, config, PipelineState()))
    return out


def _error_mk(cpu: np.ndarray, gpu: np.ndarray, t: np.ndarray, lut: Any) -> np.ndarray:
    slope = lut.lookup(t, "dlb_dt").astype(np.float64)
    return np.abs(gpu.astype(np.float64) - cpu.astype(np.float64)) / slope * 1e3


def _scene_equivalent(plane: str, values: np.ndarray, config: Any, state: Any) -> np.ndarray:
    """Whatever a stage produces, expressed as scene band radiance, so one mK budget covers all
    three. Stage 3's output is pixel power, and `invert_optics` is the oracle's own way back."""
    if plane != "flux":
        return values
    from irsim.optics.stage import invert_optics
    from irsim.pipeline.optics import housing_band_radiance

    return invert_optics(values, config.sensor.sensor, housing_band_radiance(config, state))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("stage", "fixture"), STAGE_FIXTURES)
def test_stage_matches_cpu_reference(
    warp: Any, request: pytest.FixtureRequest, stage: str, fixture: str, device: str
) -> None:
    from irsim.optics.sampling import box_downsample
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import EQUIVALENCE_OUTPUT, EQUIVALENCE_STAGES

    config = request.getfixturevalue(STAGE_CONFIG[stage])
    planes = _input_planes(stage, request.getfixturevalue(fixture), config)
    plane = EQUIVALENCE_OUTPUT[stage]
    cpu_stage, gpu_stage = EQUIVALENCE_STAGES[stage]
    state = PipelineState()
    ref = cpu_stage(planes, config, PipelineState())[plane]
    got = gpu_stage(planes, config, PipelineState(), device)[plane]
    assert got.dtype == np.float32 and got.shape == ref.shape

    rel = np.max(np.abs(got - ref) / np.abs(ref))
    # the temperature at the grid the stage's output lives on
    t = np.asarray(planes["temperature_k"])
    if t.shape != ref.shape:
        t = box_downsample(t, config.supersample)
    mk = np.max(
        _error_mk(
            _scene_equivalent(plane, ref, config, state),
            _scene_equivalent(plane, got, config, state),
            t,
            config.lut,
        )
    )
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


# ---- M10.5: stage 2 (atmosphere) and stage 3 (optics) ----------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_sky_pixels_bypass_the_atmosphere_bit_identically(
    warp: Any, config_layered: Any, gbuffer_ramp: dict[str, np.ndarray], device: str
) -> None:
    """ADR 0050: a sky pixel already carries the whole column to space, so stage 2 must not
    attenuate it again. Bit-identical, not merely close -- the CPU returns the input array."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import atmosphere_stage_warp

    planes = _input_planes("atmosphere", gbuffer_ramp, config_layered)
    sky = np.zeros(planes["radiance"].shape, dtype=bool)
    sky[:, :32] = True
    distance = np.asarray(planes["distance_m"]).copy()
    distance[sky] = np.inf  # the DistanceToCamera sentinel a sky ray comes back with
    planes = {**planes, "sky_mask": sky, "distance_m": distance}
    got = atmosphere_stage_warp(planes, config_layered, PipelineState(), device)["radiance"]
    assert np.array_equal(got[sky], planes["radiance"][sky])
    assert np.all(got[~sky] != planes["radiance"][~sky]), "the rest must have been attenuated"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("which", ["grey", "tau_override", "layered"])
def test_every_atmosphere_path_matches_its_oracle(
    warp: Any,
    request: pytest.FixtureRequest,
    gbuffer_ramp: dict[str, np.ndarray],
    which: str,
    device: str,
) -> None:
    """The three branches `atmosphere_stage` can take: one grey term, the constant-tau L1
    fallback, and MS.1's multi-term sum. A single kernel serves all three because Sum w_k = 1
    makes the per-term path radiance collapse to (1 - tau) L_air."""
    import dataclasses

    from irsim.pipeline import PipelineState
    from irsim.pipeline.atmosphere import atmosphere_stage
    from irsim_isaac.pipeline.warp_stages import atmosphere_stage_warp, atmosphere_terms

    config = request.getfixturevalue("config_layered" if which == "layered" else "config_grey")
    if which == "tau_override":
        config = dataclasses.replace(config, tau_override=0.7)
    planes = _input_planes("atmosphere", gbuffer_ramp, config)
    # a range spread that reaches optical depths where tau is small, plus the degenerate ends
    distance = np.tile(
        np.linspace(0.0, 5000.0, planes["radiance"].shape[1], dtype=np.float32),
        (planes["radiance"].shape[0], 1),
    )
    distance[0, 0] = np.inf
    planes = {**planes, "distance_m": distance}
    ref = atmosphere_stage(planes, config, PipelineState())["radiance"]
    got = atmosphere_stage_warp(planes, config, PipelineState(), device)["radiance"]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= REL_TOL
    terms = atmosphere_terms(config, PipelineState())
    assert terms is not None
    expected_terms = 3 if which == "layered" else 1
    assert terms.n_terms == expected_terms, f"{which}: {terms.n_terms} terms"


@pytest.mark.parametrize("device", DEVICES)
def test_step_edge_psf_then_box_matches_the_cpu_to_1e5(
    warp: Any, config_edge: Any, gbuffer_step_edge: dict[str, np.ndarray], device: str
) -> None:
    """M10.5's tight case: the optical PSF on the 4x grid *then* the block mean, in that order,
    across a 373/293 K edge where the blur has the most to do. A GPU path that downsampled first
    would still look plausible and would lose the sub-pixel edge profile MS.5 measures."""
    from irsim.pipeline import PipelineState
    from irsim.pipeline.optics import optics_stage
    from irsim_isaac.pipeline.warp_stages import optics_stage_warp

    assert config_edge.psf is not None and config_edge.psf.shape[0] >= 5
    planes = _input_planes("optics", gbuffer_step_edge, config_edge)
    ref = optics_stage(planes, config_edge, PipelineState())["flux"]
    got = optics_stage_warp(planes, config_edge, PipelineState(), device)["flux"]
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_vignetting_and_self_emission_come_from_the_host(
    warp: Any, config_edge: Any, device: str
) -> None:
    """cos^4 across the field and Phi_self at the corner, against `irsim.optics` directly: the
    kernel may scale and add, it may not own the aperture factor (non-negotiable #5)."""
    from irsim.optics.aperture import aperture_factor
    from irsim.optics.self_emission import self_emission_power
    from irsim.optics.stage import optics_field
    from irsim_isaac.pipeline.warp_stages import apply_optics_warp, optics_terms

    spec = config_edge.sensor.sensor
    lb_housing = float(config_edge.lut.lookup(300.0, config_edge.quantity)[()])
    terms = optics_terms(spec, lb_housing, config_edge.supersample, None)
    k, (h, w) = config_edge.supersample, spec.fpa_shape
    flux = apply_optics_warp(np.ones((h * k, w * k), np.float32), terms, device=device)

    f, tau, a_d = spec.optics.f_number, spec.optics.transmittance, spec.detector_active_area_m2
    phi_self = self_emission_power(a_d, f, tau, lb_housing)
    expected = aperture_factor(f) * tau * optics_field(spec) * a_d + phi_self
    assert np.max(np.abs(flux - expected) / expected) <= 1e-6
    corner = float(flux[0, 0] - phi_self) / float(flux[h // 2, w // 2] - phi_self)
    assert corner == pytest.approx(float(optics_field(spec)[0, 0]), rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_a_warmer_housing_raises_apparent_temperature_exactly_as_the_cpu_does(
    warp: Any, config_edge: Any, device: str
) -> None:
    """Phi_self is why a shutterless camera drifts (§8.2, §11.2). A 1 K housing step must move
    the apparent temperature by the same amount on both paths, or the GPU drift is not the
    CPU's drift and every NUC/FFC test written against the oracle stops meaning anything."""
    from irsim.isp.radiometric import apparent_temperature
    from irsim.optics.stage import invert_optics
    from irsim.pipeline import PipelineState
    from irsim.pipeline.optics import housing_band_radiance, optics_stage
    from irsim_isaac.pipeline.warp_stages import optics_stage_warp

    spec = config_edge.sensor.sensor
    k, (h, w) = config_edge.supersample, spec.fpa_shape
    scene = np.full((h * k, w * k), float(config_edge.lut.lookup(300.0, config_edge.quantity)[()]))
    planes = {"radiance": scene.astype(np.float32)}
    cal = housing_band_radiance(config_edge, PipelineState(housing_temp_k=300.0))

    def t_app(stage: Any, t_housing: float) -> np.ndarray:
        state = PipelineState(housing_temp_k=t_housing)
        args = (
            (planes, config_edge, state)
            if stage is optics_stage
            else (
                planes,
                config_edge,
                state,
                device,
            )
        )
        flux = stage(*args)["flux"]
        return apparent_temperature(
            invert_optics(flux, spec, cal), config_edge.lut, config_edge.quantity
        )

    cpu_shift = t_app(optics_stage, 301.0) - t_app(optics_stage, 300.0)
    gpu_shift = t_app(optics_stage_warp, 301.0) - t_app(optics_stage_warp, 300.0)
    assert np.max(np.abs(gpu_shift - cpu_shift)) * 1e3 <= 1.0, "within 1 mK of the CPU drift"
    # M10.5's number, and a physics check rather than a tautology: with tau_opt = 0.92 the lens
    # is a grey body of emissivity 0.08 filling the same cone, so ~8 % of a housing step arrives
    # at the detector. A sign error or a missing (1 - tau_opt) would still pass the CPU-vs-GPU
    # comparison above and would fail here.
    assert float(np.median(gpu_shift)) * 1e3 == pytest.approx(87.0, abs=2.0)
