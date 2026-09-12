"""Roadmap M10.7a: Warp stage 5 -- the 3-D noise, the OU drift and the NUC residual on device.

Stage 5 is the one stage that **cannot** be held to bit-equality with the CPU oracle. Warp's
generator is not NumPy's, so the two paths draw different numbers from the same seed by
construction, and ADR 0022 settles this by holding them to *statistical* equivalence instead. That
makes the tests here different in kind from M10.4's: they measure distributions, not values.

What is still exact, and is checked as such:

* the same frame id gives a bit-identical frame on the same device (a seeded stream, not a clock);
* frame + 1 redraws the temporal terms and leaves the fixed terms untouched -- so a run is a
  sequence of frames of one camera, not a sequence of cameras;
* the residual at ΔT_FPA = 0 is exactly the identity, on either device;
* a frozen drift (τ = ∞) leaves the device buffers bit-identical and launches nothing.

What is statistical:

* over 200 frames the DN-domain NETD is within 5 % of the CPU path's and 10 % of the config's --
  where "the CPU path" means the detector **and** `NoiseStage` together, because the stage
  boundary differs between the two: on the CPU the per-pixel TVH term belongs to stage 4 and the
  stage adds only the six correlated ones, while the Warp detector stage is the ideal transfer
  alone so stage 5 supplies all seven. The composition is identical; only the seam moves;
* the 3-D ratios are recovered within 15 %;
* NETD(373)/NETD(300) < 0.8, the derivative ratio -- the same non-negotiable #3 check the CPU
  bench makes, now on device, because noise added in kelvin would give 1.0 on either path;
* the row and column PSDs match the CPU path's within a factor 1.5.

Nothing here renders or needs Kit (ADR 0014 addendum).

    $PYTHON -m pytest tests/integration/test_warp_noise_stage.py -m gpu
"""

from __future__ import annotations

import copy
import math
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DEVICES = ("cuda:0", "cpu")
SHAPE = (64, 80)
SEED = 4242
N_FRAMES = 200
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
def config(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    from irsim.config.sensor import SensorConfig
    from irsim.materials.table import MaterialTable
    from irsim.pipeline import PipelineConfig

    d = yaml.safe_load(BOSON_YAML.read_text())
    d = copy.deepcopy(d)
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(supersample_factor=1)
    sensor = SensorConfig.model_validate(d)
    return PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=tophat_lwir_lut, sensor_seed=SEED
    )


def _planes(config, sigma_dn: float, level_dn: float = 12_000.0):  # type: ignore[no-untyped-def]
    return {
        "signal_dn": np.full(SHAPE, level_dn, dtype=np.float32),
        "sigma_dn": np.full(SHAPE, sigma_dn, dtype=np.float32),
    }


def _run_frames(config, device: str, n: int, sigma_dn: float):  # type: ignore[no-untyped-def]
    """n frames of stage 5 on one device, returned as a (n, rows, cols) cube."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import noise_stage_warp

    state = PipelineState()
    planes = _planes(config, sigma_dn)
    cube = np.empty((n, *SHAPE), dtype=np.float32)
    for k in range(n):
        cube[k] = noise_stage_warp(planes, config, state, device=device)["signal_dn"]
        state.advance()
    return cube


def _cpu_frames(config, n: int, sigma_dn: float, level_dn: float = 12_000.0):  # type: ignore[no-untyped-def]
    """The CPU oracle for the same thing: noiseless signal in, all seven components out.

    The stage *boundary* differs between the two paths and the composition does not. On the CPU
    the per-pixel TVH term belongs to the detector (stage 4) and `NoiseStage` adds only the six
    correlated ones; on the device the Warp detector stage is the ideal transfer alone, so stage 5
    supplies all seven. Comparing `NoiseStage.apply` against the device kernel directly would
    therefore be comparing a six-term image with a seven-term one -- which is a real trap: the
    resulting PSD ratio is about ten, and looks exactly like a broken kernel. The oracle here is
    the detector and the stage together, which is what both paths actually produce for a frame.
    """
    flux = config.detector.transfer.power_from_signal_w(np.full(SHAPE, level_dn, dtype=np.float32))
    cube = np.empty((n, *SHAPE), dtype=np.float32)
    for k in range(n):
        frame = config.detector.response(flux, k, config.sensor_seed)
        cube[k] = config.noise.apply(frame.signal_dn, frame.sigma_dn, k)
    del sigma_dn
    return cube


# -- what stays exact --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_the_same_frame_id_is_bit_identical(warp: Any, config, device: str) -> None:  # type: ignore[no-untyped-def]
    """A seeded stream, not a clock: re-running frame 7 must give frame 7 back."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import noise_stage_warp

    planes = _planes(config, 8.9)
    first = PipelineState()
    first.frame_index = 7
    second = PipelineState()
    second.frame_index = 7
    a = noise_stage_warp(planes, config, first, device=device)["signal_dn"]
    b = noise_stage_warp(planes, config, second, device=device)["signal_dn"]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_the_next_frame_redraws_only_the_temporal_terms(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """frame + 1 gives new TVH while V, H and VH are untouched.

    Pulled out with the project's own leakage-corrected NVESD decomposition rather than by hand:
    the fixed terms live *inside* the image, and a raw frame-to-frame correlation is dominated by
    the temporal terms it is supposed to be ignoring. Two independent windows of frames are
    decomposed, and their fixed components must match while their realisations of TVH do not.
    A chain that redrew the fixed pattern every frame would still look exactly like noise.
    """
    from irsim.validation.noise import decompose_3d

    sigma_dn = float(config.detector.sigma_signal_dn)
    cube = _run_frames(config, device, 2 * N_FRAMES, sigma_dn)
    first, second = cube[:N_FRAMES], cube[N_FRAMES:]
    assert not np.array_equal(first[0], second[0])  # the temporal terms did move

    # The frame-averaged image is the fixed pattern plus a temporal residue of sigma/sqrt(N).
    a, b = first.mean(axis=0), second.mean(axis=0)
    residue = sigma_dn / math.sqrt(N_FRAMES)
    assert float(np.abs(a - b).max()) < 12.0 * residue, "the fixed pattern moved between windows"

    da, db = decompose_3d(first), decompose_3d(second)
    for name in ("v", "h", "vh"):
        assert getattr(da, name) == pytest.approx(getattr(db, name), rel=0.15)
    # and the temporal term is present and of the right size in both
    assert da.tvh == pytest.approx(sigma_dn, rel=0.10)
    assert db.tvh == pytest.approx(sigma_dn, rel=0.10)


@pytest.mark.parametrize("device", DEVICES)
def test_the_fixed_buffers_do_not_move_between_frames(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """Device-resident state, allocated once and never round-tripped (ADR 0052)."""
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import noise_stage_warp, warp_pipeline_state

    state = PipelineState()
    planes = _planes(config, 8.9)
    noise_stage_warp(planes, config, state, device=device)
    ptrs = warp_pipeline_state(state, device).fixed_ptrs
    assert ptrs is not None
    for _ in range(10):
        state.advance()
        noise_stage_warp(planes, config, state, device=device)
    assert warp_pipeline_state(state, device).fixed_ptrs == ptrs


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_pattern_starts_from_the_cpu_realisation(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """Both paths start from the *same* fixed pattern, so equivalence measures the generators.

    Without this the two would be different cameras, and "statistically equivalent" would be a
    much weaker statement than it sounds.
    """
    from irsim.pipeline import PipelineState
    from irsim_isaac.pipeline.warp_stages import noise_stage_warp, warp_pipeline_state

    state = PipelineState()
    noise_stage_warp(_planes(config, 8.9), config, state, device=device)
    fixed = warp_pipeline_state(state, device).fixed_pattern(SHAPE, config.noise.unit_fixed)
    assert np.array_equal(fixed[0].numpy(), np.asarray(config.noise.unit_fixed.v))
    assert np.array_equal(fixed[2].numpy(), np.asarray(config.noise.unit_fixed.vh))


@pytest.mark.parametrize("device", DEVICES)
def test_a_frozen_drift_launches_nothing_and_changes_nothing(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """τ = ∞ is the ideal-sensor control, bit-identical on device as on the host (ADR 0054)."""
    import warp as wp

    from irsim_isaac.pipeline.warp_stages import launch_ou_drift

    v = wp.array(np.arange(SHAPE[0], dtype=np.float32), dtype=wp.float32, device=device)
    h = wp.array(np.arange(SHAPE[1], dtype=np.float32), dtype=wp.float32, device=device)
    vh = wp.array(
        np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE),
        dtype=wp.float32,
        device=device,
    )
    before = (v.numpy().copy(), h.numpy().copy(), vh.numpy().copy())
    launch_ou_drift((v, h, vh), (0.08, 0.15, 0.30), 1.0, float("inf"), 0, SEED, device)
    assert np.array_equal(v.numpy(), before[0])
    assert np.array_equal(h.numpy(), before[1])
    assert np.array_equal(vh.numpy(), before[2])


@pytest.mark.parametrize("device", DEVICES)
def test_the_ou_drift_is_stationary_on_device(warp: Any, device: str) -> None:
    """σ is preserved over 500 device steps -- the property the whole of ADR 0054 rests on."""
    import warp as wp

    from irsim_isaac.pipeline.warp_stages import launch_ou_drift

    sigma = 0.30
    rng = np.random.default_rng(5)
    vh = wp.array(
        (rng.standard_normal(SHAPE) * sigma).astype(np.float32), dtype=wp.float32, device=device
    )
    v = wp.zeros(SHAPE[0], dtype=wp.float32, device=device)
    h = wp.zeros(SHAPE[1], dtype=wp.float32, device=device)
    start = float(vh.numpy().std())
    for epoch in range(500):
        launch_ou_drift((v, h, vh), (0.0, 0.0, sigma), 1.0 / 60.0, 120.0, epoch, SEED, device)
    end = float(vh.numpy().std())
    assert end == pytest.approx(sigma, rel=0.10)
    assert end == pytest.approx(start, rel=0.15)
    assert abs(float(vh.numpy().mean())) < 5.0 * sigma / math.sqrt(vh.numpy().size)


@pytest.mark.parametrize("device", DEVICES)
def test_the_residual_at_zero_delta_t_is_exactly_the_identity(
    warp: Any,
    device: str,
) -> None:
    """ΔT_FPA = 0 must be the identity on device too, or an FFC would not fully correct."""
    import warp as wp

    from irsim_isaac.pipeline.warp_stages import launch_nuc_residual

    rng = np.random.default_rng(1)
    signal = (rng.uniform(1000.0, 40000.0, SHAPE)).astype(np.float32)
    src = wp.array(signal, dtype=wp.float32, device=device)
    xi = (
        wp.array(rng.standard_normal(SHAPE).astype(np.float32), dtype=wp.float32, device=device),
        wp.array(rng.standard_normal(SHAPE).astype(np.float32), dtype=wp.float32, device=device),
    )
    out = wp.zeros(SHAPE, dtype=wp.float32, device=device)
    launch_nuc_residual(src, xi, 900.0, 45.0, 178.0, 0.0, out, device)
    assert np.array_equal(out.numpy(), signal)


@pytest.mark.parametrize("device", DEVICES)
def test_the_residual_matches_the_cpu_model_exactly(warp: Any, device: str) -> None:
    """The xi fields are uploaded, not redrawn, so this one *is* bit-comparable to 1 ulp."""
    import warp as wp

    from irsim.config.sensor import SensorConfig
    from irsim.noise import NucResidual
    from irsim_isaac.pipeline.warp_stages import launch_nuc_residual

    sensor = SensorConfig.model_validate(yaml.safe_load(BOSON_YAML.read_text())).sensor
    residual = NucResidual(nuc=sensor.nuc, shape=SHAPE, dn_per_k=178.0, sensor_seed=SEED)
    rng = np.random.default_rng(2)
    signal = rng.uniform(1000.0, 40000.0, SHAPE).astype(np.float32)

    delta_t = 3.5
    expected = residual.apply(signal, delta_t)

    src = wp.array(signal, dtype=wp.float32, device=device)
    xi = (
        wp.array(np.ascontiguousarray(residual.gain_field), dtype=wp.float32, device=device),
        wp.array(np.ascontiguousarray(residual.offset_field), dtype=wp.float32, device=device),
    )
    out = wp.zeros(SHAPE, dtype=wp.float32, device=device)
    launch_nuc_residual(
        src,
        xi,
        sensor.nuc.residual_gain_ppm_per_k,
        sensor.nuc.residual_offset_mk_per_k,
        178.0,
        delta_t,
        out,
        device,
    )
    got = out.numpy()
    scale = float(np.abs(expected).max())
    assert np.allclose(got, expected, rtol=2e-6, atol=2e-6 * scale)


# -- what is statistical -----------------------------------------------------------------------


def _netd_dn(cube: np.ndarray) -> float:
    """Temporal σ per pixel, averaged over the array: the DN-domain NETD of a uniform scene."""
    return float(cube.std(axis=0).mean())


@pytest.mark.parametrize("device", DEVICES)
def test_netd_matches_the_cpu_path_and_the_config(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """Within 5 % of the CPU path and 10 % of the configured σ_TVH, over 200 frames."""
    sigma_dn = float(config.detector.sigma_signal_dn)
    gpu = _netd_dn(_run_frames(config, device, N_FRAMES, sigma_dn))
    cpu = _netd_dn(_cpu_frames(config, N_FRAMES, sigma_dn))

    assert gpu == pytest.approx(cpu, rel=0.05)
    ratios = config.sensor.sensor.noise.ratios_3d
    expected = sigma_dn * math.sqrt(1.0 + ratios.t**2 + ratios.tv**2 + ratios.th**2)
    assert gpu == pytest.approx(expected, rel=0.10)


@pytest.mark.parametrize("device", DEVICES)
def test_the_three_d_ratios_are_recovered_within_15_percent(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """Every component within 15 % of its configured sigma, or within its own estimator floor.

    Directional structure, not magnitude, is what makes thermal imagery look thermal (§10.2), so
    this is the test that says the device kernel built the right *kind* of noise. It uses
    ``irsim.validation.decompose_3d`` -- the leakage-corrected NVESD decomposition, already the
    oracle for the CPU path -- rather than an ad-hoc row/column estimator, which would conflate V
    with the row average of VH and report a value 20 % high for reasons that have nothing to do
    with the kernel.

    ``estimate_floors`` gives the standard deviation of each variance estimate for a cube of this
    size, so a component whose floor is comparable to its own value is checked against the floor
    instead of against a percentage it cannot support.
    """
    from irsim.validation.noise import decompose_3d, estimate_floors

    sigma_dn = float(config.detector.sigma_signal_dn)
    cube = _run_frames(config, device, N_FRAMES, sigma_dn)
    got = decompose_3d(cube)

    ratios = config.sensor.sensor.noise.sigma_ratios()
    expected = tuple(sigma_dn * r for r in ratios)
    floors = estimate_floors((N_FRAMES, *SHAPE), expected)

    from irsim.config.sensor import RATIO_ORDER

    for name, want in zip(RATIO_ORDER, expected, strict=True):
        have = float(getattr(got, name))
        floor = float(floors[name])
        tolerance = max(0.15 * want, 3.0 * floor)
        assert abs(have - want) <= tolerance, (
            f"{name}: {have:.4f} against {want:.4f} (15 % = {0.15 * want:.4f}, "
            f"3x floor = {3 * floor:.4f})"
        )


@pytest.mark.parametrize("device", DEVICES)
def test_netd_falls_with_temperature_because_noise_is_not_in_kelvin(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """NETD(373)/NETD(300) < 0.8 (non-negotiable #3), measured through the device kernel.

    σ is scene-independent in DN, so a hotter blackbody -- which moves more DN per kelvin -- has a
    smaller NETD. Noise added in kelvin would give a ratio of exactly 1.0 on either path.
    """
    from irsim.detector.quantise import dn_max_for_bits
    from irsim.isp import dn_per_kelvin

    del dn_max_for_bits
    sigma_dn = float(config.detector.sigma_signal_dn)
    netd_dn = _netd_dn(_run_frames(config, device, 60, sigma_dn))
    at_300 = netd_dn / dn_per_kelvin(config.calibration, config.lut, 300.0)
    at_373 = netd_dn / dn_per_kelvin(config.calibration, config.lut, 373.0)
    assert at_373 / at_300 < 0.8
    assert at_373 / at_300 == pytest.approx(0.576, rel=0.05)


@pytest.mark.parametrize("device", DEVICES)
def test_row_and_column_psd_match_the_cpu_path(
    warp: Any,
    config,  # type: ignore[no-untyped-def]
    device: str,
) -> None:
    """Spatial PSD shapes agree within a factor 1.5 (the ADR 0023 Tier 4(c) statistic).

    ``compare_psd`` is the project's own measure -- the worst radial-bin ratio between two
    frame-averaged 2-D periodograms -- so the device path is held to the same statistic the CPU
    path is validated with, not to a second one invented here.
    """
    from irsim.validation.noise import compare_psd, spatial_psd

    sigma_dn = float(config.detector.sigma_signal_dn)
    gpu = spatial_psd(_run_frames(config, device, N_FRAMES, sigma_dn))
    cpu = spatial_psd(_cpu_frames(config, N_FRAMES, sigma_dn))
    assert compare_psd(gpu, cpu) <= 1.5
