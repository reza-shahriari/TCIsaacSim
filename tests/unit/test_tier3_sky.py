"""Tier 3 sky phenomenology on the aerial fixture (MS.8).

These are the qualitative-but-decisive checks of the ir-sim-testing skill's Tier 3 list, in the
sky-target half of the scope: the sky is not a constant background, cloud is warmer than clear
sky, a target's contrast falls with range as τ(R)/R², and a two-pixel target aliases with its
sub-pixel position unless the optical PSF is doing its job.

Each assertion is stated against an independent oracle -- MS.2's sky model evaluated directly,
the analytic τ(R)/R² law, flux conservation -- rather than against a stored number, so a wrong
sky, a wrong atmosphere or a dropped PSF fails them.

docs/physics-model.md §15 (Tier 3), §5.3; ADR 0044, 0050, 0070, 0071
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.config.gbuffer import UNMAPPED_MATERIAL_ID, GBuffer
from irsim.materials.table import MaterialTable
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.radiometry.lut import BandLUT
from irsim.validation.aerial_scene import SceneTarget, build_aerial_gbuffer

BORESIGHT_DEG = 20.0


def _config(
    sensor_cfg,  # type: ignore[no-untyped-def]
    sky,  # type: ignore[no-untyped-def]
    materials: MaterialTable,
    lut: BandLUT,
    *,
    noise: bool = False,
    psf: bool = False,
    seed: int = 11,
) -> PipelineConfig:
    return PipelineConfig.from_sensor(
        sensor_cfg,
        materials,
        lut=lut,
        noise_enabled=noise,
        psf_enabled=psf,
        atmosphere=sky.atmosphere,
        sky=sky,
        sensor_seed=seed,
    )


def _state(cfg: PipelineConfig) -> PipelineState:
    return PipelineState(housing_temp_k=cfg.t_housing_cal_k, t_s=0.0)


# -- the fixture itself ----------------------------------------------------------------------


def test_fixture_honours_the_contract_in_its_sky_aware_form(
    gbuffer_aerial: dict[str, np.ndarray], aerial_scene
) -> None:  # type: ignore[no-untyped-def]
    """The G-buffer validates, and `material_id` 0 appears **only** under the sky mask -- the
    ADR 0050 contract. (The ground fixtures forbid id 0 outright; a sky scene cannot.)"""
    g = GBuffer.from_dict(dict(gbuffer_aerial))
    assert g.sky_mask is not None and g.sky_mask.dtype == np.bool_
    ids = gbuffer_aerial["material_id"]
    sky = gbuffer_aerial["sky_mask"]
    assert np.all(ids[sky] == UNMAPPED_MATERIAL_ID), "sky pixels carry the background id"
    assert np.all(ids[~sky] >= 1), "every geometry pixel is a mapped material"
    assert np.all(gbuffer_aerial["distance_m"][sky] == 0.0), "no distance sentinel (ADR 0050)"
    assert np.all(np.isfinite(gbuffer_aerial["distance_m"]))
    assert gbuffer_aerial["temperature_k"].dtype == np.float32
    # the elevation grid spans the horizon-to-mid-sky range the profile tests need
    deg = aerial_scene.column_elevation_deg()
    assert deg[0] > deg[-1], "row 0 looks highest: elevation decreases down the image"
    assert 3.0 < deg[-1] < 6.0 and 34.0 < deg[0] < 38.0, deg[[0, -1]]
    assert len(aerial_scene.resolved) == 1 and not aerial_scene.point_targets


def test_sky_is_colder_upward_and_matches_the_sky_model(
    aerial_scene, aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut: BandLUT
) -> None:  # type: ignore[no-untyped-def]
    """Rendered apparent temperature on sky pixels equals MS.2's `T_sky(θ)` within 1 mK: stage 1
    gives ε = 1 under the mask, stage 2 leaves those pixels untouched, and the radiometric branch
    divides the vignetting back out -- so the whole chain is an identity on the sky."""
    cfg = _config(aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut)
    out = run_frame(dict(aerial_scene.planes), cfg, _state(cfg))
    assert out.apparent_t is not None
    col = aerial_scene.shape[1] // 2
    rows = np.where(aerial_scene.sky_mask[:, col])[0]
    rendered = out.apparent_t[rows, col].astype(np.float64)
    elevation = aerial_scene.elevation_rad[rows, col]
    expected = np.asarray(aerial_sky.apparent_temperature_k(0.0, elevation))
    assert np.max(np.abs(rendered - expected)) * 1e3 < 1.0, "sky profile is an identity"
    assert np.all(np.diff(rendered) > 0.0), "colder toward zenith (row 0 is highest)"
    assert rendered[-1] - rendered[0] > 20.0, "a real gradient, not a flat background"


def test_scale_free_profile_and_snr_per_degree(
    aerial_scene, aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut: BandLUT
) -> None:  # type: ignore[no-untyped-def]
    """The *shape* of the sky profile, normalised by its own horizon-to-top span, is what a
    real image can be compared against (ME.4 reports it scale-free because gain and offset are
    unknown). It must match MS.2's shape to 1 %, and the per-degree gradient divided by the
    flat-sky noise must be a finite, positive detectability number.
    """
    cfg = _config(aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut)
    out = run_frame(dict(aerial_scene.planes), cfg, _state(cfg))
    assert out.apparent_t is not None
    col = aerial_scene.shape[1] // 2
    rows = np.where(aerial_scene.sky_mask[:, col])[0]
    rendered = out.apparent_t[rows, col].astype(np.float64)
    deg = np.degrees(aerial_scene.elevation_rad[rows, col])
    model = np.asarray(aerial_sky.apparent_temperature_k(0.0, np.radians(deg)))

    def scale_free(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min())

    assert np.max(np.abs(scale_free(rendered) - scale_free(model))) < 0.01

    # the gradient is steepest near the horizon: that is the saturating column of MS.1
    gradient = np.gradient(rendered, deg)  # K per degree, negative upward
    assert np.all(gradient < 0.0)
    assert abs(gradient[-1]) > 2.0 * abs(gradient[0]), "steepest near the horizon"

    noisy_cfg = _config(aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut, noise=True)
    noisy = run_frame(dict(aerial_scene.planes), noisy_cfg, _state(noisy_cfg))
    assert noisy.apparent_t is not None
    # flat-sky sigma: a narrow elevation band near the top, where the profile is flattest
    band = rows[:6]
    residual = noisy.apparent_t[band, :].astype(np.float64) - out.apparent_t[band, :].astype(
        np.float64
    )
    sigma_k = float(residual.std())
    assert sigma_k > 0.0, "the noisy chain actually adds noise"
    snr_per_deg = abs(float(np.median(gradient))) / sigma_k
    assert math.isfinite(snr_per_deg) and snr_per_deg > 0.0


def test_clouds_read_warmer_than_the_clear_sky(
    aerial_sensor, aerial_cloudy_sky, aerial_materials, tophat_lwir_lut: BandLUT
) -> None:  # type: ignore[no-untyped-def]
    """Cloud at its base temperature against a clear sky tens of kelvin colder: the clutter a
    sky-target detector has to survive (MS.3). The covered pixels must also be *structured* --
    a 1/f^β field, not salt and pepper -- so neighbouring pixels agree far more often than chance.
    """
    scene = build_aerial_gbuffer(
        aerial_sensor.sensor,
        aerial_cloudy_sky,
        aerial_materials,
        boresight_elevation_deg=BORESIGHT_DEG,
        cloud_seed=7,
    )
    cfg = _config(aerial_sensor, aerial_cloudy_sky, aerial_materials, tophat_lwir_lut)
    out = run_frame(dict(scene.planes), cfg, _state(cfg))
    assert out.apparent_t is not None
    t = out.apparent_t.astype(np.float64)
    cloudy = scene.cloud_mask
    clear = scene.sky_mask & ~cloudy
    assert cloudy.sum() > 100 and clear.sum() > 100
    assert t[cloudy].mean() - t[clear].mean() > 20.0
    base = aerial_cloudy_sky.cloud_base_temperature_k(0.0)
    assert abs(t[cloudy].mean() - base) < 0.5, "thick cloud reads its base temperature"
    # structure: horizontal neighbours share the cloud state far more often than for a random
    # mask of the same coverage
    agree = float(np.mean(cloudy[:, 1:] == cloudy[:, :-1]))
    c = float(cloudy.sum()) / cloudy.size
    chance = c * c + (1 - c) * (1 - c)
    assert agree > chance + 0.15, (agree, chance)


def test_target_contrast_falls_as_tau_over_range_squared(
    aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut: BandLUT
) -> None:  # type: ignore[no-untyped-def]
    """A sub-pixel target of fixed size and temperature, moved out in range.

    Against a **grey** atmosphere the excess signal it puts in the frame obeys the closed form
    φ τ(R) (L_t − L_air) exactly, so excess·R²/τ(R) is constant: that pins the 1/R² geometry, the
    Beer-Lambert attenuation and MS.6's injection in one number.

    Against the **layered** atmosphere the same target decays measurably *slower* than that law,
    because the sky column the target occults is itself dimmer at longer range (MS.6's per-class
    L_beyond term). Asserting the sign and size of that departure is what distinguishes the two
    models; a simulator that quietly used the grey law everywhere would fail the second half.
    """
    from irsim.atmosphere.model import Atmosphere

    sensor = aerial_sensor.sensor
    elevation_rad = math.radians(BORESIGHT_DEG)
    ranges = [400.0, 800.0, 1600.0, 3200.0]
    grey = Atmosphere(aerial_sky.atmosphere.preset, aerial_sky.weather, {"lwir": tophat_lwir_lut})

    def excess_for(atmosphere, range_m: float) -> float:  # type: ignore[no-untyped-def]
        scene = build_aerial_gbuffer(
            sensor,
            aerial_sky,
            aerial_materials,
            boresight_elevation_deg=BORESIGHT_DEG,
            targets=[
                SceneTarget("painted_composite", 330.0, 0.5, range_m, (32.5, 24.5), elevation_rad)
            ],
        )
        assert len(scene.point_targets) == 1, "0.5 m at these ranges stays sub-pixel"
        cfg = PipelineConfig.from_sensor(
            aerial_sensor,
            aerial_materials,
            lut=tophat_lwir_lut,
            noise_enabled=False,
            psf_enabled=False,
            atmosphere=atmosphere,
            sky=aerial_sky,
        )
        with_target = run_frame(
            dict(scene.planes), cfg, _state(cfg), point_targets=list(scene.point_targets)
        )
        without = run_frame(dict(scene.planes), cfg, _state(cfg))
        assert with_target.radiance is not None and without.radiance is not None
        return float(
            (with_target.radiance.astype(np.float64) - without.radiance.astype(np.float64)).sum()
        )

    def grey_tau(range_m: float) -> float:
        return float(np.asarray(grey.transmittance("lwir", 0.0, range_m))[()])

    def layered_tau(range_m: float) -> float:
        atm = aerial_sky.atmosphere
        return float(np.asarray(atm.transmittance("lwir", 0.0, range_m, elevation_rad))[()])

    grey_excess = [excess_for(grey, r) for r in ranges]
    grey_scaled = [e * r**2 / grey_tau(r) for e, r in zip(grey_excess, ranges, strict=True)]
    assert max(grey_scaled) / min(grey_scaled) - 1.0 < 0.01, grey_scaled
    assert all(b < a for a, b in zip(grey_excess[:-1], grey_excess[1:], strict=True))

    layered = aerial_sky.atmosphere
    excess = [excess_for(layered, r) for r in ranges]
    scaled = [e * r**2 / layered_tau(r) for e, r in zip(excess, ranges, strict=True)]
    assert all(b < a for a, b in zip(excess[:-1], excess[1:], strict=True)), excess
    # the atmosphere costs more than geometry alone: signal falls faster than 1/R^2
    assert excess[0] / excess[-1] > (ranges[-1] / ranges[0]) ** 2
    # ... but slower than tau(R)/R^2, because the occulted sky dims with range
    assert all(b > a for a, b in zip(scaled[:-1], scaled[1:], strict=True)), scaled
    assert 0.02 < scaled[-1] / scaled[0] - 1.0 < 0.30, scaled


def test_two_pixel_target_aliases_and_the_psf_spreads_it(
    aerial_sensor, aerial_sky, aerial_materials, tophat_lwir_lut: BandLUT
) -> None:  # type: ignore[no-untyped-def]
    """A 2 px target swept through sub-pixel phases on a 4x supersampled render: total flux is
    conserved but the peak pixel is not -- the aliasing a point-sampled rasteriser cannot avoid
    (ADR 0071) -- and switching MS.4's optical PSF on lowers the peak and widens the footprint.
    """
    import copy

    raw = copy.deepcopy(aerial_sensor.model_dump(mode="json"))
    raw["sensor"]["optics"]["supersample_factor"] = 4
    from irsim.config.sensor import SensorConfig

    sensor_cfg = SensorConfig.model_validate(raw)
    sensor = sensor_cfg.sensor
    range_m = 700.0
    size_m = 2.0 * range_m * sensor.fpa.pitch_um * 1e-6 / (sensor.optics.focal_length_mm * 1e-3)

    def render(phase: float, psf: bool) -> np.ndarray:
        scene = build_aerial_gbuffer(
            sensor,
            aerial_sky,
            aerial_materials,
            boresight_elevation_deg=BORESIGHT_DEG,
            targets=[
                SceneTarget("painted_composite", 330.0, size_m, range_m, (20.0 + phase, 20.0))
            ],
            supersample=4,
        )
        assert len(scene.resolved) == 1
        cfg = _config(sensor_cfg, aerial_sky, aerial_materials, tophat_lwir_lut, psf=psf)
        out = run_frame(dict(scene.planes), cfg, _state(cfg))
        assert out.radiance is not None
        base = build_aerial_gbuffer(
            sensor,
            aerial_sky,
            aerial_materials,
            boresight_elevation_deg=BORESIGHT_DEG,
            supersample=4,
        )
        ref = run_frame(dict(base.planes), cfg, _state(cfg))
        assert ref.radiance is not None
        return out.radiance.astype(np.float64) - ref.radiance.astype(np.float64)

    def lit(e: np.ndarray) -> int:
        return int(np.count_nonzero(e > 0.01 * e.max()))

    phases = [0.0, 0.25, 0.5, 0.75]
    sharp = [render(p, psf=False) for p in phases]
    flux = [float(e.sum()) for e in sharp]
    footprint = [lit(e) for e in sharp]
    assert max(flux) / min(flux) - 1.0 < 0.02, f"flux conserved across phase: {flux}"
    # 2 px on a 4x grid always fills one native pixel completely, so the *peak* is phase-
    # independent; the aliasing shows in the footprint, which grows from 2x2 to 3x3 as the
    # target straddles pixel boundaries. That is the artefact a detector sees as size jitter.
    assert min(footprint) == 4, f"pixel-aligned, a 2 px target lights 2x2: {footprint}"
    assert max(footprint) >= 6, f"straddling a boundary it lights more: {footprint}"

    blurred = render(0.0, psf=True)
    assert float(blurred.max()) < float(sharp[0].max()), "the PSF lowers the peak"
    assert abs(float(blurred.sum()) / flux[0] - 1.0) < 0.02, "and conserves the flux"
    assert lit(blurred) > footprint[0], "and spreads it over more pixels"


@pytest.mark.skip(
    reason="ME.5 landed (docs/validation/reference-stats-2026-09-15.md) and the comparison it "
    "enables is a whole-set acceptance run, not a unit assertion: it lives in M12.2 "
    "(docs/validation/tier4-2026-09-15.md) and it currently **fails**, with the discriminator "
    "naming the signal path rather than the physics. Asserting a passing version of that "
    "comparison here would contradict the report, so this stays a pointer to it."
)
def test_frame_statistics_inside_the_reference_bands() -> None:  # pragma: no cover
    raise AssertionError("unreachable")
