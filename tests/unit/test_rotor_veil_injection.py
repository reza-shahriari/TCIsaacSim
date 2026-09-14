"""Stage 2c: rotor veils composited onto the k× radiance plane (ADR 0081).

The veil is a *blend*, not MS.6's excess, and that choice is what most of this file checks. The
plane already carries the correctly attenuated background at every pixel — sky beyond the disc for
some, the aircraft's own arm for others — so the blend is exact for both at once, where an excess
form carrying a `sky_beyond` term would subtract a sky column that is not behind the airframe
pixels at all. Two tests put a rotor half over sky and half over a warm arm to hold that.

The other load-bearing test is that windowing is invisible: the coverage is built on the ellipse's
bounding box for memory, and a full-frame reference must agree bit for bit.
"""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere import Atmosphere, load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.optics.rotor import DiscEllipse, RotorDisc, coverage_map, swept_angle_rad, veil_radiance
from irsim.pipeline import PipelineConfig, PipelineState, run_frame
from irsim.pipeline.rotor_veil import (
    RotorVeil,
    blade_radiance_at_sensor,
    inject_rotor_veils,
    veil_window,
)
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

DISC = RotorDisc()
SHAPE = (160, 200)
ELLIPSE = DiscEllipse(
    centre_px=(100.0, 80.0),
    semi_major_px=40.0,
    semi_minor_px=12.0,
    rotation_deg=0.0,
    tilt_rad=math.acos(12.0 / 40.0),
)
#: A bolometer frame on a 3000 rpm prop: 1.67 blade spacings, the smooth-annulus regime.
BOLO_SWEPT = swept_angle_rad(3000.0, 1.0 / 60.0)
#: A 2 ms cooled integration: 0.20 spacings, the resolved-arc regime.
COOLED_SWEPT = swept_angle_rad(3000.0, 0.002)


def _weather(t_air: float = 288.15) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, 0.3, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _sensor() -> SensorConfig:
    return SensorConfig.model_validate(copy.deepcopy(BOSON))


def _lb(lut: BandLUT, t: float) -> float:
    return float(lut.lookup(np.float64(t))[()])


def _veil(lut: BandLUT, **kw: object) -> RotorVeil:
    args: dict = {
        "disc": DISC,
        "ellipse": ELLIPSE,
        "swept_rad": BOLO_SWEPT,
        "blade_radiance": _lb(lut, 290.0),
        "range_m": 20.0,
    }
    args.update(kw)
    return RotorVeil(**args)  # type: ignore[arg-type]


def _sky_plane(lut: BandLUT, t: float = 230.0) -> np.ndarray:
    return np.full(SHAPE, _lb(lut, t), dtype=np.float64)


# --------------------------------------------------------------------------------------------


def test_no_veils_returns_the_very_same_plane(tophat_lwir_lut: BandLUT) -> None:
    """A scene without rotors must pay nothing and leave every golden untouched."""
    plane = _sky_plane(tophat_lwir_lut)
    assert inject_rotor_veils(plane, (), None, "lwir", 0.0, tophat_lwir_lut) is plane


def test_the_windowed_coverage_equals_a_full_frame_composite(tophat_lwir_lut: BandLUT) -> None:
    """The bounding box is a memory optimisation and must be invisible in the result.

    A 4x supersampled Boson frame is 2560 x 2048, so a full-frame float64 coverage map per rotor
    is 40 MB and four of them run to 160 MB a frame. That is the reason for the window, and this
    is the check that it costs no accuracy.
    """
    plane = _sky_plane(tophat_lwir_lut)
    veil = _veil(tophat_lwir_lut)
    got = inject_rotor_veils(plane, [veil], None, "lwir", 0.0, tophat_lwir_lut)

    alpha = coverage_map(
        DISC,
        SHAPE,
        ELLIPSE.centre_px,
        ELLIPSE.semi_major_px,
        ELLIPSE.semi_minor_px,
        BOLO_SWEPT,
        rotation_deg=ELLIPSE.rotation_deg,
    )
    expected = veil_radiance(plane, alpha, veil.blade_radiance)
    assert np.array_equal(got, expected)


def test_the_lift_is_the_projected_blade_area_times_the_contrast(tophat_lwir_lut: BandLUT) -> None:
    """Summed over the frame, the veil adds ``(sum alpha) (L_blade - L_background)``.

    This is the area identity of ADR 0081 carried all the way through the injection: whatever the
    shutter does to the *shape*, the flux it adds is set by the blade area and the contrast.
    """
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    veil = _veil(lut)
    out = inject_rotor_veils(plane, [veil], None, "lwir", 0.0, lut)

    alpha = coverage_map(
        DISC,
        SHAPE,
        ELLIPSE.centre_px,
        ELLIPSE.semi_major_px,
        ELLIPSE.semi_minor_px,
        BOLO_SWEPT,
        rotation_deg=ELLIPSE.rotation_deg,
    )
    contrast = veil.blade_radiance - _lb(lut, 230.0)
    assert float((out - plane).sum()) == pytest.approx(float(alpha.sum()) * contrast, rel=1e-9)


def test_the_shutter_changes_the_picture_and_not_the_flux(tophat_lwir_lut: BandLUT) -> None:
    """A bolometer's annulus and a cooled detector's arcs add the same total to the frame."""
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    annulus = inject_rotor_veils(plane, [_veil(lut)], None, "lwir", 0.0, lut)
    arcs = inject_rotor_veils(plane, [_veil(lut, swept_rad=COOLED_SWEPT)], None, "lwir", 0.0, lut)

    assert float((arcs - plane).sum()) == pytest.approx(float((annulus - plane).sum()), rel=0.03)
    assert float((arcs - plane).max()) > 3.0 * float((annulus - plane).max())
    assert not np.allclose(arcs, annulus)


def test_the_veil_blends_sky_and_airframe_in_one_pass(tophat_lwir_lut: BandLUT) -> None:
    """Half the disc over cold sky, half over a warm arm: each pixel keeps its own background.

    This is the case an excess form with a ``sky_beyond`` term gets wrong — over the arm there is
    no sky column to subtract — and the reason stage 2c composites rather than injects.
    """
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    plane[:80, :] = _lb(lut, 300.0)  # a warm arm across the top half of the disc
    veil = _veil(lut)
    out = inject_rotor_veils(plane, [veil], None, "lwir", 0.0, lut)

    lifted = out - plane
    # Over the cold sky the blade is warmer, so the veil lifts; over the 300 K arm a 290 K blade
    # is colder, so it must *dip*. One operator, opposite signs, same frame.
    assert float(lifted[80:, :].max()) > 0.0
    assert float(lifted[:80, :].min()) < 0.0
    # and nowhere does it push past either endpoint
    assert float(out.min()) >= min(float(plane.min()), veil.blade_radiance) - 1e-12
    assert float(out.max()) <= max(float(plane.max()), veil.blade_radiance) + 1e-12


def test_a_blade_at_ambient_over_ambient_sky_changes_nothing(tophat_lwir_lut: BandLUT) -> None:
    """Zero contrast, zero effect — whatever the coverage is."""
    lut = tophat_lwir_lut
    plane = _sky_plane(lut, 288.15)
    veil = _veil(lut, blade_radiance=_lb(lut, 288.15))
    out = inject_rotor_veils(plane, [veil], None, "lwir", 0.0, lut)
    assert np.allclose(out, plane, rtol=1e-12)


# --------------------------------------------------------------------------------------------
# The blade takes the same atmospheric path the plane took
# --------------------------------------------------------------------------------------------


def test_without_an_atmosphere_the_blade_arrives_as_authored(tophat_lwir_lut: BandLUT) -> None:
    veil = _veil(tophat_lwir_lut)
    got = blade_radiance_at_sensor(veil, None, "lwir", 0.0, tophat_lwir_lut)
    assert got == pytest.approx(veil.blade_radiance, rel=1e-12)


def test_the_blade_is_attenuated_at_its_own_range(tophat_lwir_lut: BandLUT) -> None:
    """A disc at 3 km reads closer to air temperature than the same disc at 20 m.

    The veil must not borrow the background's path: a rotor is at the *target's* range, and the
    plane behind it may be sky at infinity.
    """
    lut = tophat_lwir_lut
    atm = Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather())
    near = blade_radiance_at_sensor(_veil(lut, range_m=20.0), atm, "lwir", 0.0, lut)
    far = blade_radiance_at_sensor(_veil(lut, range_m=3000.0), atm, "lwir", 0.0, lut)
    l_air = _lb(lut, 288.15)
    blade = _lb(lut, 290.0)

    assert near == pytest.approx(blade, rel=2e-3)  # 20 m of clear air is almost nothing
    assert abs(far - l_air) < abs(near - l_air)  # 3 km pulls it towards ambient
    # and it is Beer-Lambert, not a fudge
    state = atm.state(0.0)
    tau = math.exp(-state.gamma_per_m["lwir"] * 3000.0)
    assert far == pytest.approx(tau * blade + (1.0 - tau) * l_air, rel=1e-9)


@pytest.mark.parametrize("t_blade,warmer_than_air", [(290.0, True), (270.0, False)])
def test_range_pulls_the_composited_disc_towards_ambient(
    tophat_lwir_lut: BandLUT, t_blade: float, warmer_than_air: bool
) -> None:
    """Distance moves the disc towards air temperature, which is not the same as "fainter".

    Written first as "a distant disc lifts the sky less", which is only true for a blade *above*
    ambient. Against a 230 K sky with 288 K air, a 270 K blade at 3 km reads **brighter** than the
    same blade at 20 m, because the path radiance it picks up outweighs what attenuation removes.
    The lift ratio is the contrast ratio exactly, either way.
    """
    lut = tophat_lwir_lut
    atm = Atmosphere(load_atmosphere_preset("us_standard_clear"), _weather())
    plane = _sky_plane(lut)
    blade = _lb(lut, t_blade)
    l_bg = _lb(lut, 230.0)
    assert (blade > _lb(lut, 288.15)) is warmer_than_air

    veils = [_veil(lut, blade_radiance=blade, range_m=r) for r in (20.0, 3000.0)]
    lift = [
        float((inject_rotor_veils(plane, [v], atm, "lwir", 0.0, lut) - plane).sum()) for v in veils
    ]
    arrived = [blade_radiance_at_sensor(v, atm, "lwir", 0.0, lut) for v in veils]

    assert lift[1] / lift[0] == pytest.approx((arrived[1] - l_bg) / (arrived[0] - l_bg), rel=1e-9)
    assert (lift[1] < lift[0]) is warmer_than_air


# --------------------------------------------------------------------------------------------
# Windowing, occlusion, and the edges
# --------------------------------------------------------------------------------------------


def test_the_window_covers_the_ellipse_at_every_rotation() -> None:
    """Built from the semi-major axis both ways, so no rotation can clip the disc."""
    for rotation in range(0, 180, 7):
        e = DiscEllipse((100.0, 80.0), 40.0, 12.0, float(rotation), 0.3)
        box = veil_window(e, SHAPE)
        assert box is not None
        y0, y1, x0, x1 = box
        alpha = coverage_map(
            DISC, SHAPE, e.centre_px, e.semi_major_px, e.semi_minor_px, 100.0, rotation_deg=rotation
        )
        lit = np.argwhere(alpha > 0.0)
        assert lit[:, 0].min() >= y0 and lit[:, 0].max() < y1
        assert lit[:, 1].min() >= x0 and lit[:, 1].max() < x1


def test_a_disc_entirely_off_the_frame_is_skipped(tophat_lwir_lut: BandLUT) -> None:
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    off = DiscEllipse((-500.0, 80.0), 40.0, 12.0, 0.0, 0.3)
    assert veil_window(off, SHAPE) is None
    out = inject_rotor_veils(plane, [_veil(lut, ellipse=off)], None, "lwir", 0.0, lut)
    assert np.array_equal(out, plane)


def test_a_partly_visible_disc_composites_the_visible_part(tophat_lwir_lut: BandLUT) -> None:
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    edge = DiscEllipse((10.0, 80.0), 40.0, 12.0, 0.0, 0.3)
    out = inject_rotor_veils(plane, [_veil(lut, ellipse=edge)], None, "lwir", 0.0, lut)
    assert float((out - plane).sum()) > 0.0
    assert np.array_equal(out[:, 60:], plane[:, 60:])


def test_an_occlusion_mask_keeps_the_blade_off_what_is_in_front(tophat_lwir_lut: BandLUT) -> None:
    """Without this the veil paints a blade over the motor bell it is bolted to."""
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:80, :] = True
    out = inject_rotor_veils(plane, [_veil(lut, occluded=mask)], None, "lwir", 0.0, lut)
    assert np.array_equal(out[:80, :], plane[:80, :])
    assert float((out[80:, :] - plane[80:, :]).sum()) > 0.0


def test_two_rotors_compose_independently(tophat_lwir_lut: BandLUT) -> None:
    lut = tophat_lwir_lut
    plane = _sky_plane(lut)
    left = DiscEllipse((55.0, 80.0), 25.0, 8.0, 0.0, 0.3)
    right = DiscEllipse((145.0, 80.0), 25.0, 8.0, 0.0, 0.3)
    one = inject_rotor_veils(plane, [_veil(lut, ellipse=left)], None, "lwir", 0.0, lut)
    both = inject_rotor_veils(
        plane, [_veil(lut, ellipse=left), _veil(lut, ellipse=right)], None, "lwir", 0.0, lut
    )
    assert np.array_equal(both[:, :100], one[:, :100])
    assert float((both - plane).sum()) == pytest.approx(2.0 * float((one - plane).sum()), rel=1e-9)


def test_the_plane_dtype_survives(tophat_lwir_lut: BandLUT) -> None:
    lut = tophat_lwir_lut
    plane = _sky_plane(lut).astype(np.float32)
    out = inject_rotor_veils(plane, [_veil(lut)], None, "lwir", 0.0, lut)
    assert out.dtype == np.float32


@pytest.mark.parametrize(
    "kwargs", [{"range_m": 0.0}, {"range_m": -1.0}, {"blade_radiance": -1.0}, {"swept_rad": -0.1}]
)
def test_impossible_veils_are_refused(tophat_lwir_lut: BandLUT, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        _veil(tophat_lwir_lut, **kwargs)


# --------------------------------------------------------------------------------------------
# Through run_frame — stage 2c actually wired
# --------------------------------------------------------------------------------------------


def _sized_sensor(width: int, height: int, supersample: int = 1) -> SensorConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"]["supersample_factor"] = supersample
    return SensorConfig.model_validate(d)


@pytest.mark.parametrize("psf_enabled", [False, True])
def test_the_veil_survives_the_whole_chain(tophat_lwir_lut: BandLUT, psf_enabled: bool) -> None:
    """A hot rotor over a uniform 290 K scene: the flux it adds is conserved through the optics.

    The veil is composited on the supersampled plane before the PSF and the box downsample, which
    is the same place MS.6's point targets land, so the excess measured on the *native* radiance
    plane must come back as the projected blade area times the contrast whether the PSF is on or
    off. With it on, the disc spreads; the total may not move.
    """
    lut = tophat_lwir_lut
    shape = (64, 64)
    planes = {
        "temperature_k": np.full(shape, 290.0, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.ones(shape, np.float32),
    }
    sensor_cfg = _sized_sensor(64, 64)
    ellipse = DiscEllipse((32.0, 32.0), 18.0, 6.0, 0.0, math.acos(6.0 / 18.0))
    veil = RotorVeil(
        disc=DISC,
        ellipse=ellipse,
        swept_rad=BOLO_SWEPT,
        blade_radiance=_lb(lut, 400.0),
        range_m=50.0,
    )
    cfg = PipelineConfig.from_sensor(
        sensor_cfg,
        MaterialTable.constant(1.0),
        lut=lut,
        noise_enabled=False,
        psf_enabled=psf_enabled,
    )
    with_v = run_frame(
        planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k), rotor_veils=[veil]
    )
    without = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert with_v.radiance is not None and without.radiance is not None

    alpha = coverage_map(
        DISC, shape, ellipse.centre_px, ellipse.semi_major_px, ellipse.semi_minor_px, BOLO_SWEPT
    )
    contrast = veil.blade_radiance - _lb(lut, 290.0)
    excess = with_v.radiance.astype(np.float64) - without.radiance.astype(np.float64)
    assert excess.sum() == pytest.approx(float(alpha.sum()) * contrast, rel=2e-3)
    assert excess.max() > 0.0


def test_a_frame_without_rotors_is_bit_identical(tophat_lwir_lut: BandLUT) -> None:
    """Stage 2c must cost existing scenes nothing at all, goldens included."""
    shape = (32, 32)
    planes = {
        "temperature_k": np.full(shape, 290.0, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.ones(shape, np.float32),
    }
    cfg = PipelineConfig.from_sensor(
        _sized_sensor(32, 32), MaterialTable.constant(1.0), lut=tophat_lwir_lut, noise_enabled=False
    )
    a = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k), rotor_veils=[])
    b = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
    assert a.radiance is not None and b.radiance is not None
    assert np.array_equal(a.radiance, b.radiance)
