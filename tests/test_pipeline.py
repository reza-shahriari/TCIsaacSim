"""docs/isaacsim_checklist.md Phase 1, steps 1.10-1.12 (end-to-end)."""
import numpy as np
from thermal_physics.quantize import quantize
from thermal_physics.agc import linear_agc
from thermal_physics.pipeline import render_frame, SensorParams, estimate_d_signal_d_t


def test_quantize_stays_in_bit_range():
    """Output must contain only integers in [0, 2**bits - 1], even for extreme inputs."""
    signal = np.array([-1e9, -1.0, 0.0, 5.0, 10.0, 1e9])
    dn = quantize(signal, bits=8, v_min=0.0, v_max=10.0)
    assert dn.min() >= 0 and dn.max() <= 255
    assert dn.dtype.kind == "i"


def test_agc_uses_close_to_full_display_range_despite_outlier():
    rng = np.random.default_rng(1)
    img = rng.normal(500, 50, (50, 50))
    img[0, 0] = 1e6  # single outlier pixel
    out = linear_agc(img)
    assert out.max() == 255
    # most of the frame should not be crushed near zero by the one outlier
    assert np.median(out) > 10


def test_estimate_d_signal_d_t_is_positive():
    """Hotter scenes should produce more signal -- the whole noise model depends on this."""
    assert estimate_d_signal_d_t(SensorParams()) > 0


def test_end_to_end_smoke():
    """render_frame() on a synthetic temperature gradient should produce a
    believable (H, W, 3) uint8 image -- the Phase 1 completion check."""
    h, w = 24, 32
    temperature_k = np.linspace(280, 320, w)[None, :].repeat(h, axis=0)
    emissivity = np.full((h, w), 0.95)
    range_m = np.full((h, w), 50.0)

    img = render_frame(temperature_k, emissivity, range_m, SensorParams(seed=7))

    assert img.shape == (h, w, 3)
    assert img.dtype == np.uint8
    # warm end should render brighter than cool end under white_hot
    assert img[:, -1, 0].mean() > img[:, 0, 0].mean()


def test_render_frame_reproducible_with_seed():
    h, w = 10, 10
    t = np.full((h, w), 300.0)
    e = np.full((h, w), 0.9)
    r = np.full((h, w), 20.0)
    img_a = render_frame(t, e, r, SensorParams(seed=42))
    img_b = render_frame(t, e, r, SensorParams(seed=42))
    img_c = render_frame(t, e, r, SensorParams(seed=43))
    assert np.array_equal(img_a, img_b)
    assert not np.array_equal(img_a, img_c)
