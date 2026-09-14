"""What the ISP left in the picture (ME.3b, ADR 0068).

Three extractors, each checked against the thing that produced the artefact rather than against a
stored number: the project's own `agc_linear`/`agc_plateau` for the histogram fingerprint, the
analytic answer for a 3x3 unsharp mask for the ringing, and `replace_bad_pixels` for the smoothed
footprint. Two of the three refuse to run on frames that are not a camera's display output, which
is half of what the row is for.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.config.sensor import NoiseSpec
from irsim.isp.agc import agc_linear, agc_plateau
from irsim.isp.bad_pixel import replace_bad_pixels
from irsim.isp.dde import dde
from irsim.noise.defects import generate_map
from irsim.validation import (
    agc_signature,
    edge_overshoot,
    replaced_pixel_map,
    require_display_output,
)
from irsim_eval.transcode import h264_round_trip
from irsim_eval.video import ffmpeg_available

BOX3_OVERSHOOT_PER_GAIN = 1.0 / 3.0
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not installed")


def _sky_scene(seed: int = 0, shape: tuple[int, int] = (256, 320)) -> np.ndarray:
    """A DN16 sky frame: a gentle elevation gradient, a cloud, noise, and a small hot target."""
    rng = np.random.default_rng(seed)
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    scene = 20000.0 + 1.5 * rows + rng.normal(0.0, 60.0, size=shape)
    scene += 500.0 * np.exp(-((rows - 60.0) ** 2 + (cols - 100.0) ** 2) / (2.0 * 50.0**2))
    scene[120:126, 200:206] += 5000.0
    return np.clip(scene, 0.0, 65535.0)


def _to_codes(y: object) -> np.ndarray:
    return np.clip(np.rint(np.asarray(y, dtype=np.float64) * 255.0), 0, 255)


def test_plateau_equalisation_is_separated_from_a_linear_stretch() -> None:
    """The row's discriminator, through the project's own two AGC implementations.

    Equalisation integrates a clipped histogram into a CDF, so its output is uniform over the codes
    it occupies; a linear stretch reproduces the scene's distribution, which on a sky frame is the
    noise's own bell. Measured across flat sky, a gradient, cloud and a horizon, plateau stays at
    0.003-0.007 and linear at 0.15-0.50 -- a separation of twenty times, not a hair's breadth.
    """
    scene = _sky_scene()
    linear = agc_signature(_to_codes(agc_linear(scene, 0.005, 0.995)), conversion="display")
    plateau = agc_signature(_to_codes(agc_plateau(scene, 0.012)), conversion="display")

    assert linear.verdict == "stretched"
    assert plateau.verdict == "equalised"
    assert linear.cdf_deviation > 20.0 * plateau.cdf_deviation
    assert plateau.entropy_bits > linear.entropy_bits


def test_a_naturally_uniform_scene_is_reported_as_indeterminate() -> None:
    """A strong ramp filling the frame is uniform already, so a linear stretch of it *is*
    equalisation as far as any histogram statistic can tell. Claiming otherwise would be the
    failure mode this verdict exists to avoid."""
    rng = np.random.default_rng(1)
    ramp = 20000.0 + 5.0 * np.arange(256)[:, None] + rng.normal(0.0, 60.0, size=(256, 320))
    signature = agc_signature(_to_codes(agc_linear(ramp, 0.005, 0.995)), conversion="display")
    assert signature.verdict == "indeterminate"


def test_a_y16_derived_set_may_not_report_an_agc_signature() -> None:
    """Halmstad's frames are a recorder's conversion of a Y16 stream, and its rule is itself
    unverified -- a histogram measured there describes the recorder (ME.1a's index excludes both
    analysers on that set for the same reason; this is the second lock)."""
    frame = _to_codes(agc_plateau(_sky_scene(), 0.012))
    for conversion in ("linear", "min_max", "unknown"):
        with pytest.raises(ValueError, match="not the camera's display output"):
            agc_signature(frame, conversion=conversion)
        with pytest.raises(ValueError, match="not the camera's display output"):
            edge_overshoot(frame, conversion=conversion)
    require_display_output("display")  # the one that is allowed


def _step_frame(low: float = 0.2, high: float = 0.8, size: int = 64) -> np.ndarray:
    """A cross of two clean steps, so both scan axes see one."""
    frame = np.full((size, size), low)
    frame[:, size // 2 :] = high
    frame[size // 2 :, :] = high
    frame[size // 2 :, size // 2 :] = low
    return frame


def test_dde_overshoot_is_the_analytic_third_of_the_gain() -> None:
    """A 3x3 box at the pixel beside a step reads two thirds of the way across it, so an unsharp
    mask of gain g overshoots by exactly g/3 -- a known answer, not a regression fixture."""
    frame = _step_frame()
    for gain in (0.3, 0.6, 0.9):
        measured = edge_overshoot(dde(frame, gain), conversion="display")
        assert measured.n_edges > 100
        assert measured.overshoot == pytest.approx(gain * BOX3_OVERSHOOT_PER_GAIN, rel=1e-6)
        assert measured.undershoot == pytest.approx(gain * BOX3_OVERSHOOT_PER_GAIN, rel=1e-6)
        assert measured.implied_box3_gain == pytest.approx(gain, rel=1e-6)


def test_an_undded_edge_has_no_overshoot() -> None:
    plain = edge_overshoot(dde(_step_frame(), 0.0), conversion="display")
    assert plain.overshoot == pytest.approx(0.0, abs=1e-9)
    assert plain.undershoot == pytest.approx(0.0, abs=1e-9)


def test_overshoot_survives_noise_and_eight_bit_quantisation() -> None:
    """The real case: a noisy 8-bit display frame, where the edge finder has to reject noise as an
    edge and the ringing is only 30 codes tall."""
    rng = np.random.default_rng(2)
    gain = 0.6
    frame = dde(_step_frame(), gain) + rng.normal(0.0, 0.004, size=(64, 64))
    codes = np.clip(np.rint(frame * 255.0), 0, 255)
    measured = edge_overshoot(codes, conversion="display")
    assert measured.n_edges > 50
    assert measured.overshoot == pytest.approx(gain * BOX3_OVERSHOOT_PER_GAIN, rel=0.10)


def _defective_clip(
    shape: tuple[int, int] = (128, 128), frames: int = 48, seed: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """A clip the camera has already repaired, and the mask of what it repaired."""
    rng = np.random.default_rng(seed)
    noise = NoiseSpec(
        netd_mk_at_300k=50.0,
        ratios_3d={"t": 0.02, "v": 0.08, "h": 0.15, "tv": 0.05, "th": 0.05, "vh": 0.30, "tvh": 1.0},
        fpn_drift_tau_s=float("inf"),
        bad_pixel_fraction=0.005,
        bad_pixel_cluster_lambda=1.5,
    )
    defects = generate_map(shape, noise, sensor_seed=11)
    mask = defects.mask
    clip = np.stack(
        [replace_bad_pixels(rng.normal(120.0, 3.0, size=shape), mask) for _ in range(frames)]
    )
    return clip, mask


def test_replaced_pixels_are_found_with_under_one_percent_false_positives() -> None:
    """The §10.4 smoothed footprint read back: a replaced pixel is the mean of its neighbours, so
    its Laplacian vanishes identically while a good pixel carries sqrt(20) times the pixel noise.

    Measured on 69 interior defects: every one found on float frames and 93 % after rounding to
    8-bit codes, with no false positive in either case. The rounding costs the pixels whose
    replacement lands within half a code of a neighbour's own Laplacian.
    """
    clip, mask = _defective_clip()
    interior = np.zeros(mask.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    good = interior & ~mask
    defects = interior & mask
    assert np.count_nonzero(defects) > 50, "too few defects to bound a false-positive rate"

    estimate = replaced_pixel_map(clip)
    assert estimate.mask[defects].all()
    assert np.count_nonzero(estimate.mask & good) / np.count_nonzero(good) < 0.01
    # Most replaced pixels score identically zero -- they *are* the mean of their neighbours. The
    # two that do not are a cluster's rim, filled while the inside was still invalid: merely very
    # smooth, still far under the threshold, and the reason recall is asserted and not assumed.
    assert float(np.median(estimate.score[defects])) < 1e-9
    assert estimate.score[defects].max() < estimate.threshold
    assert float(np.median(estimate.score[good])) == pytest.approx(1.0, rel=0.05)

    codes = replaced_pixel_map(np.clip(np.rint(clip), 0, 255))
    assert np.count_nonzero(codes.mask & defects) >= 0.9 * np.count_nonzero(defects)
    assert np.count_nonzero(codes.mask & good) / np.count_nonzero(good) < 0.01


@needs_ffmpeg
@pytest.mark.slow
def test_a_lossy_codec_makes_the_footprint_unmeasurable_rather_than_wrong() -> None:
    """The failure mode, measured rather than asserted.

    x264 moves a replaced pixel off the exact mean of its neighbours while flattening everyone
    else's Laplacian toward it, so the recall of a footprint that was perfect before coding falls to
    zero -- with the false-positive rate still zero. An empty result on a lossy set means "not
    measurable here", never "no bad pixels", which is why ME.5 reports it beside the codec floor.
    """
    clip, mask = _defective_clip()
    codes = np.clip(np.rint(clip), 0, 255).astype(np.uint8)
    interior = np.zeros(mask.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    assert replaced_pixel_map(codes.astype(np.float64)).mask[interior & mask].any()

    coded = replaced_pixel_map(h264_round_trip(codes, fps=60.0, crf=12).astype(np.float64))
    assert coded.count() == 0


def test_a_clean_clip_has_no_replaced_pixels() -> None:
    """The control. A detector keyed on smoothness alone would flag the quietest pixels of any
    clip; this one is normalised by the array's own typical Laplacian, so a clip with nothing
    replaced in it returns nothing."""
    rng = np.random.default_rng(6)
    clean = rng.normal(120.0, 4.0, size=(48, 128, 128))
    assert replaced_pixel_map(clean).count() == 0


def test_a_three_by_three_cluster_is_found_by_its_centre() -> None:
    """The specific case the row names, with the geometry made explicit: the centre is filled last,
    from four already-filled neighbours, so it is exactly the mean of them."""
    rng = np.random.default_rng(7)
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:23, 30:33] = True
    clip = np.stack(
        [replace_bad_pixels(rng.normal(120.0, 3.0, size=(48, 48)), mask) for _ in range(48)]
    )
    estimate = replaced_pixel_map(clip)
    assert estimate.mask[21, 31], "the last-filled pixel of the cluster is exactly interpolated"
    assert estimate.score[21, 31] < 1e-9


def test_inputs_are_checked() -> None:
    with pytest.raises(TypeError, match="float16"):
        replaced_pixel_map(np.zeros((4, 8, 8), dtype=np.float16))
    with pytest.raises(ValueError, match="one code"):
        agc_signature(np.full((8, 8), 7.0), conversion="display")
    with pytest.raises(ValueError, match="outside a 8-bit"):
        agc_signature(np.full((8, 8), 300.0), conversion="display")
