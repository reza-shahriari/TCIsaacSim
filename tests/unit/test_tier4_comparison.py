"""ME.6: the DN8 acceptance checks, and the control every one of them depends on.

ADR 0068 replaced §15's radiometric Tier 4 targets with display-domain ones. These tests pin two
things: that each statistic measures what its name says on input where the answer is known, and
that **synthetic versus itself passes everything** -- if the control fails, the thresholds are
wrong and nothing the report says about a real comparison is worth reading.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.validation.compare import (
    Tier4Report,
    Tier4Targets,
    auc_from_scores,
    compare_frames,
    esf_width_px,
    histogram_emd,
    psd_shape_ratio,
)
from irsim_eval.discriminator import FEATURE_NAMES, gap_score, patch_features

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "validation_report.py"
RNG = np.random.default_rng(20260915)


def _script():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("validation_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    base = np.random.default_rng(4).normal(0.0, 1.0, shape)
    smooth = np.zeros_like(base)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            smooth += np.roll(np.roll(base, dy, 0), dx, 1)
    return 110.0 + 12.0 * smooth / smooth.std()


# --- the statistics ------------------------------------------------------------------------------


def test_histogram_emd_reads_as_codes_of_shift() -> None:
    """The unit is the point: EMD in one dimension is the L1 distance between the CDFs, so the
    answer for a pure shift is the shift itself -- a number a reader can act on."""
    frame = np.clip(_scene() + RNG.normal(0.0, 3.0, (64, 64)), 0, 255)
    assert histogram_emd(frame, frame) == pytest.approx(0.0, abs=1e-12)
    for shift in (1.0, 5.0, 20.0):
        assert histogram_emd(frame, frame + shift) == pytest.approx(shift, rel=0.05)


def test_histogram_emd_compares_distributions_not_pixel_counts() -> None:
    """A real clip and a render are never the same number of pixels, so the histograms are
    normalised to unit mass before they are differenced."""
    big = np.clip(_scene((96, 96)) + RNG.normal(0.0, 3.0, (96, 96)), 0, 255)
    small = big[:48, :48]
    assert histogram_emd(big, small) < 3.0
    with pytest.raises(ValueError, match="empty"):
        histogram_emd(np.full((8, 8), 300.0), big)


def test_psd_shape_is_scale_free_but_catches_a_noise_mismatch() -> None:
    """ME.6's stated criterion, measured -- and it moved two thresholds.

    Multiplying a frame by a constant must *not* move the statistic; that is what makes it usable
    after an unknown AGC. Adding three times the noise must move it, because noise is white and
    the scene is not.

    ⚠️ **ADR 0068's x2 target does not catch a 3x noise mismatch.** Measured over three seeds:
    matched noise reads 1.05-1.08, 2x reads 1.30-1.36, 3x reads 1.77-1.93 and only 5x reaches
    3.2. A x2 threshold would pass a simulator whose noise is three times wrong, so the target is
    **1.5** -- which separates matched from 3x with a 1.6x margin on both sides and openly misses
    a 2x error.
    """
    scene = _scene()
    quiet = scene + RNG.normal(0.0, 1.0, scene.shape)
    loud = scene + RNG.normal(0.0, 3.0, scene.shape)
    matched = scene + RNG.normal(0.0, 1.0, scene.shape)
    assert psd_shape_ratio(quiet, quiet * 4.0) == pytest.approx(1.0, abs=1e-9)
    assert psd_shape_ratio(quiet, matched) < Tier4Targets().psd_shape_ratio
    ratio = psd_shape_ratio(quiet, loud)
    assert 1.7 < ratio < 2.0, f"3x noise mismatch read {ratio:.3f}"
    assert ratio > Tier4Targets().psd_shape_ratio


def test_a_fraction_of_peak_floor_removes_the_very_bins_the_statistic_needs() -> None:
    """Why `floor_fraction` defaults to zero, as a measurement rather than an assertion.

    On a scene with low-frequency structure the spectrum falls steeply, so "carries little power"
    and "is high-frequency" are the same bins -- and those are where per-pixel noise lives.
    """
    scene = _scene()
    quiet = scene + RNG.normal(0.0, 1.0, scene.shape)
    loud = scene + RNG.normal(0.0, 3.0, scene.shape)
    assert psd_shape_ratio(quiet, loud, floor_fraction=0.0) > 1.7
    assert psd_shape_ratio(quiet, loud, floor_fraction=0.01) < 1.2


def test_esf_width_is_the_ten_to_ninety_rise() -> None:
    ramp = np.concatenate([np.zeros(10), np.linspace(0.0, 1.0, 11), np.ones(10)])
    assert esf_width_px(ramp) == pytest.approx(8.0, abs=0.2)
    assert esf_width_px(ramp[::-1]) == pytest.approx(8.0, abs=0.2)
    step = np.concatenate([np.zeros(10), np.ones(10)])
    assert esf_width_px(step) < 1.5
    with pytest.raises(ValueError, match="flat"):
        esf_width_px(np.ones(20))


def test_auc_handles_ties_and_the_two_extremes() -> None:
    """A constant classifier must score exactly 0.5, not whatever the sort order happened to be --
    which is the only way a null control can be trusted."""
    labels = np.array([1, 1, 0, 0], dtype=bool)
    assert auc_from_scores([9.0, 8.0, 2.0, 1.0], labels) == pytest.approx(1.0)
    assert auc_from_scores([1.0, 2.0, 8.0, 9.0], labels) == pytest.approx(0.0)
    assert auc_from_scores([5.0, 5.0, 5.0, 5.0], labels) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="one sample of each"):
        auc_from_scores([1.0, 2.0], [True, True])


# --- the report ----------------------------------------------------------------------------------


def test_untestable_is_a_verdict_and_does_not_hide() -> None:
    """The failure this prevents: §15's radiometric targets silently vanishing from a DN8 report
    and the report reading as though they had passed."""
    report = Tier4Report()
    report.add("measured thing", 1.0, 2.0, ok=True)
    report.add("no data for this", None, 2.0)
    assert report.passed and len(report.untestable) == 1
    assert "untestable" in report.as_markdown()
    report.add("broken thing", 5.0, 2.0, ok=False)
    assert not report.passed and [c.name for c in report.failed] == ["broken thing"]


def test_the_radiometric_bias_target_is_always_reported_as_open() -> None:
    frame = np.clip(_scene() + RNG.normal(0.0, 3.0, (64, 64)), 0, 255)
    report = compare_frames(frame, frame)
    names = [c.name for c in report.untestable]
    assert "apparent-temperature bias (K)" in names
    assert any("no radiometric measurement" in c.note for c in report.untestable)


def test_a_shifted_histogram_is_reported_and_fails() -> None:
    frame = np.clip(_scene() + RNG.normal(0.0, 3.0, (64, 64)), 0, 255)
    report = compare_frames(frame, np.clip(frame + 20.0, 0, 255))
    emd = next(c for c in report.checks if c.name.startswith("histogram EMD"))
    assert emd.verdict == "fail"
    assert emd.measured is not None and emd.measured == pytest.approx(20.0, rel=0.15)
    assert not report.passed


# --- the discriminator ---------------------------------------------------------------------------


def test_the_feature_vector_is_finite_and_named() -> None:
    features = patch_features(np.clip(_scene((32, 32)) + RNG.normal(0.0, 3.0, (32, 32)), 0, 255))
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    with pytest.raises(ValueError, match="at least 8x8"):
        patch_features(np.zeros((4, 4)))


def _patches(sigma: float, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    scene = _scene((128, 128))
    out = []
    for _ in range(n):
        y, x = rng.integers(0, 96, size=2)
        out.append(scene[y : y + 32, x : x + 32] + rng.normal(0.0, sigma, (32, 32)))
    return out


def test_one_distribution_split_in_two_is_indistinguishable() -> None:
    """The control. ⚠️ **ME.6's planned criterion, "AUC 0.5 ± 0.03", is not achievable and is
    replaced.** At this sample size the AUC's own null standard error is already 0.04, so a
    control that lands at 0.54 is one sigma from chance and a report calling that a failure would
    be measuring its own sample size. The criterion that scales with n is
    ``abs(auc - 0.5) < 2 * null_sigma``.
    """
    result = gap_score(_patches(3.0, 40, 1), _patches(3.0, 40, 2))
    assert result.indistinguishable, f"AUC {result.auc:.3f} = {result.z:+.2f} null sigma"
    assert result.auc < Tier4Targets().discriminator_auc
    assert 0.03 < result.null_sigma < 0.08


def test_a_three_times_noise_mismatch_is_separable_and_names_the_feature() -> None:
    """The other side of the control: a gap the probe *should* find, and it must say which
    feature found it -- an AUC with no handle on it is a number nobody can act on."""
    result = gap_score(_patches(1.0, 40, 3), _patches(3.0, 40, 4))
    assert result.separable, f"AUC {result.auc:.3f}"
    assert result.auc > 0.9
    heaviest = [name for name, _ in result.importance[:4]]
    assert {"noise_scale", "std", "gradient_median"} & set(heaviest), heaviest


def test_the_probe_is_cross_validated_not_fitted_on_what_it_scores() -> None:
    """A ten-parameter probe fitted and scored on the same eighty patches reports a gap that is
    not there. Every patch is scored by a model that never saw it."""
    result = gap_score(_patches(3.0, 40, 5), _patches(3.0, 40, 6), folds=4)
    assert result.folds == 4
    assert result.n_real == 40 and result.n_synthetic == 40
    with pytest.raises(ValueError, match="two patches of each"):
        gap_score(_patches(3.0, 1, 7), _patches(3.0, 40, 8))


# --- the script -----------------------------------------------------------------------------------


def test_the_acceptance_report_passes_its_own_control(tmp_path: pathlib.Path) -> None:
    module = _script()
    assert module.main(["--self-test", "--out", str(tmp_path)]) == 0
    payload = (tmp_path / "tier4-acceptance.json").read_text()
    assert '"passed": true' in payload
    assert "untestable" in (tmp_path / "tier4-acceptance.md").read_text()


def test_the_acceptance_report_exits_non_zero_when_a_target_is_exceeded(
    tmp_path: pathlib.Path,
) -> None:
    """The whole point of a report that a pipeline can run: it fails loudly."""
    module = _script()
    real, synthetic = module._self_test_frames()
    shifted = [np.clip(f.astype(np.int16) + 30, 0, 255).astype(np.uint8) for f in synthetic]
    report = module._run(real, shifted, patch=32, per_frame=4, seed=1, targets=Tier4Targets())
    assert not report.passed
    assert any(c.name.startswith("histogram EMD") for c in report.failed)
