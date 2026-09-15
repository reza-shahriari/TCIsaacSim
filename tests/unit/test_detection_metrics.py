"""ME.7: the Tier 5 detection metrics, pinned on input where the answer is known by hand.

A detector is somebody's model; average precision at IoU 0.5 is arithmetic, and the arithmetic is
what decides what a sim-to-real number means. These tests need no GPU, no torch and nobody's labels.

The metric that carries the weight for this project is **P_d against SCR**, not AP: a drone at 2 km
subtends two pixels at an SCR of 2 and is a rounding error in an AP dominated by close, large
targets -- so two sets can agree on AP and disagree completely about the regime the simulator
exists to model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from irsim_eval.detection import (
    PROTOCOLS,
    Detection,
    average_precision,
    detection_rate_by_bin,
    false_alarms_per_frame,
    iou,
    match_detections,
)


@dataclass(frozen=True)
class Truth:
    x: float
    y: float
    width: float
    height: float
    frame: int = 0


def test_iou_on_boxes_whose_overlap_is_known_by_hand() -> None:
    a = Detection(0.0, 0.0, 10.0, 10.0, 1.0)
    assert iou([a], [Truth(0.0, 0.0, 10.0, 10.0)])[0, 0] == pytest.approx(1.0)
    assert iou([a], [Truth(20.0, 0.0, 10.0, 10.0)])[0, 0] == pytest.approx(0.0)
    # Half-overlapping: intersection 50, union 150.
    assert iou([a], [Truth(5.0, 0.0, 10.0, 10.0)])[0, 0] == pytest.approx(50.0 / 150.0)
    assert iou([], [Truth(0.0, 0.0, 1.0, 1.0)]).shape == (0, 1)


def test_a_truth_is_claimed_once_and_by_the_highest_score() -> None:
    """The rule that stops a detector inflating recall by emitting ten boxes on one target."""
    truth = [Truth(0.0, 0.0, 10.0, 10.0)]
    weak = Detection(0.0, 0.0, 10.0, 10.0, 0.4)
    strong = Detection(1.0, 1.0, 10.0, 10.0, 0.9)
    assignment, found = match_detections([weak, strong], truth)
    assert list(assignment) == [-1, 0], "the stronger box should take the target"
    assert list(found) == [True]


def test_average_precision_on_a_perfect_and_a_hopeless_detector() -> None:
    truths = [Truth(0.0, 0.0, 10.0, 10.0, frame=f) for f in range(4)]
    perfect = [Detection(0.0, 0.0, 10.0, 10.0, 0.9, frame=f) for f in range(4)]
    assert average_precision(perfect, truths) == pytest.approx(1.0)
    # Every box somewhere else: no true positive anywhere, so AP is zero, not undefined.
    misses = [Detection(500.0, 500.0, 10.0, 10.0, 0.9, frame=f) for f in range(4)]
    assert average_precision(misses, truths) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="no ground truth"):
        average_precision(perfect, [])
    assert average_precision([], truths) == 0.0


def test_average_precision_is_the_all_point_rule_not_the_eleven_point_one() -> None:
    """Two hits and two false alarms, interleaved, with the answer worked out by hand.

    Scores 0.9 (hit), 0.8 (miss), 0.7 (hit), 0.6 (miss) against two targets gives the precision
    sequence 1, 1/2, 2/3, 1/2 at recalls 1/2, 1/2, 1, 1. Under the monotone envelope the
    all-point integral is 0.5·1 + 0.5·(2/3) = 0.8333; the 11-point rule would report 0.8182 on the
    same data, which is why the rule is pinned rather than left to a library's default.
    """
    truths = [Truth(0.0, 0.0, 10.0, 10.0, frame=0), Truth(100.0, 0.0, 10.0, 10.0, frame=1)]
    detections = [
        Detection(0.0, 0.0, 10.0, 10.0, 0.9, frame=0),
        Detection(400.0, 0.0, 10.0, 10.0, 0.8, frame=0),
        Detection(100.0, 0.0, 10.0, 10.0, 0.7, frame=1),
        Detection(400.0, 0.0, 10.0, 10.0, 0.6, frame=1),
    ]
    assert average_precision(detections, truths) == pytest.approx(0.5 + 0.5 * (2.0 / 3.0), abs=1e-9)


def test_a_detection_cannot_claim_a_target_in_another_frame() -> None:
    truths = [Truth(0.0, 0.0, 10.0, 10.0, frame=0)]
    wrong_frame = [Detection(0.0, 0.0, 10.0, 10.0, 0.9, frame=1)]
    assert average_precision(wrong_frame, truths) == pytest.approx(0.0)


def test_an_empty_bin_reports_nan_and_never_zero() -> None:
    """ "No target was this small" and "every target this small was missed" are opposite
    conclusions, and a zero would let a reader draw the second from the first."""
    truths = [Truth(0.0, 0.0, 10.0, 10.0) for _ in range(4)]
    found = [True, True, False, True]
    scr = [12.0, 11.0, 3.0, 2.5]
    binned = detection_rate_by_bin(truths, found, scr, edges=[0.0, 5.0, 8.0, 15.0])
    assert list(binned.counts) == [2, 0, 2]
    rate = binned.rate
    assert rate[0] == pytest.approx(0.5)
    assert np.isnan(rate[1]), "the empty bin must be NaN"
    assert rate[2] == pytest.approx(1.0)


def test_detection_rate_falls_with_scr_which_is_the_metric_that_matters() -> None:
    """The breakdown a single AP hides: a small-SCR regime can collapse while AP barely moves."""
    truths = [Truth(0.0, 0.0, 4.0, 4.0) for _ in range(20)]
    scr = np.linspace(1.0, 20.0, 20)
    found = scr > 8.0  # a detector that simply cannot see anything fainter
    binned = detection_rate_by_bin(truths, found, scr, edges=[0.0, 5.0, 10.0, 20.0])
    rate = binned.rate
    assert rate[0] == pytest.approx(0.0)
    assert 0.0 < rate[1] < 1.0
    assert rate[2] == pytest.approx(1.0)
    assert np.all(np.diff(rate) > 0.0), rate


def test_false_alarms_are_counted_per_frame_and_respect_a_score_threshold() -> None:
    truths = [Truth(0.0, 0.0, 10.0, 10.0, frame=0)]
    detections = [
        Detection(0.0, 0.0, 10.0, 10.0, 0.9, frame=0),  # a hit
        Detection(300.0, 0.0, 10.0, 10.0, 0.8, frame=0),
        Detection(300.0, 0.0, 10.0, 10.0, 0.2, frame=1),
    ]
    assert false_alarms_per_frame(detections, truths, n_frames=2) == pytest.approx(1.0)
    assert false_alarms_per_frame(
        detections, truths, n_frames=2, score_threshold=0.5
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="n_frames"):
        false_alarms_per_frame(detections, truths, n_frames=0)


def test_the_four_protocols_are_named_and_each_says_what_it_answers() -> None:
    """§15 asks for four, and they answer different questions -- `mixed_to_real` in particular is
    not implied by the others: a simulator can have a large gap and still be worth training on."""
    assert set(PROTOCOLS) == {
        "real_to_real",
        "synthetic_to_real",
        "real_to_synthetic",
        "mixed_to_real",
    }
    assert all(len(question) > 40 for question in PROTOCOLS.values())
