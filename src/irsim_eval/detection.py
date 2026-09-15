"""Detection metrics and the four §15 Tier 5 transfer protocols (ME.7).

The metrics are **pure NumPy** and live here rather than inside a training script for one reason:
they are the part of Tier 5 that can be tested without a GPU, without torch, and without anybody's
labels, and they are the part that decides what the sim-to-real number *means*. A detector is
somebody's model; average precision at IoU 0.5 is arithmetic, and arithmetic should be pinned.

**What §15 asks for, and why P_d against SCR is the one that matters here.** A single AP over a
dataset hides exactly the regime this project exists to model: a drone at 2 km subtends two pixels
and sits at an SCR of 2, and it is a rounding error in an AP dominated by close, large targets.
:func:`detection_rate_by_bin` is the breakdown -- P_d against signal-to-clutter ratio and against
apparent size -- and a sim-to-real comparison that agrees on AP while disagreeing on the small-SCR
bins has not agreed on anything useful.

The four protocols of :data:`PROTOCOLS` are §15's, and they answer different questions: the
*synthetic to real* gap is what a simulator is for, and *real + synthetic to real* is whether it
helps, which is not implied by the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Protocol",
    "PROTOCOLS",
    "Detection",
    "iou",
    "match_detections",
    "average_precision",
    "detection_rate_by_bin",
    "false_alarms_per_frame",
]

Protocol = Literal["real_to_real", "synthetic_to_real", "real_to_synthetic", "mixed_to_real"]

#: §15's four transfer protocols and the question each one answers. Kept as data so a report can
#: print the question beside the number, which is the difference between a table and a result.
PROTOCOLS: dict[Protocol, str] = {
    "real_to_real": "the baseline: what this detector and this much real data are worth at all",
    "synthetic_to_real": "the sim-to-real gap proper -- train only on renders, test on reality",
    "real_to_synthetic": "the converse gap: does the simulator produce targets a real-trained "
    "detector recognises? A high score here with a low score above means the renders are *easier* "
    "than reality, not that they are like it",
    "mixed_to_real": "whether synthetic data *helps*, which the first three do not imply: a "
    "simulator can have a large gap and still be worth training on",
}


@dataclass(frozen=True)
class Detection:
    """One predicted box with a confidence. ``(x, y, w, h)`` in the project's pixel convention."""

    x: float
    y: float
    width: float
    height: float
    score: float
    frame: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("a detection must have positive width and height")

    @property
    def area(self) -> float:
        return float(self.width * self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def _xyxy(boxes: Sequence[Any]) -> NDArray[np.float64]:
    rows = []
    for b in boxes:
        if hasattr(b, "as_xyxy"):
            rows.append(b.as_xyxy())
        else:
            x, y, w, h = (float(v) for v in (b.x, b.y, b.width, b.height))
            rows.append((x, y, x + w, y + h))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def iou(a: Sequence[Any], b: Sequence[Any]) -> NDArray[np.float64]:
    """Pairwise intersection-over-union, ``(len(a), len(b))``."""
    box_a, box_b = _xyxy(a), _xyxy(b)
    if box_a.size == 0 or box_b.size == 0:
        return np.zeros((box_a.shape[0], box_b.shape[0]), dtype=np.float64)
    x1 = np.maximum(box_a[:, None, 0], box_b[None, :, 0])
    y1 = np.maximum(box_a[:, None, 1], box_b[None, :, 1])
    x2 = np.minimum(box_a[:, None, 2], box_b[None, :, 2])
    y2 = np.minimum(box_a[:, None, 3], box_b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.asarray(np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0))


def match_detections(
    detections: Sequence[Detection], truths: Sequence[Any], threshold: float = 0.5
) -> tuple[NDArray[np.int_], NDArray[np.bool_]]:
    """Greedy highest-score-first matching within one frame.

    Returns ``(matched_truth_index_or_-1_per_detection, truth_was_found)``. Highest score first,
    each truth claimed at most once: the standard rule, and the reason a detector cannot inflate
    its recall by emitting ten boxes on one target.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("the IoU threshold must lie in (0, 1]")
    assignment = np.full(len(detections), -1, dtype=np.int_)
    found = np.zeros(len(truths), dtype=np.bool_)
    if not detections or not truths:
        return assignment, found
    overlap = iou(detections, truths)
    order = np.argsort([-d.score for d in detections], kind="mergesort")
    for d in order:
        candidates = np.where(~found & (overlap[d] >= threshold))[0]
        if candidates.size:
            best = int(candidates[np.argmax(overlap[d][candidates])])
            assignment[d] = best
            found[best] = True
    return assignment, found


def average_precision(
    detections: Sequence[Detection],
    truths: Sequence[Any],
    *,
    threshold: float = 0.5,
    frame_of: Any = None,
) -> float:
    """VOC-style all-point average precision at one IoU threshold.

    All-point rather than the 11-point interpolation: the 11-point rule was an artefact of a 2007
    evaluation server and reports a visibly different number on a small dataset, which is the worst
    property a *comparison* metric can have.

    ``frame_of`` maps a truth to its frame index; by default a truth's ``frame`` attribute is used,
    falling back to 0. Matching is per frame -- a detection cannot claim a target in another image.
    """
    if not truths:
        raise ValueError("average precision is undefined with no ground truth")
    if not detections:
        return 0.0
    frame_key = frame_of or (lambda t: int(getattr(t, "frame", 0)))
    frames = sorted({d.frame for d in detections} | {frame_key(t) for t in truths})
    scores: list[float] = []
    is_tp: list[bool] = []
    for frame in frames:
        frame_dets = [d for d in detections if d.frame == frame]
        frame_truths = [t for t in truths if frame_key(t) == frame]
        assignment, _ = match_detections(frame_dets, frame_truths, threshold)
        for d, matched in zip(frame_dets, assignment, strict=True):
            scores.append(d.score)
            is_tp.append(bool(matched >= 0))
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    hits = np.asarray(is_tp, dtype=np.float64)[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / float(len(truths))
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Monotone envelope, then integrate over recall: the all-point rule.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    padded_recall = np.concatenate([[0.0], recall])
    return float(np.sum(np.diff(padded_recall) * precision))


@dataclass(frozen=True)
class BinnedRate:
    edges: NDArray[np.float64]
    counts: NDArray[np.int_]
    found: NDArray[np.int_]

    @property
    def rate(self) -> NDArray[np.float64]:
        """P_d per bin; NaN where no target fell in the bin, never 0."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.found / np.maximum(self.counts, 1), np.nan)


def detection_rate_by_bin(truths: Sequence[Any], found: Any, values: Any, edges: Any) -> BinnedRate:
    """P_d against any per-target quantity -- SCR, apparent size, range.

    An empty bin reports **NaN, never zero**: "no target was this small in the test set" and "every
    target this small was missed" are opposite conclusions, and a zero would let a reader draw the
    second from the first.
    """
    was_found = np.asarray(found, dtype=bool).ravel()
    quantity = np.asarray(values, dtype=np.float64).ravel()
    bounds = np.asarray(edges, dtype=np.float64).ravel()
    if was_found.size != len(truths) or quantity.size != len(truths):
        raise ValueError("found and values must have one entry per truth")
    if bounds.size < 2 or np.any(np.diff(bounds) <= 0.0):
        raise ValueError("edges must be increasing and hold at least two values")
    index = np.clip(np.digitize(quantity, bounds) - 1, 0, bounds.size - 2)
    inside = (quantity >= bounds[0]) & (quantity <= bounds[-1])
    counts = np.zeros(bounds.size - 1, dtype=np.int_)
    hits = np.zeros(bounds.size - 1, dtype=np.int_)
    np.add.at(counts, index[inside], 1)
    np.add.at(hits, index[inside & was_found], 1)
    return BinnedRate(edges=bounds, counts=counts, found=hits)


def false_alarms_per_frame(
    detections: Sequence[Detection],
    truths: Sequence[Any],
    *,
    n_frames: int,
    threshold: float = 0.5,
    score_threshold: float = 0.0,
) -> float:
    """Unmatched detections per frame above a score -- the number an operator actually feels.

    Reported per *frame* rather than per image-area or per second, because that is the unit a
    watch-stander experiences and the one that decides whether a system is usable at 60 Hz.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")

    def frame_key(truth: Any) -> int:
        return int(getattr(truth, "frame", 0))

    kept = [d for d in detections if d.score >= score_threshold]
    false_alarms = 0
    for frame in {d.frame for d in kept}:
        frame_dets = [d for d in kept if d.frame == frame]
        assignment, _ = match_detections(
            frame_dets, [t for t in truths if frame_key(t) == frame], threshold
        )
        false_alarms += int(np.count_nonzero(assignment < 0))
    return false_alarms / float(n_frames)
