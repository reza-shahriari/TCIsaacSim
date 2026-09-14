"""What the ISP did to the picture: the AGC's fingerprint, DDE's ringing, and replaced pixels.

roadmap ME.3b; docs/physics-model.md §11.3, §10.4; §15 Tier 4. ADR 0068.

§11.3 is explicit that the mapping from 14-16 bits down to 8 "is part of the sensor model, not a
display detail", so measuring it on real video is measuring the camera. The catch is that it is only
the *camera's* ISP if the frames came out of the camera's display path. Half the indexed sets are
Y16 streams that a recorder converted, and a histogram measured on those is a picture of the
recorder's conversion rule -- which for the Halmstad set is itself unverified (linear? min-max per
frame?). So :func:`agc_signature` and :func:`edge_overshoot` take the conversion as a **required
argument** and refuse anything but ``display``; the index already excludes both analysers on that
set, and this is the second lock on the same door.

**The AGC's fingerprint is flatness of the output distribution, and it is a property of the scene
as well as the ISP.** Plateau equalisation integrates a clipped histogram into a CDF, so its output
is close to uniform over its occupied range; a linear stretch reproduces the scene's own
distribution. ``cdf_deviation`` -- the largest gap between the output CDF and the straight line --
separates them by a factor of 20 or more on every sky-like frame measured (plateau 0.003-0.007,
linear 0.15-0.50 on flat sky, a gradient, cloud and a horizon). But a scene whose own histogram is
already uniform -- a strong linear ramp filling the frame -- reads 0.03 under a linear stretch,
because a linear stretch of a uniform scene *is* equalisation as far as any histogram statistic can
tell. That case is reported as ``indeterminate`` rather than guessed at.

**DDE's fingerprint is exact.** An unsharp mask with a 3x3 box and gain g overshoots a step by
exactly g/3 of the step height, one pixel wide on each side, because the box at the pixel adjacent
to a step reads 2/3 of the way across it. :func:`edge_overshoot` measures that ratio, which makes
the gain readable off a picture, and the test is a known answer rather than a regression.

**A replaced pixel is the mean of its four neighbours, so its Laplacian is exactly zero.** §10.4
asks for the replacement to be simulated because the smoothed footprint is what a detector sees;
the same fact makes the footprint findable. ``4x - (up + down + left + right)`` vanishes identically
at any pixel the camera filled in from its final neighbours, while a good pixel carries sqrt(20)
times the pixel noise. The signature is exact for an isolated defect and for the **last-filled**
pixel of a cluster; a cluster's rim was filled while its inside was still invalid, so it is not
exactly the mean of its *final* neighbours, but it is still far smoother than a good pixel. Measured
on a 128x128 array with 69 interior defects: every one of them found on float frames, 93 % after
rounding to 8-bit codes, and no false positives in either case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from irsim.validation.flat import robust_noise_scale

__all__ = [
    "AgcSignature",
    "AgcVerdict",
    "EdgeOvershoot",
    "RecorderConversion",
    "ReplacedPixels",
    "agc_signature",
    "edge_overshoot",
    "replaced_pixel_map",
    "require_display_output",
]

#: How the 8-bit frames of a set came to be. ``display`` is the camera's own output; the others are
#: a recorder's conversion of a Y16 stream, and ``unknown`` is the honest default for a set whose
#: publisher did not say.
RecorderConversion = Literal["display", "linear", "min_max", "unknown"]

#: What the output histogram's shape supports saying about the AGC.
AgcVerdict = Literal["equalised", "stretched", "indeterminate"]

#: ``cdf_deviation`` below this is as flat as equalisation makes a frame; above the second it is
#: the scene's own distribution showing through a monotone stretch. Between them nothing is claimed.
EQUALISED_BELOW = 0.02
STRETCHED_ABOVE = 0.05


def require_display_output(conversion: RecorderConversion) -> None:
    """Raise unless the frames are the camera's own display output.

    An AGC signature or an edge overshoot measured on a recorder-converted Y16 stream is a
    measurement of the recorder. Refusing is the point: the alternative is a number in a table that
    nobody can later attribute to a camera.
    """
    if conversion == "display":
        return
    if conversion in ("linear", "min_max", "unknown"):
        raise ValueError(
            f"these frames are {conversion!r}, not the camera's display output: the 16-to-8-bit "
            "conversion was the recorder's, so a histogram or overshoot measured here describes "
            "the recorder and not the ISP. Run this only on display-output sets."
        )
    raise ValueError(f"unknown recorder conversion {conversion!r}")


def _as_frame(frame: object) -> NDArray[np.float64]:
    x = np.asarray(frame)
    if x.dtype == np.float16:
        raise TypeError("frame is float16; promote to float32 or better before analysis")
    if x.ndim != 2:
        raise ValueError(f"expected one frame (V, H), got shape {x.shape}")
    return x.astype(np.float64)


@dataclass(frozen=True)
class AgcSignature:
    """Shape of an 8-bit frame's own histogram, and what it supports saying about the AGC."""

    cdf_deviation: float
    entropy_bits: float
    occupied_fraction: float
    span_codes: int
    verdict: AgcVerdict

    @property
    def equalised(self) -> bool:
        return self.verdict == "equalised"


def agc_signature(
    frame: object, *, conversion: RecorderConversion, bit_depth: int = 8
) -> AgcSignature:
    """Read the AGC's fingerprint off one display frame (§11.3).

    ``cdf_deviation`` is the Kolmogorov distance between the frame's cumulative histogram, over the
    codes it occupies, and the uniform line. ``occupied_fraction`` corroborates it: equalising a
    discrete histogram stretches sparse regions and leaves output codes unused, while a linear
    stretch of a wide input range fills every one.
    """
    require_display_output(conversion)
    if not 4 <= bit_depth <= 16:
        raise ValueError(f"bit_depth must be in 4..16, got {bit_depth}")
    x = _as_frame(frame)
    codes = np.rint(x).astype(np.int64)
    if codes.min() < 0 or codes.max() >= 2**bit_depth:
        raise ValueError(
            f"codes run {codes.min()}..{codes.max()}, outside a {bit_depth}-bit display frame"
        )
    counts = np.bincount(codes.ravel(), minlength=2**bit_depth).astype(np.float64)
    occupied = np.flatnonzero(counts)
    if occupied.size < 2:
        raise ValueError("the frame occupies one code; there is no histogram shape to read")
    span = counts[occupied[0] : occupied[-1] + 1]
    p = span / span.sum()
    cdf = np.cumsum(p)
    uniform = (np.arange(span.size, dtype=np.float64) + 1.0) / span.size
    deviation = float(np.abs(cdf - uniform).max())
    positive = p[p > 0.0]
    verdict: AgcVerdict = (
        "equalised"
        if deviation < EQUALISED_BELOW
        else "stretched"
        if deviation > STRETCHED_ABOVE
        else "indeterminate"
    )
    return AgcSignature(
        cdf_deviation=deviation,
        entropy_bits=float(-np.sum(positive * np.log2(positive))),
        occupied_fraction=float(np.count_nonzero(span) / span.size),
        span_codes=int(span.size),
        verdict=verdict,
    )


@dataclass(frozen=True)
class EdgeOvershoot:
    """Ringing either side of the clean steps in a frame, as a fraction of the step height."""

    n_edges: int
    overshoot: float
    undershoot: float

    @property
    def implied_box3_gain(self) -> float:
        """The unsharp gain that would produce this overshoot with a 3x3 box: 3x the ratio."""
        return 3.0 * self.overshoot


def _profile_edges(
    profile: NDArray[np.float64], threshold: float, ring: int, plateau: int, flat_ratio: float
) -> list[tuple[float, float]]:
    """(overshoot, undershoot) for every isolated step in one 1-D profile."""
    out: list[tuple[float, float]] = []
    steps = np.diff(profile)
    magnitude = np.abs(steps)
    span = ring + plateau
    for j in np.flatnonzero(magnitude >= threshold):
        if j < span or j + span >= profile.size - 1:
            continue
        # The ringing either side of a step is itself a difference above the threshold, and
        # measuring from one of those puts the real step inside the ring window. Only the dominant
        # difference of its own neighbourhood is an edge.
        if magnitude[j] < magnitude[max(j - ring, 0) : j + ring + 1].max():
            continue
        left_ring = profile[j - ring + 1 : j + 1]
        right_ring = profile[j + 1 : j + ring + 1]
        left_flat = profile[j - span + 1 : j - ring + 1]
        right_flat = profile[j + ring + 1 : j + span + 1]
        step = float(right_flat.mean() - left_flat.mean())
        if abs(step) < threshold:
            continue
        if max(float(left_flat.std()), float(right_flat.std())) > flat_ratio * abs(step):
            continue  # not an isolated step: there is structure in the reference plateaus
        if step > 0.0:
            over = (float(right_ring.max()) - float(right_flat.mean())) / step
            under = (float(left_flat.mean()) - float(left_ring.min())) / step
        else:
            over = (float(left_flat.mean()) - float(left_ring.max())) / step
            under = (float(right_ring.min()) - float(right_flat.mean())) / step
        out.append((over, under))
    return out


def edge_overshoot(
    frame: object,
    *,
    conversion: RecorderConversion,
    min_step_sigma: float = 8.0,
    ring: int = 2,
    plateau: int = 3,
    flat_ratio: float = 0.25,
) -> EdgeOvershoot:
    """Median over/undershoot at the isolated steps of a display frame (§11.3 DDE).

    Only steps with flat reference plateaus on both sides are used -- an edge with structure beside
    it has no unambiguous asymptote to measure ringing against -- and the step must be at least
    ``min_step_sigma`` times the frame's own pixel-to-pixel noise (a hundredth of the frame's range
    when it has none, which only a synthetic frame does), so noise is never mistaken for an edge.
    Rows and columns are both scanned, because DDE is separable and a single axis would miss a
    horizon.

    With the 3x3 box of :func:`irsim.isp.dde.dde`, ``overshoot`` is exactly ``gain/3``; reading a
    gain back off a picture is what :attr:`EdgeOvershoot.implied_box3_gain` is for. A different
    kernel gives a different constant, so the gain is only implied, never asserted.
    """
    require_display_output(conversion)
    if ring < 1 or plateau < 2:
        raise ValueError(f"need ring >= 1 and plateau >= 2, got {ring} and {plateau}")
    x = _as_frame(frame)
    noise = robust_noise_scale(x)
    span = float(np.ptp(x))
    if not span > 0.0:
        raise ValueError("the frame is constant; there are no edges in it")
    # A synthetic noiseless frame has no noise scale to set a threshold from, so fall back to a
    # hundredth of its range -- low enough to keep every real step, high enough to stay above the
    # rounding of an 8-bit plane.
    threshold = float(min_step_sigma) * noise if noise > 0.0 else 0.01 * span

    found: list[tuple[float, float]] = []
    for row in x:
        found.extend(_profile_edges(row, threshold, ring, plateau, flat_ratio))
    for column in x.T:
        found.extend(_profile_edges(column, threshold, ring, plateau, flat_ratio))
    if not found:
        return EdgeOvershoot(n_edges=0, overshoot=float("nan"), undershoot=float("nan"))
    values = np.asarray(found, dtype=np.float64)
    return EdgeOvershoot(
        n_edges=int(values.shape[0]),
        overshoot=float(np.median(values[:, 0])),
        undershoot=float(np.median(values[:, 1])),
    )


@dataclass(frozen=True)
class ReplacedPixels:
    """Pixels whose value is (nearly) the mean of their neighbours in every frame."""

    score: NDArray[np.float64]  # median |Laplacian| over t, in units of the frame's own median
    mask: NDArray[np.bool_]
    threshold: float

    @property
    def fraction(self) -> float:
        return float(np.count_nonzero(self.mask) / self.mask.size)

    def count(self) -> int:
        return int(np.count_nonzero(self.mask))


def replaced_pixel_map(cube: object, *, threshold: float = 0.25, border: int = 1) -> ReplacedPixels:
    """Estimate which pixels the camera replaced, from their vanishing Laplacian (§10.4).

    The score is each pixel's median ``|4x - (up + down + left + right)|`` over the clip, divided by
    the median of that quantity over the array, so it is scale-free and a smooth *scene* does not
    produce detections -- a uniformly smooth region lowers the numerator and the denominator alike.
    The border is excluded (a border pixel has no four neighbours); run it on the flat windows
    :func:`~irsim.validation.flat.find_flat_regions` returns, since scene structure raises the
    denominator and hides shallow defects.

    **A lossy codec destroys this signature outright, and the estimator goes silent rather than
    wrong.** Measured through x264 at CRF 12 and 18 on a clip whose footprint is recovered perfectly
    before coding: recall falls from 1.0 to 0.0 with the false-positive rate staying at zero. The
    block transform moves a replaced pixel off the exact mean of its neighbours while flattening
    everyone else's Laplacian toward it, so nothing stands out from the median any more. Read a
    set's :func:`~irsim.validation.codec.codec_floor` beside an empty result: on a lossy set, empty
    means "not measurable here", not "no bad pixels".
    """
    x = np.asarray(cube)
    if x.dtype == np.float16:
        raise TypeError("cube is float16; promote to float32 or better before analysis")
    if x.ndim != 3:
        raise ValueError(f"cube must be (T, V, H), got shape {x.shape}")
    if min(x.shape[1:]) < 3:
        raise ValueError("a Laplacian needs at least three rows and three columns")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold is a fraction of the typical |Laplacian|, got {threshold}")
    frames = x.astype(np.float64)

    laplacian = np.zeros_like(frames)
    interior = frames[:, 1:-1, 1:-1]
    laplacian[:, 1:-1, 1:-1] = (
        4.0 * interior
        - frames[:, :-2, 1:-1]
        - frames[:, 2:, 1:-1]
        - frames[:, 1:-1, :-2]
        - frames[:, 1:-1, 2:]
    )
    median = np.median(np.abs(laplacian), axis=0)
    inside = np.zeros(median.shape, dtype=bool)
    edge = max(1, int(border))
    inside[edge:-edge, edge:-edge] = True
    reference = float(np.median(median[inside]))
    if not reference > 0.0:
        raise ValueError(
            "every pixel is already the mean of its neighbours; there is no reference level to "
            "call a replacement anomalous against"
        )
    score = np.where(inside, median / reference, np.inf)
    return ReplacedPixels(
        score=np.asarray(score, dtype=np.float64),
        mask=np.asarray(score < threshold),
        threshold=float(threshold),
    )
