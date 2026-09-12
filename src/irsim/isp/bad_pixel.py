"""Bad-pixel replacement: the 4-neighbour mean, iterated for clusters (M9.5b).

docs/physics-model.md §10.4, §11.1 (``raw DN → bad-pixel replace`` is the first ISP stage).
ADR 0055 (the stencil).

§10.4's closing point is the one that matters here: cameras replace bad pixels by neighbour
interpolation, and *the replacement leaves a detectable smoothed footprint*. "Simulate the defect
**and** the replacement — the replacement artefact is what a detector actually sees." A simulator
that injects defects and leaves them, or that injects none at all, both get the same thing wrong:
the image a perception stack consumes has neither raw defects nor clean pixels in those places, it
has small patches of locally over-smooth data.

**The stencil is the mean of the valid 4-neighbours.** Three properties follow, and all three are
the reason for choosing it over the alternatives:

* For an **isolated** defect on a linear ramp it is *exact* — (left + right)/2 is the centre
  value, and so is (up + down)/2 — so replacement adds no radiometric bias on smooth scene
  content, which is most of a thermal image. Inside a cluster it is not: a pixel in a 2×2 sees
  only its outer two neighbours, never an opposing pair, so the mean is pulled toward the outside
  of the cluster. That bias is real and belongs in the model — it is part of why a cluster is
  visible where a single defect is not — and it is the reason the §10.4 cluster process of M9.5a
  matters rather than being cosmetic.
* On white noise it divides the variance by four. Copying one neighbour would leave σ² and an
  8-neighbour mean would give σ²/8; the σ²/4 signature is what an ISP-behaviour extractor (ME.3)
  can actually look for in real footage.
* That same variance suppression *is* the detectable footprint: the replaced pixel's local
  Laplacian is well below its neighbours', which is how a detector distinguishes the patch.

**Iteration, and why passes are synchronous.** A pixel in the middle of a 3×3 cluster has no valid
neighbour at all on the first pass. Replacement therefore iterates: each pass fills every bad pixel
that has at least one valid neighbour, and those become valid for the next pass. Within a pass the
stencil reads only values that were valid when the pass began, never values written during it, so
the result does not depend on scan order — otherwise a 2×2 cluster would fill differently depending
on whether the array was walked row-major or column-major, and goldens would be unreproducible.

**The mask is per frame.** Dead and hot pixels are in it always; a blinking or flickering pixel
only while it is in its bad state (M9.5a). A camera's own factory map is a different, static thing
— what it fails to contain is exactly why intermittent defects reach the image — and belongs with
the FFC controller in M9.7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["replace_bad_pixels", "MAX_PASSES"]

# A cluster spanning more than this many pixels from its edge to its centre cannot be filled by a
# 4-neighbour stencil in fewer passes; at CLUSTER_RADIUS_PX = 2 nothing near it should occur, so
# hitting the cap means something upstream is wrong and it is better to say so than to loop.
MAX_PASSES = 32


def replace_bad_pixels(
    frame: NDArray[np.number], mask: NDArray[np.bool_], max_passes: int = MAX_PASSES
) -> NDArray[np.number]:
    """Replace every pixel in ``mask`` by the mean of its valid 4-neighbours (§10.4, §11.1).

    ``frame`` may be the raw DN plane (unsigned integer) or a float signal plane; the dtype is
    preserved, with the mean computed in float64 and rounded once on the way back to an integer
    plane so that repeated replacement does not accumulate a bias. The input is never modified.

    Isolated defects and small clusters fill in one or two passes. If a cluster is still unfilled
    after ``max_passes``, or if the whole array is masked, that is an error rather than a
    best-effort fill: an unreplaced defect silently carrying a stuck value into the NUC is worse
    than a loud failure.
    """
    arr = np.asarray(frame)
    bad = np.asarray(mask, dtype=bool)
    if arr.shape != bad.shape:
        raise ValueError(f"frame shape {arr.shape} != mask shape {bad.shape}")
    if arr.ndim != 2:
        raise ValueError(f"replacement works on one 2-D plane, got {arr.ndim}-D")
    if not bad.any():
        return np.array(arr, copy=True)
    if bad.all():
        raise ValueError("every pixel is masked; there is nothing to interpolate from")

    work = np.asarray(arr, dtype=np.float64).copy()
    valid = ~bad

    for _ in range(int(max_passes)):
        remaining = ~valid
        if not remaining.any():
            break
        # Sum and count of valid 4-neighbours, from the state at the START of this pass.
        contribution = np.where(valid, work, 0.0)
        total = np.zeros_like(work)
        count = np.zeros(work.shape, dtype=np.int32)
        total[1:, :] += contribution[:-1, :]
        count[1:, :] += valid[:-1, :]
        total[:-1, :] += contribution[1:, :]
        count[:-1, :] += valid[1:, :]
        total[:, 1:] += contribution[:, :-1]
        count[:, 1:] += valid[:, :-1]
        total[:, :-1] += contribution[:, 1:]
        count[:, :-1] += valid[:, 1:]

        fillable = remaining & (count > 0)
        if not fillable.any():
            raise ValueError(
                "a masked region has no valid neighbour on any side and cannot be interpolated "
                "(a fully masked row, column or border block)"
            )
        work[fillable] = total[fillable] / count[fillable]
        valid = valid | fillable
    else:
        if not valid.all():
            raise ValueError(
                f"a defect cluster was still unfilled after {max_passes} passes; the bad-pixel "
                "map has a region far larger than the §10.4 cluster process should produce"
            )

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        rounded = np.clip(np.rint(work), info.min, info.max)
        out = rounded.astype(arr.dtype)
    else:
        out = work.astype(arr.dtype)
    # Untouched pixels must come back bit-identical, not round-tripped through float64.
    out[~bad] = arr[~bad]
    return np.asarray(out, dtype=arr.dtype)
