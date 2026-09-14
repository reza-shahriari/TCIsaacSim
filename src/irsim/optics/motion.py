"""Image-plane motion synthesised from rigid-body transforms (§13.3, §9.2; roadmap M10.1b).

The renderer on this build does not transport motion. ADR 0014's addendum measured it:
``motion_vectors`` sits at a ~6e-5 floor after a 180 px displacement and ``Motion2d`` returns
nothing, so the ``motion_px`` plane of the G-buffer had to be left empty. That plane is what a
motion MTF and an in-sim bolometer smear both read, so nothing downstream of it could be built.

It can be computed instead, and exactly, for the case that matters: **rigid** objects. If a surface
point is at ``p`` in world space now and its object moved by a rigid transform since the previous
frame, then its previous world position is ``p`` carried back through that object's own transform
pair. Project both -- through the previous and the current camera pose respectively -- and the
difference is the image-plane displacement in pixels. No renderer involved, no approximation
beyond rigidity, and it costs two matrix multiplies and two projections per pixel.

Three cases fall out of the same arithmetic, and all three are worth stating because each is a
test:

* A **static scene** gives exactly zero, because both projections are of the same point through
  the same pose.
* An **object and camera moving together** gives exactly zero as well -- a tracked target is
  stationary on the focal plane however fast it crosses the sky, which is the whole reason a
  tracking mount removes smear from the target and puts it on the background.
* The **background** is not a rigid object and has no transform. It is at infinity, so it does not
  translate: its apparent motion is the camera's *rotation* alone, and that is what it is given.
  Which is why the sky behind a tracked target smears while the target does not.

docs/physics-model.md §13.3, §9.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import DistortionSpec
from irsim.optics.projection import Intrinsics, project_usd

__all__ = ["BACKGROUND_OBJECT_ID", "transform_points", "image_plane_motion"]

#: Object index reserved for "hit nothing", matching the renderer's background instance id.
BACKGROUND_OBJECT_ID = 0


def transform_points(points: Any, matrix: Any) -> NDArray[np.float64]:
    """Apply a USD 4x4 to ``(..., 3)`` points. Row-vector convention: ``p' = p @ M``.

    USD matrices are row-vector — a point is a row and multiplies on the left — which is the
    opposite of the column convention most graphics maths is written in. Getting it backwards
    transposes every rotation and still produces plausible, wrong motion, so the convention lives
    in one function rather than at each call site.
    """
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (4, 4):
        raise ValueError(f"expected a 4x4 transform, got {m.shape}")
    return np.asarray(p @ m[..., :3, :3] + m[..., 3, :3])


def image_plane_motion(
    points_camera_usd: Any,
    object_id: Any,
    object_to_world_prev: Any,
    object_to_world_cur: Any,
    camera_to_world_prev: Any,
    camera_to_world_cur: Any,
    intrinsics: Intrinsics,
    distortion: DistortionSpec,
) -> NDArray[np.float32]:
    """``motion_px``: (H, W, 2) float32 px/frame, x right and **y down**, per the G-buffer contract.

    ``points_camera_usd`` is the per-pixel surface position in **USD camera space** (+Y up, −Z
    forward) at the current frame -- what ``Camera3dPositionSD`` delivers (ADR 0014 addendum).
    ``object_id`` indexes the two transform stacks, exactly as the instance id indexes the thermal
    bridge's temperature table; index :data:`BACKGROUND_OBJECT_ID` is the background and its entry
    is ignored.

    The displacement is *backwards-looking* -- where this surface point was one frame ago -- which
    is the convention the moving-edge fixture uses and the sign a smear kernel wants: a feature
    moving right has a positive x velocity.

    Points behind either camera come back as zero rather than NaN. A surface that was off-screen
    or behind the lens a frame ago has no defined displacement, and a NaN there would propagate
    through a convolution and take the whole frame with it.
    """
    p_cam = np.asarray(points_camera_usd, dtype=np.float64)
    if p_cam.ndim != 3 or p_cam.shape[2] != 3:
        raise ValueError(f"points must be (H, W, 3), got {p_cam.shape}")
    ids = np.asarray(object_id)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"object_id must be an integer plane, got {ids.dtype}")
    if ids.shape != p_cam.shape[:2]:
        raise ValueError(f"object_id {ids.shape} does not match points {p_cam.shape[:2]}")

    prev_stack = np.asarray(object_to_world_prev, dtype=np.float64)
    cur_stack = np.asarray(object_to_world_cur, dtype=np.float64)
    if prev_stack.shape != cur_stack.shape or prev_stack.ndim != 3:
        raise ValueError("the two transform stacks must be the same (N, 4, 4)")
    clipped = np.clip(ids, 0, prev_stack.shape[0] - 1)

    cam_cur = np.asarray(camera_to_world_cur, dtype=np.float64)
    cam_prev = np.asarray(camera_to_world_prev, dtype=np.float64)
    world_to_cam_prev = np.linalg.inv(cam_prev)

    # Geometry: carry each surface point to world, back into its object's frame, forward through
    # that object's previous pose, then into the previous camera. For the background there is no
    # object, so the point is treated as a direction at infinity and only the rotations apply.
    p_world = transform_points(p_cam, cam_cur)
    to_local = np.linalg.inv(cur_stack)[clipped]
    p_local = np.einsum("hwj,hwjk->hwk", p_world, to_local[..., :3, :3]) + to_local[..., 3, :3]
    p_world_prev = (
        np.einsum("hwj,hwjk->hwk", p_local, prev_stack[clipped][..., :3, :3])
        + prev_stack[clipped][..., 3, :3]
    )
    p_cam_prev = transform_points(p_world_prev, world_to_cam_prev)

    background = ids == BACKGROUND_OBJECT_ID
    if background.any():
        # A direction, not a point: rotate the current ray into world and back out through the
        # previous camera's rotation. Translation is dropped because a point at infinity does not
        # parallax, and `project_usd` is scale-invariant so a direction projects like a point.
        direction = transform_points(p_cam, _rotation_only(cam_cur))
        p_cam_prev = np.where(
            background[..., None],
            transform_points(direction, _rotation_only(world_to_cam_prev)),
            p_cam_prev,
        )

    u_cur, v_cur = project_usd(p_cam, intrinsics, distortion)
    u_prev, v_prev = project_usd(p_cam_prev, intrinsics, distortion)
    motion = np.stack([u_cur - u_prev, v_cur - v_prev], axis=-1)
    return np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _rotation_only(matrix: Any) -> NDArray[np.float64]:
    """The same 4x4 with its translation removed, for carrying directions rather than points."""
    m = np.array(matrix, dtype=np.float64, copy=True)
    m[..., 3, :3] = 0.0
    return m
