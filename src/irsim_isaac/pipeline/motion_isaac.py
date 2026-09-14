"""Reading per-prim transforms off a USD stage so ``motion_px`` can be synthesised (M10.1b).

The arithmetic lives in :mod:`irsim.optics.motion` and is engine-free; this is the half that has
to talk to USD. It keeps the same shape as the thermal bridge deliberately: the renderer transports
an exact integer **instance id** per pixel, and this module keeps a **stack of 4x4 transforms
indexed by that id**, exactly as :class:`~irsim_isaac.pipeline.aerial_bridge.AerialThermalBridge`
keeps a float32 temperature table. One transport mechanism, two payloads.

**Two frames of state, and the tracker owns the clock.** Motion is a difference, so a tracker that
had not yet seen two frames cannot report one; :meth:`MotionTracker.sample` returns whether the
result is usable and the first frame of any sequence reports zero rather than a guess. The prim
paths are fixed at construction because a prim that appears mid-sequence has no previous pose and
a prim that vanishes has no current one -- both are zero-motion cases and neither is an error.

ADR 0014 addendum (no motion AOV on this build); docs/physics-model.md §13.3, §9.2
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import DistortionSpec
from irsim.optics.motion import BACKGROUND_OBJECT_ID, image_plane_motion
from irsim.optics.projection import Intrinsics
from irsim_isaac.pipeline.material_ids import labels_to_paths

__all__ = ["MotionTracker"]


class MotionTracker:
    """Per-prim and camera transforms over two frames, and the ``motion_px`` plane they give.

    ``prim_paths`` are the movable prims. Anything not listed -- and anything the renderer reports
    an id for that is not on the stage -- is treated as **static**, which is the safe default
    because a static object contributes no motion and a wrong guess would contribute a plausible
    one.
    """

    def __init__(self, prim_paths: Sequence[str], camera_path: str) -> None:
        self.prim_paths = list(dict.fromkeys(prim_paths))
        self.camera_path = camera_path
        self._index = {path: i for i, path in enumerate(self.prim_paths)}
        self._prev: dict[str, NDArray[np.float64]] | None = None
        self._cur: dict[str, NDArray[np.float64]] | None = None
        self._prev_camera: NDArray[np.float64] | None = None
        self._cur_camera: NDArray[np.float64] | None = None

    @property
    def ready(self) -> bool:
        """True once two frames have been sampled, which is when a difference exists."""
        return self._prev is not None and self._prev_camera is not None

    def sample(self, stage: Any = None) -> bool:
        """Read the current transforms, pushing the previous ones back. Returns :attr:`ready`.

        Call once per rendered frame, *before* reading the plane, and exactly once: sampling twice
        between renders would throw away the pose the motion is measured against and silently
        report zero.
        """
        import omni.usd
        from pxr import Usd, UsdGeom

        if stage is None:
            stage = omni.usd.get_context().get_stage()

        def world_of(path: str) -> NDArray[np.float64]:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                return np.eye(4)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return np.asarray(matrix, dtype=np.float64)

        self._prev, self._prev_camera = self._cur, self._cur_camera
        self._cur = {path: world_of(path) for path in self.prim_paths}
        self._cur_camera = world_of(self.camera_path)
        return self.ready

    def transform_stacks(
        self, id_to_labels: Mapping[Any, Any] | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(previous, current) object-to-world stacks indexed by instance id, shape (N, 4, 4).

        Identity for the background and for any id whose prim is not tracked, which makes those
        pixels report zero motion -- see the class docstring for why that is the right default
        rather than an omission.
        """
        if self._prev is None or self._cur is None:
            raise RuntimeError("sample() twice before asking for transforms")
        paths = labels_to_paths(id_to_labels)
        size = (max(paths) if paths else 0) + 1
        prev = np.broadcast_to(np.eye(4), (size, 4, 4)).copy()
        cur = prev.copy()
        for ident, path in paths.items():
            if ident == BACKGROUND_OBJECT_ID or path not in self._index:
                continue
            prev[ident] = self._prev[path]
            cur[ident] = self._cur[path]
        return prev, cur

    def motion_px(
        self,
        positions_camera: Any,
        instance_id: Any,
        id_to_labels: Mapping[Any, Any] | None,
        intrinsics: Intrinsics,
        distortion: DistortionSpec,
    ) -> NDArray[np.float32]:
        """The G-buffer's ``motion_px`` for this frame; zeros until two frames have been sampled."""
        ids = np.asarray(instance_id)
        if not self.ready:
            return np.zeros((*ids.shape[:2], 2), dtype=np.float32)
        assert self._prev_camera is not None and self._cur_camera is not None
        prev, cur = self.transform_stacks(id_to_labels)
        return image_plane_motion(
            positions_camera,
            ids.astype(np.int64),
            prev,
            cur,
            self._prev_camera,
            self._cur_camera,
            intrinsics,
            distortion,
        )
