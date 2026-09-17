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

from collections.abc import Callable, Mapping, Sequence
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

    ``prim_paths`` are the movable prims, and they are the **leaves the renderer reports ids
    for**, not the assembly root that a driver's transform op is attached to. :meth:`sample` takes
    a per-frame ``paths`` override for the usual case, where those leaves are discovered from the
    frame's own label table.

    For a **rigid** assembly the distinction costs nothing and gains nothing: the displacement is
    built as ``inv(cur) @ prev``, and a constant local offset ``L`` cancels in the middle of that
    product -- ``inv(L @ root_cur) @ (L @ root_prev) == inv(root_cur) @ root_prev`` exactly -- so
    a child of a banking airframe reports the same pixels whichever matrices are used. (That is
    worth stating because the opposite is the intuitive answer, and it is wrong.)

    Where it matters is an **articulated** child, whose local transform differs between the two
    samples so that there is nothing left to cancel. This project has one: ``render_quad_flight``
    re-poses the rotor discs every frame from the throttle. Reading each leaf's own transform
    covers both cases; reading the root covers only the rigid one.

    Anything not listed -- and anything the renderer reports an id for that is not on the stage --
    is treated as **static**, which is the safe default because a static object contributes no
    motion and a wrong guess would contribute a plausible one.
    """

    def __init__(self, prim_paths: Sequence[str], camera_path: str) -> None:
        self.prim_paths = list(dict.fromkeys(prim_paths))
        self.camera_path = camera_path
        self._prev: dict[str, NDArray[np.float64]] | None = None
        self._cur: dict[str, NDArray[np.float64]] | None = None
        self._prev_camera: NDArray[np.float64] | None = None
        self._cur_camera: NDArray[np.float64] | None = None

    @property
    def ready(self) -> bool:
        """True once two frames have been sampled, which is when a difference exists."""
        return self._prev is not None and self._prev_camera is not None

    @staticmethod
    def _usd_reader(stage: Any = None) -> Callable[[str], NDArray[np.float64]]:
        """``path -> world-from-local 4x4``, read off a USD stage.

        The one part of this class that needs an engine, kept to four lines and behind a factory
        so :meth:`sample` can be driven without one. A prim that is absent or invalid reads as the
        identity, which is the same zero-motion default the class docstring describes for an
        untracked id -- a missing prim is not an error, it is a prim that is not moving.
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

        return world_of

    def sample(
        self,
        stage: Any = None,
        *,
        read: Callable[[str], Any] | None = None,
        paths: Sequence[str] | None = None,
    ) -> bool:
        """Read the current transforms, pushing the previous ones back. Returns :attr:`ready`.

        Call once per rendered frame, *before* reading the plane, and exactly once: sampling twice
        between renders would throw away the pose the motion is measured against and silently
        report zero.

        ``read`` replaces the USD lookup with any ``path -> 4x4``. It is what lets the wiring be
        tested on a CPU: the arithmetic in :mod:`irsim.optics.motion` was always engine-free, but
        until IG.6 the *only* way to reach it was through a running renderer, which is why it sat
        unwired behind a green in-sim test for two milestones.

        ``paths`` overrides the prim set for this frame, which is how the caller hands over the
        leaves it found in the frame's own label table rather than the assembly roots a driver
        knows about.
        """
        reader = self._usd_reader(stage) if read is None else read
        wanted = self.prim_paths if paths is None else list(dict.fromkeys(paths))
        self._prev, self._prev_camera = self._cur, self._cur_camera
        self._cur = {path: np.asarray(reader(path), dtype=np.float64) for path in wanted}
        self._cur_camera = np.asarray(reader(self.camera_path), dtype=np.float64)
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
            if ident == BACKGROUND_OBJECT_ID:
                continue
            before, after = self._prev.get(path), self._cur.get(path)
            # A prim tracked in only one of the two frames has no difference to report. It appeared
            # or it vanished, and both are zero-motion rather than errors -- see the class
            # docstring. Leaving the identity in place is what says so.
            if before is None or after is None:
                continue
            prev[ident] = before
            cur[ident] = after
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
