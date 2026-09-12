"""``IrCamera``: a USD camera, the M10.1 G-buffer, and the whole sensor chain, one frame at a time.

docs/physics-model.md §13.6 (the Warp path), §13.4 (outputs), §8.3 (supersampling), §8.4 (the lens);
roadmap M10.9a-ii; ADR 0014 (ids and geometry are the transport), ADR 0015 + addendum (the engine
applies the lens), ADR 0018 (``irsim.pipeline`` is the oracle), ADR 0032 / CLAUDE.md #6 (one
weather object).

This is the object that turns a stage into an infrared frame. Everything it needs already exists
and is tested on its own; what lives here is the wiring, and the wiring is where the silent
mistakes are, so each one is named:

* **The render product is created at ``supersample × native``** (§8.3). Rendering at native
  resolution and blurring afterwards gives the blur without the aliasing, and aliasing is exactly
  what corrupts small-target detection at range -- which is phase 1's whole subject. ``run_frame``
  refuses a G-buffer that is not on the k× grid, so this cannot drift out of step silently.
* **The camera's optics come from the sensor YAML**, not from whatever the stage author left on the
  prim. ``horizontalAperture = width · pitch`` and ``focalLength = f`` put the focal length in
  pixels at ``f / pitch``, which is what :class:`~irsim.optics.projection.Intrinsics` assumes; a
  stage whose camera was authored for a 35 mm still frame would otherwise image a different
  scene than the radiometry is computed for.
* **Distortion is written as a USD applied API schema** (ADR 0015: the engine owns the projection).
  Every attribute of that schema is written explicitly, including the ones that look like they
  could be left alone: measured on 6.1.0-rc.26, the schemas default to ``fx = 900``,
  ``cx = 1024``, ``imageSize = (2048, 1024)`` and -- on the fisheye -- a **non-zero** ``k1``, so a
  half-written schema is a lens for a different camera rather than the identity.
* **The scene supplies the atmosphere and the sky, and nothing else may.** The roadmap sketched
  this constructor as ``(sensor_yaml, atmosphere_preset, environment, scene)``; taking the preset
  separately would let a caller run one atmosphere in the transmittance and another in the sky
  while the thermal solvers ran a third. :class:`~irsim.scene.Scene` already holds all three bound
  to its single ``WeatherSeries``, so they are read off it (CLAUDE.md #6).
* **Temperature never crosses a colour AOV.** It arrives through the M10.18 bridge's float32
  table, keyed by the exact integer instance id, with the background taking ``T_sky(θ)`` from each
  pixel's own ray elevation (ADR 0014, ADR 0060).
* **The position AOV is in camera space, so the camera's rotation has to be undone.**
  ``Camera3dPositionSD`` was recorded as world-space in ADR 0014, which was true of every scene
  that measured it -- all of them had an unrotated camera at the origin, where the two frames are
  the same thing. Tilt the camera up and they separate: measured on 6.1.0-rc.26, a camera rotated
  8 degrees about X still reports the frame centre's ray as ``(0, 0, -1)``. Taken as world that
  ray has zero elevation, which puts the horizon through the middle of the picture, paints the
  upper half of the sky with the ground temperature, and tilts every ``normal_dot_view`` and
  sky-view factor with it -- a smooth, plausible, completely wrong frame. So the default here is
  ``position_frame="camera"`` with the prim's own local-to-world rotation (ADR 0014 addendum).

**Which pipeline runs.** The frame goes through ``irsim.pipeline.run_frame`` -- the CPU reference
(ADR 0018) -- including the M9 sensor chain when one is attached. The Warp twins (M10.4--M10.8)
cover stages 1--6 but **not** the chain's post-ADC half: device-side defects, the iterated
bad-pixel replacement and the FFC hold are M10.7b. Running the device stages here today would
produce a frame from a *different camera* than the reference one and label it the same, which is
the divergence this project exists to avoid. :meth:`IrCamera.planes` therefore exposes the
assembled G-buffer so the Warp path can be driven from the same scene and compared, and the
all-device frame lands when M10.7b does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.config.sensor import DistortionSpec, SensorConfig, SensorSpec
from irsim.materials.mapping import Resolution
from irsim.optics.projection import FTHETA_UNVERIFIED, Intrinsics, opencv_pinhole_coeffs
from irsim.pipeline.core import PipelineConfig, PipelineState, Planes
from irsim.pipeline.frame import Outputs, run_frame
from irsim.scene import Scene
from irsim_isaac.pipeline.aerial_bridge import AerialThermalBridge, elevation_from_rays
from irsim_isaac.pipeline.gbuffer_isaac import (
    UP_AXIS_VECTOR,
    AovReader,
    PositionFrame,
    geometry_planes,
    ray_directions,
    to_gbuffer,
)
from irsim_isaac.pipeline.material_ids import (
    labels_from_payload,
    material_id_plane,
    overlay_unmapped,
    unmapped_mask,
)

__all__ = [
    "DISTORTION_SCHEMA",
    "DISTORTION_NAMESPACE",
    "CameraOptics",
    "camera_optics",
    "distortion_attributes",
    "author_camera",
    "IrCamera",
]

#: config ``optics.distortion.model`` -> the USD applied API schema that carries it. Measured on
#: 6.1.0-rc.26 (the schema registry lists five families; these are the two whose coefficient
#: conventions are verified -- ADR 0015 addendum). ``ftheta`` is absent on purpose.
DISTORTION_SCHEMA: dict[str, str] = {
    "brown_conrady": "OmniLensDistortionOpenCvPinholeAPI",
    "kannala_brandt": "OmniLensDistortionOpenCvFisheyeAPI",
}

#: The attribute namespace each schema writes into, and the token ``omni:lensdistortion:model``
#: takes when it is applied (measured: ``allowedTokens`` holds exactly this one value).
DISTORTION_NAMESPACE: dict[str, str] = {
    "brown_conrady": "opencvPinhole",
    "kannala_brandt": "opencvFisheye",
}

#: Fisheye coefficient names, in the order a ``kannala_brandt`` block lists them.
_FISHEYE_COEFFS: tuple[str, ...] = ("k1", "k2", "k3", "k4")


@dataclass(frozen=True)
class CameraOptics:
    """What the sensor config implies for the USD camera prim and its render product.

    USD expresses focal length and aperture in the same (arbitrary) unit, so only their ratio
    matters: ``f / (width · pitch) · width = f / pitch`` is the focal length in pixels. Writing
    both in millimetres keeps the numbers readable next to the YAML they came from.
    """

    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    intrinsics: Intrinsics
    supersample: int

    @property
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` of the render product -- the supersampled grid."""
        return self.intrinsics.resolution


def camera_optics(spec: SensorSpec, supersample: int | None = None) -> CameraOptics:
    """Camera-prim optics for ``spec`` at its configured supersample factor (§8.3)."""
    k = spec.optics.supersample_factor if supersample is None else int(supersample)
    pitch_mm = spec.fpa.pitch_um * 1e-3
    return CameraOptics(
        focal_length_mm=float(spec.optics.focal_length_mm),
        horizontal_aperture_mm=float(spec.fpa.width * pitch_mm),
        vertical_aperture_mm=float(spec.fpa.height * pitch_mm),
        intrinsics=Intrinsics.from_sensor(spec, k),
        supersample=k,
    )


def distortion_attributes(
    distortion: DistortionSpec, intrinsics: Intrinsics
) -> tuple[str, dict[str, Any]]:
    """``(schema name, {attribute: value})`` for the lens ``distortion`` describes.

    Engine-free on purpose: this is the whole of the config→USD mapping, so it can be checked
    without a renderer. **Every** attribute of the schema is emitted, never a subset -- the
    schemas' own defaults describe a 2048×1024 lens with ``fx = 900`` (and a non-zero ``k1`` on
    the fisheye), so an attribute left unwritten is not "no distortion", it is someone else's
    camera. The intrinsics written here are the same ones the projection oracle uses, so the
    schema and the prim's ``focalLength``/``horizontalAperture`` cannot disagree.
    """
    if distortion.model == "ftheta":
        raise NotImplementedError(FTHETA_UNVERIFIED)
    schema = DISTORTION_SCHEMA.get(distortion.model)
    namespace = DISTORTION_NAMESPACE.get(distortion.model)
    if schema is None or namespace is None:
        raise ValueError(f"no USD lens schema is mapped for distortion model {distortion.model!r}")

    prefix = f"omni:lensdistortion:{namespace}:"
    attrs: dict[str, Any] = {
        f"{prefix}fx": float(intrinsics.fx_px),
        f"{prefix}fy": float(intrinsics.fy_px),
        f"{prefix}cx": float(intrinsics.cx_px),
        f"{prefix}cy": float(intrinsics.cy_px),
        f"{prefix}imageSize": (int(intrinsics.width), int(intrinsics.height)),
    }
    if distortion.model == "brown_conrady":
        attrs.update({f"{prefix}{n}": v for n, v in opencv_pinhole_coeffs(distortion).items()})
    else:
        values = dict.fromkeys(_FISHEYE_COEFFS, 0.0)
        for name, value in zip(_FISHEYE_COEFFS, distortion.coeffs, strict=False):
            values[name] = float(value)
        attrs.update({f"{prefix}{n}": v for n, v in values.items()})
    return schema, attrs


def author_camera(
    stage: Any, path: str, optics: CameraOptics, distortion: DistortionSpec
) -> dict[str, Any]:
    """Write ``optics`` and ``distortion`` onto the camera prim at ``path``; return what was set.

    Creates the prim if it is not there, and leaves its transform alone -- where the camera *is*
    belongs to the scene, what it *sees through* belongs to the sensor config.
    """
    from pxr import Gf, Sdf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetFocalLengthAttr().Set(optics.focal_length_mm)
    cam.GetHorizontalApertureAttr().Set(optics.horizontal_aperture_mm)
    cam.GetVerticalApertureAttr().Set(optics.vertical_aperture_mm)
    cam.GetHorizontalApertureOffsetAttr().Set(0.0)
    cam.GetVerticalApertureOffsetAttr().Set(0.0)

    prim = cam.GetPrim()
    schema, attrs = distortion_attributes(distortion, optics.intrinsics)
    if not prim.ApplyAPI(schema):
        raise RuntimeError(f"could not apply {schema} to {path}: this build does not carry it")
    for name, value in attrs.items():
        # `imageSize` is declared GfVec2i. A Python tuple is coerced to GfVec2d and the set is
        # rejected outright, so the int2 value is constructed explicitly rather than implied.
        is_int2 = isinstance(value, tuple)
        typed = Gf.Vec2i(int(value[0]), int(value[1])) if is_int2 else value
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid():
            kind = Sdf.ValueTypeNames.Int2 if is_int2 else Sdf.ValueTypeNames.Float
            attr = prim.CreateAttribute(name, kind, custom=False)
        attr.Set(typed)
    return {
        "focalLength": optics.focal_length_mm,
        "horizontalAperture": optics.horizontal_aperture_mm,
        "verticalAperture": optics.vertical_aperture_mm,
        "schema": schema,
        "model": prim.GetAttribute("omni:lensdistortion:model").Get(),
        **attrs,
    }


@dataclass
class _Frame:
    """One frame's intermediates, kept so a caller can inspect what produced an image."""

    planes: Planes
    instance_id: NDArray[np.uint32]
    material_id: NDArray[np.int32]
    unmapped: NDArray[np.bool_]
    elevation_rad: NDArray[np.float64]
    labels: dict[int, str] = field(default_factory=dict)


class IrCamera:
    """A configured IR camera on a USD stage: ``get_outputs()`` renders one frame.

    ``scene`` carries the weather, the atmosphere, the sky model and the target solvers, all bound
    to one ``WeatherSeries``; ``prim_to_target`` says which prim is which thermal node. The camera
    advances its own clock by one frame period per capture, so the FFC schedule, the pattern drift
    and the thermal solvers all run on the same time base as a real 60 Hz core would.
    """

    def __init__(
        self,
        sensor: SensorConfig,
        scene: Scene,
        *,
        pipeline: PipelineConfig,
        prim_to_target: Mapping[str, str],
        resolutions: Sequence[Resolution],
        camera_path: str = "/World/IrCamera",
        stage: Any = None,
        position_frame: PositionFrame = "camera",
        up_axis: str | None = None,
        debug_unmapped: bool = True,
        strict_materials: bool = True,
        device: str = "cpu",
    ) -> None:
        band = sensor.sensor.band.band_id
        if pipeline.sensor is not sensor:
            raise ValueError("the PipelineConfig was built for a different SensorConfig")
        if pipeline.supersample != sensor.sensor.optics.supersample_factor:
            raise ValueError(
                f"pipeline supersample {pipeline.supersample} does not match the sensor's "
                f"{sensor.sensor.optics.supersample_factor}"
            )
        for name, obj in (("atmosphere", pipeline.atmosphere), ("sky", pipeline.sky)):
            weather = getattr(obj, "weather", None)
            if weather is not None and weather is not scene.weather:
                raise ValueError(
                    f"the pipeline's {name} holds a different WeatherSeries than the scene "
                    "(CLAUDE.md #6: one weather object, injected everywhere)"
                )

        self.sensor = sensor
        self.scene = scene
        self.config = pipeline
        self.optics = camera_optics(sensor.sensor)
        self.camera_path = camera_path
        self.position_frame = position_frame
        self.debug_unmapped = debug_unmapped
        self.strict_materials = strict_materials
        self.device = device
        self.resolutions = list(resolutions)
        self.state = PipelineState(t_s=scene.t0_s)
        self.bridge = AerialThermalBridge(scene, prim_to_target, band=band)
        self.frame_period_s = 1.0 / float(sensor.sensor.fpa.frame_rate_hz)
        self._t_rel_s = 0.0
        self._reader: AovReader | None = None
        self._render_product: Any = None
        self._camera_position: NDArray[np.float64] | None = None
        self._camera_to_world: NDArray[np.float64] | None = None
        self._last: _Frame | None = None
        self._stage = stage
        self._up_axis = up_axis
        self._authored: dict[str, Any] = {}

    # -- engine set-up --------------------------------------------------------------------

    def open(self, *, settle_frames: int = 8, rt_subframes: int = 1) -> IrCamera:
        """Author the camera, create the render product, attach the annotators, settle.

        Separate from ``__init__`` so that the object can be built -- and its config validated --
        on a machine with no renderer, which is what lets most of this module be unit tested.
        """
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Usd, UsdGeom

        from irsim_isaac.geometry_probe import configure_renderer

        stage = self._stage if self._stage is not None else omni.usd.get_context().get_stage()
        self._stage = stage
        if self._up_axis is None:
            axis = str(UsdGeom.GetStageUpAxis(stage))
            self._up_axis = axis if axis in UP_AXIS_VECTOR else "Y"

        self._authored = author_camera(
            stage, self.camera_path, self.optics, self.sensor.sensor.optics.distortion
        )
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_path))
        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self._camera_position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
        # USD matrices are row-vector (p_world = p_camera @ M), and `ray_directions` applies its
        # rotation as `vec @ rot.T`, so the transpose of the upper-left 3x3 is what it wants.
        self._camera_to_world = np.asarray(matrix, dtype=np.float64)[:3, :3].T

        configure_renderer()
        width, height = self.optics.resolution
        self._render_product = rep.create.render_product(self.camera_path, (width, height))
        path = getattr(self._render_product, "path", None) or str(self._render_product)
        self._reader = AovReader(
            path,
            device=self.device,
            required=("distance", "position", "normal", "instance"),
            expected_shape=(height, width),
        ).attach(settle_frames=settle_frames, rt_subframes=rt_subframes)
        return self

    def close(self) -> None:
        if self._reader is not None:
            self._reader.detach()
            self._reader = None

    def __enter__(self) -> IrCamera:
        return self if self._reader is not None else self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def authored(self) -> dict[str, Any]:
        """Exactly what :func:`author_camera` wrote -- for the M10.9b renderer audit."""
        return dict(self._authored)

    @property
    def t_rel_s(self) -> float:
        """Render time in seconds since the scene start."""
        return self._t_rel_s

    @property
    def last_frame(self) -> _Frame | None:
        return self._last

    # -- one frame ------------------------------------------------------------------------

    def planes(self, *, step: bool = True, rt_subframes: int = 1) -> Planes:
        """Render one frame and assemble the M0.6 G-buffer from it.

        Order matters: the renderer is stepped first, then the thermal clock is moved to the time
        that frame represents, so the temperatures written into the plane are the ones for the
        geometry that was just rendered rather than one frame stale.
        """
        if self._reader is None:
            raise RuntimeError("call open() first (it needs a running Kit application)")
        if step:
            self._reader.step(frames=1, rt_subframes=rt_subframes)
        aovs = self._reader.read()

        rays = ray_directions(
            aovs.position,
            frame=self.position_frame,
            camera_position=self._camera_position,
            camera_to_world=self._camera_to_world,
        )
        geometry = geometry_planes(
            aovs,
            up_axis=self._up_axis or "Y",
            position_frame=self.position_frame,
            camera_position=self._camera_position,
            camera_to_world=self._camera_to_world,
        )
        if aovs.instance_id is None:  # pragma: no cover - `required` already guarantees it
            raise RuntimeError("the instance-id channel produced no data this frame")
        instance_id = np.asarray(aovs.instance_id, dtype=np.uint32)
        labels = labels_from_payload(aovs.device_handles.get("instance"))
        material_id = material_id_plane(
            instance_id, labels, self.resolutions, strict=self.strict_materials
        )
        elevation = elevation_from_rays(rays, up=UP_AXIS_VECTOR[self._up_axis or "Y"])

        self.bridge.advance_to(self._t_rel_s)
        temperature = self.bridge.temperature_plane(
            instance_id,
            labels,
            sky_mask=geometry.sky_mask,
            elevation_rad=elevation,
            strict=self.strict_materials,
        )
        # `to_gbuffer` validates the M0.6 contract; the stages consume the plane dict.
        # An UNMAPPED prim has no emissivity, and `MaterialTable` refuses to invent one
        # (ADR 0047). In debug mode those pixels are handed to stage 1 as blackbody-equivalent --
        # the eps = 1 ADR 0047 specifies, applied to the prim's *own* temperature, which is
        # already in the plane -- and then painted magenta, so they are visibly not physics
        # rather than plausibly wrong. Without debug mode the frame raises instead.
        unmapped = np.asarray(unmapped_mask(material_id, geometry.sky_mask), dtype=np.bool_)
        if self.debug_unmapped and unmapped.any():
            geometry = replace(
                geometry, sky_mask=np.asarray(geometry.sky_mask | unmapped, dtype=np.bool_)
            )
        planes = to_gbuffer(geometry, temperature_k=temperature, material_id=material_id).to_dict()
        self._last = _Frame(
            planes=planes,
            instance_id=instance_id,
            material_id=material_id,
            unmapped=unmapped,
            elevation_rad=elevation,
            labels=labels,
        )
        return planes

    def get_outputs(self, *, step: bool = True, rt_subframes: int = 1) -> Outputs:
        """One frame, all the way to ``radiance`` / ``apparent_t`` / ``dn16`` / ``display8``.

        The clock advances by one frame period afterwards, so a sequence of calls is a real
        sequence: the FFC fires on its schedule, the fixed pattern drifts, the bolometer's
        membrane carries its lag from frame to frame.
        """
        planes = self.planes(step=step, rt_subframes=rt_subframes)
        self.state.t_s = self.scene.t0_s + self._t_rel_s
        outputs = run_frame(planes, self.config, self.state)
        overlay = (
            self.debug_unmapped
            and outputs.display8 is not None
            and self._last is not None
            and bool(self._last.unmapped.any())
        )
        if overlay:
            assert outputs.display8 is not None and self._last is not None  # narrowed by `overlay`
            outputs = replace(
                outputs,
                display8=self._downsample_mask_overlay(outputs.display8, self._last.unmapped),
            )
        self._t_rel_s += self.frame_period_s
        return outputs

    def _downsample_mask_overlay(
        self, display8: NDArray[np.uint8], mask: NDArray[np.bool_]
    ) -> NDArray[np.uint8]:
        """Paint the UNMAPPED magenta at the native grid (ADR 0047, display branch only).

        The mask lives on the k× G-buffer and the picture on the native one, so a native pixel is
        marked when **any** of its k×k samples was unmapped: the debug overlay must over-report a
        forgotten prim, never hide one behind three good samples.
        """
        k = self.config.supersample
        if k > 1:
            h, w = mask.shape[0] // k, mask.shape[1] // k
            mask = np.asarray(
                mask[: h * k, : w * k].reshape(h, k, w, k).any(axis=(1, 3)), dtype=np.bool_
            )
        return overlay_unmapped(display8, mask)
