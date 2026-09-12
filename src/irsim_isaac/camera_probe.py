"""Survey the lens-distortion schemas a build carries, and what their attributes default to.

docs/physics-model.md §8.4; roadmap M10.9a; ADR 0015 addendum.

``optics.distortion`` has to become USD, and which schema carries which model -- and what its
attributes are called -- is a property of the build, not of the documentation. This is the
measurement ADR 0015's addendum records, kept as code so the next build can be re-measured rather
than argued about.

Two findings from the first run (6.1.0-rc.26) are why the probe reports **defaults** and not just
names. The schemas do not default to an identity lens: ``fx = 900``, ``cx = 1024``,
``imageSize = (2048, 1024)``, and ``OmniLensDistortionOpenCvFisheyeAPI`` defaults ``k1`` to
0.00245, a real curvature. An attribute left unwritten is therefore not "no distortion", it is a
different camera -- which is why :func:`irsim_isaac.pipeline.ir_camera.distortion_attributes`
emits every attribute of the schema every time. And ``omni:lensdistortion:model`` is set by
``ApplyAPI`` itself and its ``allowedTokens`` holds exactly one value per schema, so the model
token cannot disagree with the schema that was applied.

Engine imports live inside the functions, so this module imports on a machine with no Isaac Sim.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DISTORTION_API_SCHEMAS", "survey_distortion_schemas", "registered_schema_names"]

#: The families worth reporting on. The probe still reports any *other* schema whose name mentions
#: distortion, so a build that adds one is not silently missed.
DISTORTION_API_SCHEMAS: tuple[str, ...] = (
    "OmniLensDistortionOpenCvPinholeAPI",
    "OmniLensDistortionOpenCvFisheyeAPI",
    "OmniLensDistortionKannalaBrandtK3API",
    "OmniLensDistortionRadTanThinPrismAPI",
    "OmniLensDistortionFthetaAPI",
    "OmniLensDistortionLutAPI",
)


def registered_schema_names() -> list[str]:
    """Every USD schema type this build registers whose name mentions distortion or f-theta."""
    from pxr import Plug, Tf

    base = Tf.Type.FindByName("UsdSchemaBase")
    names = [t.typeName for t in Plug.Registry().GetAllDerivedTypes(base)]
    return sorted(n for n in names if "istortion" in n or "theta" in n.lower())


def survey_distortion_schemas(
    stage: Any = None, *, schemas: tuple[str, ...] = DISTORTION_API_SCHEMAS
) -> dict[str, Any]:
    """Apply each schema to a throwaway camera and record what appears, with defaults.

    Each schema goes on its **own** camera prim: ``omni:lensdistortion:model`` is single-valued,
    so applying two families to one prim would report whichever won rather than what each does.
    A schema this build does not carry is recorded as an error, not raised -- an absent family is
    a result.
    """
    import omni.usd
    from pxr import UsdGeom

    if stage is None:
        stage = omni.usd.get_context().get_stage()

    report: dict[str, Any] = {"registered": registered_schema_names(), "schemas": {}}
    for index, schema in enumerate(schemas):
        entry: dict[str, Any] = {}
        try:
            prim = UsdGeom.Camera.Define(stage, f"/World/_DistortionProbe_{index}").GetPrim()
            before = {a.GetName() for a in prim.GetAttributes()}
            entry["applied"] = bool(prim.ApplyAPI(schema))
            attrs = sorted({a.GetName() for a in prim.GetAttributes()} - before)
            entry["attributes"] = {}
            for name in attrs:
                attr = prim.GetAttribute(name)
                entry["attributes"][name] = {
                    "type": str(attr.GetTypeName()),
                    "default": _plain(attr.Get()),
                    "allowed_tokens": _plain(attr.GetMetadata("allowedTokens")),
                }
        except Exception as exc:  # noqa: BLE001 - an absent schema is data, not a crash
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["schemas"][schema] = entry
    return report


def _plain(value: Any) -> Any:
    """USD values (Gf/Vt types) as something ``json.dump`` will accept."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return str(value)
