"""Walk a USD stage into the engine-free :class:`~irsim.materials.mapping.PrimRecord` dump.

docs/physics-model.md §13.3; roadmap M10.2; ADR 0047 (precedence and the coverage gate).

ADR 0047 fixed the mapping precedence -- explicit override, then semantic class, then a glob on
the bound material's name, then a recorded miss -- and deliberately kept the resolver engine-free
by having it consume plain records. This module is the other half: the only place that opens a USD
stage and produces those records. Nothing in ``irsim`` imports USD, and this module imports it
inside functions, so ``import irsim_isaac`` still works on a machine with no Isaac Sim.

``pxr`` is provided by Kit and is **not importable outside a running Kit application** on this
build (the same is true of ``warp`` -- ADR 0014), so the USD path needs the engine even though
nothing about reading a stage is inherently graphical. The two-step flow exists for that reason:
dump the records inside Kit once, then audit them anywhere.

**The override attribute.** ``thermal:material`` on the prim, holding a library material name. It
is authored in the asset (or by a scene-prep script) and beats every heuristic, which is what makes
a wrong glob fixable without editing the rules for everyone. It is a plain USD attribute rather
than an applied schema so that any DCC can write it without our schema installed.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from irsim.materials.mapping import PrimRecord

__all__ = [
    "THERMAL_MATERIAL_ATTR",
    "prim_records",
    "bound_material_name",
    "semantic_class_of",
    "override_of",
    "dump_prim_records",
]

#: The per-prim override attribute (ADR 0047 precedence rule 1).
THERMAL_MATERIAL_ATTR = "thermal:material"


def bound_material_name(prim: Any) -> str | None:
    """Name of the material bound to this prim, or None when nothing is bound.

    The *name* is used, not the path, because the glob rules in ``configs/materials/mapping.yaml``
    match how artists name materials (``Car_Paint_Red``, ``Glass_Clear``), which survives being
    re-parented far better than an absolute path does.
    """
    from pxr import UsdShade

    try:
        binding = UsdShade.MaterialBindingAPI(prim)
        material = binding.ComputeBoundMaterial()[0]
    except Exception:  # noqa: BLE001 - an unbindable prim is a miss, not a crash
        return None
    if not material or not material.GetPrim().IsValid():
        return None
    return str(material.GetPrim().GetName())


def semantic_class_of(prim: Any) -> str | None:
    """The prim's ``class`` semantic label, via the 6.x API with the legacy schema as fallback."""
    try:
        from isaacsim.core.experimental.utils.semantics import get_labels

        labels = get_labels(prim)
        if isinstance(labels, dict):
            value = labels.get("class")
            if value:
                return str(value[0] if isinstance(value, (list, tuple)) else value)
    except Exception:  # noqa: BLE001 - fall through to the legacy schema
        pass
    try:
        from pxr import Semantics

        for name in prim.GetAppliedSchemas():
            if not name.startswith("SemanticsAPI:"):
                continue
            api = Semantics.SemanticsAPI.Get(prim, name.split(":", 1)[1])
            if api and api.GetSemanticTypeAttr().Get() == "class":
                data = api.GetSemanticDataAttr().Get()
                if data:
                    return str(data)
    except Exception:  # noqa: BLE001
        return None
    return None


def override_of(prim: Any) -> str | None:
    """The ``thermal:material`` override, or None. An empty string counts as absent."""
    attr = prim.GetAttribute(THERMAL_MATERIAL_ATTR)
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    return str(value) if value else None


def prim_records(
    stage: Any = None, *, root: str = "/", include_invisible: bool = False
) -> list[PrimRecord]:
    """Every renderable prim under ``root`` as an engine-free record.

    Only geometry is reported -- a ``Gprim`` (mesh, sphere, cube, ...). Xforms, scopes, cameras,
    lights and the material prims themselves are not surfaces and would dilute the coverage
    fraction ADR 0047 gates on, making an asset look better mapped than it is.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    if stage is None:
        stage = omni.usd.get_context().get_stage()
    start = stage.GetPrimAtPath(root)
    if not start or not start.IsValid():
        raise ValueError(f"no prim at {root!r}")

    records: list[PrimRecord] = []
    for prim in Usd.PrimRange(start):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if not include_invisible:
            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
        records.append(
            PrimRecord(
                path=str(prim.GetPath()),
                material_name=bound_material_name(prim),
                semantic_class=semantic_class_of(prim),
                override=override_of(prim),
            )
        )
    return records


def dump_prim_records(records: list[PrimRecord], path: str | os.PathLike[str]) -> pathlib.Path:
    """Write the records as the JSON ``scripts/audit_materials.py`` already consumes."""
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "path": r.path,
            "material_name": r.material_name,
            "semantic_class": r.semantic_class,
            "override": r.override,
        }
        for r in records
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
