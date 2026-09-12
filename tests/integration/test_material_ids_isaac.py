"""Roadmap M10.2: material-ID transport measured on a stage that exercises every mapping rule.

The engine-free half -- the id remap, the sentinel, the magenta overlay -- is covered in
``tests/unit/test_material_ids.py``. What needs a renderer is the rest of the chain: that a USD
stage walk sees the bindings, semantics and ``thermal:material`` overrides that were authored;
that the resolver's precedence (ADR 0047) holds on a real stage; and that the instance ids the
renderer produces decode to those materials exactly, on every pixel, at a supersampled resolution.

The failure this guards against is quiet. If ids blended at object edges, or if a prim silently
picked up a default emissivity, the resulting thermal image would look completely reasonable and
be made of the wrong substances.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

SUPERSAMPLE = 4
NATIVE = 128
RESOLUTION = SUPERSAMPLE * NATIVE


@pytest.fixture(scope="module")
def scene(simulation_app: Any) -> Any:
    del simulation_app
    from irsim_isaac.material_probe import build_material_scene

    built = build_material_scene(resolution=RESOLUTION)
    assert built.errors == {}, built.errors
    return built


@pytest.fixture(scope="module")
def resolver() -> Any:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MaterialResolver, load_mapping_rules
    from irsim.materials.table import MaterialTable

    library = MaterialLibrary.load()
    names = MaterialTable.from_library(library, "lwir").names
    rules = load_mapping_rules(known_materials=library.names)
    return MaterialResolver(rules, names)


@pytest.fixture(scope="module")
def records(scene: Any) -> Any:
    from irsim_isaac.pipeline.materials_usd import prim_records

    return prim_records(root="/World/Targets")


@pytest.fixture(scope="module")
def frame(scene: Any) -> Any:
    """One settled render: the instance-id plane and the idToLabels table behind it."""
    import omni.replicator.core as rep

    from irsim_isaac.geometry_probe import configure_renderer
    from irsim_isaac.pipeline.gbuffer_isaac import AovReader

    configure_renderer()
    rp = rep.create.render_product(scene.camera_path, (RESOLUTION, RESOLUTION))
    rp_path = rp.path if hasattr(rp, "path") else str(rp)
    reader = AovReader(
        rp_path,
        device="cpu",
        required=("instance",),
        expected_shape=(RESOLUTION, RESOLUTION),
    ).attach(settle_frames=24)
    aovs = reader.read()
    yield {"reader": reader, "aovs": aovs}
    reader.detach()


# --- the stage walk ------------------------------------------------------------------------------


def test_stage_walk_sees_every_binding_semantic_and_override(scene: Any, records: Any) -> None:
    """`prim_records` must report exactly what was authored, or the resolver decides on fiction."""
    by_path = {r.path: r for r in records}
    assert len(records) == len(scene.targets), sorted(by_path)
    for name, target in scene.targets.items():
        record = by_path[target.prim_path]
        assert record.material_name == target.material_name, name
        assert record.semantic_class == target.semantic_class, name
        assert record.override == target.override, name


def test_stage_walk_skips_non_geometry(scene: Any, records: Any) -> None:
    """Cameras, lights, scopes and material prims are not surfaces; counting them inflates
    coverage and would let an unmapped asset pass the ADR 0047 gate."""
    from irsim_isaac.pipeline.materials_usd import prim_records

    whole_stage = prim_records(root="/")
    assert {r.path for r in whole_stage} == {t.prim_path for t in scene.targets.values()}
    assert all("/World/Looks" not in r.path for r in whole_stage)
    assert all("Camera" not in r.path and "Light" not in r.path for r in whole_stage)


# --- precedence (ADR 0047) ------------------------------------------------------------------------


def test_resolver_precedence_holds_on_a_real_stage(scene: Any, records: Any, resolver: Any) -> None:
    """Override beats the `*glass*` glob; semantics beat an unmatched name; the rest is a miss."""
    by_path = {r.path: r for r in records}
    for name, target in scene.targets.items():
        resolution = resolver.resolve(by_path[target.prim_path])
        assert resolution.rule == target.expected_rule, (name, resolution)
        assert resolution.material == target.expected_material, (name, resolution)

    window = resolver.resolve(by_path[scene.targets["Window"].prim_path])
    assert window.material == "bare_aluminium", "the thermal:material override must beat *glass*"
    assert window.rule == "override"


def test_audit_fails_on_the_unmapped_prim_and_names_it(
    scene: Any, records: Any, resolver: Any
) -> None:
    from irsim.materials.mapping import audit

    report = audit(records, resolver)
    assert report.total == 5
    assert report.mapped == 4
    assert report.coverage == pytest.approx(0.8)
    assert not report.passed, "0.8 is below the 0.95 gate; the miss must fail CI"
    assert scene.targets["Mystery"].prim_path in report.miss_paths
    assert "UNMAPPED" in report.render()


# --- the transport -------------------------------------------------------------------------------


def test_material_ids_decode_exactly_at_supersampled_resolution(
    scene: Any, frame: Any, records: Any, resolver: Any
) -> None:
    """Every pixel of every target carries its material id, with nothing in between."""
    from irsim_isaac.material_probe import expected_material_ids
    from irsim_isaac.pipeline.material_ids import labels_from_payload, material_id_plane

    aovs = frame["aovs"]
    labels = labels_from_payload(aovs.device_handles["instance"])
    assert labels, "instance_segmentation returned no idToLabels table"

    resolutions = resolver.resolve_all(records)
    plane = material_id_plane(aovs.instance_id, {str(k): v for k, v in labels.items()}, resolutions)
    expected = expected_material_ids(scene, resolver)

    assert plane.dtype == np.int32
    assert set(np.unique(plane).tolist()) <= set(expected.values()) | {0}
    for name in scene.targets:
        rows, cols = scene.patch_of(name, half_px=3)
        patch = plane[rows, cols]
        assert np.all(patch == expected[name]), (name, np.unique(patch), expected[name])


def test_object_edges_never_blend_two_materials(
    scene: Any, frame: Any, records: Any, resolver: Any
) -> None:
    """Scan the row through the quad centres: every pixel is an authored id or the background."""
    from irsim_isaac.material_probe import expected_material_ids
    from irsim_isaac.pipeline.material_ids import labels_from_payload, material_id_plane

    aovs = frame["aovs"]
    labels = labels_from_payload(aovs.device_handles["instance"])
    plane = material_id_plane(
        aovs.instance_id,
        {str(k): v for k, v in labels.items()},
        resolutions=resolver.resolve_all(records),
    )
    expected = expected_material_ids(scene, resolver)

    row = scene.pixel_of("Body")[1]
    allowed = set(expected.values()) | {0}
    scan = set(np.unique(plane[row, :]).tolist())
    assert scan <= allowed, sorted(scan - allowed)
    assert len(scan) >= 3, "the scan line must actually cross several materials"


def test_the_unmapped_prim_is_magenta_and_nothing_else_is(
    scene: Any, frame: Any, records: Any, resolver: Any
) -> None:
    """Magenta marks exactly the forgotten prim -- not the sky, and not its neighbours."""
    from irsim_isaac.pipeline.material_ids import (
        MAGENTA_RGBA,
        labels_from_payload,
        material_id_plane,
        overlay_unmapped,
        unmapped_mask,
    )

    aovs = frame["aovs"]
    labels = labels_from_payload(aovs.device_handles["instance"])
    plane = material_id_plane(
        aovs.instance_id, {str(k): v for k, v in labels.items()}, resolver.resolve_all(records)
    )
    sky = np.asarray(aovs.instance_id) == 0
    mask = unmapped_mask(plane, sky_mask=sky)

    rows, cols = scene.patch_of("Mystery", half_px=3)
    assert np.all(mask[rows, cols]), "the unmapped prim must be flagged"
    for name in ("Body", "Window", "Road", "Trim"):
        r, c = scene.patch_of(name, half_px=3)
        assert not mask[r, c].any(), f"{name} resolved and must not be flagged"
    assert not mask[sky].any(), "the background is not a mapping miss (M0.6 contract)"

    display = np.zeros((RESOLUTION, RESOLUTION, 4), dtype=np.uint8)
    painted = overlay_unmapped(display, mask)
    assert tuple(painted[rows, cols][0, 0]) == MAGENTA_RGBA
    assert np.all(painted[~mask] == 0)


def test_supersampled_ids_survive_decimation_to_native(
    scene: Any, frame: Any, records: Any, resolver: Any
) -> None:
    """Decimating the 4x id plane to native must still give authored ids, never an average."""
    from irsim_isaac.material_probe import expected_material_ids
    from irsim_isaac.pipeline.material_ids import labels_from_payload, material_id_plane

    aovs = frame["aovs"]
    labels = labels_from_payload(aovs.device_handles["instance"])
    plane = material_id_plane(
        aovs.instance_id, {str(k): v for k, v in labels.items()}, resolver.resolve_all(records)
    )
    native = plane[::SUPERSAMPLE, ::SUPERSAMPLE]
    expected = expected_material_ids(scene, resolver)
    assert native.shape == (NATIVE, NATIVE)
    assert set(np.unique(native).tolist()) <= set(expected.values()) | {0}
