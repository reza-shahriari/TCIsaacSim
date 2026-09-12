"""Engine-free tests for instance-id → material-id transport (roadmap M10.2, ADR 0047).

The property that matters is that ids are *decoded*, never *interpolated*. A pixel on the boundary
between material 0 and material 7 is one or the other; a blended 3 or 4 is a different substance
with a different emissivity, and the resulting image looks entirely plausible. These tests assert
that no value outside the authored set can ever appear, that a miss keeps the loud UNMAPPED
sentinel instead of a convenient default, and that the magenta overlay touches display pixels only.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.materials.mapping import Resolution
from irsim.materials.table import UNMAPPED_MATERIAL_ID
from irsim_isaac.pipeline.material_ids import (
    MAGENTA_RGBA,
    id_coverage,
    labels_to_paths,
    material_id_plane,
    overlay_unmapped,
    unmapped_mask,
)

# Three ids spanning the range the packed table can hold, including the sentinel's neighbour.
IDS = {1: ("/World/A", 7), 2: ("/World/B", 200), 3: ("/World/C", 1)}


def _resolutions() -> list[Resolution]:
    return [Resolution(path, f"mat_{mid}", mid, "pattern", "*x*") for path, mid in IDS.values()]


def _labels() -> dict[str, str]:
    return {str(k): v[0] for k, v in IDS.items()}


def test_ids_decode_exactly_on_every_pixel() -> None:
    """Every pixel gets the authored id; nothing in between ever appears."""
    rng = np.random.default_rng(0)
    instance = rng.choice(np.array([0, 1, 2, 3], dtype=np.uint32), size=(64, 64))
    plane = material_id_plane(instance, _labels(), _resolutions())

    assert plane.dtype == np.int32
    assert set(np.unique(plane).tolist()) <= {0, 1, 7, 200}
    for ident, (_, material) in IDS.items():
        where = instance == ident
        assert np.all(plane[where] == material), ident
    assert np.all(plane[instance == 0] == UNMAPPED_MATERIAL_ID)


def test_an_edge_between_two_materials_never_blends() -> None:
    """The 0/7 boundary must not produce 3 or 4 -- the failure this transport exists to avoid."""
    instance = np.zeros((16, 16), dtype=np.uint32)
    instance[:, 8:] = 1  # left half background, right half material 7
    plane = material_id_plane(instance, _labels(), _resolutions())

    values = set(np.unique(plane).tolist())
    assert values == {0, 7}
    assert not values & {3, 4}
    assert np.all(plane[:, :8] == 0) and np.all(plane[:, 8:] == 7)


def test_supersampled_ids_downsample_by_nearest_not_by_mean() -> None:
    """A 4x supersampled id block decimates to one of its members, never to their average."""
    block = np.zeros((8, 8), dtype=np.uint32)
    block[:4] = 1  # material 7
    block[4:] = 2  # material 200
    plane = material_id_plane(block, _labels(), _resolutions())
    decimated = plane[::4, ::4]
    assert set(np.unique(decimated).tolist()) <= {7, 200}
    assert abs(float(plane.mean()) - 103.5) < 1e-6  # the mean IS 103.5 -- and is never an id
    assert 103 not in np.unique(plane).tolist()


def test_an_unresolved_prim_keeps_the_unmapped_sentinel() -> None:
    """A prim the resolver missed must not acquire a plausible default emissivity."""
    resolutions = [
        Resolution("/World/A", "mat_7", 7, "pattern", "*x*"),
        Resolution("/World/B", None, UNMAPPED_MATERIAL_ID, "miss", None),
    ]
    instance = np.array([[1, 2], [1, 2]], dtype=np.uint32)
    plane = material_id_plane(instance, {"1": "/World/A", "2": "/World/B"}, resolutions)
    assert np.all(plane[:, 0] == 7)
    assert np.all(plane[:, 1] == UNMAPPED_MATERIAL_ID)


def test_an_id_with_no_label_is_unmapped_and_strict_mode_raises() -> None:
    instance = np.array([[1, 9]], dtype=np.uint32)
    plane = material_id_plane(instance, _labels(), _resolutions())
    assert plane[0, 1] == UNMAPPED_MATERIAL_ID
    with pytest.raises(KeyError, match="idToLabels"):
        material_id_plane(instance, _labels(), _resolutions(), strict=True)


def test_background_is_unmapped_but_is_not_a_mapping_miss() -> None:
    """Sky pixels carry id 0 and must not be counted as a forgotten prim (M0.6 contract)."""
    instance = np.array([[0, 0], [1, 1]], dtype=np.uint32)
    sky = instance == 0
    plane = material_id_plane(instance, _labels(), _resolutions())
    mask = unmapped_mask(plane, sky_mask=sky)
    assert not mask.any()
    assert id_coverage(plane, sky_mask=sky) == pytest.approx(1.0)
    # without the sky mask the same frame looks half unmapped, which is why the mask exists
    assert id_coverage(plane) == pytest.approx(0.5)


def test_labels_to_paths_handles_both_payload_shapes() -> None:
    paths = labels_to_paths({"1": "/World/A", "2": {"class": "car_body"}, "3": None, "x": "/W"})
    assert paths == {1: "/World/A", 2: "car_body"}


# --- the magenta overlay -----------------------------------------------------------------------


def test_magenta_paints_the_mask_and_nothing_else() -> None:
    rgba = np.full((4, 4, 4), 30, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 2] = True
    out = overlay_unmapped(rgba, mask)

    assert tuple(out[1, 2]) == MAGENTA_RGBA
    untouched = np.ones((4, 4), dtype=bool)
    untouched[1, 2] = False
    assert np.all(out[untouched] == 30)
    assert np.all(rgba == 30), "the input image must not be modified in place"


def test_overlay_refuses_a_non_rgba8_image() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    with pytest.raises(TypeError, match="uint8"):
        overlay_unmapped(np.zeros((2, 2, 4), dtype=np.float32), mask)
    with pytest.raises(ValueError, match=r"\(H, W, 4\)"):
        overlay_unmapped(np.zeros((2, 2, 3), dtype=np.uint8), mask)
    with pytest.raises(ValueError, match="mask"):
        overlay_unmapped(np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((3, 3), dtype=bool))


def test_material_id_plane_refuses_a_float_id_plane() -> None:
    with pytest.raises(TypeError, match="integer"):
        material_id_plane(np.zeros((2, 2), dtype=np.float32), _labels(), _resolutions())
