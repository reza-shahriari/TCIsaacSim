"""``Sequence`` / ``Frame``: the one shape every validation analyser reads.

roadmap ME.1b; ADR 0003.

The indexed sets agree on nothing. Halmstad ships MATLAB ``.mat`` labels beside mp4; Anti-UAV410
ships per-sequence JSON; the single-frame sets ship folders of images and masks. Writing five
analysers against five layouts would mean five places for a box convention to be wrong, and a box
convention that is wrong by half a pixel quietly changes every size-versus-range number.

So this module defines **our** canonical form and nothing else reads a publisher's layout: a
per-dataset converter writes a sequence directory, and every analyser from ME.2 onward reads
:class:`Sequence`. The format is deliberately boring -- a JSON index plus one array per frame --
because its job is to be unambiguous, not compact.

**Frames are 8-bit and the reader insists on it.** Every set in the index stores 8-bit frames;
that is a fact about the data, not a limitation of this reader, and it is the single most important
thing to keep visible. A pipeline that quietly widened them to float would invite a radiometric
claim the data cannot support (ADR 0003: the 8-bit quantisation and the codec are a floor under
every noise statistic measured here). Where a set really is 16-bit, that is a new field in the
index and a deliberate change here, not an accident of dtype promotion.

**Boxes are top-left ``(x, y)`` plus ``(w, h)`` in pixels, in the same pixel-edge convention as
the rest of the project**: the centre of pixel ``(row, col)`` is at ``(col + 0.5, row + 0.5)``, so
a box covering exactly the first pixel is ``x=0, y=0, w=1, h=1`` and its centre is ``(0.5, 0.5)``.
Converters state their own convention and translate; the analysers never have to ask.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterator, Mapping
from collections.abc import Sequence as Seq
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SEQUENCE_SCHEMA_VERSION",
    "FrameFormat",
    "Box",
    "Frame",
    "Sequence",
    "write_sequence",
    "read_sequence",
]

SEQUENCE_SCHEMA_VERSION = 1

#: How frame arrays are stored. ``npy`` needs no decoder and round-trips exactly, which is why it
#: is the default and the only format the test suite depends on; ``png`` is for converters that
#: want something a person can look at, and reading it needs the ``validation`` extra.
FrameFormat = Literal["npy", "png"]


@dataclass(frozen=True)
class Box:
    """An annotated target: top-left corner and size, in pixel-edge units."""

    x: float
    y: float
    w: float
    h: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not (self.w > 0.0 and self.h > 0.0):
            raise ValueError(f"box must have positive size, got {self.w} x {self.h}")

    @property
    def area_px(self) -> float:
        return float(self.w * self.h)

    @property
    def centre(self) -> tuple[float, float]:
        """``(x, y)`` of the box centre, in the project's pixel-edge convention."""
        return (self.x + 0.5 * self.w, self.y + 0.5 * self.h)

    @property
    def extent_px(self) -> float:
        """The side of a square of the same area -- the size a detection metric bins on."""
        return float(np.sqrt(self.area_px))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"x": self.x, "y": self.y, "w": self.w, "h": self.h}
        if self.label is not None:
            out["label"] = self.label
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Box:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            w=float(data["w"]),
            h=float(data["h"]),
            label=data.get("label"),
        )


@dataclass(frozen=True)
class Frame:
    """One frame: its image, its boxes, and whatever the publisher said about it."""

    index: int
    image: NDArray[np.uint8]
    boxes: tuple[Box, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.image.shape[0]), int(self.image.shape[1]))


def _require_uint8(image: Any, where: str) -> NDArray[np.uint8]:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"{where}: validation frames are 8-bit (got {arr.dtype}). Every indexed set stores "
            "8-bit frames, and that quantisation is a floor under every statistic measured on "
            "them (ADR 0003) -- widening it here would hide the floor, not remove it."
        )
    if arr.ndim != 2:
        raise ValueError(f"{where}: frames must be single-channel (H, W), got {arr.shape}")
    return arr


class Sequence:
    """A clip: frames on disk or in memory, with boxes and per-frame attributes.

    Frames are read on demand rather than all at once -- a ten-second Halmstad clip is 600 frames
    and a whole indexed set is thousands of clips, so an analyser that wants one frame should not
    pay for the rest. :meth:`images` is the opt-in for the analysers that genuinely need the cube
    (a temporal PSD does; a box-size histogram does not).
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, int],
        *,
        frames: Seq[Frame] | None = None,
        directory: str | os.PathLike[str] | None = None,
        files: Seq[str] | None = None,
        boxes: Seq[tuple[Box, ...]] | None = None,
        attributes: Seq[Mapping[str, Any]] | None = None,
        source_dataset: str | None = None,
        frame_format: FrameFormat = "npy",
    ) -> None:
        if (frames is None) == (directory is None):
            raise ValueError("a Sequence is built either from frames in memory or from a directory")
        self.name = name
        self.shape = shape
        self.source_dataset = source_dataset
        self.frame_format = frame_format
        self._frames = None if frames is None else tuple(frames)
        self._directory = None if directory is None else pathlib.Path(directory)
        self._files = None if files is None else tuple(files)
        self._boxes = None if boxes is None else tuple(boxes)
        self._attributes = None if attributes is None else tuple(attributes)

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        images: Any,
        *,
        name: str = "sequence",
        boxes: Seq[Seq[Box]] | None = None,
        attributes: Seq[Mapping[str, Any]] | None = None,
        source_dataset: str | None = None,
    ) -> Sequence:
        """Build in memory from a ``(N, H, W)`` uint8 stack -- the synthetic-clip entry point."""
        stack = np.asarray(images)
        if stack.ndim != 3:
            raise ValueError(f"images must be (N, H, W), got {stack.shape}")
        count = int(stack.shape[0])
        if boxes is not None and len(boxes) != count:
            raise ValueError(f"{len(boxes)} box lists for {count} frames")
        if attributes is not None and len(attributes) != count:
            raise ValueError(f"{len(attributes)} attribute maps for {count} frames")
        frames = [
            Frame(
                index=i,
                image=_require_uint8(stack[i], f"frame {i}"),
                boxes=tuple(boxes[i]) if boxes is not None else (),
                attributes=dict(attributes[i]) if attributes is not None else {},
            )
            for i in range(count)
        ]
        return cls(
            name,
            (int(stack.shape[1]), int(stack.shape[2])),
            frames=frames,
            source_dataset=source_dataset,
        )

    # -- access -----------------------------------------------------------------------------

    def __len__(self) -> int:
        if self._frames is not None:
            return len(self._frames)
        assert self._files is not None
        return len(self._files)

    def __getitem__(self, index: int) -> Frame:
        if self._frames is not None:
            return self._frames[index]
        assert self._files is not None and self._directory is not None
        count = len(self._files)
        i = index + count if index < 0 else index
        if not 0 <= i < count:
            raise IndexError(f"frame {index} outside a {count}-frame sequence {self.name!r}")
        image = self._load(self._directory / self._files[i], f"frame {i} of {self.name!r}")
        if image.shape != self.shape:
            raise ValueError(
                f"frame {i} of {self.name!r} is {image.shape}, the index says {self.shape}"
            )
        return Frame(
            index=i,
            image=image,
            boxes=() if self._boxes is None else self._boxes[i],
            attributes={} if self._attributes is None else self._attributes[i],
        )

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]

    def images(self) -> NDArray[np.uint8]:
        """Every frame as one ``(N, H, W)`` uint8 cube. Loads the whole clip; ask deliberately."""
        return np.stack([frame.image for frame in self], axis=0).astype(np.uint8, copy=False)

    def boxes_at(self, index: int) -> tuple[Box, ...]:
        """Boxes without decoding the image -- what a size or SCR histogram actually needs."""
        if self._boxes is not None:
            return self._boxes[index]
        if self._frames is not None:
            return self._frames[index].boxes
        return ()

    # -- storage ----------------------------------------------------------------------------

    @staticmethod
    def _load(path: pathlib.Path, where: str) -> NDArray[np.uint8]:
        if path.suffix == ".npy":
            return _require_uint8(np.load(path, allow_pickle=False), where)
        if path.suffix == ".png":
            return _require_uint8(_decode_png(path), where)
        raise ValueError(f"{where}: unknown frame format {path.suffix!r}")

    def write(
        self, directory: str | os.PathLike[str], *, frame_format: FrameFormat = "npy"
    ) -> pathlib.Path:
        return write_sequence(directory, self, frame_format=frame_format)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        where = "memory" if self._frames is not None else str(self._directory)
        return f"Sequence({self.name!r}, {len(self)} frames of {self.shape}, from {where})"


def _decode_png(path: pathlib.Path) -> NDArray[Any]:
    """Decode an 8-bit PNG frame. Needs the ``validation`` extra; ``npy`` needs nothing.

    Lazily imported and with an error that names the remedy, because the default gate must keep
    running on a machine with no image stack at all (CLAUDE.md #1 is about the core, but the same
    reason applies here: a test suite nobody can run is a test suite nobody runs).
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            f"reading {path.name} needs a PNG decoder: pip install -e '.[validation]'. "
            "Sequences written as .npy need no decoder."
        ) from exc
    return np.asarray(iio.imread(path))


def write_sequence(
    directory: str | os.PathLike[str],
    sequence: Sequence,
    *,
    frame_format: FrameFormat = "npy",
) -> pathlib.Path:
    """Write a sequence to the canonical layout and return the index file.

    ``frames/000000.npy`` and friends are zero-padded so the directory sorts in frame order in
    every tool, and the JSON index carries the boxes and attributes so that a reader never has to
    parse a filename for meaning.
    """
    out = pathlib.Path(directory)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for frame in sequence:
        rel = f"frames/{frame.index:06d}.{frame_format}"
        path = out / rel
        image = _require_uint8(frame.image, f"frame {frame.index}")
        if frame_format == "npy":
            np.save(path, image, allow_pickle=False)
        elif frame_format == "png":
            from irsim.io.png import write_png

            write_png(path, image)
        else:  # pragma: no cover - Literal is checked by mypy
            raise ValueError(f"unknown frame format {frame_format!r}")
        entries.append(
            {
                "index": frame.index,
                "file": rel,
                "boxes": [b.to_dict() for b in frame.boxes],
                "attributes": dict(frame.attributes),
            }
        )

    index = {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "name": sequence.name,
        "source_dataset": sequence.source_dataset,
        "frame_format": frame_format,
        "shape": list(sequence.shape),
        "frames": entries,
    }
    index_path = out / "sequence.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index_path


def read_sequence(directory: str | os.PathLike[str]) -> Sequence:
    """Read a sequence directory written by :func:`write_sequence`."""
    out = pathlib.Path(directory)
    index_path = out / "sequence.json" if out.is_dir() else out
    data = json.loads(index_path.read_text(encoding="utf-8"))
    version = int(data.get("schema_version", 0))
    if version != SEQUENCE_SCHEMA_VERSION:
        raise ValueError(
            f"sequence schema_version {version}, this reader speaks {SEQUENCE_SCHEMA_VERSION}"
        )
    entries = sorted(data["frames"], key=lambda e: int(e["index"]))
    shape = (int(data["shape"][0]), int(data["shape"][1]))
    return Sequence(
        name=str(data["name"]),
        shape=shape,
        directory=index_path.parent,
        files=[str(e["file"]) for e in entries],
        boxes=[tuple(Box.from_dict(b) for b in e.get("boxes", [])) for e in entries],
        attributes=[dict(e.get("attributes", {})) for e in entries],
        source_dataset=data.get("source_dataset"),
        frame_format=data.get("frame_format", "npy"),
    )
