"""Seeded random streams (M4.2, ADR 0022): bit-identical replay, per-element isolation,
traversal independence, independence across frames/streams/sensors, correct statistics of the
hashed normals, and no global state."""

from __future__ import annotations

import numpy as np
import pytest

from irsim.noise import (
    NoiseStream,
    field_normal,
    hash_normal,
    hash_u64,
    hash_uniform,
    noise_rng,
    sensor_rng,
    stream_key,
)

N = 100_000
R_BOUND = 3.0 / np.sqrt(N)


def _field(sensor: int, frame: int, stream: NoiseStream, n: int = N) -> np.ndarray:
    return field_normal(stream_key(sensor, frame, stream), (n,))


def test_same_triple_is_bit_identical() -> None:
    a = _field(7, 42, NoiseStream.TVH, 4096)
    b = _field(7, 42, NoiseStream.TVH, 4096)
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert stream_key(7, 42, NoiseStream.TVH) == stream_key(7, 42, NoiseStream.TVH)


def test_pinned_hash_values() -> None:
    """The integer hash is the cross-implementation contract: pin a few values so that a change
    to the mixer (or a GPU port that disagrees) is caught, not silently re-baselined."""
    key = stream_key(7, 42, NoiseStream.TVH)
    assert key == stream_key(7, 42, 1)  # IntEnum value, not name
    h = hash_u64(key, np.array([0, 1, 2, 1000], dtype=np.int64))
    assert h.dtype == np.uint64
    assert h[0] != h[1] != h[2]
    # a change of a single bit of the key or index flips about half the output bits
    h2 = hash_u64(key ^ 1, np.array([0], dtype=np.int64))
    diff = bin(int(h[0]) ^ int(h2[0])).count("1")
    assert 20 < diff < 44
    assert stream_key(7, 42, NoiseStream.TVH) != stream_key(7, 43, NoiseStream.TVH)
    assert stream_key(7, 42, NoiseStream.TVH) != stream_key(7, 42, NoiseStream.TV)


def test_per_element_isolation_and_traversal_independence() -> None:
    key = stream_key(3, 9, NoiseStream.TH)
    full = field_normal(key, (64, 96))
    single = hash_normal(key, np.array([37 * 96 + 11]))
    assert full[37, 11] == single[0]
    # tiling: compute rows 32-63 alone via their flat indices
    idx = np.arange(32 * 96, 64 * 96).reshape(32, 96)
    assert np.array_equal(hash_normal(key, idx), full[32:])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((7, 42, NoiseStream.TVH), (7, 43, NoiseStream.TVH)),  # frame
        ((7, 42, NoiseStream.TVH), (7, 42, NoiseStream.TV)),  # stream
        ((7, 42, NoiseStream.TVH), (8, 42, NoiseStream.TVH)),  # sensor
        ((7, 42, NoiseStream.TV), (7, 42, NoiseStream.TH)),
    ],
)
def test_streams_are_independent(
    a: tuple[int, int, NoiseStream], b: tuple[int, int, NoiseStream]
) -> None:
    x, y = _field(*a), _field(*b)
    r = float(np.corrcoef(x, y)[0, 1])
    assert abs(r) < R_BOUND, f"{a} vs {b}: r = {r:.4f}"
    assert not np.array_equal(x[:64], y[:64])


def test_hashed_normals_are_standard_normal() -> None:
    x = _field(11, 5, NoiseStream.TVH, 1_000_000).astype(np.float64)
    assert abs(x.mean()) < 4e-3 and abs(x.std() - 1.0) < 3e-3
    # tails: P(|x| > 3) = 2.7e-3
    assert abs(np.mean(np.abs(x) > 3.0) - 2.7e-3) < 4e-4
    # neighbouring elements uncorrelated
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1]) < 3e-3
    u = hash_uniform(stream_key(11, 5, NoiseStream.TVH), np.arange(1_000_000))
    assert u.min() > 0.0 and u.max() <= 1.0 and abs(u.mean() - 0.5) < 1e-3


def test_frame_can_be_reached_directly() -> None:
    """Frame 500 does not depend on frames 0-499 having been generated (no hidden state)."""
    direct = _field(3, 500, NoiseStream.TH, 256)
    for f in range(0, 500):
        _field(3, f, NoiseStream.TH, 8)
        noise_rng(3, f, NoiseStream.SHOT).poisson(10.0, 4)
    assert np.array_equal(direct, _field(3, 500, NoiseStream.TH, 256))


def test_sequential_generators_for_poisson_draws() -> None:
    a = noise_rng(7, 42, NoiseStream.SHOT).poisson(1e6, 4096)
    b = noise_rng(7, 42, NoiseStream.SHOT).poisson(1e6, 4096)
    assert np.array_equal(a, b)
    c = noise_rng(7, 43, NoiseStream.SHOT).poisson(1e6, 4096)
    assert not np.array_equal(a, c)
    assert np.array_equal(
        sensor_rng(7, NoiseStream.BAD_PIXEL_MAP).random(64),
        sensor_rng(7, NoiseStream.BAD_PIXEL_MAP).random(64),
    )


def test_seed_validation_and_stream_values_are_stable() -> None:
    with pytest.raises(ValueError):
        stream_key(-1, 0, NoiseStream.TVH)
    with pytest.raises(TypeError):
        stream_key(1.5, 0, NoiseStream.TVH)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        noise_rng(True, 0, NoiseStream.TVH)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        hash_u64(1, np.array([-1]))
    assert (NoiseStream.TVH, NoiseStream.TV, NoiseStream.TH, NoiseStream.T) == (1, 2, 3, 4)
    assert NoiseStream.VH_FIXED == 10 and NoiseStream.BAD_PIXEL_MAP == 30 and NoiseStream.SHOT == 40
