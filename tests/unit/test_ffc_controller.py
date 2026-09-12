"""FFC controller (M9.7): the schedule, the freeze, and the ΔT_FPA each mode leaves behind.

§11.2 is unusually direct about priorities here -- "that freeze is a real behavioural artefact that
a perception stack must survive. Simulating it is worth more than another decimal place of
radiometry" -- so the tests are about behaviour, at the frame level:

* at 60 Hz / 180 s / 700 ms the shutter closes on frame 10800 and the output is held for exactly
  42 frames, bit-identical while the input keeps changing underneath;
* the rounding rule is round, not ceil: 9 Hz x 0.7 s is 6 frames, not 7;
* the residual immediately after the event is below one LSB (it is exactly zero);
* 179 s after the event under a 0.05 K/s drift the residual is M9.6's prediction within 5 %;
* ``ideal`` never freezes and never has a residual;
* ``shutterless`` never freezes and its residual at 540 s is at most 1.5x its 180 s value, because
  a scene-based correction bounds it rather than letting it grow.

docs/physics-model.md §11.2, §12.2. ADR 0057, ADR 0056.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.isp import FfcController
from irsim.noise import NucResidual

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 777
DN_PER_K = 178.0  # the committed Boson chain at 300 K (ADR 0056); pinned by test_nuc_residual


def _nuc(**over: Any):  # type: ignore[no-untyped-def]
    d = copy.deepcopy(BOSON)
    d["sensor"]["nuc"].update(over)
    return SensorConfig.model_validate(d).sensor.nuc


class _Spy:
    """A Resettable that records when it was reset."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def reset(self, frame_index: int) -> None:
        self.calls.append(frame_index)


# -- the schedule ----------------------------------------------------------------------------


def test_boson_schedule_at_60_hz() -> None:
    """180 s at 60 Hz is 10800 frames; 700 ms is 42."""
    c = FfcController(nuc=_nuc(), fps=60.0)
    assert c.interval_frames == 10_800
    assert c.freeze_frames == 42
    assert c.fires_on(10_800)
    assert not c.fires_on(10_799)
    assert c.fires_on(21_600)


def test_the_rounding_rule_at_9_hz() -> None:
    """round, not ceil: 9 x 0.7 = 6.3 -> 6 frames. A freeze is a duration sampled by the clock."""
    c = FfcController(nuc=_nuc(), fps=9.0)
    assert c.freeze_frames == 6


@pytest.mark.parametrize(
    ("fps", "freeze_ms", "expected"),
    [(60.0, 700.0, 42), (9.0, 700.0, 6), (30.0, 500.0, 15), (25.0, 1000.0, 25), (60.0, 0.0, 0)],
)
def test_freeze_frames_across_rates(fps: float, freeze_ms: float, expected: int) -> None:
    assert FfcController(nuc=_nuc(ffc_freeze_ms=freeze_ms), fps=fps).freeze_frames == expected


def test_frame_zero_never_fires() -> None:
    """The camera was calibrated when it started; firing at 0 would freeze every sequence's open."""
    c = FfcController(nuc=_nuc(), fps=60.0)
    assert not c.fires_on(0)


def test_an_interval_below_one_frame_is_refused() -> None:
    c = FfcController(nuc=_nuc(ffc_interval_s=0.001), fps=60.0)
    with pytest.raises(ValueError, match="under one frame"):
        _ = c.interval_frames


# -- the freeze ------------------------------------------------------------------------------


def _run(c: FfcController, n_frames: int, start: int = 10_780) -> list[tuple[np.ndarray, Any]]:
    """Feed a frame whose value is its own index, so a held frame is instantly recognisable."""
    out = []
    for i in range(start, start + n_frames):
        frame = np.full((4, 4), float(i), dtype=np.float32)
        out.append(c.process(frame, i))
    return out


def test_the_freeze_lasts_exactly_42_frames_and_holds_the_last_good_one() -> None:
    """Held, not blanked: the image goes stale rather than obviously wrong.

    That staleness is the artefact -- a tracker that survives a blank may still fail on a hold.
    """
    c = FfcController(nuc=_nuc(), fps=60.0)
    results = _run(c, 80)

    frozen_at = [10_780 + i for i, (_, e) in enumerate(results) if e.frozen]
    assert frozen_at == list(range(10_800, 10_842))  # exactly 42 frames, starting at the event
    assert len(frozen_at) == 42

    held_value = 10_799.0  # the last live frame before the shutter closed
    for i, (frame, event) in enumerate(results):
        index = 10_780 + i
        if event.frozen:
            assert np.all(frame == held_value), f"frame {index} is not the held one"
        else:
            assert np.all(frame == float(index))


def test_every_held_frame_is_bit_identical_while_the_input_changes() -> None:
    c = FfcController(nuc=_nuc(), fps=60.0)
    results = _run(c, 80)  # the window must outlast the 42-frame freeze
    held = [f for f, e in results if e.frozen]
    assert len(held) == 42
    for frame in held[1:]:
        assert np.array_equal(frame, held[0])


def test_the_event_is_reported_on_exactly_one_frame() -> None:
    c = FfcController(nuc=_nuc(), fps=60.0)
    fired = [e.fired for _, e in _run(c, 80)]
    assert sum(fired) == 1
    assert fired.index(True) == 10_800 - 10_780


def test_resettables_are_called_once_per_event() -> None:
    c = FfcController(nuc=_nuc(), fps=60.0)
    spy_a, spy_b = _Spy(), _Spy()
    c.register(spy_a, spy_b)
    _run(c, 80)
    assert spy_a.calls == [10_800]
    assert spy_b.calls == [10_800]


def test_an_object_without_a_reset_hook_is_refused() -> None:
    c = FfcController(nuc=_nuc(), fps=60.0)
    with pytest.raises(TypeError, match="no reset"):
        c.register(object())  # type: ignore[arg-type]


def test_the_output_does_not_alias_the_input() -> None:
    """A held frame that aliased the caller's buffer would change when the caller reused it."""
    c = FfcController(nuc=_nuc(), fps=60.0)
    frame = np.zeros((4, 4), dtype=np.float32)
    out, _ = c.process(frame, 1)
    frame[:] = 5.0
    assert np.all(out == 0.0)


# -- the residual across the event -------------------------------------------------------------


def test_the_residual_is_below_one_lsb_right_after_the_event() -> None:
    """ΔT_eff returns to zero at the shutter, so the residual is exactly zero.

    Which is well under the roadmap's one-LSB bar, and not by tuning: it is what ΔT = 0 means.
    """
    nuc = _nuc()
    c = FfcController(nuc=nuc, fps=60.0)
    residual = NucResidual(nuc=nuc, shape=(32, 32), dn_per_k=DN_PER_K, sensor_seed=SEED)
    c.register(residual)

    # Drift for a while so there is something to correct, then fire.
    t_fpa = 300.0
    for _ in range(1, 10_800):
        t_fpa += 0.05 / 60.0
        c.update_delta_t(t_fpa, 1.0 / 60.0)
    assert c.delta_t_eff_k > 8.0  # ~9 K of drift accumulated

    c.process(np.zeros((32, 32), dtype=np.float32), 10_800, t_fpa_k=t_fpa)
    assert c.delta_t_eff_k == 0.0
    assert float(residual.offset_dn(c.delta_t_eff_k).std()) < 1.0
    assert float(np.abs(residual.gain(c.delta_t_eff_k) - 1.0).max()) == 0.0


def test_179_s_after_the_event_the_residual_matches_the_m9_6_prediction() -> None:
    """0.05 K/s for 179 s is 8.95 K, so 45 mK/K gives 402.75 mK, within 5 %."""
    nuc = _nuc()
    c = FfcController(nuc=nuc, fps=60.0)
    residual = NucResidual(nuc=nuc, shape=(128, 128), dn_per_k=DN_PER_K, sensor_seed=SEED)
    c.register(residual)

    t_fpa = 300.0
    dt = 1.0 / 60.0
    c.process(np.zeros((4, 4), dtype=np.float32), 10_800, t_fpa_k=t_fpa)
    for _ in range(int(round(179.0 * 60.0))):
        t_fpa += 0.05 * dt
        c.update_delta_t(t_fpa, dt)

    assert c.delta_t_eff_k == pytest.approx(8.95, rel=0.01)
    sigma_mk = float(residual.offset_dn(c.delta_t_eff_k).std()) / DN_PER_K * 1e3
    assert sigma_mk == pytest.approx(45.0 * 8.95, rel=0.05)


def test_the_next_event_resets_it_again() -> None:
    nuc = _nuc()
    c = FfcController(nuc=nuc, fps=60.0)
    spy = _Spy()
    c.register(spy)
    t_fpa = 300.0
    for i in range(1, 3 * 10_800 + 1):
        t_fpa += 0.05 / 60.0
        c.process(np.zeros((2, 2), dtype=np.float32), i, t_fpa_k=t_fpa)
        c.update_delta_t(t_fpa, 1.0 / 60.0)
    assert spy.calls == [10_800, 21_600, 32_400]


# -- ideal -----------------------------------------------------------------------------------


def test_ideal_never_freezes_and_never_drifts() -> None:
    nuc = _nuc(mode="ideal")
    c = FfcController(nuc=nuc, fps=60.0)
    spy = _Spy()
    c.register(spy)
    t_fpa = 300.0
    for i in range(1, 12_000):
        t_fpa += 0.05 / 60.0
        frame = np.full((2, 2), float(i), dtype=np.float32)
        out, event = c.process(frame, i, t_fpa_k=t_fpa)
        assert not event.frozen and not event.fired
        assert np.all(out == float(i))
        assert c.update_delta_t(t_fpa, 1.0 / 60.0) == 0.0
    assert spy.calls == []
    assert c.freeze_frames == 0


# -- shutterless -----------------------------------------------------------------------------


def test_shutterless_never_freezes() -> None:
    c = FfcController(nuc=_nuc(mode="shutterless"), fps=60.0)
    for i in range(1, 12_000):
        _, event = c.process(np.zeros((2, 2), dtype=np.float32), i)
        assert not event.frozen and not event.fired
    assert c.freeze_frames == 0


def test_shutterless_residual_is_bounded_not_growing() -> None:
    """At 540 s it is at most 1.5x its 180 s value; an uncorrected core would be at 3x.

    This is the difference a scene-based correction makes [R32, R33]: the residual settles where
    the estimator's convergence balances the drift, instead of tracking ΔT_FPA without limit.
    """
    nuc = _nuc(mode="shutterless", shutterless_tau_s=120.0)
    c = FfcController(nuc=nuc, fps=60.0)
    dt = 1.0 / 60.0
    rate = 0.05
    t_fpa = 300.0
    at: dict[int, float] = {}
    for i in range(1, int(round(600.0 * 60.0)) + 1):
        t_fpa += rate * dt
        value = c.update_delta_t(t_fpa, dt)
        seconds = int(round(i * dt))
        if abs(i * dt - seconds) < dt / 2 and seconds in (180, 540):
            at[seconds] = value

    assert at[540] / at[180] <= 1.5
    # and it really has saturated near r * tau, rather than merely being slow
    assert at[540] == pytest.approx(rate * nuc.shutterless_tau_s, rel=0.05)
    # The uncorrected comparison: a shuttered core that never fired would be at 3x.
    assert (540.0 * rate) / (180.0 * rate) == pytest.approx(3.0)


def test_shutterless_needs_a_positive_time_constant() -> None:
    nuc = _nuc(mode="shutterless").model_copy(update={"shutterless_tau_s": 0.0})
    with pytest.raises(ValueError, match="shutterless_tau_s"):
        FfcController(nuc=nuc, fps=60.0)


# -- guards ----------------------------------------------------------------------------------


def test_a_non_positive_frame_rate_is_refused() -> None:
    for fps in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="fps must be positive"):
            FfcController(nuc=_nuc(), fps=fps)
