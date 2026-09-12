"""Microbolometer thermal time constant (M9.1): the exponential step response against the closed
form, the sampled IIR against an RK4 integration of the membrane ODE, the trailing decay length of
a moving edge, the absence of any tail on the photon path, and float32 state after 1000 steps."""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig
from irsim.detector import PhotonParams, fpa_params_from_config, photoelectrons
from irsim.detector.lowpass import (
    BolometerLowPass,
    alpha_for,
    responsivity_rolloff,
    trailing_decay_length_px,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())

# §9.2 reference core: tau_th = 10 ms at 60 Hz, so dt/tau = 1.6667 and one frame reaches 81 %.
TAU_S = 0.010
DT_S = 1.0 / 60.0


def test_step_response_matches_the_closed_form() -> None:
    """Settled at S_0 = 0, the response to a unit step is 1 - e^{-k dt/tau} at frame k to 1e-6.

    This is the test that would fail if the IIR used dt/tau in place of 1 - e^{-dt/tau} (a
    linearisation that is 20 % wrong at this dt/tau), or applied the blend weight to the wrong
    term. k = 1 pins the 0.8111 the spec's "0.6 frames" phrasing obscures (spec issue S8).
    """
    f = BolometerLowPass(TAU_S)
    f.step(np.zeros((4, 4), np.float32), DT_S)  # settle at zero
    ratio = DT_S / TAU_S
    assert ratio == pytest.approx(1.66667, abs=1e-4)
    for k in range(1, 11):
        out = f.step(np.ones((4, 4), np.float32), DT_S)
        expected = -np.expm1(-k * ratio)
        assert np.allclose(out, expected, atol=1e-6), (k, out[0, 0], expected)
    assert -np.expm1(-1.0 * ratio) == pytest.approx(0.8111, abs=1e-4)


def test_alpha_and_rolloff_limits() -> None:
    """alpha -> dt/tau as the frame gets short and -> 1 as it gets long; R(f) is the -3 dB form."""
    assert alpha_for(1e-9, TAU_S) == pytest.approx(1e-9 / TAU_S, rel=1e-6)
    assert alpha_for(100.0, TAU_S) == pytest.approx(1.0, abs=1e-12)
    assert responsivity_rolloff(0.0, TAU_S) == pytest.approx(1.0)
    # the half-power point sits at f = 1/(2 pi tau)
    assert responsivity_rolloff(1.0 / (2 * np.pi * TAU_S), TAU_S) == pytest.approx(
        1 / np.sqrt(2.0), rel=1e-12
    )
    decade = responsivity_rolloff(np.array([10.0, 100.0]) / (2 * np.pi * TAU_S), TAU_S)
    assert decade[0] / decade[1] == pytest.approx(10.0, rel=1e-2), "-20 dB/decade far above f_c"
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            alpha_for(DT_S, bad)
    with pytest.raises(ValueError):
        responsivity_rolloff(-1.0, TAU_S)


def test_matches_rk4_of_the_membrane_ode() -> None:
    """Over 100 frames of a varying scene the sampled IIR tracks an RK4 integration of
    C_th dDT/dt = alpha_abs Phi - G_th DT to 1e-4.

    The IIR is the exact zero-order-hold solution, so any disagreement beyond RK4's own error
    means the update, the sign of the exponent or the state handling is wrong. Driving it with a
    varying input is what makes this stronger than the step test: a filter that merely settled to
    the right steady state would still pass a step.
    """
    rng = np.random.default_rng(20260912)
    drive = rng.uniform(0.0, 1.0, size=100).astype(np.float64)

    f = BolometerLowPass(TAU_S)
    f.step(np.zeros((1, 1), np.float32), DT_S)
    iir = np.array([float(f.step(np.full((1, 1), d, np.float32), DT_S)[0, 0]) for d in drive])

    # RK4 on dDT/dt = (Phi - DT)/tau with Phi held constant across each frame (zero-order hold)
    substeps = 200
    h = DT_S / substeps
    y = 0.0
    rk4 = np.empty_like(drive)
    for n, phi in enumerate(drive):
        for _ in range(substeps):
            k1 = (phi - y) / TAU_S
            k2 = (phi - (y + 0.5 * h * k1)) / TAU_S
            k3 = (phi - (y + 0.5 * h * k2)) / TAU_S
            k4 = (phi - (y + h * k3)) / TAU_S
            y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        rk4[n] = y

    assert np.max(np.abs(iir - rk4)) < 1e-4, float(np.max(np.abs(iir - rk4)))


def test_moving_edge_trailing_decay_length() -> None:
    """A bar moving at v px/frame leaves a tail decaying with length v*tau/dt, within 2 %.

    This is the phenomenology M9.1 exists for -- the visible edge trailing of an uncooled core --
    and it is measured from the rendered trail rather than asserted from the formula, so a filter
    that produced the right step response but smeared in the wrong direction would fail.
    """
    width, bar, v, frames = 240, 40, 4, 40
    expected = trailing_decay_length_px(v, DT_S, TAU_S)
    assert expected == pytest.approx(v * TAU_S / DT_S, rel=1e-12)

    f = BolometerLowPass(TAU_S)
    f.step(np.zeros((1, width), np.float32), DT_S)
    x = np.arange(width)
    out = np.zeros((1, width), np.float32)
    for n in range(frames):
        pos = 60 + v * n
        scene = ((x < pos) & (x >= pos - bar)).astype(np.float32)[None, :]
        out = f.step(scene, DT_S)

    trailing_edge = 60 + v * (frames - 1) - bar
    # the tail is a staircase: the v pixels vacated in the same frame share a value, so sample one
    # pixel per frame group (d = v m) rather than every pixel
    d = v * np.arange(1, 6)
    tail = out[0, trailing_edge - d]
    assert np.all(tail > 1e-4) and np.all(np.diff(tail) < 0.0), "the tail must decay monotonically"
    slope = np.polyfit(d, np.log(tail.astype(np.float64)), 1)[0]
    measured = -1.0 / slope
    assert measured == pytest.approx(expected, rel=0.02), (measured, expected)

    ahead = out[0, 60 + v * (frames - 1) + 2]
    assert ahead < tail[-1], "the smear trails the motion; it must not lead it"


def test_photon_path_has_no_tail() -> None:
    """A photon detector's transfer is memoryless: the same flux gives the same electrons whatever
    preceded it (§15 T3 -- LWIR smears, cooled MWIR does not)."""
    d = copy.deepcopy(BOSON)
    d["sensor"]["band"].update(lambda_min_um=3.0, lambda_max_um=5.0, regime="mixed")
    d["sensor"]["fpa"] = {
        "type": "photon",
        "width": 8,
        "height": 8,
        "pitch_um": 15.0,
        "fill_factor": 1.0,
        "frame_rate_hz": 30,
        "bit_depth": 14,
        "quantum_efficiency": 0.7,
        "well_capacity_e": 1.0e7,
        "integration_time_ms": 5.0,
        "dark_current_model": "arrhenius",
    }
    params: Any = fpa_params_from_config(SensorConfig.model_validate(d))
    assert isinstance(params, PhotonParams)

    cold = np.full((8, 8), 2.0e18)  # photon-form Lb_q (photons s^-1 m^-2 sr^-1), not energy form
    hot = np.full((8, 8), 2.0e20)
    first = photoelectrons(cold, params, 1.4, 0.9)
    photoelectrons(hot, params, 1.4, 0.9)  # a hot frame in between must leave nothing behind
    assert np.array_equal(photoelectrons(cold, params, 1.4, 0.9), first)


def test_state_stays_float32_and_shape_checked() -> None:
    f = BolometerLowPass(TAU_S)
    frame = np.full((6, 5), 300.0, np.float32)
    for _ in range(1000):
        out = f.step(frame, DT_S)
    assert f.state is not None and f.state.dtype == np.float32
    assert out.dtype == np.float32
    assert np.allclose(out, 300.0, atol=1e-3), "a constant scene must not drift over 1000 frames"

    # float64 input is accepted (better than float32) but float16 is refused outright
    assert f.step(np.full((6, 5), 300.0, np.float64), DT_S).dtype == np.float32
    with pytest.raises(TypeError, match="float16"):
        f.step(np.full((6, 5), 300.0, np.float16), DT_S)
    with pytest.raises(TypeError):
        f.step(np.zeros((6, 5), np.int32), DT_S)
    with pytest.raises(ValueError, match="reset"):
        f.step(np.zeros((3, 3), np.float32), DT_S)

    f.reset()
    assert f.step(np.zeros((3, 3), np.float32), DT_S).shape == (3, 3), "reset re-adopts any shape"
    with pytest.raises(ValueError):
        BolometerLowPass(0.0)


def test_settled_start_has_no_opening_transient() -> None:
    """The first frame adopts its input rather than ramping from zero: a camera already staring at
    the scene is in equilibrium with it, and an artificial first-frame ramp would otherwise appear
    in every golden fixture and every dataset's opening frame."""
    f = BolometerLowPass(TAU_S)
    scene = np.full((4, 4), 42.0, np.float32)
    assert np.array_equal(f.step(scene, DT_S), scene)
    assert np.array_equal(f.step(scene, DT_S), scene)
