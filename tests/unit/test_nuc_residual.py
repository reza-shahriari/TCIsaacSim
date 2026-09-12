"""NUC residual (M9.6): what survives the two-point correction, and in which units.

§11.2 says "model the residual, not the ideal". The residual is what makes a shutterless core
drift and a shuttered one need an FFC at all, so the tests pin its three defining behaviours:

* it is **zero at ΔT_FPA = 0** by construction, not by cancellation -- a camera that has just
  shuttered is perfectly corrected;
* it is **linear in ΔT_FPA**, so the offset std at 2 K is exactly twice the 1 K value and lands on
  the configured millikelvin figure;
* it is applied in **DN, not kelvin** (non-negotiable #3, ADR 0056). The headline test for this is
  ``test_residual_not_kelvin_flat``: a fixed DN error is a *smaller* apparent-temperature error at
  373 K than at 300 K, in the ratio of the thermal derivatives, because ∂L/∂T rises with
  temperature. A residual added in kelvin would be flat across the range and would fail it.

And the FFC signature: a reset draws a new realisation, so the pattern after the event is
uncorrelated with the pattern before -- the fixed pattern is replaced, not faded.

docs/physics-model.md §11.2, §2, §12.2. ADR 0053, ADR 0056.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.sensor import SensorConfig, SensorSpec
from irsim.isp import RadiometricCalibration, apparent_temperature, dn_per_kelvin
from irsim.noise import RESIDUAL_REFERENCE_K, NucResidual
from irsim.radiometry.lut import BandLUT

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())
SEED = 4242


@pytest.fixture(scope="module")
def sensor() -> SensorSpec:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=64, height=48)
    return SensorConfig.model_validate(d).sensor


@pytest.fixture(scope="module")
def cal(sensor: SensorSpec, tophat_lwir_lut: BandLUT) -> RadiometricCalibration:
    return RadiometricCalibration.from_scene_range(sensor, tophat_lwir_lut, 233.15, 473.15, 300.0)


@pytest.fixture(scope="module")
def dn_per_k(cal: RadiometricCalibration, tophat_lwir_lut: BandLUT) -> float:
    return dn_per_kelvin(cal, tophat_lwir_lut, RESIDUAL_REFERENCE_K)


@pytest.fixture
def residual(sensor: SensorSpec, dn_per_k: float) -> NucResidual:
    return NucResidual(nuc=sensor.nuc, shape=sensor.fpa_shape, dn_per_k=dn_per_k, sensor_seed=SEED)


def _signal_at(cal: RadiometricCalibration, lut: BandLUT, t_k: float) -> np.ndarray:
    radiance = np.full(cal.sensor.fpa_shape, float(lut.lookup(t_k)[()]), dtype=np.float32)
    return np.asarray(cal.signal_from_radiance(radiance))


# -- zero at ΔT = 0 --------------------------------------------------------------------------


def test_a_freshly_shuttered_camera_has_no_residual(residual: NucResidual) -> None:
    """ΔT_FPA = 0 -> gain ≡ 1 and offset ≡ 0, exactly. Not approximately, not on average."""
    assert np.all(residual.gain(0.0) == 1.0)
    assert np.all(residual.offset_dn(0.0) == 0.0)
    signal = np.full((48, 64), 12345.0, dtype=np.float32)
    assert np.array_equal(residual.apply(signal, 0.0), signal)


# -- linearity in ΔT_FPA ---------------------------------------------------------------------


def test_offset_std_is_the_configured_millikelvin_figure(
    residual: NucResidual, sensor: SensorSpec, dn_per_k: float
) -> None:
    """45 mK/K at ΔT = 2 K is 90 mK, within 2 %, and exactly twice the 1 K value."""
    assert sensor.nuc.residual_offset_mk_per_k == 45.0

    sigma_mk_2 = float(residual.offset_dn(2.0).std()) / dn_per_k * 1e3
    sigma_mk_1 = float(residual.offset_dn(1.0).std()) / dn_per_k * 1e3
    assert sigma_mk_2 == pytest.approx(90.0, rel=0.02)
    assert sigma_mk_2 == pytest.approx(2.0 * sigma_mk_1, rel=1e-6)
    assert residual.offset_sigma_mk(2.0) == pytest.approx(90.0, rel=1e-12)


@pytest.mark.parametrize("delta_t", [0.5, 1.0, 3.0, -2.0])
def test_the_residual_is_linear_in_delta_t(residual: NucResidual, delta_t: float) -> None:
    """Both terms scale with ΔT_FPA and nothing else; a negative ΔT flips the pattern's sign."""
    unit_gain = residual.gain(1.0) - 1.0
    unit_offset = residual.offset_dn(1.0)
    # The gain is stored as 1 + eps in float32, so recovering eps by subtracting 1 costs about
    # one float32 ulp at 1.0 (1.2e-7) each time. The absolute tolerance is set from that rather
    # than guessed; the offset carries no such cancellation and is held far tighter.
    assert np.allclose(residual.gain(delta_t) - 1.0, delta_t * unit_gain, rtol=1e-3, atol=2e-6)
    assert np.allclose(residual.offset_dn(delta_t), delta_t * unit_offset, rtol=1e-5, atol=1e-9)


def test_the_gain_residual_scales_with_the_signal(residual: NucResidual) -> None:
    """g_ij multiplies, so the error is proportional to DN -- exactly, per pixel.

    Stated as a ratio between two uniform fields rather than across a ramp: the per-pixel form is
    exact, where comparing slab means of a ramp would be limited by how the |xi| sample happens to
    fall in each slab and could only support a percent-level claim about an exact property.
    """
    gain_only = NucResidual(
        nuc=residual.nuc.model_copy(update={"residual_offset_mk_per_k": 0.0}),
        shape=residual.shape,
        dn_per_k=residual.dn_per_k,
        sensor_seed=SEED,
    )
    low = np.full(residual.shape, 10_000.0, dtype=np.float32)
    high = np.full(residual.shape, 30_000.0, dtype=np.float32)
    err_low = gain_only.apply(low, 3.0).astype(np.float64) - low
    err_high = gain_only.apply(high, 3.0).astype(np.float64) - high
    # atol is float32's spacing at DN 3e4 (about 2e-3), a few times over: the error is
    # recovered by subtracting two numbers of that magnitude, so it cannot be resolved
    # more finely than the representation allows.
    assert np.allclose(err_high, 3.0 * err_low, rtol=1e-4, atol=1e-2)
    assert float(np.abs(err_low).mean()) > 0.0


def test_the_offset_residual_does_not_scale_with_the_signal(residual: NucResidual) -> None:
    """The control for the test above: o_ij is additive, so the same error appears at any level."""
    offset_only = NucResidual(
        nuc=residual.nuc.model_copy(update={"residual_gain_ppm_per_k": 0.0}),
        shape=residual.shape,
        dn_per_k=residual.dn_per_k,
        sensor_seed=SEED,
    )
    low = np.full(residual.shape, 10_000.0, dtype=np.float32)
    high = np.full(residual.shape, 30_000.0, dtype=np.float32)
    err_low = offset_only.apply(low, 3.0).astype(np.float64) - low
    err_high = offset_only.apply(high, 3.0).astype(np.float64) - high
    # atol is float32's spacing at DN 3e4 (about 2e-3), a few times over: the error is
    # recovered by subtracting two numbers of that magnitude, so it cannot be resolved
    # more finely than the representation allows.
    assert np.allclose(err_high, err_low, rtol=1e-4, atol=1e-2)
    assert float(np.abs(err_low).mean()) > 0.0


# -- the headline: DN, not kelvin ------------------------------------------------------------


def test_residual_not_kelvin_flat(
    cal: RadiometricCalibration, tophat_lwir_lut: BandLUT, residual: NucResidual, dn_per_k: float
) -> None:
    """The same DN residual is a *smaller* apparent-T error at 373 K than at 300 K (within 5 %).

    ∂L/∂T rises with temperature, so a fixed DN error divides down. The predicted ratio is
    (∂DN/∂T at 300) / (∂DN/∂T at 373), and a residual that had been added in kelvin would instead
    be flat -- which is the failure non-negotiable #3 exists to prevent, and which no single-
    temperature test would reveal.
    """
    lut = tophat_lwir_lut
    offset_only = NucResidual(
        nuc=residual.nuc.model_copy(update={"residual_gain_ppm_per_k": 0.0}),
        shape=residual.shape,
        dn_per_k=dn_per_k,
        sensor_seed=SEED,
    )

    errors = {}
    for t in (300.0, 373.0):
        clean = _signal_at(cal, lut, t)
        dirty = offset_only.apply(clean.astype(np.float32), 2.0)
        t_clean = apparent_temperature(cal.radiance_from_signal(clean), lut).astype(np.float64)
        t_dirty = apparent_temperature(cal.radiance_from_signal(dirty), lut).astype(np.float64)
        errors[t] = float(np.std(t_dirty - t_clean))

    predicted = dn_per_kelvin(cal, lut, 300.0) / dn_per_kelvin(cal, lut, 373.0)
    assert errors[373.0] / errors[300.0] == pytest.approx(predicted, rel=0.05)
    assert predicted < 0.8  # the effect is large, not a rounding artefact
    # And at the reference temperature the error is the configured millikelvin figure.
    assert errors[300.0] * 1e3 == pytest.approx(90.0, rel=0.05)


def test_dn_per_kelvin_rises_with_temperature(
    cal: RadiometricCalibration, tophat_lwir_lut: BandLUT
) -> None:
    """The conversion reference is itself temperature-dependent.

    Which is the entire reason ADR 0056 exists: if dDN/dT were constant, applying the residual in
    kelvin and in DN would be the same thing and the unit question would not arise.
    """
    values = [dn_per_kelvin(cal, tophat_lwir_lut, t) for t in (250.0, 300.0, 373.0, 450.0)]
    assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))
    assert all(v > 0 for v in values)


# -- the FFC signature -----------------------------------------------------------------------


def test_ffc_reset_decorrelates_the_pattern(residual: NucResidual) -> None:
    """|r| < 0.05 between the pattern before and after the shutter: replaced, not faded."""
    before_g = residual.gain_field.copy()
    before_o = residual.offset_field.copy()
    residual.ffc_reset()
    for before, after in (
        (before_g, residual.gain_field),
        (before_o, residual.offset_field),
    ):
        r = float(np.corrcoef(before.ravel(), after.ravel())[0, 1])
        assert abs(r) < 0.05


def test_ffc_reset_preserves_the_configured_amplitude(
    residual: NucResidual, dn_per_k: float
) -> None:
    """A new epoch is a new realisation of the same distribution, within 5 %."""
    before = float(residual.offset_dn(2.0).std())
    residual.ffc_reset()
    after = float(residual.offset_dn(2.0).std())
    assert after == pytest.approx(before, rel=0.05)
    assert after / dn_per_k * 1e3 == pytest.approx(90.0, rel=0.05)


def test_the_gain_and_offset_fields_are_independent(residual: NucResidual) -> None:
    """Separate streams: a camera whose gain and offset errors were the same pattern would show a
    residual that vanishes at one signal level, which is not what a real FPA does."""
    r = float(np.corrcoef(residual.gain_field.ravel(), residual.offset_field.ravel())[0, 1])
    assert abs(r) < 0.05


def test_the_fields_are_unit_variance_and_zero_mean(residual: NucResidual) -> None:
    for xi in (residual.gain_field, residual.offset_field):
        assert float(xi.std()) == pytest.approx(1.0, rel=0.05)
        assert abs(float(xi.mean())) < 5.0 / np.sqrt(xi.size)


# -- determinism and contracts ---------------------------------------------------------------


def test_the_residual_is_deterministic_per_sensor_seed(sensor: SensorSpec, dn_per_k: float) -> None:
    def build(seed: int) -> NucResidual:
        return NucResidual(
            nuc=sensor.nuc, shape=sensor.fpa_shape, dn_per_k=dn_per_k, sensor_seed=seed
        )

    assert np.array_equal(build(SEED).gain_field, build(SEED).gain_field)
    assert not np.array_equal(build(SEED).gain_field, build(SEED + 1).gain_field)


def test_float16_and_integer_planes_are_refused(residual: NucResidual) -> None:
    with pytest.raises(TypeError, match="float16"):
        residual.apply(np.zeros(residual.shape, dtype=np.float16), 1.0)
    with pytest.raises(TypeError, match="float signal plane"):
        residual.apply(np.zeros(residual.shape, dtype=np.uint16), 1.0)  # type: ignore[arg-type]


def test_output_is_float32(residual: NucResidual) -> None:
    out = residual.apply(np.full(residual.shape, 1000.0, dtype=np.float32), 2.0)
    assert out.dtype == np.float32


def test_a_non_positive_conversion_reference_is_refused(sensor: SensorSpec) -> None:
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="dn_per_k"):
            NucResidual(nuc=sensor.nuc, shape=sensor.fpa_shape, dn_per_k=bad, sensor_seed=SEED)


def test_a_mismatched_shape_is_refused(residual: NucResidual) -> None:
    with pytest.raises(ValueError, match="!= residual shape"):
        residual.apply(np.zeros((4, 4), dtype=np.float32), 1.0)
