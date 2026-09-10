"""R(λ) file contract (ADR 0009) and the Band object.

The loader's job is to make unit and normalisation mistakes impossible to load: nanometres,
area-normalised or QE-scaled responses all produce plausible-looking arrays and wrong radiance.
Each such file must raise; an exact top-hat file must reproduce the closed-form band radiance.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import DEFAULT_DATA_DIR, load_sensor_config
from irsim.radiometry.band import EDGE_TOLERANCE_UM, Band
from irsim.radiometry.planck import band_radiance_tophat, spectral_radiance
from irsim.radiometry.spectral_response import RESAMPLE_DL_UM, load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_CSV = DEFAULT_DATA_DIR / "spectra" / "responses" / "boson_vox.csv"
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"


def _write(
    tmp_path: pathlib.Path, name: str, lam: np.ndarray, r: np.ndarray, header: str = ""
) -> pathlib.Path:
    p = tmp_path / name
    body = "\n".join(f"{x:.6f},{y:.9f}" for x, y in zip(lam, r, strict=True))
    p.write_text(header + body + "\n")
    return p


def _tophat(lo: float, hi: float, dl: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    lam = np.round(np.arange(lo, hi + 1e-9, dl), 6)
    return lam, np.ones_like(lam)


def test_exact_tophat_file_integrates_to_closed_form(tmp_path: pathlib.Path) -> None:
    """∫R·B dλ on the file grid == band_radiance_tophat within 0.1 % -- fails if the loader read
    nm, a per-nm response, or area-normalised the shape."""
    lam, r = _tophat(8.0, 12.0)
    sr = load_spectral_response(_write(tmp_path, "tophat.csv", lam, r, "# exact top-hat\n"))
    for T in (300.0, 500.0):
        quad = np.trapezoid(sr.response * spectral_radiance(sr.wavelength_um, T), sr.wavelength_um)
        closed = band_radiance_tophat(8.0, 12.0, T)
        assert abs(quad / closed - 1.0) < 1e-3, f"T={T}: {quad} vs {closed}"


def test_resampled_tophat_integral_equals_width(tmp_path: pathlib.Path) -> None:
    lam, r = _tophat(8.0, 12.0)
    sr = load_spectral_response(_write(tmp_path, "tophat.csv", lam, r))
    grid = np.arange(7.0, 13.0 + 1e-9, RESAMPLE_DL_UM)
    assert abs(np.trapezoid(sr.resampled(grid), grid) - 4.0) < 1e-12
    assert sr.integral_um() == pytest.approx(4.0, abs=1e-12)
    assert np.all(sr.resampled(np.array([7.5, 12.5])) == 0.0), "zero outside support"


def test_rejects_nanometre_file(tmp_path: pathlib.Path) -> None:
    lam = np.arange(7500.0, 13500.0 + 1, 10.0)
    with pytest.raises(ValueError, match="MICROMETRES"):
        load_spectral_response(_write(tmp_path, "nm.csv", lam, np.ones_like(lam)))


def test_rejects_area_normalised_response(tmp_path: pathlib.Path) -> None:
    lam, r = _tophat(8.0, 12.0)
    r = r / np.trapezoid(r, lam)  # integral 1, peak 0.25
    with pytest.raises(ValueError, match="peak"):
        load_spectral_response(_write(tmp_path, "area.csv", lam, r))


def test_rejects_qe_scaled_response(tmp_path: pathlib.Path) -> None:
    """A curve with peak 0.8 is a QE-scaled response; QE lives in the config, not the file."""
    lam, r = _tophat(0.9, 1.7)
    with pytest.raises(ValueError, match="quantum_efficiency"):
        load_spectral_response(_write(tmp_path, "qe.csv", lam, 0.8 * r))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda lam, r: (lam[::-1], r), "increasing"),
        (lambda lam, r: (lam, np.where(lam > 10, 1.2, r)), r"\[0, 1\]"),
        (lambda lam, r: (lam, np.where(lam < 9, -0.01, r)), r"\[0, 1\]"),
        (lambda lam, r: (lam, np.where(lam < 9, np.nan, r)), "NaN"),
    ],
)
def test_rejects_malformed_tables(tmp_path: pathlib.Path, mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    lam, r = _tophat(8.0, 12.0, 0.1)
    lam2, r2 = mutate(lam, r)
    with pytest.raises(ValueError, match=match):
        load_spectral_response(_write(tmp_path, "bad.csv", lam2, r2))


def test_rejects_three_columns_and_single_row(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "three.csv"
    p.write_text("8.0,1.0,0.5\n9.0,1.0,0.5\n")
    with pytest.raises(ValueError, match="two columns"):
        load_spectral_response(p)
    p.write_text("wavelength_um,response\n10.0,1.0\n")
    with pytest.raises(ValueError, match="two samples"):
        load_spectral_response(p)


def test_header_and_provenance_are_parsed(tmp_path: pathlib.Path) -> None:
    lam, r = _tophat(8.0, 12.0, 0.5)
    p = _write(
        tmp_path, "h.csv", lam, r, "# source: bench X\n# MEASURED 2026\nwavelength_um,response\n"
    )
    sr = load_spectral_response(p)
    assert sr.provenance == ("source: bench X", "MEASURED 2026")
    assert sr.sha256 and sr.source_path == str(p.resolve())
    assert sr.wavelength_um.dtype == np.float64 and sr.response.dtype == np.float64


# --- the committed Boson estimate ---------------------------------------------------------


def test_boson_file_loads_and_half_power_points_match_yaml_edges() -> None:
    sr = load_spectral_response(BOSON_CSV)
    assert any("ESTIMATED" in line for line in sr.provenance), "must be marked as an estimate"
    lo, hi = sr.half_power_points_um()
    assert abs(lo - 7.5) < 0.2 and abs(hi - 13.5) < 0.2, (lo, hi)
    assert sr.response.max() == 1.0
    assert np.all(np.diff(sr.wavelength_um) > 0)
    assert abs(np.diff(sr.wavelength_um).max() - RESAMPLE_DL_UM) < 1e-9
    # symmetric raised-cosine edges centred on the band edges integrate to the top-hat width
    assert sr.integral_um() == pytest.approx(6.0, abs=1e-6)


def test_band_from_boson_spec_and_edge_mismatch_detection(tmp_path: pathlib.Path) -> None:
    cfg = load_sensor_config(BOSON_YAML)
    band = Band.from_spec(cfg.sensor.band)
    assert band.id == "lwir" and band.regime == "emissive"
    assert (band.lambda_min_um, band.lambda_max_um) == (7.5, 13.5)
    assert band.response.sha256 == load_spectral_response(BOSON_CSV).sha256
    # attach an MWIR response to the LWIR spec: the half-power points disagree by > 0.5 um
    lam, r = _tophat(3.0, 5.0, 0.05)
    with pytest.raises(ValueError, match="disagree"):
        Band.from_spec(cfg.sensor.band, _write(tmp_path, "mwir.csv", lam, r))
    assert EDGE_TOLERANCE_UM == 0.5
