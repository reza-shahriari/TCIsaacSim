"""n/k tables and the band-effective Fresnel emissivity built on them (M7.5, MM.1).

The material under test is water, because it is the one the maritime lane rests on and the one
§4.2 singles out as impossible to model with a constant emissivity. The tests that matter here are
the ones about *reduction*: that the band average is not a formality, that interpolation happens on
the data rather than on something derived from it, and that a table is never quietly extrapolated.

docs/physics-model.md §4.2, §12.3; ADR 0041, ADR 0079
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.fresnel import directional_emissivity_fresnel
from irsim.materials.nk import (
    band_directional_emissivity,
    fresnel_from_table,
    load_nk_table,
)
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_RESPONSE = REPO / "data" / "spectra" / "responses" / "boson_vox.csv"

#: Segelstein's own row at 10.0 µm. A grid point, so `at` must return it exactly, not nearly.
WATER_10UM = (1.193164, 0.05079)


@pytest.fixture(scope="module")
def water():
    return load_nk_table("water")


@pytest.fixture(scope="module")
def boson():
    return load_spectral_response(BOSON_RESPONSE)


def test_the_water_table_loads_with_provenance(water) -> None:
    """A table with no `# source:` is unusable, so the loader treats provenance as data."""
    assert "Segelstein" in water.source
    assert "1981" in water.source
    # EXTENDED 2026-09-15 from 2.0 um down to 0.65 um, re-truncated from the same Segelstein
    # download, because a maritime scene in SWIR or NIR needs water's reflectance over 0.7-1.8 um
    # and the loader refuses to extrapolate. The 365 rows at and above 2.0 um are unchanged.
    assert water.support_um == (0.6501, 15.6)
    assert water.wavelength_um.size == 535
    assert np.all(np.diff(water.wavelength_um) > 0)
    assert np.all(water.n > 0) and np.all(water.k >= 0)


def test_a_table_without_a_source_header_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "nameless.csv"
    path.write_text("wavelength_um,n,k\n8.0,1.3,0.03\n12.0,1.1,0.13\n")
    with pytest.raises(ValueError, match="source"):
        load_nk_table(str(path))


def test_grid_points_come_back_exactly(water) -> None:
    n, k = water.at(10.0)
    assert float(n) == pytest.approx(WATER_10UM[0], abs=1e-9)
    assert float(k) == pytest.approx(WATER_10UM[1], abs=1e-12)


def test_n_and_k_are_interpolated_separately(water) -> None:
    """Interpolating ε or R instead would make the answer depend on the table's sampling.

    Halfway between two rows must give the mean of n and the mean of k — which is *not* the same
    as the mean of the two emissivities, and this test pins the difference so a future refactor
    cannot quietly switch to interpolating the derived quantity.
    """
    i = int(np.searchsorted(water.wavelength_um, 10.5))
    lo, hi = water.wavelength_um[i], water.wavelength_um[i + 1]
    mid = 0.5 * (lo + hi)

    n_mid, k_mid = water.at(mid)
    assert float(n_mid) == pytest.approx(0.5 * (water.n[i] + water.n[i + 1]), abs=1e-12)
    assert float(k_mid) == pytest.approx(0.5 * (water.k[i] + water.k[i + 1]), abs=1e-12)

    eps_mid = float(directional_emissivity_fresnel(n_mid, k_mid, 1.0))
    eps_mean = 0.5 * (
        float(directional_emissivity_fresnel(water.n[i], water.k[i], 1.0))
        + float(directional_emissivity_fresnel(water.n[i + 1], water.k[i + 1], 1.0))
    )
    assert eps_mid != pytest.approx(eps_mean, abs=1e-9)


def test_outside_the_table_raises_rather_than_extrapolating(water) -> None:
    """20 µm from a 15.6 µm table is a fabricated number, and fabricated numbers propagate.

    This refusal is not theoretical: it is what stopped a maritime SWIR render outright rather
    than letting it invent water's optical constants, and extending the table from the published
    source was the fix. The bound below the band moved from 2.0 µm to 0.65 µm; the behaviour did
    not.
    """
    with pytest.raises(ValueError, match="outside the table"):
        water.at(20.0)
    with pytest.raises(ValueError, match="outside the table"):
        water.at(0.4)
    # ...and the range a reflective band actually needs is inside it now.
    for lam in (0.75, 1.0, 1.55, 1.8):
        n, k = water.at(lam)
        assert 1.29 < float(n) < 1.34, (lam, n)
        assert 0.0 <= float(k) < 1e-3, (lam, k)


def test_fresnel_from_table_agrees_with_the_direct_call(water) -> None:
    mu = np.cos(np.radians([0.0, 45.0, 80.0]))
    from_table = fresnel_from_table(water, 10.0, mu)[0]
    direct = 1.0 - directional_emissivity_fresnel(*WATER_10UM, mu)
    assert np.allclose(from_table, direct, atol=1e-9)


def test_band_effective_water_emissivity_over_the_boson_band(water, boson) -> None:
    """ε_B(θ) for water: near-blackbody at nadir, collapsing past 70°.

    The nadir value is the checkable one — published sea-surface emissivities in the LWIR sit at
    0.98–0.99 looking straight down, and this lands inside that.
    """
    angles = np.array([0.0, 30.0, 60.0, 70.0, 80.0, 85.0, 89.0])
    eps = band_directional_emissivity(water, boson, np.cos(np.radians(angles)))

    assert eps.shape == angles.shape
    assert eps[0] == pytest.approx(0.988, abs=0.003)
    assert np.all(np.diff(eps) < 0.0)  # monotone, no bumps
    assert eps[4] == pytest.approx(0.673, abs=0.01)  # 80°
    assert eps[-1] < 0.15  # 89°: a mirror


def test_the_band_average_is_not_a_formality(water, boson) -> None:
    """Band-effective ε differs from Fresnel-at-10-µm by enough to matter, and worst at angle.

    Over the Boson band water's n runs 1.28 → 1.16 and k spans an order of magnitude, so a single
    "representative wavelength" is not a cheap approximation of the band value — at 80° the gap is
    0.039, which against a 60 K sea-to-sky contrast is more than 2 K of apparent temperature.
    """
    mu = np.cos(np.radians([0.0, 80.0]))
    band = band_directional_emissivity(water, boson, mu)
    mono = 1.0 - fresnel_from_table(water, 10.0, mu)[0]

    assert abs(band[0] - mono[0]) > 0.002
    assert abs(band[1] - mono[1]) > 0.03
    assert np.all(band < mono)  # the band average pulls ε down, at every angle


def test_hemispherical_emissivity_brackets_the_spec_table_value(water, boson) -> None:
    """2∫ε(θ)cosθ sinθ dθ = 0.945, below the normal 0.988 — §16.2's 0.96 sits between them.

    This is the §16.2 scalar-ambiguity ADR 0043 exists to resolve: a single tabulated 0.96 for
    water is neither the normal nor the hemispherical value, and using it as either is a real
    error in the energy balance.
    """
    from numpy.polynomial.legendre import leggauss

    x, w = leggauss(64)
    theta = 0.25 * np.pi * (x + 1.0)
    weight = w * 0.25 * np.pi
    eps = band_directional_emissivity(water, boson, np.cos(theta))
    hemi = 2.0 * np.sum(weight * eps * np.cos(theta) * np.sin(theta))

    assert hemi == pytest.approx(0.945, abs=0.005)
    assert hemi < float(band_directional_emissivity(water, boson, 1.0))
    assert 0.93 < hemi < 0.96


def test_a_response_wider_than_the_table_is_refused(water, boson, tmp_path) -> None:
    """Averaging over held end values would be inventing optical constants."""
    narrow = tmp_path / "narrow.csv"
    narrow.write_text(
        "# source: a deliberately short table for this test\n"
        "wavelength_um,n,k\n8.0,1.29,0.034\n9.0,1.24,0.040\n"
    )
    short = load_nk_table(str(narrow))
    with pytest.raises(ValueError, match="only covers"):
        band_directional_emissivity(short, boson, 1.0)
