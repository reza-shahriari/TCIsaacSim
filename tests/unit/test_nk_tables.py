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


# --- the two proxy tables (M7.5: glass and the paint proxy) ---------------------------------------

#: Franta's own row nearest 10.0 µm, and Zhang's. These pin the *data*, not the code: a re-fetch
#: that silently picked up a different dataset from the same database changes these numbers.
GLASS_10UM = (2.526835, 0.082695)
PAINT_10UM = (1.499792, 0.042477)

#: §12.1's nominal band ranges. A table that stops inside one of them makes that band raise rather
#: than extrapolate, so coverage is the property worth asserting.
BAND_SPAN_UM = (0.75, 13.5)


@pytest.fixture(scope="module")
def glass():
    return load_nk_table("glass")


@pytest.fixture(scope="module")
def paint_proxy():
    return load_nk_table("paint_proxy")


@pytest.mark.parametrize(("name", "expected"), [("glass", GLASS_10UM), ("paint_proxy", PAINT_10UM)])
def test_the_proxy_tables_carry_the_dataset_they_claim(name, expected) -> None:
    """A check value from each source dataset, at the wavelength its header quotes.

    The tables are fetched from a third-party database by
    ``scripts/fetch_nk_tables.py``, so what needs pinning is not arithmetic but *identity*: which
    dataset is in the file. Several silica and PMMA datasets in the same database disagree about k
    at 10 µm by a factor of five, and picking up the wrong one would change every angular
    emissivity in the band while still looking like plausible glass.
    """
    table = load_nk_table(name)
    n, k = table.at(10.0)
    assert float(n) == pytest.approx(expected[0], abs=5e-3)
    assert float(k) == pytest.approx(expected[1], rel=2e-2)


@pytest.mark.parametrize("name", ["glass", "paint_proxy"])
def test_a_proxy_table_declares_that_it_is_a_proxy(name) -> None:
    """Neither substance is what the material is made of, and the file has to say so.

    ADR 0041 treats the ``# source:`` header as data. For these two it carries a second claim
    beyond provenance -- that fused silica stands in for soda-lime and PMMA for a clearcoat -- and
    a reader who does not know that would take a band emissivity from the table as measured truth
    for the material named in the library.
    """
    table = load_nk_table(name)
    assert "PROXY" in table.source, table.source
    assert "refractiveindex.info" in table.source.lower()


@pytest.mark.parametrize("name", ["glass", "paint_proxy"])
def test_the_proxy_tables_span_every_configured_band(name) -> None:
    """Coverage, not accuracy: the loader refuses to extrapolate, so a short table raises."""
    table = load_nk_table(name)
    lo, hi = table.support_um
    assert lo <= BAND_SPAN_UM[0] and hi >= BAND_SPAN_UM[1], (lo, hi)
    n, k = table.at(np.linspace(*BAND_SPAN_UM, 64))
    assert np.all(np.isfinite(n)) and np.all(np.isfinite(k))
    # Not `n > 1`: silica's n dips to 0.35 inside its reststrahlen band, which is physics, not a
    # bad row -- see `test_the_glass_table_carries_the_reststrahlen_band`.
    assert np.all(n > 0.0) and np.all(k >= 0.0)


def test_glass_and_the_paint_proxy_are_different_materials(glass, paint_proxy, boson) -> None:
    """Both are transparent dielectrics in the visible; only one has a reststrahlen band.

    This is what makes glass worth a table of its own. Silica's Si-O stretch drives n to ~2.5 and
    k to ~0.08 across the LWIR, so its emissivity falls away from normal several times faster than
    the polymer's. A single proxy shared by both would flatten exactly the feature the angular
    model exists to reproduce.
    """
    cos_70 = float(np.cos(np.radians(70.0)))
    glass_shape = float(band_directional_emissivity(glass, boson, cos_70)) / float(
        band_directional_emissivity(glass, boson, 1.0)
    )
    paint_shape = float(band_directional_emissivity(paint_proxy, boson, cos_70)) / float(
        band_directional_emissivity(paint_proxy, boson, 1.0)
    )
    assert glass_shape < paint_shape - 0.05, (glass_shape, paint_shape)
    assert 0.7 < glass_shape < 0.82, glass_shape
    assert 0.84 < paint_shape < 0.90, paint_shape


def test_the_glass_table_carries_the_reststrahlen_band(glass) -> None:
    """Silica's Si-O stretch, the feature that makes glass unlike every other dielectric here.

    Across the resonance near 9 µm the real index falls *below one* (to 0.35) while k rises above
    1.6 -- the medium responds like a metal over a narrow band, and reflectance there reaches ~0.7
    where the material is otherwise ε ≈ 0.9. It sits inside the LWIR window, so it is not a
    curiosity: it is why the LWIR angular shape differs from the other three bands, and why a
    table that had been smoothed or resampled onto a coarse grid would be the wrong table.
    """
    inside = (glass.wavelength_um >= 8.0) & (glass.wavelength_um <= 10.0)
    assert inside.sum() > 20, "the band is under-sampled; check the fetch truncation"
    assert float(glass.n[inside].min()) < 0.6
    assert float(glass.k[inside].max()) > 1.2
    n, k = glass.at(8.798)
    reflectance = ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)
    assert 0.6 < float(reflectance) < 0.8, float(reflectance)


def test_the_paint_proxy_has_no_such_band(paint_proxy) -> None:
    """The polymer has absorption bands here too, but they never turn it metallic.

    PMMA's C-O stretch puts k at 0.33 near 8.7 µm -- not negligible -- yet n stays between 1.36
    and 1.71 across the window, where silica's runs 0.35 to 2.5. It is the excursion of **n**, not
    the presence of absorption, that separates a reststrahlen band from an ordinary vibrational
    one, and it is what drives the two materials' angular shapes apart. Stated as its own test
    because an accidental swap of the two files would otherwise pass everything above.
    """
    inside = (paint_proxy.wavelength_um >= 8.0) & (paint_proxy.wavelength_um <= 10.0)
    assert float(paint_proxy.n[inside].min()) > 1.3
    assert float(paint_proxy.n[inside].max()) < 2.0
    assert float(paint_proxy.k[inside].max()) < 0.5
