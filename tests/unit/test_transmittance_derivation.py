"""RP.7 — a material's transmittance either comes from its n/k table or says that it does not.

Spec issue S13 asked for glass's τ_mwir to be computed by Beer-Lambert over 5 mm of the checked-in
k(λ). Doing that shows the proposed resolution cannot be carried out: `data/nk/glass.csv` is **fused
silica** standing in for soda-lime and, by the deliberate design recorded in the material file
itself, supplies the angular *shape* while `emissivity_per_band` supplies the magnitude. Fused
silica lacks the iron and alkali that make real windscreen glass absorb in the near infrared, so the
proxy is transparent where soda-lime is not.

That leaves a number sitting next to a table that looks like its source and is not — `mwir: 0.02`
was in fact authored six commits before the table existed. The fix is not to change the number to
match the proxy; it is to make the file *declare* which of the two it is, and to hold that
declaration against the recomputation so neither can drift unnoticed.

docs/physics-model.md §4.4, §12.1; spec issues S13, S11.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from irsim.config.materials import AngularModel, FresnelAngular
from irsim.materials.library import load_material
from irsim.materials.nk import band_slab_transmittance, load_nk_table
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
MATERIALS = REPO / "configs" / "materials"
SPEC_ISSUES = REPO / "docs" / "spec-issues.md"

#: One representative response per band — the detectors the project actually ships configs for.
BAND_RESPONSE = {
    "nir": "nir_si.csv",
    "swir": "ingaas.csv",
    "mwir": "insb.csv",
    "lwir": "boson_vox.csv",
}

#: How far a *derived* transmittance may sit from its own table before it is not derived from it.
DERIVED_TOLERANCE = 0.10


def _response(band: str):
    return load_spectral_response(REPO / "data" / "spectra" / "responses" / BAND_RESPONSE[band])


def _transmissive_materials() -> list[tuple[str, object]]:
    """Materials that author a transmittance *and* carry an n/k table to check it against."""
    out = []
    for path in sorted(MATERIALS.glob("*.yaml")):
        # `configs/materials/` also holds the prim→material mapping rules, whose own entries say
        # `material: <name>` -- so the block has to be matched at the top level, not anywhere.
        if not re.search(r"^material:", path.read_text(), re.MULTILINE):
            continue
        spec = load_material(path).spec
        optical = spec.optical
        if optical.transmittance_per_band and isinstance(optical.angular_model, FresnelAngular):
            out.append((path.name, spec))
    return out


def _table_for(model: AngularModel):
    return load_nk_table(pathlib.Path(model.n_k_file).stem)


def _issue_status(ident: str) -> str:
    """The status cell of one `docs/spec-issues.md` row.

    Only the two main tables are searched. The short "coding-blocking fixes" table at the top of
    that file repeats three ids (S1, S2, T4) in a three-column form with no status, and matching
    those would silently return the wrong cell.
    """
    main = SPEC_ISSUES.read_text().split("**Physics (fix the spec):**", 1)[1]
    for line in main.splitlines():
        match = re.match(rf"^\|\s*{ident}\s*\|.*\|\s*([^|]*?)\s*\|\s*$", line)
        if match:
            return match.group(1)
    raise AssertionError(f"{ident} has no row in the main tables of {SPEC_ISSUES.name}")


# --- the invariant ------------------------------------------------------------------------------


def test_at_least_one_material_is_actually_checked() -> None:
    """A walk over an empty set passes for ever; this is what stops that."""
    assert _transmissive_materials(), "no transmissive material carries an n/k table"


def test_a_transmittance_is_derived_from_its_table_or_declares_that_it_is_not() -> None:
    for name, spec in _transmissive_materials():
        optical = spec.optical
        declared = optical.transmittance_derivation
        if declared is not None and declared.startswith("authored:"):
            issue = declared.split(":", 1)[1].strip()
            assert issue in SPEC_ISSUES.read_text(), (
                f"{name} names spec issue {issue}, which has no row in {SPEC_ISSUES.name}"
            )
            assert "open" in _issue_status(issue), (
                f"{name} defers its transmittance to {issue}, but that issue is no longer open — "
                "either derive the number or reopen the issue"
            )
            continue
        table = _table_for(optical.angular_model)
        for band, authored in optical.transmittance_per_band.items():
            derived = band_slab_transmittance(table, _response(band), spec.thermal.thickness_m)
            assert abs(derived - authored) <= DERIVED_TOLERANCE, (
                f"{name} {band}: authored {authored:.3f} against {derived:.3f} recomputed from "
                f"{optical.angular_model.n_k_file} over {spec.thermal.thickness_m * 1e3:.1f} mm. "
                "Either fix the number or set transmittance_derivation: 'authored: <issue>'."
            )


# --- S13 itself ---------------------------------------------------------------------------------


def _glass():
    return load_material(MATERIALS / "glass_windshield.yaml").spec


def test_glass_declares_its_transmittance_authored_against_an_open_issue() -> None:
    assert _glass().optical.transmittance_derivation == "authored: S13"
    assert "open" in _issue_status("S13")


@pytest.mark.parametrize(
    ("band", "expected"),
    [("nir", 0.9347), ("swir", 0.9359), ("mwir", 0.1090), ("lwir", 0.0000)],
)
def test_what_the_proxy_table_actually_says(band: str, expected: float) -> None:
    """Pinned, so a change to the table or to the authored 5 mm cannot pass unnoticed.

    These are the numbers S13's row quotes. If one moves, the row is wrong.
    """
    got = band_slab_transmittance(load_nk_table("glass"), _response(band), 0.005)
    assert got == pytest.approx(expected, abs=5e-4)


def test_only_lwir_is_reproduced_and_the_near_infrared_is_not() -> None:
    """The one band the proxy earns its keep in, and the three where it cannot supply a magnitude.

    In LWIR the Si-O reststrahlen band makes fused silica and soda-lime both opaque, which is why
    the table is usable for the angular shape at all (ADR 0042). Shortwards the proxy has none of
    soda-lime's iron, so it transmits where a real windscreen absorbs — the gap this test records
    is the evidence S13 stays open on.
    """
    spec = _glass()
    authored = spec.optical.transmittance_per_band
    derived = {
        band: band_slab_transmittance(
            load_nk_table("glass"), _response(band), spec.thermal.thickness_m
        )
        for band in authored
    }
    assert abs(derived["lwir"] - authored["lwir"]) < 1e-3
    assert derived["mwir"] > 5 * authored["mwir"]
    for band in ("nir", "swir"):
        assert derived[band] - authored[band] > 0.15


# --- the physics the check rests on ---------------------------------------------------------------


def test_the_surfaces_are_part_of_the_answer() -> None:
    """Internal absorption alone calls a glass slab in NIR a perfect transmitter.

    About 8 % of the light never gets into the glass. A `transmittance_per_band` is what leaves the
    far side — `1 - eps - rho` by Kirchhoff — so dropping the Fresnel loss would make every
    authored number look 8 points too low against its own table.
    """
    table = load_nk_table("glass")
    response = _response("nir")
    lam = np.linspace(*response.support_um, 2001)
    _, k = table.at(lam)
    internal_only = float(np.exp(-4.0 * np.pi * k / lam * 5000.0).min())
    assert internal_only > 0.999  # the glass itself absorbs essentially nothing here
    assert band_slab_transmittance(table, response, 0.005) < 0.95


def test_a_thicker_slab_transmits_less() -> None:
    table, response = load_nk_table("glass"), _response("mwir")
    thin, thick = (band_slab_transmittance(table, response, d) for d in (0.001, 0.02))
    assert thick < thin
    assert band_slab_transmittance(table, response, 1.0) < 1e-6


def test_a_transparent_slab_is_the_two_surface_fresnel_limit() -> None:
    """k → 0 must leave exactly the surface loss, which is the closed form the sum converges to."""
    table = load_nk_table("glass")
    clear = type(table)(
        name=table.name,
        wavelength_um=table.wavelength_um,
        n=table.n,
        k=np.zeros_like(table.k),
        source=table.source,
        path=table.path,
    )
    response = _response("nir")
    got = band_slab_transmittance(clear, response, 0.005)
    # A non-absorbing slab: (1-R)^2 / (1-R^2) = (1-R)/(1+R). Compared against the unweighted mean
    # rather than a band average, which is why the tolerance is 2e-3 and not tighter -- n barely
    # moves across the NIR response, so the two weightings agree to about that.
    lam = np.linspace(*response.support_um, 2001)
    n, _ = clear.at(lam)
    r = ((n - 1.0) / (n + 1.0)) ** 2
    assert got == pytest.approx(float(np.mean((1.0 - r) / (1.0 + r))), abs=2e-3)


def test_a_thickness_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="thickness must be positive"):
        band_slab_transmittance(load_nk_table("glass"), _response("lwir"), 0.0)
