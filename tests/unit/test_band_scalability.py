"""M11.1 — "bands are data, not code", asserted rather than asserted-in-prose.

Two halves, and the second is the one that will fail first:

* a **dynamic** half parametrised over every YAML in ``configs/sensors/``: each one loads,
  classifies into a canonical band by overlap, hashes reproducibly and distinctly, and tabulates
  a band LUT -- all through the same functions, with no branch anywhere on which band it is.
  Adding ``example_swir_ingaas_640.yaml`` next to the LWIR Boson exercised exactly this path and
  needed no code change; this test is what keeps that true.
* a **static** half over ``irsim/{radiometry,optics,detector,noise,isp,pipeline}``: those packages
  may not contain a band name, in a string or in an identifier, nor a nominal band edge written
  as a wavelength. The registry in ``irsim.config.bands`` is the single place that knows the four
  bands; a kernel that learns one has broken the abstraction CLAUDE.md calls the main scalability
  requirement of the project, and it breaks it silently -- the code still works, for the bands
  that were thought of.

docs/physics-model.md §12.1, §12.2 line 900, §16.4 step 10
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pytest

from irsim.config.bands import BAND_IDS, NOMINAL_RANGES_UM, band_id_for, overlap_fraction
from irsim.config.loader import band_hash, config_hash, load_sensor_config
from irsim.config.sensor import BolometerFpa, PhotonFpa
from irsim.radiometry.constants import C_LIGHT, H_PLANCK
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "sensors"
DATA_DIR = REPO / "data"
CORE_PACKAGES = ("radiometry", "optics", "detector", "noise", "isp", "pipeline")

# A coarse grid: this half of the suite is about *whether* a band tabulates, not about quadrature
# accuracy, which tests/unit/test_planck.py owns at the full 0.05 K spacing.
COARSE_N = 161

SENSOR_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.stem for p in SENSOR_CONFIGS]


@pytest.fixture(scope="module")
def configs():  # type: ignore[no-untyped-def]
    return {p.stem: load_sensor_config(p, DATA_DIR) for p in SENSOR_CONFIGS}


def test_the_config_directory_is_not_empty() -> None:
    """Guards the guard: every parametrised test below would vacuously pass on an empty glob."""
    assert SENSOR_CONFIGS, f"no sensor YAML under {CONFIG_DIR}"


def test_more_than_one_band_is_configured(configs) -> None:  # type: ignore[no-untyped-def]
    """A scalability claim tested on one band is not tested at all."""
    ids = {c.sensor.band.band_id for c in configs.values()}
    assert len(ids) >= 2, f"only band {ids} is configured; M11.1 adds SWIR beside LWIR"


# ---------------------------------------------------------------------------------------------
# dynamic half: every configured band goes through the same code
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_loads_and_classifies_by_overlap(name, configs) -> None:  # type: ignore[no-untyped-def]
    band = configs[name].sensor.band
    derived = band_id_for(band.lambda_min_um, band.lambda_max_um)
    assert derived in BAND_IDS
    assert band.band_id == derived
    # Classification is by *maximal* overlap, so the winner must beat every other band, not
    # merely clear the 50 % floor.
    others = {b: overlap_fraction(band.lambda_min_um, band.lambda_max_um, b) for b in BAND_IDS}
    assert others[derived] == max(others.values())
    assert others[derived] > max(v for b, v in others.items() if b != derived)


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_hashes_reproducibly(name, configs) -> None:  # type: ignore[no-untyped-def]
    cfg = configs[name]
    assert config_hash(cfg, DATA_DIR) == config_hash(
        load_sensor_config(CONFIG_DIR / f"{name}.yaml", DATA_DIR), DATA_DIR
    )
    assert len(band_hash(cfg, DATA_DIR)) == 64


def test_the_configs_hash_distinctly(configs) -> None:  # type: ignore[no-untyped-def]
    """Distinct *cameras*, and a band hash that collides **exactly when the bands are the same**.

    The second half used to assert every band hash was distinct, which held only while every
    config was a different band. It is not the invariant, and M12.1's Halmstad camera -- a Boson
    320x256 with the same LWIR response as the 640 -- is the case that shows why: two cameras in
    one band *must* share a band hash, because that is what lets `make luts` build one LUT bundle
    for both instead of two identical ones. The invariant is the iff.
    """
    hashes = {name: config_hash(c, DATA_DIR) for name, c in configs.items()}
    assert len(set(hashes.values())) == len(hashes), hashes

    bands = {name: band_hash(c, DATA_DIR) for name, c in configs.items()}
    blocks = {
        name: json.dumps(c.model_dump(mode="json")["sensor"]["band"], sort_keys=True)
        for name, c in configs.items()
    }
    for a in configs:
        for b in configs:
            same_block = blocks[a] == blocks[b]
            same_hash = bands[a] == bands[b]
            assert same_hash == same_block, (a, b, bands[a], bands[b])


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_every_config_tabulates_a_lut_with_no_code_change(name, configs) -> None:  # type: ignore[no-untyped-def]
    """The same ``BandLUT.build`` over the same four quantities, whatever the band."""
    response = load_spectral_response(configs[name].sensor.band.spectral_response)
    lut = BandLUT.build(response, n=COARSE_N)
    assert lut.lb.dtype == np.float32 and lut.lb_q.dtype == np.float32
    assert np.all(np.diff(lut.lb) > 0.0), "Lb must rise with temperature in every band"
    assert np.all(np.diff(lut.lb_q) > 0.0)
    assert np.all(lut.dlb_dt > 0.0) and np.all(lut.dlb_q_dt > 0.0)
    # A band with no signal at all would pass the monotonicity checks as a table of zeros.
    assert float(lut.lb[0]) > 0.0


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_the_response_curve_covers_the_configured_band(name, configs) -> None:  # type: ignore[no-untyped-def]
    band = configs[name].sensor.band
    lo, hi = load_spectral_response(band.spectral_response).support_um
    assert lo <= band.lambda_min_um and hi >= band.lambda_max_um


@pytest.mark.parametrize("name", CONFIG_IDS)
def test_the_declared_regime_is_physically_possible_for_the_band(name, configs) -> None:  # type: ignore[no-untyped-def]
    """§12.1: no self-emission below ~2.5 um, no reflected sun beyond ~3 um."""
    from irsim.config.sensor import EMISSIVE_MIN_LAMBDA_MAX_UM, REFLECTIVE_MAX_LAMBDA_MIN_UM

    band = configs[name].sensor.band
    if band.regime == "emissive":
        assert band.lambda_max_um > EMISSIVE_MIN_LAMBDA_MAX_UM
    if band.regime == "reflective":
        assert band.lambda_min_um < REFLECTIVE_MAX_LAMBDA_MIN_UM


# ---------------------------------------------------------------------------------------------
# the SWIR config itself
# ---------------------------------------------------------------------------------------------


def test_the_swir_config_is_a_photon_fpa_with_no_bolometer_fields(configs) -> None:  # type: ignore[no-untyped-def]
    fpa = configs["example_swir_ingaas_640"].sensor.fpa
    assert isinstance(fpa, PhotonFpa) and not isinstance(fpa, BolometerFpa)
    assert fpa.thermal_time_constant_ms is None and fpa.tcr_per_k is None
    assert fpa.quantum_efficiency > 0 and fpa.well_capacity_e > 0
    assert fpa.integration_time_ms <= 1000.0 / fpa.frame_rate_hz


def test_the_swir_config_is_reflective_and_claims_no_apparent_temperature(configs) -> None:  # type: ignore[no-untyped-def]
    """Inverting Lb gives a temperature only where the signal *is* self-emission."""
    sensor = configs["example_swir_ingaas_640"].sensor
    assert sensor.band.regime == "reflective"
    assert sensor.outputs.apparent_temperature is False


def test_a_300_k_scene_is_invisible_in_swir(configs) -> None:  # type: ignore[no-untyped-def]
    """The known answer that makes "reflective" a fact rather than a label.

    Compares the two configured bands' self-emission from the same 300 K blackbody, with the
    SWIR side converted out of photon units by its own band-mean photon energy -- so the ratio
    is energy over energy and the photon path is exercised, not bypassed.
    """
    swir = configs["example_swir_ingaas_640"].sensor.band
    lwir = configs["flir_boson_640_lwir"].sensor.band
    r_swir = load_spectral_response(swir.spectral_response)
    r_lwir = load_spectral_response(lwir.spectral_response)
    lut_swir = BandLUT.build(r_swir, n=COARSE_N)
    lut_lwir = BandLUT.build(r_lwir, n=COARSE_N)

    photon_energy_j = H_PLANCK * C_LIGHT / (r_swir.mean_wavelength_um() * 1e-6)
    lb_swir_from_photons = float(lut_swir.lookup(300.0, "lb_q")[()]) * photon_energy_j
    ratio = lb_swir_from_photons / float(lut_lwir.lookup(300.0)[()])
    assert ratio < 1e-6, f"300 K self-emission ratio SWIR/LWIR = {ratio:.3e}"

    # And the photon-unit table agrees with the energy-unit one to the band-mean photon energy:
    # a band this narrow in hc/lambda is well described by a single mean.
    assert lb_swir_from_photons == pytest.approx(float(lut_swir.lookup(300.0)[()]), rel=0.3)


def test_swir_self_emission_becomes_significant_at_exhaust_temperatures(configs) -> None:  # type: ignore[no-untyped-def]
    """The other side of the same coin: "reflective" is a statement about 300 K, not about the band.

    Between 300 K and 900 K the SWIR band's own emission rises by many orders of magnitude while
    the LWIR band's rises by about one -- which is why M11.2 refuses to cull emission in a
    reflective band.
    """
    lut_swir = BandLUT.build(
        load_spectral_response(configs["example_swir_ingaas_640"].sensor.band.spectral_response),
        n=COARSE_N,
    )
    lut_lwir = BandLUT.build(
        load_spectral_response(configs["flir_boson_640_lwir"].sensor.band.spectral_response),
        n=COARSE_N,
    )
    swir_gain = float(lut_swir.lookup(900.0)[()]) / float(lut_swir.lookup(300.0)[()])
    lwir_gain = float(lut_lwir.lookup(900.0)[()]) / float(lut_lwir.lookup(300.0)[()])
    assert swir_gain > 1e6
    assert lwir_gain < 100.0


def test_the_nir_config_is_a_photon_fpa_that_claims_no_apparent_temperature(configs) -> None:  # type: ignore[no-untyped-def]
    nir = configs["example_nir_si_1280"].sensor
    assert isinstance(nir.fpa, PhotonFpa) and not isinstance(nir.fpa, BolometerFpa)
    assert nir.fpa.thermal_time_constant_ms is None and nir.fpa.tcr_per_k is None
    assert nir.band.regime == "reflective"
    assert nir.outputs.apparent_temperature is False


def test_nir_is_the_extreme_case_of_reflective(configs) -> None:  # type: ignore[no-untyped-def]
    """The fourth band, and the one that makes the point hardest.

    "Reflective" is a statement about how much of the signal is the scene's own emission, and it
    is quantitative. At 300 K the SWIR band's self-emission is already 1e-9 of the LWIR band's;
    the NIR band's is smaller again by several more orders, because Planck's Wien tail falls
    another factor of e^(hc/kT (1/0.9 - 1/1.3)) between the two. A camera whose own band cannot
    see a 300 K object at all is one for which *every* photon is borrowed -- sun, moon or
    airglow -- which is what makes the M11.3/M11.4 illumination terms load-bearing rather than a
    refinement.
    """
    bands = {
        name: load_spectral_response(configs[name].sensor.band.spectral_response)
        for name in ("example_nir_si_1280", "example_swir_ingaas_640", "flir_boson_640_lwir")
    }
    luts = {name: BandLUT.build(response, n=COARSE_N) for name, response in bands.items()}

    def energy_at(name: str, temp: float) -> float:
        photon_j = H_PLANCK * C_LIGHT / (bands[name].mean_wavelength_um() * 1e-6)
        return float(luts[name].lookup(temp, "lb_q")[()]) * photon_j

    lwir_300 = float(luts["flir_boson_640_lwir"].lookup(300.0)[()])
    nir_300 = energy_at("example_nir_si_1280", 300.0)
    swir_300 = energy_at("example_swir_ingaas_640", 300.0)
    assert nir_300 / lwir_300 < 1e-12, f"NIR/LWIR at 300 K = {nir_300 / lwir_300:.3e}"
    assert nir_300 < swir_300 / 1e3, "NIR must be far deeper into the Wien tail than SWIR"

    # ...and, as with SWIR, emission is not culled: the same band lights up at flame temperature,
    # which is why the regime gates the *illumination* terms and never the emissive one.
    nir_gain = float(luts["example_nir_si_1280"].lookup(900.0)[()]) / float(
        luts["example_nir_si_1280"].lookup(300.0)[()]
    )
    assert nir_gain > 1e9, f"NIR 300 -> 900 K gain = {nir_gain:.3e}"


# ---------------------------------------------------------------------------------------------
# static half: the core may not know a band's name or its edges
# ---------------------------------------------------------------------------------------------

#: Names whose value is a wavelength. A nominal band edge is only evidence of a hard-coded band
#: when it is *used as a wavelength*; 1.0 and 3 are otherwise the most ordinary numbers there are.
WAVELENGTH_NAME_MARKERS = ("lambda", "wavelength", "_um", "micron")
NOMINAL_EDGES_UM = frozenset(e for pair in NOMINAL_RANGES_UM.values() for e in pair)


def _core_sources() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for package in CORE_PACKAGES:
        out += sorted((REPO / "src" / "irsim" / package).rglob("*.py"))
    return out


def _is_wavelength_name(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in WAVELENGTH_NAME_MARKERS)


def test_the_core_source_scan_finds_files() -> None:
    sources = _core_sources()
    assert len(sources) > 30, f"only {len(sources)} core modules found under {REPO}"


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_no_band_name_appears_as_a_literal_or_an_identifier(package: str) -> None:
    """``if band == "lwir"`` and ``SWIR_CUTOFF`` are the same mistake wearing different clothes."""
    offences: list[str] = []
    for path in sorted((REPO / "src" / "irsim" / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.strip().lower() in BAND_IDS
            ):
                offences.append(f"{path.name}:{node.lineno} string {node.value!r}")
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.arg):
                names = [node.arg]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            for name in names:
                parts = name.lower().replace("-", "_").split("_")
                if any(part in BAND_IDS for part in parts):
                    offences.append(f"{path.name}:{node.lineno} identifier {name!r}")
    assert not offences, (
        f"irsim.{package} names a band; the registry in irsim.config.bands is the only place "
        f"that may:\n  " + "\n  ".join(sorted(set(offences)))
    )


@pytest.mark.parametrize("package", CORE_PACKAGES)
def test_no_nominal_band_edge_is_hard_coded_as_a_wavelength(package: str) -> None:
    """A band edge reaching the core as a number is the same coupling, written numerically."""
    offences: list[str] = []
    for path in sorted((REPO / "src" / "irsim" / package).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            targets: list[tuple[str, ast.expr]] = []
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.append((target.id, node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    targets.append((node.target.id, node.value))
            elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
                for comparator in node.comparators:
                    targets.append((node.left.id, comparator))
            for name, value in targets:
                if not _is_wavelength_name(name):
                    continue
                for constant in ast.walk(value):
                    if not isinstance(constant, ast.Constant):
                        continue
                    number = constant.value
                    if isinstance(number, bool) or not isinstance(number, (int, float)):
                        continue
                    if float(number) in NOMINAL_EDGES_UM:
                        offences.append(f"{path.name}:{node.lineno} {name} = {number}")
    assert not offences, (
        f"irsim.{package} hard-codes a nominal band edge in micrometres; band limits belong in "
        f"the sensor YAML:\n  " + "\n  ".join(sorted(set(offences)))
    )


def test_the_edge_guard_would_catch_a_planted_band_edge(tmp_path: pathlib.Path) -> None:
    """The guard above is worth only as much as its ability to fail."""
    planted = tmp_path / "planted.py"
    planted.write_text("lambda_cut_um = 13.5\n", encoding="utf-8")
    tree = ast.parse(planted.read_text(encoding="utf-8"), str(planted))
    found = [
        n.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and _is_wavelength_name(node.targets[0].id)
        for n in ast.walk(node.value)
        if isinstance(n, ast.Constant) and isinstance(n.value, float)
    ]
    assert any(float(v) in NOMINAL_EDGES_UM for v in found)


def test_the_illumination_switch_is_on_the_regime_not_the_band() -> None:
    """§5.2's "compile-time flag" as a runtime function of the regime alone (ADR 0063)."""
    from irsim.config.bands import DEFAULT_REGIME, enabled_illumination_terms

    for band in BAND_IDS:
        terms = enabled_illumination_terms(DEFAULT_REGIME[band])
        assert terms, band
    # Two bands that share a regime share a term set exactly -- there is no band-specific tail.
    assert enabled_illumination_terms("reflective") == enabled_illumination_terms("reflective")
    assert enabled_illumination_terms(DEFAULT_REGIME["nir"]) == enabled_illumination_terms(
        DEFAULT_REGIME["swir"]
    )
