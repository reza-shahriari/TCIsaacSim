# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `irsim.radiometry.band_integration`: the reference oracle — `band_radiance`, `band_photon_radiance`,
  `d_band_radiance_dT`, `d_band_photon_radiance_dT` by composite Simpson on an odd, edge-aligned 0.01 µm
  grid (NumPy only, float64, vectorised over T). Top-hat vs closed form: 7.5-13.5 um @ 300 K: 3.3e-11 rel, 0.000 mK; 3.0-5.0 um @ 300 K: 3.1e-11 rel, 0.000 mK; 3.0-5.0 um @ 500 K: 5.7e-11 rel, 0.000 mK; 0.9-1.7 um @ 300 K: 1.5e-06 rel, 0.016 mK.
  Derivative vs FD < 1e-5 over 200–1000 K; dLb/dT(373)/dLb/dT(300) = 1.736 for 7.5–13.5 µm.
- `irsim.radiometry.spectral_response`: R(λ) file contract and loader (two-column CSV, `#` provenance,
  µm strictly increasing, R in [0, 1], peak == 1 ± 1e-6 asserted never renormalised, zero outside support,
  half-power points, resampling) and `irsim.radiometry.band.Band` (config edges must agree with the file's
  half-power points to 0.5 µm). `data/spectra/responses/boson_vox.csv`: ESTIMATED Boson VOx curve,
  7.5–13.5 µm with ±0.25 µm raised-cosine edges (ADR 0009).
- `SIGMA_Q` (4π ζ(3) k³/(h³c²), derived in `constants.py`), `fractional_photon_exitance` and
  `band_photon_radiance_tophat` (§3.2 a): the closed-form oracle for the photon band table. Photon
  Stefan–Boltzmann closes to 1e-6; `Lb_q(7.5–13.5 µm, 300 K) = 2.915e21` known answer; `Lb/Lb_q = hc/λ_eff`
  with λ_eff inside the band.
- `d_spectral_photon_radiance_dT` (§3.4, §9.4): photon-form thermal derivative sharing the energy form's
  expm1/clip guards. Identity `dL_q/dT · hc/λ == dL/dT` to 1e-12; central FD < 1e-6; finite over
  200–2000 K × 0.4–20 µm.
- GPU-free CI: `.github/workflows/check.yml` runs `make check` on plain CPython 3.10 and 3.12; `make ci`
  reproduces it locally in a `python3.10` venv. Interpreter matrix recorded in ADR 0002.
- `docs/spec-issues.md`: the 37 physics and 15 tooling issues found in the spec, with proposed
  resolutions and applied status, for the spec owner (roadmap open question 10).
- `irsim.config.bands`: canonical band ids with §12.1 nominal ranges and per-band defaults,
  `band_id_for` by maximal overlap (< 50 % rejected), optional `band.id` validated against the edges,
  and `enabled_illumination_terms(regime)` so kernels never switch on a band name (§5.2, S17).
- `irsim.config.loader`: `load_sensor_config` (data paths resolved against `data_dir` /
  `$IRSIM_DATA_DIR` / `<repo>/data`, missing files named), `dump_sensor_config`, `config_hash` (canonical
  JSON, files by content hash) and `band_hash` (band block + spectral bytes: the LUT key). Every numeric
  and enumerated leaf of the Boson file is tested to move the hash (ADR 0008).
- `irsim.config.sensor.SensorConfig`: pydantic v2 schema for §12.2 (frozen, unknown keys rejected,
  units-in-names, discriminated bolometer/photon FPA, regime-vs-wavelength and 3-D ratio checks, derived
  pixel area / Nyquist / HFOV / frame period / DN max, deliberately no aperture factor). Boson file loads
  exactly; HFOV 30.7° vs datasheet 32° recorded (ADR 0007). `types-PyYAML` added to dev extras.
- `irsim.config.gbuffer.GBuffer`: the frozen G-buffer contract (`temperature_k`, `normal_dot_view`,
  `distance_m`, `material_id`, `sky_view_factor`; optional `encoded_t`, `motion_px`, `semantic_id`,
  `radiance*`). float16 refused on temperature/encoded/distance/radiance planes, upcast elsewhere;
  `encoded_t` checked against `temperature_k` to 10 mK; id 0 reserved as UNMAPPED.
- Synthetic G-buffer fixtures: `gbuffer_two_material`, `gbuffer_sphere` (analytic cos θ to 1e-6, reaches
  grazing), `gbuffer_step_edge` (1024², 5.5° tilt, ideal two-level), `gbuffer_moving_edge` (8 frames,
  2 px/frame with `motion_px`); all fixtures now carry `encoded_t` and use material id 1+.
  `tests/unit/test_gbuffer_schema.py` freezes the key set the Isaac adapter must emit.
- `irsim.radiometry.encoding`: float32 temperature encode/decode `c = (T − 200)/800` with the constants
  defined once in `constants.py` (plus the LUT grid constants); float16 and integer inputs refused; fp16
  coarse/fine pair codec as the §13.3 fallback. Round trip 0.05 mK; negative controls show 125 mK (raw
  kelvin) and ≥ 40 mK (encoded) through fp16 (ADR 0006).
- Stefan–Boltzmann identity test at 1e-6 relative via Simpson quadrature plus the closed-form tail
  (measured 3e-11); `fractional_exitance` gains a small-x Bernoulli branch so it is double-precision
  accurate in the Rayleigh–Jeans tail, and it and `band_radiance_tophat` now reject metre-valued
  wavelengths and Celsius temperatures (ADR 0005).
- Layering guard hardened in both directions: `src/irsim` may not import engine modules (now including
  `carb`), `irsim_isaac`, or the ML/imaging stack; default-gate tests may not import engines; glue files
  must keep engine imports inside functions. Scanner self-tested on synthetic offending modules.
- `irsim_isaac.env` (`has_isaac`, `has_warp`, `require_*`, force-off env flags) and
  `tests/integration/conftest.py`, which auto-marks and skips the directory without Isaac Sim.
- `isaac` optional extra (`warp-lang`, floor provisional until M2.1); mypy now covers `src/irsim_isaac`.
- Golden fixture helper (`tests/golden/conftest.py`, ADR 0004): `.npy` + JSON sidecar with config hash;
  STALE (inputs changed) reported distinctly from FAILING (values changed); actual array dumped to
  `outputs/golden/` on failure; float16/float64 refused on disk; `--update-golden` registered so
  `make golden-update` works. `tests/golden` now runs in `make test`.
- Project scaffold: physics specification (`docs/physics-model.md`), `CLAUDE.md`, skills, build tooling.
- `irsim.radiometry.constants` — physical constants with sources.
- `irsim.radiometry.planck` — spectral radiance (energy and photon forms), fractional exitance,
  thermal derivative. Analytic identity tests passing.
- Layering test enforcing that `src/irsim` imports no engine modules.
- ADR 0002: NumPy ≥ 2.0 floor and Isaac Sim's bundled Python as the project interpreter.
- `docs/roadmap.md`: phased implementation roadmap — 15 milestones, 157 one-commit steps with
  verification tolerances, risk register, ADR backlog, non-negotiable enforcement map, spec issue list.
- ADR 0003: sky targets first, validated against public anti-UAV thermal data (no camera); adds the
  evaluation-harness (ME) and sky/clouds/MTF/point-target (MS) milestones; revised after a four-critic
  adversarial review (72 findings applied); the twelve subsystem maps under `docs/maps/`.

### Changed
- `docs/physics-model.md` spec fixes raised before coding (M0.10): §5.3(a) sky temperature now
  `cos^q(θ_zen)` (coldest at zenith, S1); §6.1 absorbed solar written `α_sol Q_sol` (S2); §12.2
  spectral-response path `spectra/responses/boson_vox.csv` relative to `data/` (T4).
- Boson YAML `spectral_response` now `spectra/responses/boson_vox.csv`, the canonical layout under the
  data root (spec issue T4); the file itself arrives with M1.3.
- Golden helper implementation moved to `tests/golden/golden_store.py` (conftest keeps the fixture) so
  its self-tests import it unambiguously.
- Makefile targets run through a `PYTHON` variable (`make check PYTHON=.../python.sh`); `make test`
  prints the ten slowest tests so the 30 s budget stays visible.
- NumPy floor raised from 1.24 to 2.0 (the Planck tests use `np.trapezoid`).
- ruff ignores `N802` alongside `N803`/`N806` (physics notation such as `d_spectral_radiance_dT`).

### Fixed
- `SpectralResponse.resampled` snaps grid points within 1e-9 µm of the support edges: a grid built as
  `lo + k·dl` lands 2e-16 µm past the last sample and lost the endpoint (a 5 % error for SWIR at 300 K).
- `make check` is green on the scaffold: three files reformatted, one `Any` return in
  `irsim.radiometry.planck` typed explicitly.
