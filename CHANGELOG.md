# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Project scaffold: physics specification (`docs/physics-model.md`), `CLAUDE.md`, skills, build tooling.
- `irsim.radiometry.constants` — physical constants with sources.
- `irsim.radiometry.planck` — spectral radiance (energy and photon forms), fractional exitance,
  thermal derivative. Analytic identity tests passing.
- Layering test enforcing that `src/irsim` imports no engine modules.
- ADR 0002: NumPy ≥ 2.0 floor and Isaac Sim's bundled Python as the project interpreter.

### Changed
- Makefile targets run through a `PYTHON` variable (`make check PYTHON=.../python.sh`); `make test`
  prints the ten slowest tests so the 30 s budget stays visible.
- NumPy floor raised from 1.24 to 2.0 (the Planck tests use `np.trapezoid`).
- ruff ignores `N802` alongside `N803`/`N806` (physics notation such as `d_spectral_radiance_dT`).

### Fixed
- `make check` is green on the scaffold: three files reformatted, one `Any` return in
  `irsim.radiometry.planck` typed explicitly.
