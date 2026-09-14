# ADR 0064 — Solar spectral irradiance: modelled files, named after the model

**Status:** Accepted
**Date:** 2026-09-15

## Context

M11.3 needs a solar spectrum to band-integrate: E_B = ∫ R(λ) E(λ) dλ, in both energy and photon
form, for whatever band a sensor YAML configures. The roadmap named two files —
`astm_e490_toa.csv` (the extraterrestrial standard) and a G-173 ground-level shape.

Both are **tabulated standards, not formulas**. The authoring machine has no network. Writing a
file of plausible digits and naming it `astm_e490_toa.csv` would put a fabricated table under a
standard's name, where the next reader — and every provenance check in this repository — would
take it as the real thing. That is worse than having no file.

## Decision

**Generate both spectra from stated models, and name the files after the models.**
`scripts/generate_solar_spectra.py` writes:

| file | model |
|---|---|
| `data/spectra/solar/toa_planck_5778k.csv` | 5778 K Planck exitance, normalised so ∫E dλ = 1361 W m⁻² |
| `data/spectra/solar/direct_normal_am1p5.csv` | that × a clear-sky direct-beam band model at AM1.5, normalised to 900.1 W m⁻² |

Each file's header says, in its first line of status, that it is MODELLED and NOT the standard,
and says how to swap the real table in: drop a two-column CSV in its place. `irsim.radiometry.solar`
only integrates `(wavelength, irradiance)` tables, so nothing else in the repository changes.

**The extraterrestrial shape is the effective-temperature model, and its error is anisotropic in
a way that suits this simulator.** A 5778 K Planck is within a few percent of the real spectrum
above ~0.7 µm and badly wrong in the ultraviolet — it puts 12.0 % of the solar constant below
0.4 µm against a real ~8 %. Every band `irsim.config.bands` knows starts at 0.75 µm or above. So
the limitation is made **enforced rather than documented**: `SolarSpectrum` refuses a spectral
response reaching below `TRUSTED_MIN_UM = 0.70 µm`, naming E-490 as the fix. A visible band would
have to supply real data; the infrared ones do not.

**The ground-level shape is a named band model, and its integral is a calibration, not a result.**
Rayleigh scattering in Bird & Riordan's (1984) analytic form; Ångström aerosol at the G-173
reference turbidity (AOD 0.084 at 500 nm, α = 1.14); the ozone Chappuis band; and the major water,
O₂ and CO₂ bands as Gaussians in ln λ. Band **centres** are spectroscopy. Band **strengths** carry
one common water multiplier, bisected so the AM1.5 integral is the 900.1 W m⁻² that defines
G-173's direct-normal reference.

Because that integral is fitted, it cannot be evidence for the file. The checks that *are*
independent are on shape, and they were not constrained by the fit:

| measured | value | why it matters |
|---|---|---|
| τ(1.38 µm) | **0.025** | the water band that makes SWIR cameras blind there |
| τ(1.87 µm) | **0.008** | ditto |
| τ(1.06 µm) | 0.924 | the SWIR imaging window stays open |
| τ(1.55 µm) | 0.944 | ditto |
| peak wavelength | 0.502 µm | Wien for 5778 K, so the file is on the right wavelength unit |

**τ_sun for the direct beam uses Kasten–Young air mass, not ADR 0051's plane-parallel sec θ.**
`airmass` diverges at the horizon and refuses beyond 85°, which is right for the band-averaged
slant path and wrong for the sun: 09:00 is not an error condition, and a camera watching a sunlit
target does not stop working because the sun is 8° up. `airmass_kasten_young` is finite to the
horizon (m = 37.9 at 90°). The band-averaged path is untouched, so nothing measured against ADR
0051 moves. One artefact is accepted and asserted: the empirical fit gives m(0°) = 0.99972 rather
than 1, i.e. **0.03 % in transmittance at zenith**, far inside the ESTIMATED preset's own error.

**The term enters stage 1 as an incident radiance, with ρ applied once.** §5.4 writes the reflected
result as `(ρ_B/π) E_B τ_sun cos θ_s S`. This module produces everything except ρ_B, as
`l_sun = E_B τ_sun cos θ_s S / π`, so ADR 0063's bundle can add it to the environment and night
terms and reflect the sum once. Three cases are **exactly** zero — sun below the horizon, surface
facing away, full shadow — because a horizon that leaks 10⁻¹² of a 300 W m⁻² beam becomes a visible
seam after AGC.

## What this step measured

* The **spectral crossover** between reflected sunlight and self-emission, for a 300 K, ρ = 0.3
  surface in full sun, is at **4.14 µm**. That is the fact behind §12.1's three regimes, and it is
  the least model-dependent number here — both sides are steep, so a few percent of solar-shape
  error moves it by tens of nanometres. At the band level: SWIR reflected/emitted = **1.6e8**,
  LWIR = **0.0035**.
* A disagreement worth recording. Band-integrating these two files gives a SWIR band
  transmittance of τ(AM1.5) = 0.701, hence **0.789 at zenith** — against the atmosphere presets'
  `solar.zenith_transmittance['swir'] = 0.88`. The presets' value is an ESTIMATED §7.2 table
  midpoint, and a broadband midpoint does not know that the **1.38 µm water band sits inside
  0.9–1.7 µm**. Neither number is measured, so this is a flag, not a correction: a test holds the
  two within 20 % and asserts the sign, so a future change to either is noticed.

## Consequences

* `shadow_mask` (S in [0, 1], **1 = lit**, the same polarity as `thermal.solar.solar_loading`)
  and `sun_cos_incidence` (n·ŝ) join the G-buffer's optional keys. Optional, so every fixture and
  adapter written before them is unchanged.
* The two CSVs are generated, committed and reproducible — rerun the script and diff — rather than
  hand-authored like `boson_vox.csv` was.
* The band-integrated solar path currently ignores in-band spectral **redistribution**: E_B is
  integrated at the top of the atmosphere and attenuated by one band-averaged τ_sun, so the fact
  that the 1.38 µm band removes a *slice* of the SWIR band rather than a fraction of all of it is
  not represented. Using the ground-level file as the integrand instead is a one-argument change
  (`spectrum_file=AM15_DIRECT_FILE`) and is the right answer whenever the sun is the dominant
  source; it is not the default because the preset's τ_sun would then be applied twice.

## Revisit when

* The network is available, or the tables are otherwise obtained. Replace both CSVs, rerun the
  suite, and record in this ADR what moved — in particular the 4.14 µm crossover and the 0.789
  SWIR zenith transmittance above, which are the two numbers this simulator will lean on.
* A visible band is configured. `TRUSTED_MIN_UM` will refuse it, correctly, and E-490 becomes a
  prerequisite rather than an improvement.
* The in-band redistribution above starts to matter — the first time a SWIR scene is compared
  against real imagery (M12).
