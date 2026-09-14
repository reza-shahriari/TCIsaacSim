# ADR 0065 — Night illumination: what the airglow level means, and a first-order moon

**Status:** Accepted
**Date:** 2026-09-15

## Context

§5.5 is the band-specific piece it says people forget: at night in SWIR the scene is lit by
airglow, and "getting this right is what makes a SWIR channel behave like a real SWIR camera at
night rather than like a dark visible camera". It gives the model as

    E_night(λ) = E_airglow(λ)·k_cloud + E_moon(λ, φ) + E_star(λ) + E_artificial

with a configurable airglow level, "reported … roughly 3.5–39 nW/cm²", default ~10.

Three things are undefined, and each of them is a factor of several:

1. **What interval the nW/cm² is measured over.** 10 nW/cm² in a 0.1 µm line and 10 nW/cm² over
   2 µm are different skies (spec issue S30).
2. **What `k_cloud` does at full overcast.** §5.5 says cloud "attenuates it, but it does not have
   a directional shadow" — so overcast must not be black, and the schema's `k_cloud ∈ [0, 1]`
   multiplied in linearly would make it black at 1.
3. **What "first-order moon" means.** The spec gives no lunar level at all.

## Decision

**1. The level is defined over the shape file's own support, and a band takes its fraction.**
`data/spectra/airglow_oh_meinel.csv` covers 0.6–2.6 µm; the configured level is the irradiance
integrated over that, and `AirglowSpectrum.band_fraction` gives what a given R(λ) sees.

The alternative — "the level is whatever the configured band sees" — is superficially the more
literal reading of "in-band", and it is wrong the moment a second band exists: it would hand the
LWIR camera the same 10 nW/cm² of OH emission at 10 µm, where there is none. The level would then
be a property of the camera rather than of the sky.

The fraction is not a detail. The modelled InGaAs camera sees **61 %** of the level (66 % for an ideal 0.9–1.7 µm top-hat);
an ideal top-hat extended to 2.5 µm sees **89 %** — a **third** more light — because the strongest OH
sequence (Δv = 1) sits at 1.4–2.0 µm and a 1.7 µm cut-off throws part of it away. That is a real
reason extended-InGaAs parts exist, and it only exists in the model because the band heads are in
the right places.

**2. The shape is an OH-Meinel band model, generated and labelled as one.**
`scripts/generate_airglow_spectrum.py` places the Δv = 1..4 sequence heads (spectroscopy) and
gives them 55/27/12/6 % of the OH emission (modelled), plus the O₂(a¹Δ) band at 1.27 µm and a
flat 4 % continuum. Like the solar files (ADR 0064), it is committed, reproducible and replaceable
by a measured near-infrared sky spectrum with no code change.

**3. Cloud attenuates to a floor.** `1 − k_cloud · cloud · (1 − CLOUD_FLOOR)` with
`CLOUD_FLOOR = 0.10` — solid overcast under `k_cloud = 1` still returns a tenth. Cloud **scatters**
airglow rather than absorbing it, and a night sky under thick cloud is dimmer, never black. The
floor is ESTIMATED and is stated as a floor, not a measurement.

**4. The lunar level is derived from magnitudes, not authored.** m_sun = −26.74 and
m_moon = −12.74 give a flux ratio of 10^(14/2.5) = 3.98e5 and E_full = **3.42e-3 W m⁻²**. The
moon then borrows the *solar* spectral shape, so the lunar albedo cancels between the shape and
the level and never has to be authored. Phase follows Lane & Irvine,
Δm(α) = 0.026|α| + 4e-9 α⁴, with α = arccos(2f − 1) from the schema's illuminated fraction.

That phase law is violently non-linear and that is the point: a **quarter moon is 0.091 of a full
moon**, not 0.5. A model that used the illuminated fraction directly would make every
quarter-moon scene five times too bright, and would look entirely reasonable doing it.

**5. Airglow has no shadow parameter, and the signature is the assertion.** §5.5 says airglow
"is not blocked in the same way as moonlight". `NightIllumination.incident_radiance` therefore
takes no `shadow` and no sun direction at all — there is nothing to pass, so a scene cannot darken
the night sky by geometry. The moon does take an elevation and sets.

**6. Stars and artificial lighting are deferred**, as §5.5's own equation allows. Starlight is
roughly a tenth of airglow in this band and artificial lighting is a scene asset rather than an
illumination model; neither changes a sky-target or maritime scene's ordering.

## A disagreement recorded rather than resolved (spec issue S38)

§5.5 asserts, citing [R7], that "at full moon, moonlight and airglow radiation densities are
comparable". Built as above, they are not:

| quantity, 0.9–1.7 µm | value |
|---|---|
| airglow at the 10 nW/cm² default | 6.09e-5 W m⁻² |
| full moon at zenith | 8.46e-4 W m⁻² |
| **full moon / airglow** | **13.9** in energy units, **12.6** in photon units |
| full moon / airglow at §5.5's *upper* level (39 nW/cm²) | 3.2–3.6 |
| **quarter moon / airglow** | **1.26** energy, **1.15** photons |
| spectral-density ratio at the OH band peaks (1.27–1.87 µm) | 2–5 |

Three readings make the spec's claim nearly true — the upper end of its own airglow range, the
*spectral density* at the OH peaks rather than the band integral, and a moon at a realistic
elevation rather than at zenith. None of them is the plain band-integrated reading.

Nothing here is tuned to meet it. Instead the test asserts what the models **do** reproduce — that
airglow is about a quarter moon in SWIR, which is the rule of thumb SWIR night imaging is actually
sold on — and pins the full-moon ratio to the measured 12.6 (photon units) so that a change to either model is
noticed and the disagreement stays visible.

## Consequences

* Two more generated, reproducible data files join E-490's and G-173's stand-ins. Every one of
  them says MODELLED in its first status line.
* `SolarSpectrum` and `AirglowSpectrum` now share `SpectralTable`: one merged-grid band integral,
  one photon convention. Two sources integrating a band two slightly different ways is how they
  come to disagree about what a band is.
* The airglow's slow variation is three seeded sinusoids at 1800/5400/17000 s, not per-frame
  noise. Airglow drifts over tens of minutes; a random draw per frame would look like sensor
  noise, which the simulator already has plenty of.

## Revisit when

* A measured near-infrared sky spectrum is available (a Gemini or VLT sky model). Replace the CSV
  and record what moves — in particular the 61 % SWIR fraction and the 13.4 ratio above.
* A scene needs the lunar **reddening**. The moon's albedo roughly doubles between 0.5 µm and
  1.6 µm, so borrowing the solar shape understates moonlight in SWIR by something like a factor of
  1.5 — which happens to push the S38 disagreement the wrong way, and is the first thing to check
  when it is revisited.
* Artificial lighting matters, i.e. the first ground/urban night scene.
