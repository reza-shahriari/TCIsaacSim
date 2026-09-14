# ADR 0063 — The illumination switch is a runtime function of the regime, and emission is never culled

**Status:** Accepted
**Date:** 2026-09-15

## Context

§5.2 describes the shading core as "one core, two illumination paths" and puts the choice behind a
**compile-time flag**: a build for a thermal band evaluates ε·B(T), a build for a reflective band
evaluates the reflected sun and night sources.

That is incompatible with the property CLAUDE.md calls the main scalability requirement of this
project — "bands are data, not code". M11.1 added a SWIR camera as a YAML file and a CSV, and
`tests/unit/test_band_scalability.py` now refuses to let a band's *name* appear anywhere in
`irsim/{radiometry,optics,detector,noise,isp,pipeline}`. A compile-time flag cannot be set by a
configuration file that did not exist when the binary was built, and a runtime `if band == "swir"`
is exactly the literal the guard exists to catch.

There is a second, less obvious problem with the §5.2 framing. It describes the two paths as
alternatives, which invites an implementation that evaluates self-emission **only** in an emissive
or mixed band. `enabled_illumination_terms` in the band registry encourages the same reading: it
returns `{solar, night}` for a reflective band, with no `self_emission` in the set.

That reading is wrong, and wrong in a way that produces a plausible picture. "Reflective" is a
statement about a **300 K scene**, not about the band. Measured on this repository's own tables
(M11.1, and `tests/unit/test_illumination_bundle.py`): between 300 K and 900 K, SWIR self-emission
rises by more than 10⁶ while LWIR rises by under 10². Against §5.5's 10 nW/cm² airglow, a ρ = 0.1
surface in 0.9–1.7 µm crosses over at **330 K** — below that the reflected airglow dominates by
15× at 300 K, above it the surface's own emission dominates by **11 235×** at 500 K. A jet exhaust,
a flare, a brake disc or a muzzle flash is a *self-luminous* object in SWIR, at night, with no sun
and no airglow at all. A simulator that culls ε·B(T) in a reflective band renders those as
invisible, and nothing about the resulting frame looks broken.

## Decision

**1. The gate is a runtime function of `sensor.band.regime` alone, applied in exactly one place.**
`Illumination.for_regime(regime, quantity, …)` consults `enabled_illumination_terms` at
construction and drops the terms the regime does not enable. Every consumer downstream — the
stage-1 kernel included — receives a bundle that has *already* been gated and never sees a regime
or a band id.

**2. A dropped term is absent, not zero.** `for_regime` sets it to `None`, and an empty bundle's
`total_incident()` is `None`, which makes stage 1 take its emission-only branch. So an LWIR frame
rendered with a solar plane attached is **bit-identical** to one rendered without, rather than
equal to within rounding. That distinction is the difference between a test that can fail and one
that cannot.

**3. Self-emission is unconditional.** `ε L_B(T)` is evaluated in every band in every regime. The
`self_emission` entry that `enabled_illumination_terms` returns is carried on the bundle for
provenance and means nothing to the kernel. Two tests pin both sides of the 330 K crossover, in the
*same* band, so the assertion is about temperature and not about SWIR.

**4. Everything in the bundle is an incident band radiance, in one declared quantity.** Three
terms — `l_env` (the thermal environment, §5.3 a), `l_sun` (the direct beam as an *equivalent
isotropic incident radiance*, E_B τ_sun cos θ_s S / π, §5.4) and `l_night` (isotropic night
sources, §5.5). The kernel sums them and multiplies by ρ once. Dividing the solar irradiance by π
here rather than carrying §5.4's combined `(ρ_B/π) E_B τ cos θ_s S` is what makes the Lambertian
identity ρ = 1 → E_B/π an identity **of the code** rather than of the algebra, and it keeps ρ from
being applied twice by a caller who did not read §5.4 closely.

**5. The quantity is a checked tag, not a convention.** The bundle carries `lb` or `lb_q` and the
kernel refuses a mismatch. `lb` and `lb_q` differ by ~1e19 at infrared wavelengths; a solar
spectrum integrated in W m⁻² and handed to a photon-unit kernel does not raise, does not produce
NaN, and after AGC does not look wrong — it produces a scene that is uniformly, invisibly
mis-scaled. This is the one error in the illumination path that no picture can reveal.

**6. `l_env` is not gated.** The thermal environment is what every surface in every band sees; in
SWIR it is simply small. Gating it would be a second band switch wearing the regime's clothes.

## Consequences

* Adding a band still touches only YAML, a CSV and `irsim.config.bands`. M11.1's static guard
  keeps that honest, and this ADR is the reason the guard can pass.
* `band_radiance` grows an `illumination=` parameter that is **mutually exclusive** with the older
  `l_env=`, and raises if both are given. Two ways to say the same thing is how the two drift
  apart; the older spelling stays because it is the natural one for a direct kernel call in a test.
* A uniform (0-d) environment is now accepted and broadcast, because isotropic airglow over a whole
  frame is the common case and materialising a full-frame constant is waste. **Only** 0-d — general
  broadcasting would let a `(H, 1)` column through as if it were a plane, which is the
  misalignment the shape check exists to catch.
* `l_sun` and `l_night` arrive as *optional planes*, exactly as `motion_px` and `radiance_behind`
  do, so a scene that has not asked for illumination costs nothing and renders identically. M11.3
  and M11.4 write them.
* §5.2's compile-time framing is recorded in `docs/spec-issues.md` as diverged-from deliberately.

## Revisit when

* A band needs a term that is neither environment, sun nor night — a laser designator, a flare
  illuminating the scene rather than being the target, an active NIR emitter. The bundle takes a
  fourth incident term and the registry a fourth `IlluminationTerm`; nothing else changes.
* Profiling shows the per-term sum matters. It is three adds per pixel over planes that are
  usually two constants and one array; it will not.
