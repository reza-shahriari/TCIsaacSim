# 0086 — An isotropic scattered-sunlight sky for the reflective bands

Date: 2026-09-15
**Status:** Accepted
Roadmap: M11.10 (§5.3, §5.4)

## Context

`SkyModel` (ADR 0044) is a **thermal** sky: the clear sky is the atmospheric column's own emission,
`L_clear(θ) = L_path(∞, θ)`. That is exactly right for LWIR and essentially zero at 0.9 µm.

M10.22 put the reflected-solar term on the render path, and the first NIR render made the
consequence obvious: **a brightly sunlit quadrotor on a black sky.** That is backwards. In the near
infrared the daytime sky is the brightest thing in the frame and an aircraft against it is usually
a dark silhouette — which is the *whole* detection problem in this band, and the opposite of the
LWIR one. The target was lit and nothing lit the sky.

ADR 0071 already recorded that the layered atmosphere "carries no scattered sunlight". This is that
gap, now load-bearing.

## Decision

The sky gains an isotropic scattered-sunlight radiance,

```
L_sky,scatter = f_B · DHI / π
```

where `DHI` is the diffuse horizontal irradiance already carried by the scene's one
`WeatherSeries`, and `f_B` is the fraction of the diffuse spectrum inside the band, per watt.

1. **Isotropy is chosen for the integral, not for convenience.** For an isotropic sky,
   ∫ L cos θ dΩ over the hemisphere is exactly `L·π`, so this form puts back precisely the diffuse
   irradiance the weather file measured. It is the one angular distribution guaranteed correct in
   the integral, and the integral is the quantity that was measured. A test integrates the
   hemisphere numerically and checks it.
2. **The diffuse spectrum is the solar spectrum weighted by λ⁻⁴.** Rayleigh scattering is strongly
   wavelength-dependent, so skylight is much bluer than the beam that made it — which is decisive
   here, because it means a near-infrared band receives a far smaller share of DHI than of DNI.
   Only the *shape* is used: the in-band integral is divided by the broadband one, so any constant
   multiplying the Rayleigh law cancels and no calibration is implied.
3. **The ground-level spectrum, not the top-of-atmosphere one.** Skylight is made by scattering a
   beam that has already crossed the column, so the absorption bands the beam lost are missing from
   the sky too.
4. It is added inside `clear_radiance`, so **one place** feeds both the sky background and the
   reflected environment term, and the tilt-integrated sky picks it up for free.
5. `skylight_for_sensor` returns **`None` for an emissive band**, so every LWIR render stays
   bit-identical. Measured: against a clear-sky LWIR column of ~50 W m⁻² sr⁻¹ the scattered term is
   below 1e-5 W m⁻² sr⁻¹, eight orders down and two orders below float32's spacing there.

## Consequences — what this model gets wrong, deliberately

An isotropic sky is wrong in its *distribution* in two known ways:

* **The horizon is brighter than the zenith** on a real clear sky, because the scattering path is
  longer. An isotropic model under-renders the horizon and over-renders the zenith.
* **The circumsolar aureole** — forward Mie scattering within roughly 25° of the sun — can carry a
  large fraction of DHI on a hazy day and is entirely absent here. A target passing near the sun is
  therefore rendered against a sky that is too dim.

Both are distribution errors under a correct total. A Preetham-style distribution (which
`irsim_isaac.visible_sky` already implements for the *visible* dome) would fix the shape, but it is
regressed on daylight and extending it to 1.7 µm would be using a fit outside its data — which this
project has refused before (ADR 0064 on the solar spectra). The honest sequence is: correct
integral now, measured distribution when there is something to measure it against.

**Aerosol scattering is greyer than Rayleigh** (Ångström exponents of 0.5–1.5 against 4), so a pure
Rayleigh weighting *understates* a hazy sky's near-infrared share. The exponent is a parameter, so
the sensitivity can be measured; the default is the clear-sky law.

## Alternatives considered

* **Use the direct-beam spectrum's band share for DHI.** Simpler and wrong in the direction that
  matters: it would give a NIR sky roughly as bright, relative to the beam, as a visible one, which
  is the error this band is least able to absorb.
* **Add the term to `Illumination` instead of to the sky.** It would light the surfaces and leave
  the sky background black — half the bug.
