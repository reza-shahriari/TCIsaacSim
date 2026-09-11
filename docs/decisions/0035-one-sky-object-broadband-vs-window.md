# ADR 0035 — One sky object serves both paths; broadband downwelling from a clear-sky emissivity relation, not the LWIR-window T_sky

**Status:** Accepted
**Date:** 2026-09-11

## Context

Two consumers need "the sky": the surface energy balance (§6.1) needs the **broadband** downwelling
longwave irradiance Q_LW↓ (all wavelengths, what a pyrgeometer measures), and the image needs the
**in-band** sky radiance L_sky,B(θ) (§5.3, the reflected term and the sky background). §5.3(a)
gives a practical LWIR-window sky temperature, T_sky = T_air − ΔT_clear (1 − cloud) cos^q θ_zen
with ΔT ≈ 55–70 K. That window value is *not* a broadband quantity: the 8–14 µm window is where the
clear sky is most transparent and therefore coldest; outside the window (H₂O rotation band, CO₂
15 µm) the sky radiates near T_air. Using the zenith window temperature as a broadband sky gives
σ(288 − 60)⁴ = 153 W m⁻² against 260–295 W m⁻² from the emissivity relations at 288 K, a 100+ W m⁻²
error that would over-cool every night-time surface by tens of kelvin.

## Options considered

1. Use the LWIR-window T_sky for the balance as well — simple, wrong by 100+ W m⁻² (above).
2. **Clear-sky broadband emissivity relation** (chosen): Brunt (1932) ε = 0.52 + 0.065 √e (e in hPa)
   as the default, Idso (1981) ε = 0.70 + 5.95e-5 e exp(1500/T) as an option; cloud blended
   linearly to ε = 1 (overcast radiates as a blackbody at T_air); Q_LW↓ = V_s ε_sky σT_air⁴ +
   (1 − V_s) σT_surround⁴.
3. Integrate the layered atmosphere (MS.1) spectrally over all wavelengths — the right answer in
   phase 2, and it is how `SkyModel.broadband_downwelling()` (MS.2) may be upgraded; today the
   layered model exists per sensor band only.

## Decision

Option 2 in `irsim.thermal.longwave`, fed by the same `WeatherSample` (vapour pressure via M8.2's
Magnus form) that the atmosphere uses, so both paths see one humidity. The MS.2 `SkyModel`
will expose `broadband_downwelling()` by delegating here and `L_sky_B(θ)` from the layered
atmosphere: one object, two physically different quantities, never one substituted for the other.

## Consequences

- Clear dry sky at 288 K: 259 W m⁻² (Brunt) / 294 W m⁻² (Idso); overcast 390 W m⁻². The two
  relations differ by ~35 W m⁻² in dry air -- that spread is the model uncertainty (Brunt-type
  relations are ±20–30 W m⁻² against pyrgeometers without local calibration).
- The cloud blend assumes cloud base at T_air; a high cold cloud radiates less. Cloud *type*
  and base height are not in the weather file.
- The window T_sky of §5.3(a) remains available for the *image* fast path (MS.2), fitted rather
  than authored.

## Revisit when

MS.1's layered atmosphere is integrated over the full spectrum (then option 3 replaces the
relation with a consistent one), or when pyrgeometer data for a target site allows calibrating
the coefficients.
