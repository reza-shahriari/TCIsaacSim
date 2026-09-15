# ADR 0080 — Sea skin temperature: the cool skin, the diurnal warm layer, and T_bulk as an input

**Status:** Accepted
**Date:** 2026-09-15

## Context

A maritime scenario knows its **bulk** sea surface temperature. That is what a buoy reports, what a
ship's intake thermometer reads, what a satellite SST product is calibrated to, and what the public
datasets this project validates against publish. `configs/environments` authors it as
`ground.bulk_sst_k` (MM.5).

An infrared camera does not see it. It sees the top fraction of a millimetre, and that skin is not
the bulk:

* **The cool skin is always there.** Heat leaves the ocean by conduction through a sublayer that
  turbulence cannot penetrate, so a temperature gradient *must* exist to carry it. The skin sits
  0.1–0.6 K below the water a millimetre beneath it whenever the surface is losing heat, which at
  sea is almost always.
* **The diurnal warm layer is sometimes there.** On a sunny, calm day absorbed shortwave accumulates
  in a shallow surface layer faster than wind can mix it down, and the skin runs up to ~3 K *above*
  the bulk. It vanishes at night and is gone above about 6 m/s of wind.

Neither is negligible at this project's scale. A 50 mK NETD sees 0.3 K as six noise-equivalent
temperature differences, and the error is one-sided and covers the whole lower half of every
maritime frame, so it does not average out the way noise does. Before MM.4, `SeaModel` took a
constant `cool_skin_k` that defaulted to **zero** — so every maritime frame rendered so far has used
the bulk SST as if it were the skin.

The spec is silent here: `docs/physics-model.md` has no cool-skin section, because the maritime
lane (MM) was added after it. This ADR therefore carries the physics, not just the decision.

## Options considered

1. **Author the skin temperature beside the bulk.** One more number in the environment preset.
   Rejected: it lets a scene hold a dead-calm skin offset under a 15 m/s wind, and the *same* wind
   is already driving the Cox–Munk slope statistics three lines away (MM.2). That is precisely the
   class of contradiction CLAUDE.md non-negotiable #6 exists to prevent.
2. **Solve the upper-ocean thermal structure.** A multi-layer diffusion model with a prognostic warm
   layer (Price–Weller–Pinkel, or the COARE warm-layer routine). Correct, and it is what an ocean
   model does. Rejected for now: it needs state, a spin-up and a mixing closure, and it would be the
   only stateful object in the background model. The cool skin is a *diagnostic* quantity — it
   responds to the instantaneous flux in seconds — so it does not need any of that.
3. **Saunders' cool skin plus an empirical warm layer, both diagnostic.** Chosen.

## Decision

`T_skin = T_bulk − ΔT_cool(U, Q_net) + ΔT_warm(Q_sw, U)`, in `irsim/thermal/sea_skin.py`, with
`T_bulk` staying an authored scenario input and the skin **derived**, never authored. The
`cool_skin_k` constructor argument is gone: there is nowhere to type a contradicting value.

**The cool skin is Saunders (1967).** The sublayer thickness is `δ = λ ν / u*`, the deficit is
Fourier's law across it, `ΔT = Q_net δ / k_w`, and the water-side friction velocity comes from the
wind through the continuity of stress across the surface, `ρ_a C_D U² = ρ_w u*²`. With λ = 6 this
gives δ ≈ 0.93 mm at 5 m/s, which is what the measurements show, and deficits of 0.11 K at 5 m/s
and 0.23 K at 1 m/s under the ~79 W/m² of net longwave a clear night over a 290 K sea produces.

**The low-wind divergence is bounded, and how it is bounded matters.** Saunders' form sends δ → ∞
as u* → 0, because it describes a sublayer under a shear that is vanishing; what actually limits
transport there is free convection, which it does not model. The thickness is capped at the top of
the observed range (2 mm) through `δ_max tanh(δ_Saunders/δ_max)` rather than a `min`. The two agree
to third order wherever the Saunders term is small, so the wind-stirred regime is untouched; what
the smooth form buys is that the deficit stays **strictly** decreasing in wind instead of acquiring
a flat shelf and a corner. Calm, clear nights are the conditions a long-range maritime scenario
cares about most, so the behaviour there should be smooth and bounded rather than clipped.

**The warm layer is empirical and labelled as such.** `ΔT_warm = ΔT_max · min(1, Q_sw/Q_ref) ·
max(0, 1 − (U/U_c)²)` with ΔT_max = 3 K, Q_ref = 1000 W/m², U_c = 6 m/s. It carries the three
properties the observations agree on — proportional to absorbed irradiance, zero at night,
suppressed by wind above a threshold — and claims nothing else. It is not derived and does not
pretend to be.

**Under net warming the cool skin is zero, not negative.** Conduction against an *outgoing* flux is
what makes the skin cool, so the mechanism stops when the flux reverses. A surface genuinely warmer
than the water beneath it is the warm layer, a different mechanism with a different depth scale;
letting the deficit go negative would count it twice.

## Consequences

**What this makes easy.** A maritime scene's skin temperature now moves with its own weather, from
the one `WeatherSeries` everything else reads. Turning the wind up thins the sublayer and roughens
the surface together, because both read the same number.

**The error this introduces, and its direction.** `Q_net` is the net *longwave* the scene's sky model
implies (M6.5). The real net heat loss also carries sensible and latent turbulent fluxes, and at sea
the latent term is usually the largest of the three; a full-flux `Q_net` runs roughly **twice** the
longwave-only value. The deficit is exactly linear in `Q_net` — asserted by a test — so this is a
bounded omission and not an unknown one: **the modelled cool skin is a lower bound, low by about a
factor of two.** The function takes `Q_net` as an argument precisely so a caller that has the
turbulent fluxes can pass their sum with no other change.

Three smaller approximations, in order of size:

* λ is held at its windy-limit value of 6. Fairall et al. (1996) make it a function of the surface
  buoyancy flux, rising under free convection; that is the principled version of the 2 mm cap above
  and the natural upgrade.
* The net longwave is evaluated at the **bulk** temperature, not the skin, which is circular by
  0.2 K in the radiating temperature — 0.3 % in σT⁴, well under a millikelvin in the deficit. Named
  rather than iterated.
* The warm layer has no memory. The real one integrates the day's heating, peaks in mid-afternoon
  and decays for an hour or two after sunset; this form responds instantaneously and is exactly zero
  the moment the sun sets. A lagged form would be more faithful in the afternoon and would break
  "zero at night", which is the property a night-time maritime scene actually depends on.

**A measured correction to MM.4's own acceptance criterion.** The step asked for a sensitivity
ordering: a 1 K error in `T_bulk` should move nadir apparent temperature by more than 0.9 K and a
0.2°-depression patch by less than 0.15 K. The ordering holds with room to spare — measured 0.98 K
at nadir against 0.20 K at 0.2° — but **the 0.15 K threshold is not reachable at any camera
height**, and the test records why instead of asserting it. Sensitivity near the horizon is set by
slant **range**, not by the angle: a 20 m camera's 0.2° ray is 6.8 km out and reads 0.20 K; getting
under 0.15 K needs about 13 km of path, and at the ~100 m height where 13 km corresponds to 0.5°,
0.2° is above the horizon and sees no water at all. The useful statement is the regime, not the
number: **SST accuracy matters looking down and stops mattering within a degree of the horizon**,
where emissivity collapse and atmospheric path have already replaced most of the signal.

## Revisit when

- A maritime scenario needs the turbulent fluxes — then `Q_net` becomes sensible + latent + net
  longwave and the deficit roughly doubles. Nothing else in this model changes.
- A scene needs the afternoon peak or the post-sunset decay of the warm layer: that is the
  prognostic Price–Weller–Pinkel form, and it brings state with it.
- Tier 4 comparison against public maritime imagery shows a systematic sea-temperature bias of the
  0.1–0.5 K size — the cool skin is the first thing to check, since the model is knowingly a lower
  bound.
- Anyone reaches for a scene in genuinely dead calm (under ~0.14 m/s), where the tanh bound
  saturates in float64 and the deficit stops varying with wind at all.
