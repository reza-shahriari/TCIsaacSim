# ADR 0073 — A generated environment dome for the companion visible frame

**Status:** Accepted
**Date:** 2026-09-14

## Context

The phase-1 aerial stage (M10.19) deliberately has no sky geometry and no ground plane. ADR 0014
and ADR 0060 explain why: every colour AOV on this build is float16 and exposure-scaled, so pushing
the sky through one quantises it to ~100 mK against a 50 mK NETD, and the error is worst at the
horizon where the elevation gradient is steepest and where the targets of interest are. The
infrared background is therefore *computed* from the MS.2 sky model at each pixel's own ray
elevation, and the only geometry in the stage is the targets themselves.

That is right for the infrared path and it made the **companion visible frame** (the registered
RGB capture) useless. With an untextured grey dome light and a distant light pointing nowhere in
particular, it rendered as a flat grey field with six grey squares in it. It showed no sky, no
horizon, no sun, no time of day — nothing a reader could use to check where the camera was pointing
or what scene the infrared frame was a picture of. The visible frame exists precisely to be the
human-legible half of the pair, and it was not doing that job.

There is also a second, sharper reason to fix it. The infrared frame *cannot* be eyeballed for
correctness — that is the premise of this whole repository. The companion frame is the one place a
reviewer can apply ordinary visual judgement, and only if it depicts the same scene: the same sun,
at the same hour, through the same air.

## Options considered

**1. How the visible background is produced.**

  a. Import an HDRI sky photograph. Realistic for free, but it is *a* sky, not *this* scene's sky:
     its sun sits wherever the photographer stood, contradicting the NOAA sun the thermal solver
     is loading the targets with. It also needs a licensed asset the repository does not have and
     a network fetch CI cannot make.
  b. Add a real sky dome and ground plane as **geometry**. Reverses ADR 0014/0060 for the infrared
     path, which is the one thing this stage exists not to do.
  c. **Generate a lat-long environment map from the scene's own state and bind it to a
     `UsdLux.DomeLight`, with a `UsdLux.DistantLight` for the solar disc** (chosen). A light is not
     geometry: a ray that sees it still reports instance id 0 and an infinite `DistanceToCameraSD`,
     so the infrared background keeps coming from the sky model. Verified as bit equality, not as
     an argument — see `test_the_dome_does_not_touch_one_infrared_pixel`.
  d. Composite the sky into the RGB frame ourselves, from the same ray directions the infrared path
     uses. Exactly consistent by construction, but the geometry would be tone-mapped by RTX and the
     background by us, so the two halves of one picture would be exposed differently. Rejected for
     now; it becomes attractive if a linear HDR colour AOV ever works on this build.

**2. What sky model.**

  a. **Preetham, Shirley & Smits (SIGGRAPH 1999), "A Practical Analytic Model for Daylight"**
     (chosen). A Perez luminance distribution scaled to an absolute zenith luminance, with CIE xy
     chromaticity distributed by the same form. Its two inputs — solar zenith angle and turbidity —
     are both quantities this repository already has. Closed-form, no tables, no dependency.
  b. Hosek-Wilkie (2012). Better near the horizon and at high turbidity, which is where this scene
     actually looks; but it is a large coefficient dataset, and the extra fidelity buys nothing
     that is measured, since the visible frame is not a radiometric output.
  c. A hand-tuned gradient. Cheapest, and it would have had to be re-tuned for every hour of the
     day while never being tied to the scene at all.

**3. Where turbidity comes from.**

  Turbidity is the ratio of *column* optical depths, `T = (tau_molecular + tau_aerosol) /
  tau_molecular`, and the shared `WeatherSeries` reports *visibility*, a ground-level extinction
  coefficient. Taking the ground-level ratio directly — `3.912 / (V gamma_Rayleigh)` — gives
  T = 14.2 for a clear 23 km day, which is a dense industrial haze and would render a clear June
  morning as a white sky. The two layers are therefore given their own scale heights (8 km
  molecular, 1.2 km continental aerosol) before the ratio is taken, which puts the same day at
  T = 2.98. The preset's visible `gamma0_per_m` of 1.2e-5 /m gives a column depth of 0.096 against
  the textbook Rayleigh 0.0973 at 550 nm, so the two really are the same quantity.

**4. Exposure.**

  The sky's absolute luminance moves by two decades between dawn and noon and any fixed dome
  intensity blacks out one end of it. The texture keeps honest cd/m2 on disk and the dome light's
  `intensity` carries an **auto-exposure** on the median daylight sky. This is a display choice
  and is the reason the visible frame carries no absolute brightness information. The twilight
  fade is divided back out of the reference first, so a night scene still renders dark rather than
  being exposed up to look like noon.

## Decision

`irsim_isaac.visible_sky` generates a float32 lat-long environment map — Preetham sky above the
horizon, Lambertian terrain hazed into the horizon sky by Koschmieder's contrast transmittance
below it — from the scene's NOAA sun position, the shared weather's visibility and irradiance, and
the atmosphere preset's Rayleigh coefficient. `irsim_isaac.stage` writes it to an EXR, binds
it to the stage's dome light, aims a distant light with a 0.53 degree cone along the same sun
direction for the disc, and gives each target a `UsdPreviewSurface` so the companion frame shows a
white airframe and a black carbon one rather than six identical grey squares.

The map is authored in **direction space**: every texel is turned into a stage-space unit vector
and the sky is evaluated there, so the texture layout is the renderer's business and the physics
never has to know it.

**Nothing here is in `src/irsim/`.** It is not infrared physics, it implements no section of
`docs/physics-model.md`, and no part of the sensor chain reads it. Only the *geometry* of the
visible frame is a calibrated claim — it is box-filtered onto the infrared pixel grid, so the pair
is registered by construction. Its photometry is a daylight-appearance model tone-mapped by the RTX
path tracer and is not traceable to a radiometric unit.

## Consequences

**The renderer's lat-long convention had to be measured, and is recorded as a measurement.** On
Isaac Sim 6.1.0 with `omni:rtx:domeLight:mode = infinite`, the RTX dome light samples the texture
with its polar axis on the stage's **+Z** and its azimuth running from **+X toward -Y** — so on a
Y-up stage the texture's poles sit on the horizon and its equator passes through the zenith. This
is not the USD documented behaviour for a Y-up stage and the public documentation for this build
does not settle it. The first implementation assumed elevation/azimuth and produced a companion
frame filled entirely with ground, because the whole camera field fell inside one texture pole.
`DOME_POLE_AXIS` records the measurement and `test_the_sun_renders_where_noaa_puts_it` re-measures
it against the renderer rather than trusting the constant: with the camera pointed down the sun's
own azimuth, the solar disc must land at `f_px tan(tilt - elevation)` below the frame centre.

**Two absences are deliberate**, so the pair does not lie to the reader:

* **No cloud on the dome.** MS.3's structured cloud field is not wired into the infrared
  background, so painting cloud on the visible dome would show a sky the infrared frame does not
  have. *(Correction, ADR 0076: "the clear-sky profile only" overstated it. `SkyModel.radiance`
  has always applied the uniform blend (1 − cε) L_clear + cε L_base — the mean cloud effect was
  there all along; only the structure was missing. The structure is wired in as of ADR 0076, and
  the dome should follow.)*
* **No solar disc in the texture.** Preetham's distribution carries the aureole around the sun but
  not the disc itself; the disc is the distant light. The two therefore do not double-count.

**What it makes easy.** Looking at the RGB frame and immediately seeing where the camera points,
where the horizon is, what hour it is and which side the sun is on — and therefore noticing when
one of those is wrong. It also gives the targets real solar shading in the visible band, so a
silhouetted drone looks silhouetted.

**What it makes hard / what it does not do.** The terrain is painted on a dome at infinity, so it
has no parallax: a target at 120 m does not move against it, and nothing below the horizon casts
or receives a shadow. The infrared frame still has no ground model at all below the horizon — one
temperature, per ADR 0060 — so the visible and infrared halves disagree about how much structure
the ground has. Adding real ground *geometry* would fix both and would reverse ADR 0060's
background treatment for those pixels; that is a separate decision and is not taken here.

**Error introduced: none, in the sense that matters.** The infrared outputs are bit-identical with
and without the dome, asserted at every render. The visible frame's own error is unbounded and
unquantified, which is acceptable exactly because nothing measures it.

## Revisit when

* A linear, working HDR colour AOV appears on this build — then option 1(d) becomes strictly
  better than a dome light, and the exposure stops being a guess.
* The infrared background gains cloud (MS.3 into `aerial_bridge`) — then the dome should gain the
  same cloud field, from the same seed and the same cloud fraction.
* Ground geometry is added to the stage — then the terrain hemisphere of the dome becomes a
  backdrop for it rather than a stand-in, and its haze model has to agree with the atmosphere the
  infrared path applies to those same rays.
* A scene needs high turbidity or a sun within a few degrees of the horizon, where Preetham's fit
  is weakest — Hosek-Wilkie is the replacement.
