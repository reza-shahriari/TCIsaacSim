# ADR 0081 — A spinning rotor as a time-averaged veil, not as geometry

**Status:** Accepted
**Date:** 2026-09-14

## Context

ADR 0074 built the quadrotor out of primitives and left the propellers off, on the grounds that a
solid disc of the right diameter is 33 px across at 20 m and hides the motor bells underneath it —
which are the entire subject of an infrared picture of a multirotor. It deferred them to the motion
path: *"doing that properly needs M10.1b. Omitted until then: nothing is more honest than a wrong
thing."*

ADR 0077 then landed within-frame motion smear and, in its own "revisit when", claimed the deferral
was now discharged: *"a blade at flight rpm sweeps its whole disc within one integration, so the
annulus it smears into is this operator's job and the reason ADR 0074 left props out."*

**That claim was wrong, and finding out why is the substance of this decision.**
`apply_motion_smear` averages each pixel along a straight segment with one tap per pixel of travel,
capped at 65. A blade tip at 3000 rpm on a 28-inch prop travels **112 m/s**, so during a 16.7 ms
bolometer frame it sweeps 1.86 m of arc — 109 pixels at 20 m through a Boson, on a *circular* path
that wraps the disc 1.7 times. A 65-tap linear boxcar is not a coarse version of that; it is a
different operator, and the failure is not mainly the tap count but the shape of the path. The
smear operator and the rotor diverge exactly where the motion stops being small compared with the
structure it is moving through.

## Options considered

**1. What to draw.**

  a. **Blade geometry, smeared by ADR 0077's operator.** What ADR 0077 assumed. Needs a few
     hundreds of taps along an arc the operator does not follow, and the renderer would first have
     to resolve the blade: its chord at three-quarter radius is 27 mm, which is **1.6 pixels** at
     20 m, so what would be smeared is itself an aliased sliver. Rejected on both counts.
  b. **Renderer motion blur with many sub-frame samples.** RTX can do it, and at ~2 revolutions per
     frame the sample count needed to avoid banding is in the hundreds per frame. It also puts the
     answer inside the engine, where the physics core cannot test it and the Unreal port would have
     to reproduce it.
  c. **A static disc at some nominal transparency.** Cheap, and it throws away the one thing that
     distinguishes the two detector families here — see the decision below.
  d. **A time-averaged occluding veil, computed analytically** (chosen). The detector reports the
     mean over its window, so compute the mean directly and composite it. Engine-free, closed-form,
     and testable against area identities rather than against a previous render.

**2. What the veil's alpha means.**

  It is **not** a transmittance and the blade is not semi-transparent. It is the fraction of the
  integration window during which opaque blade material stood between that pixel and the
  background. The blend is therefore linear in **radiance**, which is what the detector integrates —
  the same reasoning as CLAUDE.md #3's rule about noise. Blending apparent temperatures instead is
  a different number: for a 3 % veil of 290 K blade over a 230 K sky, the radiance blend reads
  233.0 K and the temperature blend 231.9 K, so the naive version under-reports the disc in the
  direction that would hide it.

**3. How viewing tilt enters.**

  a. A cosine fudge on the coverage. Untestable and would have hidden the result below.
  b. **The azimuthal mean of the projected area of a pitched plate** (chosen). A blade at azimuth φ
     has normal `n = cos β z − sin β t(φ)`, so `d · n = A + B sin φ` and the mean of its modulus is
     closed-form. Face-on it is `cos β`; edge-on it is `(2/π) sin β`.

## Decision

`irsim.optics.rotor` models a spinning rotor as a veil whose coverage is the **running mean of the
blade-passage pulse train over the angle swept during the integration**, scaled by the projected
area of a pitched plate:

    alpha(r, psi) = clip( k(tilt, pitch) / cos(tilt) · duty(psi; swept, c(r)/r, N),  0, 1 )
    L             = alpha · L_blade + (1 − alpha) · L_behind

It lives in `irsim/optics/` beside `smear.py` because it is the same §8.3 within-integration
average applied to a rotating occluder instead of a translating scene, and it shares that module's
`smear_duty` reasoning about which window a detector actually integrates over.

**One formula covers both detector families, and the window length alone selects between them.** A
bolometer has no shutter, so it integrates the whole frame: a 3000 rpm prop sweeps 300° = 1.67
blade spacings and the picture is a smooth annulus. A cooled photon detector integrating 2 ms
sweeps 36° = 0.20 spacings and the same expression resolves into two distinct arcs, five times the
mean brightness. Nothing switches; the window does. This is ADR 0077's LWIR/MWIR split a second
time, and it is the reason option 2(c)'s static disc was rejected — a fixed transparency cannot
express it.

## Consequences

**The invariant is mean preservation, not flatness.** A running mean cannot change the mean of a
periodic function, so the azimuthal mean coverage equals the local solidity for *every* window
length — verified from a frozen shutter to ten revolutions. The shutter moves the disc's flux
around; it never changes how much there is. That is what makes the annulus and the arcs
comparable, and it is the property a brightness fudge would have broken.

**The ripple is real physics, not a numerical artefact.** A window of 1.67 blade spacings crosses
some azimuths twice and others once, so the annulus is banded in the ratio exactly 2:1 — one whole
blade pass, not a fraction of one. It flattens only when the window is a whole number of spacings.
A real prop on real video bands for the same reason.

**Coverage is exactly tilt-invariant until the disc is within `pitch` degrees of edge-on.** The
blade's projected area and the ellipse's area shrink by the same cosine, so `k/cos(tilt)` is
`cos(pitch)` throughout, and a banking rotor changes shape without changing brightness. This module
was first written asserting the opposite — that coverage rises with foreshortening — and the
rasteriser contradicted it: the peak at 0° and at 45° was the same number to three digits. Only
past the crossover does the annulus densify, towards the clip at 1.

That crossover is not academic here. The demo stage looks 75° off the disc axis against 18° of
blade pitch, where the threshold is 72°, so the scene sits just past it and the branch is exercised
by the geometry it was written for. A **zero-pitch** blade would be invisible edge-on, which is the
cleanest check that pitch is carrying the projection rather than decorating it.

**The disc is faint, as ADR 0074 predicted.** A two-blade 28-inch prop is 3.2 % solid at
three-quarter radius. A 290 K blade over a 230 K sky lifts the apparent temperature by **3.1 K** —
a low-contrast annulus around a hot motor, which is what the phenomenology says and what the solid
disc of option 2(c) would have buried.

**What is not modelled, stated rather than implied:**

* **Constant rpm through the window.** The ripple pattern assumes it. Spooling a motor within one
  frame would smear the bands; nothing here does that.
* **An isothermal blade.** Real blades have a spanwise gradient and warm tips; `L_blade` is one
  radiance for the whole disc. The thermal model has no rotor node, so there is nothing better to
  read yet.
* **Arc width uses the planform chord `c/r`, not the projected `c cos(pitch)/r`,** because the
  projection is carried entirely by `k` and counting it twice would be worse. The resolved arcs are
  therefore about 5 % wider in azimuth than they should be at 18° of pitch. **The mean is exact
  regardless**, so the error is in the shape of a short-exposure picture and not in any flux.
* **Depth ordering is the caller's.** `veil_radiance` takes an `occluded` mask; the module does not
  know which half of the airframe is in front of the disc plane. Without that mask a blade would be
  painted over the motor bell it is bolted to.
* **No blade-to-airframe or blade-to-sky reflection**, and no downwash. The veil sees only what is
  directly behind it.

**The disc is projected through the lens oracle, not by `f R / Z`.** `disc_ellipse` measures both
semi-axes by projecting rim points with `irsim.optics.projection.project`, so the authored
distortion model and the off-axis scale come from the same forward model ADR 0015 already calls the
oracle for the lens the engine is handed. A barrel lens 8 m off axis at 20 m shrinks the disc by
4.3 % and pulls it 17 px back towards the axis; an `f R / Z` stand-in sees neither. The major axis
is taken along `axis x view` — the one diameter in the disc plane square to the line of sight, and
therefore the only one that is not foreshortened.

That leaves one approximation, stated as a measurement: the projected conic is assumed centred on
the projected centre with perpendicular axes, which is exact only for an orthographic camera. A
0.71 m rotor seen 75° off its axis at 20 m has rim points up to **0.079 px** off the fitted
ellipse, falling quadratically to 0.013 px at 50 m, and the axis ratio differs from `cos(tilt)` by
at most 3e-4. Both sit far below the veil's own modelling uncertainty. An exact conic fit is the
upgrade if a disc ever has to be *measured* rather than drawn. (The docstring first claimed "under
a hundredth of a pixel"; the measurement said 0.079, and the measurement is what is recorded.)

**The veil composites; it does not inject an excess.** The obvious reuse was MS.6's point-target
machinery (`irsim.pipeline.point_target`), which adds `phi tau (L_t − L_beyond)` — and `alpha` *is*
a per-pixel fill fraction, so the shapes line up. It is the wrong operator here. A sub-pixel target
occults a **sky column** whose radiance the plane does not carry separately, which is why the
excess form needs a `sky_beyond` term at all. A rotor disc does not: by stage 2c the plane already
holds the correctly attenuated background at every pixel — sky beyond the disc over some of it, the
aircraft's own arm or motor bell over the rest, each having travelled its own path. So the blend

    L = L_plane + alpha (L_blade,at-sensor − L_plane)

is exact for both cases at once and needs no decision about what is behind. The excess form applied
over a pixel where the blade veils the airframe would subtract a sky column that is not there. A
test puts one rotor half over cold sky and half over a 300 K arm: the veil lifts on one side and
*dips* on the other, from one operator in one pass.

The blade's own path is stage 2's, reused rather than re-derived — `blade_radiance_at_sensor` calls
the same `apply_layered_gbuffer` / `apply_atmosphere` the plane went through, on a one-element
array — so the veil and the pixels under it cannot disagree about the atmosphere. Range moves the
disc *towards ambient*, which is not the same as fainter: against a 230 K sky in 288 K air, a 290 K
blade at 3 km reads dimmer than at 20 m and a 270 K blade reads **brighter**. That test was written
the first way round and only the first way round.

**Coverage is built on the ellipse's bounding box.** A 4× supersampled Boson frame is 2560 × 2048,
so a full-frame float64 coverage map per rotor is 40 MB and four rotors would be 160 MB a frame.
The window is an optimisation, so a test asserts the result is bit-identical to a full-frame
composite rather than merely close.

## Mounting them on a stage

`irsim_isaac.pipeline.rotor_isaac` turns a prim's transform into veils, and
`QuadrotorSpec.rotor_mounts` puts four of them above the motor bells. Four consequences are worth
recording because each was a decision and two of them were nearly mistakes.

**Nothing is authored.** A mount adds no prim, no mesh and no material to the stage — only a
position, an axis and an rpm — because a spinning rotor is not geometry. That buys the whole
sub-pixel and occlusion story for free, and it costs one honest asymmetry: **the discs appear in
the infrared frame and not in the companion visible frame**, which is rendered by RTX from stage
geometry. The visible frame therefore shows a quadrotor with no propellers. ADR 0073 made the
companion frame the human-legible half of the pair, so this is a real gap rather than a cosmetic
one; option 1(b) (real blade geometry, motion-blurred by the renderer) is what would close it, at
the cost the options list already rejected.

**Occlusion is a plane intersection, not a range comparison.** Each pixel's ray is intersected with
the disc *plane* and compared with the depth the renderer reported. Comparing against the disc
centre's range instead would be wrong by up to a disc radius across the ellipse — 0.36 m at 20 m,
about 18 pixels of arm drawn on the wrong side of the aircraft. The mask is meaningless outside the
ellipse (an infinite plane is met kilometres away by a grazing ray) and does not need to be: it is
only read where the coverage is non-zero. A test asserting otherwise failed on arithmetic that was
correct, which is recorded in the test rather than papered over.

**The integration window is the detector's, not the capture interval's.** ADR 0074 films this
scene as a time-lapse, one frame per six seconds of scene time, and `IrCamera.frame_period_s`
carries that interval so every stage knows the truth about it. Using it here would sweep a
3000 rpm prop through three hundred revolutions and produce a perfectly uniform annulus with no
banding at all — a *plausible* picture with the ripple physics quietly deleted. The sweep is taken
over `1 / frame_rate_hz` scaled by the integration duty instead.

**Rpm follows the throttle by a square root.** A propeller's thrust goes as rpm², and the flight
profile's `u` is a fraction of maximum *thrust* — it is what ADR 0072's `ΔT_max u²` motor law
reads. So `rpm = rpm_max √u`, and a hovering aircraft at u = 0.5 turns at 2121 rpm, not the 1500 a
linear reading would give. That is a 40 % error in the swept angle, which moves which regime the
discs are filmed in, so it is not a detail.

One input is **ESTIMATED**: the blade's sky-view factor, taken as 0.5 for a blade seen mostly
edge-on. `propeller_rubber` is ε = 0.95 in LWIR, so the reflected term is a twentieth of what
leaves the blade and the choice is worth about a kelvin.

### What the first render found, that no unit test did

**Sky carries `distance_m = 0`, not infinity.** `irsim.config.gbuffer` says so in its first
paragraph, and for a good reason: the sentinel is zero so a consumer that ignores `sky_mask` still
computes τ = 1. Read as a *distance*, zero is a surface at the camera — nearer than everything —
and the first `occlusion_mask` duly marked **every pixel of the frame** occluded and erased all
four discs. The rendered frames were bit-identical with and without rotors.

The unit test that should have caught it used `inf`, which is what the **AOV** reports before the
adapter translates it. It was a plausible fixture rather than the contract, it passed, and it
tested the one value that could not fail. It is now parameterised over both, with the zero case
named as the dangerous one, plus a sky-only frame — the demo's actual configuration, since the
quadrotor is the only geometry on the stage.

**Measured, with the fix:** 6444 native pixels change, peaking at **+10.1 K** of apparent
temperature, and occlusion falls from the whole frame to the 15 786 pixels of airframe that really
are in front of a disc.

**The disc is brightest at its root**, because local solidity `N c(r) / (2 π r)` rises inward as the
circumference shrinks while the chord does not: 3.2 % at three-quarter radius and 19 % just outside
the motor bell. That is why the peak lift is 10 K and the median over the disc is 0.02 K — the
outer annulus is genuinely almost invisible, and the inner one is not.

**And it is invisible at the demo's own display span.** ADR 0074 spans the *target's* nodes, 18 to
62 °C, so a 273 K disc over a 263 K sky clips to black along with the sky; plateau equalisation
gives it no codes either. The picture is only wrong if you read the display as the data. At a
scene-context span (`--span-c -15 40`) the same frames differ by up to **45 display codes** and the
four annuli are plainly there. This is the AGC lesson again and it is worth stating: **a veil that
is real in radiance can be absent from the picture**, and which of those you measure decides
whether you believe the feature works.

## Revisit when

* The thermal model grows a rotor node — then `L_blade` should vary along the span, and the veil
  needs a radiance per radius rather than a scalar.
* A scene runs a detector whose integration window is short enough to freeze a blade rather than
  arc it (under roughly 0.1 ms here), where the "which azimuth is the blade at" question becomes a
  scene-state question rather than a statistical one, and `phase_rad` stops being cosmetic.
* A helicopter main rotor is modelled: it turns an order of magnitude slower, so a bolometer frame
  covers well under one blade spacing and the *arcs* become the normal case rather than the
  cooled-detector special case.
* Anything needs the disc's effect on what is behind it beyond occlusion — downwash on a sea
  surface, or the rotor wash warming a wall.
* The companion visible frame has to show the propellers. Nothing here authors geometry, so RTX has
  nothing to draw; closing that gap means option 1(b) and the sample count it costs, or compositing
  the same veil into the RGB frame ourselves — which ADR 0073's option 1(d) already describes, and
  which would inherit that option's exposure mismatch.
* Blade aerodynamic heating matters. At 3000 rpm the tip does 112 m/s (M = 0.33), whose recovery
  temperature is about 5.6 K above ambient, and three-quarter radius about 3.1 K — computable today
  with `irsim.thermal.aerial.recovery_temperature_k`. The blade currently takes the airframe node's
  temperature flat, which is the right order but not the right gradient, and the gradient is the
  same span-wise variation the first item here would carry.
