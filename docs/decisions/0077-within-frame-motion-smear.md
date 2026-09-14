# ADR 0077 — Within-frame motion smear, spatially varying

**Status:** Accepted
**Date:** 2026-09-14

## Context

`irsim.optics.mtf.mtf_motion` — `|sinc(v t_int ξ)|` for linear image-plane motion during the
integration — has been in the MTF cascade since M5. **Nothing ever called it.** A grep for its name
outside its own definition and `__all__` returned the export list and nothing else, so every frame
this simulator has produced was sharp regardless of how fast the scene crossed it.

That is not a small omission for the scenes this project is built for. The aircraft stage (ADR
0075) sweeps the boresight at 34 °/s at closest approach and a Boson pixel is 0.049°, so the scene
crosses **eleven pixels in one frame period**. Motion blur is also one of the most obvious
signatures separating real thermal video of a moving target from synthetic video of one, which
puts it squarely in the sim-to-real path this whole lane exists to serve.

It is distinct from, and additional to, the bolometer's frame-to-frame lag. The membrane IIR (§9.2)
carries a target's history across successive frames; this is the blur laid down **within** a single
integration. Both are real and they compose.

## Options considered

**1. Frequency domain, folded into the PSF.** The cascade is already built that way, and it is what
`mtf_motion` is written for. It requires *one* velocity for the whole frame. That is exactly wrong
for the scenes here: under a tracking mount the target is stationary on the focal plane while the
sky sweeps past it, so a single kernel serves neither. Kept for the MTF bench, which measures a
uniform-motion figure of merit; not usable for a render.

**2. A spatially varying line average** (chosen). Each pixel is averaged along **its own** motion
vector, from −s/2 to +s/2 with s = |v|·duty. Costs one gather per tap and handles the tracked case
correctly by construction.

**3. Where it sits.** After the PSF, before the box filter. Both are convolutions laid down during
the integration and they commute, so the order between *them* is arbitrary; that both precede the
box filter is not, because the box filter is the detector sampling the result.

**4. Centred or trailing.** Centred. A trailing segment displaces every moving feature by half its
smear — a shift that looks like a timing error because it is one.

**5. The duty, which is where the two detector families part.** A microbolometer has no shutter and
no integration window — `integration_time_ms` is `None` for one in the schema, deliberately —
because it integrates continuously, so the scene smears over the **whole frame period**. A cooled
photon detector integrates briefly inside the frame and is idle for the rest, so it smears over
that fraction and comes out sharper. Reading a missing integration time as zero would have made
every uncooled camera in the repository sharper than it is, which is the flattering direction and
therefore the dangerous one. This is the mechanism behind §16's checklist line *"lateral motion
smears LWIR, not cooled MWIR"*.

## Decision

`irsim.optics.smear.apply_motion_smear` is the spatially varying operator; `smear_duty` is the
integration fraction, with `None` meaning a bolometer and therefore 1. `irsim.pipeline.optics`
scales the G-buffer's `motion_px` by the duty and hands it to `apply_optics`, which applies it
between the PSF and the box filter. `motion_px` stays optional in the G-buffer (M0.6), so a still
scene takes the previous path exactly and every committed golden is unchanged.

## Consequences

**It agrees with the cascade term it implements**, which is the only reason to trust it: measured
against `|sinc(s f)|` over smears of 3–20 px and frequencies of 1/48–1/12 cyc/px, worst deviation
**0.015**, mostly under 0.006. Two descriptions of one effect that disagreed would be worse than
one, because a picture cannot tell you which is wrong.

**The cross-check found a real error.** Taps placed at the segment's *endpoints* look natural and
are wrong: N taps spanning length s sit s/(N−1) apart, so the comb implements a boxcar of length
s + s/(N−1). Measured MTF 0.7182 where sinc said 0.7842 — exactly the Dirichlet kernel of the
longer smear. The operator was self-consistent and describing the wrong smear, which no amount of
"does it look blurred" would have caught. The taps are now at sub-interval midpoints.

**Cost.** One bilinear gather per tap, with one tap per pixel of travel up to 65. On a 4×
supersampled Boson frame with an 11 px native smear that is ~45 gathers of 5 M points. Scenes
without a `motion_px` plane pay nothing, and motion below a quarter pixel returns the input
untouched.

**What it does not model.** Rotation within the integration is approximated as translation at each
pixel's own velocity — correct to first order, and the error grows with the rotation rate times
the integration time. Occlusion is ignored: a smeared foreground edge blends with whatever the
background *currently* is rather than with what was actually behind it during the sweep, which is
the standard approximation for post-hoc motion blur and is visible only at high-contrast
silhouettes. There is no sub-frame *scene* motion — the target moves, the world does not deform.

## Revisit when

* A Warp port of stage 3 lands — this is a gather-heavy operator and is the natural candidate.
* Rotation rates get high enough that the per-pixel-translation approximation shows, most likely
  on a spinning propeller rather than a slewing mount.
* Propellers are modelled: a blade at flight rpm sweeps its whole disc within one integration, so
  the annulus it smears into is this operator's job and the reason ADR 0074 left props out.
