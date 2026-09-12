# 0055 — The bad-pixel cluster process, the type mix, and where defects are injected

- Status: accepted
- Date: 2026-09-12
- Step: M9.5a (`feat(noise): bad-pixel map by a Poisson cluster process and defect injection`)
- Spec: docs/physics-model.md §10.4, §11.1

## Context

§10.4 is four lines long. It names four defect classes — dead, hot, flickering, blinking — gives a
typical density of 0.05–0.5 % of pixels, and says they are "clustered slightly (use a Poisson
cluster process, not uniform)". It does not say *which* cluster process, how the classes behave
frame to frame, or where in the signal chain the defects enter. All three are ours.

The clustering instruction is the one that carries weight downstream. §10.4's own closing point is
that cameras replace bad pixels by neighbour interpolation and the replacement leaves "a detectable
smoothed footprint" — and a 4-neighbour stencil (M9.5b) handles an isolated defect almost perfectly
while a 2×2 or 3×3 cluster defeats it. A simulator that scatters defects uniformly therefore
produces essentially no clusters at these densities and systematically *understates* the artefact a
perception stack actually sees. Clustering is the point, not decoration.

## Decision

### Neyman–Scott, with the parent count set from the target fraction

Parents land uniformly at random; each carries `1 + Poisson(λ)` offspring placed uniformly in a
disc of radius `CLUSTER_RADIUS_PX = 2` pixels, with λ = `noise.bad_pixel_cluster_lambda`. The
parent count is `round(fraction · rows · cols / (1 + λ))`, so the expected offspring total is the
configured fraction.

- **Radius 2 px** is chosen to make pairs, 2×2s and small 3×3s — the configurations that defeat a
  4-neighbour stencil — while keeping a cluster a cluster rather than a diffuse sprinkle.
- **`1 + Poisson(λ)`**, not `Poisson(λ)`, so a parent always produces at least one defect; a
  cluster process whose clusters can be empty just re-thins the parent process.
- **Uniform in the disc** uses `r = R√u`, not `r = Ru`, or offspring pile up at the centre and the
  cluster is tighter than the radius advertises.

**The realised count runs a few per cent under the target** (1479 against 1573 at 1024² and
0.0015 — 6 %) because colliding offspring merge into one defect and offspring outside the array are
dropped. This is deliberately *not* compensated for. Inflating the parent count to hit an exact
total would change the cluster-size distribution, and the cluster-size distribution is the part
that matters downstream; the total is the part that does not. The shortfall is bounded and is
asserted by the tests rather than hidden.

### The four classes, and what "flickering" versus "blinking" means

§10.4 names both but distinguishes them only as "random telegraph noise" and "intermittent". The
split adopted here:

- **Dead** — stuck at the DN floor, every frame, forever. **Hot** — stuck at the ceiling likewise.
  Both are findable by a single calibration frame, which is why real cameras ship a factory map.
- **Flickering (RTS)** — the pixel *still responds to the scene*; its offset hops between two
  levels. This is the class that survives a map built from one calibration frame, because a good
  fraction of the time it looks fine. Amplitude is `noise.bad_pixel_rts_amplitude_dn`.
- **Blinking** — the same two-state chain, but in its bad state the pixel is stuck like a dead one
  rather than merely offset.

Both stateful classes are driven by one two-state Markov chain, parameterised by its **stationary
occupancy** (`bad_pixel_rts_occupancy`) and its **mean dwell in the bad state**
(`bad_pixel_rts_dwell_frames`), from which the two switch probabilities follow by detailed balance.
Parameterising it this way rather than by the raw probabilities means both dwell times are
geometric by construction, which is the defining statistic of random telegraph noise and what
separates it from a pixel that is simply noisy. The initial state is drawn from the stationary
distribution, so a short clip is not biased by a burn-in.

Classes are assigned to defects **independently of their cluster**: the mix describes a population,
nothing in §10.4 says a cluster shares a failure mode, and a mixed-class cluster is the harder case
for M9.5b.

### Injection is on the raw DN plane

§11.1's order is `raw DN → bad-pixel replace`, so defects enter immediately before that. A stuck
pixel is physically stuck *before* the ADC, but the ADC is monotone and clipping, so stuck-low in
signal space and DN 0 are the same observable. Working in DN makes "bit-identical across 100
frames" exact rather than approximate, and puts the defect and its repair on the same plane.

### Testing a discrete distribution

The roadmap asks for "geometric dwell (KS p > 0.01)". The *analytic* KS p-value assumes a
continuous distribution; applied to integer dwell times it reports p ≈ 1e-174 for a perfectly
geometric sample, because the empirical CDF is a step function and the statistic picks up the step
at x = 1 (of height p) rather than any misfit. The test therefore calibrates the null distribution
by Monte Carlo over simulated geometric samples of the same size, which is valid for discrete data
— and a memoryless-pixel control is run through the same estimator to show it still rejects.

## Consequences

- `noise.bad_pixel_rts_{occupancy, dwell_frames, amplitude_dn}` are new (sensor schema v7) and all
  three are **ESTIMATED**: no published figures exist. ME.3's bad-pixel extractor is what should
  eventually set them from public imagery.
- Because occupancy and dwell jointly fix the switch probabilities, not every pair is a valid
  chain; an impossible pair raises rather than silently clamping.
- The Markov chain is advanced only over the stateful pixels. A 640×512 array with a few hundred
  defects would otherwise draw 327k uniforms per frame to move a few dozen bits.
- Like M9.4's drift, the defect state is **sequential**: frame N's state depends on every frame
  before it. The map itself is counter-based and reproducible from `sensor_seed` alone.
- Dead and hot pixels being pinned in DN means they survive the NUC unchanged, which is correct —
  a two-point correction cannot rescue an unresponsive pixel — and is what makes M9.5b necessary.

## Revisit when

ME.3 measures a real bad-pixel population and either the cluster-size distribution or the RTS
occupancy and dwell fall outside what this process can produce; or a camera is modelled whose
factory map is applied before the data leaves the core, in which case the *residual* (undetected,
mostly flickering) population is what should be simulated rather than the full one.

## Addendum (M9.5b): the replacement stencil

The replacement is the **mean of the valid 4-neighbours, iterated until clusters fill**
(`irsim.isp.replace_bad_pixels`). Three properties made it the choice over copy-one-neighbour and
over an 8-neighbour mean, and all three are asserted:

- **Exact on a linear field, for an isolated defect.** (left + right)/2 is the centre value, and so
  is (up + down)/2, so replacement introduces no radiometric bias on smooth scene content — which
  is most of a thermal image. *Inside a cluster it is not exact*: a pixel in a 2×2 sees only its
  two outer neighbours, never an opposing pair, so the mean is pulled toward the outside of the
  cluster. That bias is real, is kept, and is part of why a cluster is visible where a single
  defect is not — the concrete reason the cluster process above is not cosmetic.
- **σ²/4 on white noise.** Copy-one-neighbour leaves σ² and an 8-neighbour mean gives σ²/8. The
  σ²/4 signature is something ME.3's ISP-behaviour extractor can look for in real footage, so
  choosing the stencil fixes a testable prediction rather than an arbitrary smoothing.
- **A suppressed local Laplacian**, well under half that of untouched pixels. This *is* §10.4's
  "detectable smoothed footprint" — the artefact a perception stack actually sees, and the thing a
  simulator that injects defects without replacing them, or that injects none at all, both fail to
  present.

**Passes are synchronous.** Each pass fills every masked pixel with at least one valid neighbour,
reading only values that were valid when the pass began — never values written during it. A 3×3's
centre therefore needs two passes and an isolated defect one. Without this, a 2×2 would fill
differently row-major than column-major and goldens would not reproduce; the test transposes the
problem and requires the answer to transpose with it.

**A partial fill is an error, not an output.** If a masked region is still unfilled after
`MAX_PASSES`, or has no valid neighbour on any side, the call raises. An unreplaced defect silently
carrying a stuck floor or ceiling value into the NUC is worse than a loud failure, and at
`CLUSTER_RADIUS_PX = 2` no region that large should exist.

The mask is **per frame**: dead and hot always, blinking and flickering only while bad
(`active_defect_mask`). This is the *ideal* mask — what a camera would replace if its map were
perfect. A real core's factory map is static, and what it fails to contain is exactly why
intermittent defects reach the image; that distinction belongs with the FFC controller in M9.7.
