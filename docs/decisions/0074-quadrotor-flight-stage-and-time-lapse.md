# ADR 0074 — A parametric quadrotor, its flight scene, and why the video is a time-lapse

**Status:** Accepted
**Date:** 2026-09-14

## Context

The phase-1 aerial stage (M10.19, ADR 0003) answers *can the camera detect this at all*: six
targets from 120 m to 2.5 km, most of them a pixel or less, which is the anti-UAV range problem.
It says nothing about the other question a sensor trade study asks — *what does a drone look like
once you can see it* — and it cannot, because its targets are flat quads at one temperature.

That second question is not cosmetic. What an infrared sensor keys on in a multirotor is not the
silhouette, it is that the **motors are hot, the speed controllers are warm, the battery is warm,
and the airframe is at air temperature**. Four thermal nodes on one object. Three of them were
already modelled — `irsim.thermal.aerial` has held §6.6's ΔT = ΔT_max·u^n law since M6.6 (ADR
0072) — and none of them were reachable: `heat_source_solver` and `airframe_solver` were imported
only by their own unit tests, because `irsim.config.scene` had no solver kind that named them. A
scene could ask for a Newton node or a typed temperature schedule and nothing else.

So a demo of a resolved, running drone needed three things that did not exist: an airframe whose
parts are separate objects, a way for a scene to ask for a motor, and a way to watch a
minutes-long thermal process in a video a person will actually sit through.

## Options considered

**1. Where the airframe comes from.**

  a. Download a quadrotor mesh (Sketchfab/CC0, NVIDIA's SimReady library, which is reachable from
     this machine). Realistic silhouette for free. But a mesh arrives as whatever sub-prims an
     artist happened to make, so every part's material and thermal node is assigned by hand and
     re-assigned for the next mesh; it arrives in arbitrary units, and angular size is the
     load-bearing quantity in this whole project; and it carries a licence to track.
  b. **Build it parametrically from USD primitives** (chosen). Every part is separate *by
     construction* — four motor bells, four speed controllers, four arms, a hub, a pack — so the
     thermal nodes attach themselves. Dimensions are *stated* rather than measured off a bounding
     box, which is what lets a test assert an angular size. No licence, no import step, no scale
     bug. At the ranges that matter a drone is a handful of pixels wide, so the fidelity a mesh
     buys over primitives is mostly invisible in the band this simulator models.
  c. Keep flat quads. Cannot express a per-part temperature at all, which is the entire subject.

  Downloaded meshes remain the right answer for an **aircraft** and a **bird**, where silhouette
  genuinely matters and the shape is not parameterisable. That is not this ADR.

**2. Propellers.**

  Modelled as solid discs, a 28-inch prop is 33 px across at 20 m and hides the motors underneath
  it — which are the subject. Physically, a prop at flight rpm sweeps its disc many times within a
  60 Hz integration period, so a thermal camera records a faint low-contrast annulus, not a blade.
  Doing that properly needs the motion path (M10.1b). **Omitted until then**: nothing is more
  honest than a wrong thing.

**3. How a scene asks for a motor.**

  a. Build the solvers in the render script and inject them into the `Scene`. Puts a flight
     profile in Python, where no config can reach it and nothing validates it.
  b. Bake the derived temperature schedule into the YAML as a `prescribed` node. Works today, and
     throws away the model: the config would hold several hundred numbers that no longer say what
     produced them, and a test comparing them against the law would be comparing a table with
     itself.
  c. **Two new solver kinds, `heat_source` and `airframe`, that name a throttle profile**
     (chosen). The config says *what the pilot did*; the temperature is derived through ADR 0072's
     law and refined until piecewise-linear interpolation reproduces it to 1 mK. Changing the
     flight is changing the flight. Scene schema v2 → v3.

**4. Real time or time-lapse.**

  This is the one that is a physics decision wearing a rendering decision's clothes.

  ADR 0072's law is `T_node(t) = T_air(t) + ΔT_max·u(t)^n` — a **steady-state relation**. There is
  no thermal time constant in it, so the node follows the throttle instantaneously. That is a fine
  approximation while the throttle moves slowly against a real motor's response, which is minutes.
  It is not fine at 60 Hz: a profile that swings the throttle over ten seconds would show 45 K of
  temperature change no lump of metal could follow, and the video would be a picture of the
  model's invalid region.

  a. Speed up the flight profile so the whole mission fits in a 10-second real-time clip. Looks
     the best and is a lie about what metal can do.
  b. **Author the flight over 1800 s and capture one frame every 6 s of it** (chosen). The
     throttle then moves slowly compared with the response the model assumes, and the result plays
     back as 10 seconds at 30 fps. `IrCamera` takes a `frame_period_s` override and **every stage
     is told the truth about the interval** — the FFC fires on its real schedule, the fixed
     pattern drifts by a real amount, and the temporal noise decorrelates between frames exactly
     as it would between two captures six seconds apart. It is a time-lapse camera, not a
     fast-forward button, and the burnt-in caption says so.
  c. Add a thermal time constant to the node model so a real-time clip becomes defensible. The
     right long-term answer and a change to a published model with its own tests and ADR. Not
     bundled into a demo.

**5. How the video is displayed.**

  Both of §11.3's AGC modes rescale from the current frame's own histogram. For a video whose
  entire subject is a temperature *change*, that is fatal in two ways: the gain follows the target
  and cancels the change, and plateau equalisation allocates display codes by population, so an
  aircraft covering under 1 % of the frame gets almost none of them and the whole airframe
  saturates to flat white with the motors indistinguishable from the arms. Measured, not
  predicted — it is what the first render of this stage looked like.

  The main video is therefore a **fixed span**, held for every frame: apparent temperature mapped
  through the ISP's own palette. *(Superseded in detail: the span was first taken as ±50 K about
  ambient, which spends half of 256 levels on sky-to-ambient and leaves every part of the target
  squeezed into the rest — measured at 76 display codes of spread across the airframe. It is now
  taken from the target's own nodes over the sequence, which gives 128. See ADR 0075.)* This is
  what an operator does when they
  switch to manual level and gain. The camera's own AGC output is filmed alongside it as a second
  video, because the difference between the two is itself the lesson.

## Decision

* `irsim_isaac.quadrotor` — a parametric heavy-lift quadrotor (1.8 m span, 110 mm bells), layout
  engine-free and unit-tested, USD authoring separate. Motor bells default to a high-emissivity
  anodised material: **bare aluminium is ε = 0.09 in this library** and would render a 70 °C motor
  as barely above the reflected sky.
* `irsim_isaac.quad_flight` — the stage: dome, tilted camera, one aircraft on boresight, tracked.
  Attitude is driven by the same throttle the thermal model reads, so the picture and the physics
  cannot disagree about what the aircraft is doing.
* `irsim.config.scene` v3 — `heat_source` and `airframe` solver kinds, making ADR 0072's model
  reachable from configuration for the first time.
* `configs/scenes/quad_flight_clear_noon.yaml` — a 30-minute mission authored as throttle, at
  midday so the companion visible frame has a blue sky (ADR 0073).
* `irsim_eval.video` — burnt-in readout and ffmpeg encoding. In `irsim_eval` because it needs PIL,
  which CLAUDE.md keeps out of the physics core.
* `scripts/render_quad_flight.py` — the command.

## Consequences

**What it makes easy.** Seeing the thing the band is actually about: four motors climbing from
ambient to +45 K and back over a mission while the airframe stays at air temperature. The three
heat sources share one throttle history and differ only in ΔT_max, so motor > ESC > battery >
airframe holds by construction at every instant, and there is a test that says so at 120 sampled
points of the flight.

**What it does not do, and what would be wrong to read into it.**

* **No thermal time constant.** The node temperature follows throttle instantaneously. Every
  transient in the video is a *throttle* transient. A motor that really was stepped to full power
  would lag by minutes, and neither the model nor the video shows that.
* **ΔT_max is ESTIMATED** (45 / 30 / 15 K, ADR 0072). No public dataset of instrumented multirotor
  motor temperatures was available. The *ordering* is the defensible part; the magnitudes are the
  first thing a Tier 4 comparison against public aerial IR should re-fit.
* **No propellers**, so nothing occludes or shadows as a rotor would.
* **The camera tracks the aircraft**, so in-frame motion is tracking error and attitude, never
  translation. A drone on a 30-minute mission covers kilometres; nothing that flies realistically
  stays in a 33° field.
* **The fixed span is a display choice and carries no measurement.** The burnt-in gauge and the
  JSON sidecar carry the temperatures; the colours carry only contrast.

**Error introduced:** none in the radiometric chain — this adds a stage and a solver kind, and
changes no existing physics. The scene schema bump is breaking for any v2 config; the one in the
repository is migrated in the same commit.

## Revisit when

* A motor thermal time constant lands — then a real-time clip becomes defensible and option 4(c)
  replaces the time-lapse framing, and the throttle profile can carry step changes.
* Public instrumented multirotor thermal data turns up — re-fit ΔT_max before anything else.
* M10.1b delivers the motion path — then propellers can be modelled as the smeared annulus they
  actually are, and the tracking jitter can become real motion blur.
* An aircraft or a bird is added — those want downloaded meshes, not this, and will need the
  scale-and-orientation assertion that a parametric airframe makes unnecessary.
