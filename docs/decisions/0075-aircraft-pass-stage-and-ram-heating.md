# ADR 0075 — A fixed-wing pass stage, aerodynamic skin heating, and a mount that slews

**Status:** Accepted
**Date:** 2026-09-14

## Context

ADR 0074 built a quadrotor stage whose variable is *throttle*: four motors a sixteenth of the span
across, resolvable at close range, warming over half an hour. An aircraft is the opposite problem
in every respect that matters, and running it through the same machinery would have produced a
believable picture of the wrong physics.

* Its hot part is an exhaust nozzle a **fortieth** of the span across but hundreds of kelvin above
  ambient — a near-point source, bright because it is hot rather than because it is large.
* That nozzle is **occluded from the front** by its own nacelle, which is why a real aircraft's
  measured infrared signature varies by a large factor around the clock and why rear-aspect
  detection ranges are the ones quoted. Nothing about the *target* changes during a pass; the
  variable is **aspect**.
* Its skin is **not at air temperature**. `irsim.thermal.aerial.airframe_solver` says an unpowered
  skin sits at ambient, and justifies it by forced convection at flight speed — which is true at
  20 m/s and false at 200. The stagnating boundary layer heats the skin to its recovery
  temperature, 10 K above ambient at M 0.43 and 34 K at M 0.8.

`docs/physics-model.md` has nothing to say about any of this. §6.6 is an automotive heat-source
table, and the spec has no aircraft section at all; the whole aerial lane (ADR 0003, 0072, 0074) is
an extension the project made deliberately. This ADR records another one.

## Options considered

**1. Where the airframe comes from.** Same answer as ADR 0074 and for the same reasons —
parametric primitives, so every part is separate by construction, dimensions are *stated* rather
than measured off an imported bounding box, and there is no licence or unit-scale bug to carry.
A downloaded mesh remains the right call when silhouette fidelity becomes the point; it is not yet.
Rear-fuselage engines rather than under-wing pods, deliberately: both nozzles then sit on the
centreline, so occlusion is a clean function of one aspect angle instead of a wing-shadowing
problem.

**2. The skin temperature.**

  a. Reuse `airframe` with a hand-tuned `offset_k`. Works, and throws away the reason for the
     offset: the number would be typed rather than derived, would not follow the airspeed, and
     would not follow the shared weather.
  b. **A `ram_skin` node computing the adiabatic wall temperature** (chosen):
     `T_r = T_air (1 + r (γ−1)/2 M²)` with `r = Pr^⅓ ≈ 0.892` for a turbulent boundary layer. The
     config names an **airspeed**; the Mach number is taken against the shared `WeatherSeries`'s
     own air temperature, so one true airspeed is correctly a different Mach number on a cold day.
     Standard boundary-layer result (e.g. White, *Viscous Fluid Flow*, ch. 7).
  c. A full skin energy balance with solar gain and radiation to the sky. More correct and much
     more machinery; for a cruising aircraft those terms are small against the ram term. Noted
     below as the bound this model does not carry.

  The two skin models are kept **separate and non-interchangeable** in the schema: `airframe`
  refuses an airspeed, because it *assumes* a slow one and silently ignoring a stated 200 m/s is
  exactly the failure this is meant to prevent.

**3. The engine nodes.** Prescribed constants. A jet at steady power has a steady nozzle, the pass
lasts ten seconds, and saying so in the config is more honest than dressing a constant up as a
schedule. Values are **ESTIMATED**: nacelle 340 K, nozzle 450 K.

**4. How the camera follows the target.**

  a. Rotate the whole scene onto a fixed boresight — the cheap trick, and what was built first.
     It puts the target at the right range and aspect, and holds **one sky** behind an aircraft
     whose true elevation sweeps 17° → 37° → 17°. The sky is tens of kelvin colder at the top of
     that. A perfectly plausible frame of a camera that is not the one being simulated.
  b. **Fly the aircraft on its true track and slew the mount onto it** (chosen). Needs
     `IrCamera.refresh_pose()`, because the camera pose that every pixel's ray direction is built
     from is cached at `open()` — without a refresh the geometry AOVs follow the re-aimed prim
     while the elevations, the sky temperature and every view cosine stay at the opening aim. The
     aim is an azimuth/elevation construction rather than the minimal rotation between two
     vectors: the minimal rotation is shorter and rolls the horizon as it slews, which no
     two-axis pedestal does.

**5. Real time or time-lapse.** ADR 0074 filmed a *thermal* process, which is slow, so a time-lapse
was the only honest framing. This process is *geometric* and fast: ten seconds captured at 30 Hz
and played at 30 fps, with no speed-up to declare. Tracking is also what disposes of the motion
blur objection — a stabilised line of sight holds the target still on the focal plane and smears
the featureless sky behind it instead.

## Decision

* `irsim_isaac.airframe` — `Part` and `author_parts`, extracted from `quadrotor.py` so that the
  `thermal:material` override and the prim→node map have exactly one implementation. A second
  authoring path would be a second chance to leave a surface with no emissivity or no temperature.
* `irsim_isaac.aircraft` — a parametric light business jet, 16 m span, rear-mounted engines.
* `irsim_isaac.aircraft_pass` — the track, the look-at mount, and the stage.
* `irsim.thermal.aerial` — `speed_of_sound_m_s`, `mach_number`, `recovery_temperature_k`,
  `ram_skin_solver`; `R_SPECIFIC_AIR`, `GAMMA_AIR`, `PRANDTL_AIR` added to
  `irsim.radiometry.constants` with sources, never inline.
* `irsim.config.scene` **v4** — the `ram_skin` solver kind.
* `IrCamera.refresh_pose()` and `configs/scenes/aircraft_pass_clear_noon.yaml`,
  `scripts/render_aircraft_pass.py`.

## Consequences

**What it makes easy.** Seeing the aspect dependence that dominates aircraft infrared detection:
the same aircraft at the same range shows several times more nozzle from behind than from ahead,
and its hottest pixel is tens of kelvin warmer. Measured off the instance-id plane, so it is a
count and not an impression. The pass also sweeps range 2:1, which exercises the inverse-square
fall-off and the atmospheric path against a target of known size and temperature at no extra cost.

**A defect this work exposed.** `PipelineConfig.from_sensor` defaults `flat_field_enabled=False`,
and both flight scripts had simply not passed it — so the AGC companion videos carried the full
21 % cos⁴ vignetting that M9.12 exists to remove, stretched into dark corners by plateau
equalisation. The radiometric branch was unaffected — it divides cos⁴ out analytically, and
`tests/unit/test_flat_field.py` already asserted that `radiance`, `apparent_t` and `dn16` come out
bit-identical either way while `display8` changes — so the main fixed-span videos were always
correct and only the companion was wrong. Both scripts now enable it, with a `--no-flat-field`
ablation. The lesson is narrower than "call the function": a default that is right for a library
(the radiometric branch does not need a flat field) was wrong for every caller that also wanted a
picture.

**What is not modelled, and must not be read into a frame.**

* **No exhaust plume.** The gas behind the nozzle radiates in CO₂ and H₂O bands, not as a grey
  surface, and it is not a surface the renderer can carry. For a rear aspect in MWIR the plume can
  rival the nozzle; in LWIR it matters less. §6.6 makes the same point about automotive exhaust.
* **ΔT is ESTIMATED.** Nacelle 340 K and nozzle 450 K are plausible descent-power figures for a
  low-bypass turbofan; published nozzle-skin temperatures span roughly 400–700 K with bypass ratio
  and power setting. The *ordering* — nozzle > nacelle > skin > air — is the defensible part.
* **The recovery temperature is the flat-plate one**, applied to the whole skin. A real leading
  edge runs hotter still, an intake is a cavity that reads warmer than the skin, and neither solar
  gain nor radiation to a cold sky is in the node. For a cruising aircraft those are small against
  the ram term; for a slow one they are not, and `airframe` with an offset is then the honester
  model.
* **The nozzle material is a stand-in.** A turbofan nozzle is oxidised Inconel or titanium and the
  library has no high-temperature alloy, so painted aluminium (ε = 0.90 LWIR) carries it. Close to
  a hot oxidised metal, and very far from the ε = 0.09 *bare* aluminium a careless mapping would
  pick, which would render an 800 K nozzle as barely warm.
* **No motion blur.** Tracking makes this nearly moot for the target and irrelevant for a
  featureless sky, but a resolved background would smear and does not.

**The display span is taken from the target, not the scene.** ADR 0074 spanned ±50 K about
ambient; with eight bits that spends half of 256 levels on sky-to-ambient — two flat regions — and
leaves every part of the target in the rest. Measured on the quadrotor at full throttle, like for
like over the airframe's own 1000 pixels: the old rule gave 76 display codes of spread between the
10th and 90th percentile, the new one gives 128, and the cool parts now start near black instead of
mid-grey. The floor sits a quarter of the node spread *below* the coldest node rather than on it,
because a node at the very bottom of the span is invisible and at the start of a flight every node
is at ambient. The sky falls below the floor and clips to black: it is the region with least to see
in and it was costing the target all of its contrast. `--span-c` restores a scene-context span.

For the aircraft this helps and cannot fully succeed: skin and nozzle are 136 K apart, so a linear
8-bit span that reaches the nozzle leaves the skin dark (codes 47 / 84 / 236 for skin / nacelle /
nozzle). That is the honest consequence of the physics rather than a display bug, and it is why a
real operator re-spans rather than looking for one setting.

**Error introduced:** none in the existing chain. `ram_skin` is a new node kind, `refresh_pose` is
a no-op unless the camera is moved, and the schema bump is breaking only for v3 configs — both in
the repository are migrated in the same commit.

## Revisit when

* A band with a plume-dominated signature is wired up (MWIR) — the plume stops being optional.
* Public instrumented aircraft infrared imagery turns up — re-fit the nacelle and nozzle
  temperatures before anything else, and check the aspect ratio of the signature against it.
* A high-temperature alloy material is added — the nozzle should use it rather than paint.
* Anything flies above about M 1 — the recovery temperature is still right, but the skin then has
  a real radiation balance to solve rather than an equilibrium to read off.
