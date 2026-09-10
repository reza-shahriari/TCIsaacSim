# ADR 0003 — Sky targets first, validated against public data

**Status:** Accepted
**Date:** 2026-09-10

## Context

The simulator's first application is detecting objects in the sky — drones, aircraft, birds — against a
sky background, from a ground-based or vehicle-mounted LWIR camera. Ground scenes and the automotive use
case in `docs/physics-model.md` remain the long-term target but come second.

No IR camera is available to the team. One may be obtainable later, but planning around it would stall
validation indefinitely. What *is* available, free, on the internet, is a large amount of real thermal
video of exactly the target class: Anti-UAV410 (410 thermal videos, 438 K boxes; sensor undocumented,
frames are the camera's display output), CST Anti-UAV (220 sequences, tiny targets, cloud clutter), the
Halmstad drone-detection dataset (Svanström et al. 2021, CC0: 365 ten-second clips at 320×256 from a
FLIR Breach PTQ-136 / Boson core at 60 fps, with birds and aircraft as confusers), LRDDv3 (air-to-air,
ranges ≤ ~200 m with GPS truth), and the single-frame small-target sets IRSTD-1k / NUAA-SIRST /
NUDT-SIRST (sensor unknown; usable for target size and contrast priors only).

What these frames are must be recorded per set before any statistic is trusted. The Halmstad clips are
the Boson's raw Y16 stream converted to 8 bits by the recorder and stored as mp4 — so they carry the
core's noise, striping and FFC freezes but **not** its AGC, DDE or palette. Anti-UAV410 / CST frames
are display output through an unknown ISP, stored lossy. Flat-sky regions in all of them are the
equivalent of a laboratory blackbody for noise structure, within the limits of 8-bit quantisation and
the codec.

The physics consequences of the sky-first scope are not neutral. It removes the need for a ground thermal
solver and a broad material library at first, but it makes the sky radiance model the *background*
(elevation-dependent, per band, with clouds as the dominant clutter), pushes ranges to 0.5–5 km on slant
paths — past the ~500 m validity the spec accepts for band-averaged Beer–Lambert (Appendix A #2) — and
makes sub-pixel target rendering (MTF, sampling, aliasing, point-target radiometry) first-order rather
than a late refinement. §5.3 of the spec already warns that a constant sky background "badly mismodels a
system's ability to detect small aerial targets".

## Options considered

1. **Follow §16.4 literally (ground/automotive first), add sky targets later.** Cleanest against the
   spec; but every early validation would be against ground phenomenology we cannot measure, and the
   first deliverable would be a camera nobody can point at a drone.
2. **Sky targets first, validated against public anti-UAV video; ground phase second.** Reorders the
   roadmap, adds a public-data evaluation harness early, adds clouds, slant-path atmosphere and
   point-target radiometry, defers the ground thermal solver and material breadth. Tier 2 benches remain
   self-consistency until a camera exists; Tier 4 and Tier 5 become possible now.
3. **Buy or borrow a Boson and do the §15 lab benches first.** Best evidence per hour once the hardware
   exists; not available.

## Decision

Option 2. Concretely:

- **Validation data:** public datasets only, tracked under `data/validation/` with licence and
  provenance notes; no plan step may depend on owning hardware. If a camera later becomes available,
  the §15 lab benches are added as a bonus tier, not a prerequisite.
- **What is compared:** on the simulator's 8-bit output after the *same signal path as the reference
  set* — the camera's AGC/DDE where the frames are display output, or a linear recorder conversion
  where they are Y16-derived — followed by the same codec, never on radiance. Measured on real frames
  and reproduced in sim: temporal and spatial noise (3-D decomposition, PSD, striping lines, with a
  codec/quantisation floor stated per set), between-FFC fixed-pattern growth, bad-pixel footprints,
  FFC freeze length (and interval, where sequences are long enough), AGC histogram signature and DDE
  overshoot (display-output sets only), scale-free sky brightness profile vs elevation, cloud clutter
  spectra, target size and signal-to-clutter ratio vs range (to ~200 m — beyond that the range regime
  is modelled, not validated), contrast polarity, edge spread and motion smear (edge asymmetry vs
  velocity).
- **Task-level evaluation is a deliverable:** a detector train/evaluate harness with the
  synthetic→real, real→real, real+synthetic→real protocol and a one-feature-at-a-time ablation so that
  fidelity spending is steered by measured AP, not by taste.
- **Ordering:** phase 1 (sky) = baseline, radiometry, evaluation harness, radiance kernel, noise, ISP,
  sky/clouds/slant-path/point targets, sensor dynamics, Isaac integration, Tier 4/5 on public data.
  Phase 2 (ground/automotive) = ground thermal solver, full material library, Fresnel angular models,
  second and third bands, ground phenomenology.

## Consequences

- Makes easy: an honest sim-to-real number within the first release; evaluation code that outlives the
  scope change; noise/ISP realism getting the attention §15 says it deserves.
- Makes hard: absolute radiometric validation (public frames carry no temperature truth) — Tier 2 SITF
  and NETD remain self-consistency against ESTIMATED datasheet values until hardware exists. The error
  this introduces cannot be bounded from public data; it is stated in README limitations.
- Introduces new physics with its own unbounded approximations: a cloud-clutter model (no spec section
  exists; it will be authored and recorded in its own ADR), elevation/air-mass-dependent path radiance
  beyond the spec's 500 m guidance, and analytic point-target injection below ~0.1 px.
- The Isaac Sim interpreter (`/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh`, Python
  3.12, NumPy 2.3) is the project interpreter for both halves of the repo; the Makefile gains a
  `PYTHON` variable.

## Revisit when

- A real IR camera becomes available (add the §15 lab benches; re-anchor NETD and 3-D ratios to
  measurements and replace every `ESTIMATED` field).
- The ground/automotive phase starts (re-instate §16.4 ordering for the deferred milestones).
- Tier 5 shows that a deferred fidelity item (e.g. Fresnel angular emissivity, cubemap reflections)
  moves detector AP by more than the noise/ISP items do.
