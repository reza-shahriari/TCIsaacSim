---
name: sensor-noise-chain
description: Detector and signal-chain modelling for the IR camera simulator — photon detectors vs microbolometers, responsivity/NEP/D*, NETD anchoring, the NVESD three-dimensional noise model, fixed-pattern noise, bad pixels, NUC and flat-field correction, AGC/plateau equalisation, DDE and palettes. Use this skill whenever the task involves noise, NETD, SNR, detector response, integration time, well capacity, quantum efficiency, bolometer time constant, striping or fixed-pattern artefacts, dead pixels, shutter/FFC behaviour, dynamic range compression, or converting radiance into 8-bit display imagery. Low-level noise structure dominates sim-to-real transfer more than radiometric accuracy, so consult this before tuning radiometry to fix an appearance problem.
---

# Detector and signal chain

Everything from photons to pixels. This layer creates most of the *visual character* of a thermal image
and most of the sim-to-real gap. Networks overfit to noise structure, NUC residual and AGC behaviour far
more than to a 3 K error on a car door.

Reference: `docs/physics-model.md` §9, §10, §11.

## Two detector classes, one interface

Implement `Detector.response(band_power_or_photons, dt) -> counts` with two implementations.

### Photon detector — cooled MWIR / SWIR, and NIR

```
N_e = η · A_d · t_int · (π·τ_opt/(4F² + 1)) · ∫ R(λ)·L_q(λ) dλ  +  i_dark·t_int/q
DN  = clip(floor((N_e + noise) / N_well · 2**bits), 0, 2**bits − 1)
```

Uses the **photon** form of Planck. Dark current follows Arrhenius `i_dark ∝ T^1.5·exp(−E_g/2k_BT)` —
model the cooldown transient if startup behaviour matters. Expose `cold_shield_efficiency ∈ [0,1]`;
mismatch between the cold stop and the lens raises background flux and degrades NETD, and it is a real
trade-study knob.

### Microbolometer — uncooled LWIR

Different physics: absorbed power heats a thermally isolated membrane, changing its resistance. Uses the
**energy** form of Planck.

```
C_th · dΔT/dt = α_abs·Φ(t) − G_th·ΔT
τ_th = C_th / G_th                          # typically 8–12 ms
```

Implement as a per-pixel exponential IIR across frames:

```
S_n = S_{n−1} + (S_ideal_n − S_{n−1}) · (1 − exp(−Δt/τ_th))
```

Two lines, and it produces genuine motion smear: at 60 Hz with τ_th = 10 ms a moving object smears over
~0.6 frames. For a vehicle-mounted camera at speed, edges visibly trail. Most simulators omit this and
it is one of the strongest "this is a real uncooled camera" cues available.

Also model FPA temperature coupling: bolometer gain and offset are functions of the FPA's own
temperature, which drifts with ambient and self-heating. This is the physical origin of shutterless
drift — model it and NUC behaviour falls out naturally instead of having to be faked.

## NETD: anchor, don't generate

NETD is a **calibration handle**, not a noise generator. The workflow:

1. Build the full physical chain with best-estimate parameters.
2. Compute predicted NETD at a 300 K reference: `NETD = σ_total / (∂N_e/∂T)`.
3. Scale the noise terms so predicted NETD matches the datasheet number.
4. Now the spatial and spectral *structure* of the noise is physical and its *magnitude* matches the
   real device.

Critically: **NETD must fall as scene temperature rises**, because `∂L/∂T` rises with T. A real uncooled
core measured ≈39 mK at ambient and ≈23 mK against a 100 °C blackbody. If your simulated NETD is flat
across scene temperature, noise is being added in the wrong domain — find it and move it into radiance
or electron space.

When comparing against published NETD figures, check the convention: derivations differ in whether
target and atmosphere temperatures are treated as independent.

## The 3D noise model — use this, not a single sigma

NVESD decomposition into components along temporal (T), vertical (V) and horizontal (H) directions.
This is what real EO/IR test equipment reports, so it can be **parameterised from measurements of the
actual camera you intend to model**.

| Component | Meaning | How to synthesise |
|---|---|---|
| `σ_TVH` | random spatio-temporal (pixel temporal noise) | i.i.d. Gaussian, per pixel, per frame |
| `σ_VH` | fixed 2-D pattern (pixel non-uniformity, 1/f) | one fixed 2-D field, slowly drifting |
| `σ_V` | fixed row noise (line-to-line non-uniformity) | one value per row, fixed |
| `σ_H` | fixed column noise | one value per column, fixed |
| `σ_TV` | temporal row noise (ROIC line noise) | new value per row, per frame |
| `σ_TH` | temporal column noise | new value per column, per frame |
| `σ_T` | frame-to-frame bounce | one global value per frame |

`σ_total² = σ_T² + σ_V² + σ_H² + σ_TV² + σ_TH² + σ_VH² + σ_TVH²`

Configure as **ratios relative to σ_TVH**, with overall magnitude set by the NETD anchor. Starting point
for an uncooled LWIR core after NUC: `TVH:VH:H:V ≈ 1.0 : 0.3 : 0.15 : 0.08`. Cooled MWIR has smaller
`σ_VH` and `σ_TVH` dominant.

**Directional noise (row/column striping) is what makes thermal imagery look thermal**, and it is what
detection networks latch onto. Getting the ratios wrong is a larger contributor to sim-to-real gap than
radiometric error.

Do not freeze the fixed-pattern terms — give them a slow random walk (correlation time seconds to
minutes). That produces the characteristic "pattern breathing" between NUC events.

Seed all noise deterministically from `(frame_id, sensor_id)` so runs are reproducible and a failing
test can be replayed exactly.

## Bad pixels

0.05–0.5% of pixels, slightly clustered (Poisson cluster process, not uniform). Types: dead (stuck low),
hot (stuck high), flickering (random telegraph), blinking (intermittent).

**Simulate the defect *and* the camera's replacement of it.** Cameras replace bad pixels by neighbour
interpolation, leaving a detectable smoothed footprint — and that footprint is what a perception stack
actually sees. Modelling only the defect, or only clean pixels, both miss the real artefact.

## NUC and flat-field correction

```
DN_corr = G·(DN − O)
```

with two-point coefficients from two blackbody temperatures. **Model the residual, not the ideal:**

- Coefficients calibrated at one FPA temperature, applied at another → drift proportional to ΔT_FPA.
- Residual non-uniformity growing between shutter events.
- The FFC event itself: shutter closes, **image freezes for 0.5–1 s**, pattern resets.

That freeze is a real behavioural artefact a perception stack must survive. Simulating it is worth more
than another decimal place of radiometry. For shutterless cores, model slow drift plus a scene-based
correction residual instead of periodic freezes.

## AGC — part of the sensor model, not a display detail

A 14–16 bit radiometric image must become 8 bits, and the mapping changes the image drastically.

- **Linear with percentile clipping**: clip at 0.5%/99.5%, normalise, optional gamma.
- **Plateau equalisation**: histogram equalisation with per-bin counts clipped at a plateau value before
  integrating the CDF. Low plateau → near-linear; high → full HE. **This is what most real thermal cores
  use** and it is the algorithm behind the characteristic thermal look.
- **DDE**: high-pass boost added back over the compressed image. Unsharp mask with configurable gain.
  It is why real thermal imagery looks crunchy.

Model AGC as **global by default**, because that reproduces a genuine failure mode: a hot exhaust
entering frame collapses contrast on everything else. That is a real hazard for perception and it only
appears in simulation if AGC is modelled. Offer ROI-weighted and locally-adaptive variants as options.

**AGC must match between training and deployment.** Same algorithm, same plateau, same palette. A model
trained on linear-AGC synthetic data and deployed on plateau-EQ real data underperforms for reasons that
look like a physics problem and are not.

## Pipeline order

```
raw DN → bad-pixel replace → NUC (2-pt) → temporal filter
       → [linearise → T_apparent]                      # radiometric branch
       → AGC/DRC → gamma → DDE → polarity → palette    # display branch
```

Emit **both** branches. Perception usually consumes the 8-bit display image; validation needs the linear
one. A camera config declares whether it is radiometric — non-radiometric cores expose only the display
branch, and simulating that restriction honestly is part of modelling a specific camera.

## Tests

- NETD anchor: measured NETD from simulated two-blackbody imagery matches the config within 10%.
- NETD vs. scene temperature **decreases** — assert the slope, not just the value.
- 3D noise round trip: synthesise with known ratios, decompose 200 frames of a uniform field, recover
  the ratios within 5%.
- Bolometer smear: a step edge moving at known velocity produces the analytically predicted trailing
  profile for the configured τ_th.
- Determinism: same seed and frame id → bit-identical output.
- Well saturation: a very hot source clips at `2**bits − 1` and does not wrap or go negative.
- AGC: plateau = 0 behaves like linear; increasing plateau monotonically increases output entropy.
- FFC: a freeze of the configured duration occurs at the configured interval, and fixed-pattern residual
  is reset afterwards.
