# Thermal (Infrared) Camera Simulation Model

A point-wise radiometric pipeline: temperature, emissivity, and range are per-scene-point quantities, and every stage below is evaluated per pixel from those point values — not once per object.

## Notation

| Symbol | Meaning | Typical units |
|---|---|---|
| `T` | temperature | K |
| `ε(λ)` | spectral emissivity | – (0–1) |
| `λ` | wavelength | m |
| `L(λ,T)` | spectral radiance | W·m⁻²·sr⁻¹·m⁻¹ |
| `τ(λ,R)` | atmospheric transmittance | – (0–1) |
| `γ(λ)` | extinction coefficient | m⁻¹ |
| `R` | range, point → camera | m |
| `N` | optical f-number | – |
| `E` | focal-plane irradiance | W·m⁻² |
| `NETD` | noise-equivalent temperature difference | K |
| `DN` | raw digital count | counts |

Physical constants: `h = 6.626e-34 J·s`, `c = 2.998e8 m/s`, `k_B = 1.381e-23 J/K`, `σ = 5.67e-8 W·m⁻²·K⁻⁴` (Stefan-Boltzmann).

## 0. Core Architecture — Point-Wise, Not Object-Wise

Give every surface sample (mesh vertex, texel, terrain grid cell) its own state:

- `T(x,y,z)` — temperature field, e.g. a texture/heightmap-correlated function, exactly like an albedo map but for temperature instead of color
- `ε(x,y,z,λ)` — emissivity, usually gray-body (a single scalar per material rather than a full spectral curve)
- Position/normal, for projection and (optionally) angular effects

At render time, for each output pixel: find the visible point (rasterize + z-buffer, or ray march), pull its `T`, `ε`, and computed range `R`, then run steps 1–5 below for that pixel alone. Steps 6–11 are the only parts that operate on the whole frame (blur mixes neighboring points; noise and color mapping are display-level operations). This is what makes "one ground object with warm and cool patches" work — the ground is one mesh, but temperature is a field over it, not a scalar property of it.

## 1. Surface Emission — Planck's Law

Blackbody spectral radiance:

```
L_bb(λ,T) = (2hc²/λ⁵) · 1 / (exp(hc/(λ·k_B·T)) − 1)
```

Real point on a surface (opaque, diffuse, gray-body: ε+ρ=1, no transmission):

```
L_point(λ) = ε(λ)·L_bb(λ, T_point) + (1−ε(λ))·L_bb(λ, T_env)
```

- `ε`: ~0.95–0.98 for skin, vegetation, most paints/soils; can drop to 0.05–0.2 for polished/bare metal — low-ε surfaces mostly show a reflection of their surroundings, not their own temperature
- `T_env`: effective temperature of whatever the point reflects (sky, surrounding terrain). Treating it as one blackbody temperature is a simplification — see §14 for a full reflection model
- Gotcha worth simulating correctly: glass is transparent in visible light but opaque and highly emissive in LWIR — model windows as ordinary opaque emitters in this band, not as see-through

**Band integration.** The sensor sees a band, not one wavelength. Weight by system spectral response `R_sys(λ) = τ_optics(λ)·R_detector(λ)`:

```
L_band(T) = ∫(band) L_bb(λ,T)·R_sys(λ) dλ
```

Precompute `L_band(T)` as a lookup table or fitted curve over your working temperature range (e.g. 250–400 K) once, rather than re-integrating per pixel per frame. Common bands: MWIR 3–5 μm, LWIR 8–14 μm (LWIR is what most uncooled/consumer thermal cameras use).

*Sanity check only:* total broadband exitance is `M = εσT⁴` (Stefan-Boltzmann) — don't use this for the actual signal, since the sensor only responds to `L_band`, a narrow slice of the spectrum.

## 2. Geometric Projection — Per-Point Range

Standard pinhole projection. Camera position `C`, rotation `R_cam`, intrinsics `(f_x,f_y,c_x,c_y)`:

```
p_cam = R_camᵀ · (p_point − C)
u = f_x · p_cam.x/p_cam.z + c_x
v = f_y · p_cam.y/p_cam.z + c_y        (valid when p_cam.z > 0)

R(u,v)       = ‖p_point − C‖                                  (range)
θ_field(u,v) = angle between line-of-sight and optical axis   (used in §4)
```

`R` and `θ_field` are per-pixel by construction — on a single ground mesh this varies continuously from a few meters underfoot to kilometers at the horizon. Where multiple points project to the same pixel, keep the nearest (ordinary z-buffer/depth test), same as any rasterizer.

## 3. Atmospheric Propagation — The Distance Effect

Radiative transfer along a homogeneous path of length `R` (absorption/emission only, no scattering-in — standard for clear-to-moderate haze, moderate range):

```
τ(λ,R) = exp(−γ(λ)·R)                                  (Beer-Lambert transmittance)
L_received(λ) = τ(λ,R)·L_point(λ) + (1−τ(λ,R))·L_bb(λ,T_atm)
```

The second term is path radiance / "airlight" — the atmosphere itself glows along the path and eventually swamps the target signal at long range. Because `R` is per-pixel, both terms are per-pixel: a nearby ground point is barely attenuated; a farther point on the same mesh is attenuated more and mixed with more path radiance. That's the "distance effect," and it falls out naturally once range is point-wise.

`γ(λ)` = absorption (water-vapor continuum dominates LWIR; H₂O/CO₂ bands matter in MWIR) + scattering (aerosols; Rayleigh scattering is negligible at IR wavelengths). For simulation, an effective in-band coefficient per weather condition is a practical middle ground; for real fidelity, replace with MODTRAN/LOWTRAN-derived values. Illustrative starting orders of magnitude (not measured constants):

| Condition | γ_eff LWIR (km⁻¹) | γ_eff MWIR (km⁻¹) |
|---|---|---|
| Clear, dry | 0.05–0.15 | 0.1–0.3 |
| Humid / haze | 0.2–0.4 | 0.3–0.6 |
| Fog / light rain | 1+ | 1+ |

`T_atm`: local air temperature is fine for short/medium range; integrate a vertical profile for long slant paths through a stratified atmosphere if needed.

## 4. Optics — Radiance to Irradiance

Standard radiometric camera equation (scene effectively at infinity, magnification ≈ 0):

```
E(u,v) = L_received(u,v) · τ_optics · π/(4·N²) · cos⁴(θ_field(u,v))
```

- `N = f/D`: f-number. IR lenses (germanium/chalcogenide) are commonly f/1.0–f/1.4 — "faster" than visible-light lenses because thermal photon/power levels are much lower
- `τ_optics`: lens transmittance, ~0.7–0.9 typical
- `cos⁴θ_field`: classic off-axis illumination falloff (vignetting) — point-wise, since `θ_field` depends on pixel position
- **Narcissus effect (Artifact)**: The cooled detector array reflects off the camera's own lens surfaces back onto itself. This appears as a subtle, dark (cold) circular vignette or spot in the center of the image. To simulate, apply a radial, low-frequency negative offset to `E` centered on the optical axis.

## 5. Detector — Irradiance to Raw Signal

Pick the architecture matching your target sensor.

**(a) Uncooled microbolometer** (LWIR — the common case, e.g. FLIR Boson/Lepton/Tau-class):

```
P_abs(u,v) = A_pixel · α_abs · E(u,v)                    (α_abs ≈ 0.9 typical)

C_th · d(ΔT_px)/dt = P_abs(t) − G_th·ΔT_px(t)             τ_th = C_th/G_th   (~8–12 ms typical)
steady state:  ΔT_px = P_abs / G_th
```

Because `τ_th` is comparable to a frame period, fast-moving scene content should really be integrated through the ODE rather than assumed steady-state — this thermal lag is a genuine, sensor-specific motion-blur source.

```
ΔR/R₀ ≈ TCR · ΔT_px                    (TCR ≈ −2 to −4 %/K for VOx / a-Si)
```

The readout (bridge circuit) turns `ΔR` into a signal roughly linear in `ΔT_px`, hence roughly linear in `E`. For simulation, collapse the chain into one responsivity constant:

```
Signal_raw(u,v) = K_resp · E(u,v) + Offset₀
```

Get `K_resp` from the physical parameters above, or — easier, and usually good enough — fit it so the model reproduces your target NETD (§8).

**(b) Cooled photon detector** (MWIR — InSb/HgCdTe/T2SL, higher-end systems):

```
N_electrons(u,v) = (A_pixel·t_int·η/(h·c)) · ∫(band) E_λ(u,v)·λ dλ
```

`η` = quantum efficiency. Microsecond-level response (no thermal-lag blur), but needs cryogenic cooling to suppress dark current; typically lower NETD than bolometers.

## 6. Blur — Point Spread Function / MTF

Neighboring points genuinely mix at this stage. Multiply these MTFs in the frequency domain:

```
Diffraction (circular aperture):
  MTF_diff(f) = (2/π)[cos⁻¹(f/f_c) − (f/f_c)·√(1−(f/f_c)²)],  f ≤ f_c
  f_c = 1/(λ·N)

Detector footprint (pixel pitch d, ~12–17 μm typical LWIR):
  MTF_det(f) = |sinc(f·d)|
```

Add a small extra kernel (Gaussian/exponential) for bolometer thermal cross-talk between neighboring pixels if you want to match a real part's datasheet MTF.
- **Thermal Blooming (Cross-talk)**: Extreme heat (e.g., the sun, a fire) conducts laterally through the sensor substrate, heating adjacent pixels. Convolve the raw signal with a tight, high-magnitude exponential kernel specifically for pixels that hit the saturation limit.
- **Shallow Depth of Field (DoF)**: Because IR lenses are extremely "fast" (f/1.0 - f/1.4), they have a very shallow Depth of Field. Instead of a single system MTF PSF, you need a depth-dependent blur (circle of confusion) based on the per-pixel range (`R`). Focusing on a near object should heavily blur the background.

System MTF ≈ product of the above; convert to an equivalent PSF (or depth-dependent blur) and convolve the noise-free point-wise image with it.

Apply this **before** noise, not after — noise originates at the detector plane at full pixel resolution and is uncorrelated pixel-to-pixel, so blurring it afterward would smear something that physically isn't blurred.

## 7. Fixed-Pattern Noise / Non-Uniformity

```
Signal_measured(u,v) = G(u,v)·Signal_raw(u,v) + O(u,v)
```

- `G(u,v)`: residual gain non-uniformity after factory NUC, e.g. Gaussian(mean=1, σ≈1–3%)
- `O(u,v)`: fixed-pattern offset (this is the per-pixel deviation from the flat `Offset₀` in §5) — often structured, not fully independent per pixel: per-column or per-row components (shared readout electronics) are common and show up as visible streaking, on top of a fully independent per-pixel term
- Bad/dead pixels: flag a small random fraction (<0.1–1%) as stuck; a real camera then interpolates over these — include or skip depending on whether you're simulating pre- or post-correction output
- **NUC Shutter Freeze (Artifact)**: Uncooled bolometers drift with ambient temperature. To fix this, they periodically drop a mechanical shutter (a uniform temperature reference) over the sensor to re-zero the offsets. To simulate this highly realistic effect, trigger a periodic event where the video freezes for ~0.5 to 1.0 seconds, a flat field is shown, and the fixed-pattern noise `O(u,v)` resets to near-zero, slowly growing again over time.

## 8. Temporal Noise

NETD is the standard figure of merit: the ΔT that would produce a signal change equal to the noise. Typical uncooled bolometer: 30–50 mK; cooled photon detectors can be <20 mK.

```
σ_noise = NETD · (dSignal/dT)      [evaluate the derivative of the full chain (§1–§5) at your reference scene T]
```

Add zero-mean noise with this `σ`, independently per pixel per frame, as the baseline "temporal, spatially-white" component. Other sources worth layering on:

- Photon shot noise (photon detectors): Poisson, `σ = √N_electrons` — cheap to sample directly instead of Gaussian-approximating
- Dark current + its own shot noise (photon detectors; grows sharply with detector temperature — why cryocooling matters)
- Johnson-Nyquist / 1/f (flicker) noise (bolometers): slow, correlated drift — model as a random-walk/AR(1) process across frames, not white noise, if simulating video
- Row/column-correlated noise: one shared random value per row or column per frame, layered on top of the per-pixel term
- Quantization noise (§9): `σ_quant ≈ LSB/√12`

This decomposition (independent-per-pixel / row / column / frame-to-frame) is the standard "3D noise model" framework used to characterize real IR imagers — useful vocabulary if you want to match a real sensor's published spec.

## 9. Quantization (ADC)

```
DN_raw(u,v) = round(clamp((Signal(u,v) − V_min)/(V_max−V_min) · (2^bits−1), 0, 2^bits−1))
```

Radiometric thermal cameras typically output 14–16 bit raw (to preserve temperature resolution) even though the displayed image is usually 8-bit after §10.

## 10. AGC — Dynamic Range Compression

A scene's real temperature span is usually a small slice of the sensor's full dynamic range, so cameras auto-scale before display:

```
Linear (percentile-clipped, e.g. 1st/99th to reject outlier hot/cold pixels):
  DN_8(u,v) = 255 · clamp((DN_raw − DN_low)/(DN_high − DN_low), 0, 1)

Histogram equalization:
  DN_8(u,v) = 255 · CDF(DN_raw(u,v)) / CDF_max
```

Histogram equalization uses contrast better; a common refinement ("plateau equalization") caps how much any single histogram bin can stretch, so one small hot spot doesn't wash out the rest of the scene. For video, smooth `DN_low`/`DN_high` across frames (e.g. an exponential moving average) so the picture doesn't visibly flicker as the AGC readjusts.

## 11. Palette — Signal to Color

```
RGB(u,v) = Palette[DN_8(u,v)]
```

- Grayscale: "White Hot" (`RGB = DN,DN,DN`), "Black Hot" (`RGB = 255−DN` ×3)
- Pseudocolor ("Ironbow"-style or similar): define N control-point colors across [0,255] and linearly interpolate channel-wise between them for any DN. Open colormaps like matplotlib's inferno/magma/plasma or Google's Turbo look and behave like the proprietary vendor palettes and come with reusable formulas/LUTs
- Isotherm highlighting: an easy add-on — override the palette with a flat contrasting color wherever apparent temperature crosses a threshold you set

## 12. Full Pipeline — Per-Pixel Flow

For every output pixel `(u,v)`:

```
1. Find visible surface point → T, ε, R, θ_field         (§1–2)
2. Point emission + reflected environment → L_point(λ)   (§1)
3. Atmosphere: τ(R), + path radiance → L_received         (§3)
4. Optics → irradiance E                                  (§4)
5. Detector → raw signal                                  (§5)
```

Then, over the whole frame:

```
6. Convolve with system PSF                (§6)
7. Apply per-pixel gain/offset FPN          (§7)
8. Add temporal + row/column noise          (§8)
9. Quantize                                 (§9)
10. AGC                                     (§10)
11. Palette → final RGB                     (§11)
```

Steps 1–5 are where your "one ground mesh, many temperatures" lives; steps 6–11 are whole-array operations layered on top.

## 13. Typical Parameters — Starting Points

| Parameter | Uncooled LWIR bolometer | Cooled MWIR photon detector |
|---|---|---|
| Band | 8–14 μm | 3–5 μm |
| NETD | 30–50 mK | <20 mK |
| Pixel pitch | 12–17 μm | 15–30 μm |
| f-number | ~1.0–1.4 | ~2–4 |
| Frame rate | 30/60 Hz | up to 100s Hz |
| Thermal/response time | ~8–12 ms | μs-level |
| Raw bit depth | 14–16 bit | 12–16 bit |

Representative, not universal — use a datasheet if you're matching a specific part.

## 14. Extending Further

- **Realistic `T(x,y,z)` field**: derive it from a surface energy balance (solar absorption − IR emission − convection ± conduction) instead of hand-painting it, if you want physically-driven hot/cool regions — sunlit vs. shaded ground, evaporative cooling, engine heat. This is a separate model from the camera itself, but it's the natural source for the per-point field everything above consumes. **→ See `thermal_object_dynamics.md`** for the full time-varying model: engine warm-up/cool-down, water/sea thermal mass, ice phase-change, and fire's volumetric emission.
- **Full reflection**: replace the single-`T_env` approximation in §1 with a real hemispherical integral or environment map, important for low-ε (shiny/metal) materials
- **Directional emissivity**: real materials' emissivity rises toward grazing viewing angles (e.g. water looks progressively less "hot" and more mirror-like near the horizon)
- **Solar glint (MWIR)**: the sun's own blackbody spectrum has a non-trivial tail into 3–5 μm; specular reflections off smooth/metallic surfaces can appear as bright false hot-spots
- **Slant-path atmosphere**: integrate `γ(λ)` and `T_atm` along the true ray through a stratified atmosphere instead of assuming one homogeneous path, for large altitude differences
- **Target motion blur**: separate from optical PSF — a hot point moving across multiple pixels during the integration time smears along the motion direction

## References

**Textbooks (whole-document background)**
- M. Vollmer & K.-P. Möllmann, *Infrared Thermal Imaging: Fundamentals, Research and Applications*, 2nd ed., Wiley-VCH, 2018 — the standard field-wide reference. https://onlinelibrary.wiley.com/doi/book/10.1002/9783527630868
- G. C. Holst, *Electro-Optical Imaging System Performance*, 6th ed., SPIE Press, 2017 — heavier on system/sensor performance (NETD, MTF, range prediction). https://spie.org/Publications/Book/2588947
- P. W. Kruse & D. D. Skatrud (eds.), *Uncooled Infrared Imaging Arrays and Systems*, Academic Press, 1997 — the classic bolometer-physics reference. https://shop.elsevier.com/books/uncooled-infrared-imaging-arrays-and-systems/willardson/978-0-12-752155-8

**Closest match to this exact project (start here)**
- L. Leja, V. Purlans, R. Novickis, A. Cvetkovs, K. Ozols, "Mathematical Model and Synthetic Data Generation for Infra-Red Sensors," *Sensors*, 22(23):9458, 2022 (open access) — a full mathematical model of an uncooled IR sensor: FPA, bolometer readout, optics, environment, non-uniformity, dead pixels, noise. https://www.mdpi.com/1424-8220/22/23/9458
- A. Upadhyay, M. Sharma, P. Mukherjee, A. Singhal, B. Lall, "A Comprehensive Survey on Synthetic Infrared Image Synthesis," arXiv:2408.06868, 2024 — survey of the whole field: emission physics, sensor modeling, atmospheric attenuation, and existing simulation tools (DIRSIG, OSSIM, etc.). https://arxiv.org/abs/2408.06868
- C. Garnier, R. Collorec, J. Flifla, C. Mouclier, F. Rousee, "Infrared sensor modeling for realistic thermal image synthesis," *ICASSP 1999*, vol. 6, pp. 3513–3516 — the original point-wise (per-pixel, geometric + radiometric) IR sensor model that §0 of this document follows. https://ieeexplore.ieee.org/document/757600/

**Emission & the full radiometric equation (§1, §3)**
- J. Sova & M. Kolaříková, "Thermography Equation: From Conceptual Relation to Quantitative Formulation via the Optogeometric Factor," arXiv:2508.11455, 2025 — derives the emission + reflection + atmosphere equation used here, including the directional-emissivity extension mentioned in §14. https://arxiv.org/abs/2508.11455

**Atmospheric propagation / distance effect (§3)**
- "Refining Atmosphere Profiles for Aerial Target Detection Models," PMC, 2021 — accessible walkthrough of MODTRAN and path radiance and why they matter for contrast at range. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8588161/
- "Quantitative atmospheric rendering for real-time infrared scene simulation," *Infrared Physics & Technology*, 2021 — closest match to your use case: real-time Beer-Lambert + path-radiance rendering, validated against MODTRAN. https://www.sciencedirect.com/science/article/abs/pii/S1350449520306587

**Detector physics — microbolometers (§5)**
- A. Rogalski, *Infrared Detectors*, 2nd ed., CRC Press, 2010 — broad detector-physics reference covering bolometers and photon detectors alike.
- P. V. K. Yadav et al., "Advancements of uncooled infrared microbolometer materials: A review," *Sensors and Actuators A: Physical*, 2022 — includes the NETD formula in terms of f-number, optics transmittance, and detector noise used in §8. https://www.sciencedirect.com/science/article/abs/pii/S0924424722002497

**Noise (§8)**
- J. M. Mooney & F. D. Shepherd, "Characterizing IR FPA nonuniformity and IR camera spatial noise," *Infrared Physics & Technology*, 1996 — origin of the "3D noise model" referenced in §8. https://www.sciencedirect.com/science/article/abs/pii/1350449595001336

**Optics / MTF (§4, §6)**
- Optris, "Modulation Transfer Function in Infrared Imaging" — short, accessible walkthrough of diffraction-limited MTF and f-number tradeoffs specific to LWIR. https://optris.com/lexicon/modulation-transfer-function-mtf/
- UC San Diego CSE252A, "Cameras and Radiometry" (lecture notes) — clean derivation of the f-number-based irradiance equation used in §4. https://cseweb.ucsd.edu/classes/fa11/cse252A-a/lec4.pdf

**Color mapping (§11)**
- Google Research, "Turbo, An Improved Rainbow Colormap for Visualization," 2019 — open, published colormap construction, a good substitute for proprietary "Ironbow"-style palettes. https://research.google/blog/turbo-an-improved-rainbow-colormap-for-visualization/
