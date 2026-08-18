# Object Temperature Dynamics — Engines, Water, Ice, and Fire

Companion to **thermal_camera_model.md**. That document takes a per-point temperature `T` as a given input to the camera/sensor chain; this one covers where `T` actually comes from when it changes over time — objects heating up (an engine), or materials that don't fit "one evolving scalar" at all (melting ice, radiating fire, a thermally sluggish sea).

## 1. The General Case — Lumped Thermal Capacitance

Same idea as the bolometer pixel in the other document's §5, just applied to a whole object instead of a detector element — heat in, heat out, one energy balance:

```
C_th · dT/dt = Q_in(t) − Q_out(t)
```

- `C_th = m·c_p` — thermal capacitance [J/K] (mass × specific heat capacity)
- `Q_in(t)` — heat generated or absorbed (combustion, solar, electrical, friction...)
- `Q_out(t)` — heat lost to the surroundings, normally dominated by convection (Newton's law of cooling), with radiation as a second term:

```
Q_out ≈ G_th·(T − T_ambient),     G_th = h·A + 4εσA·T_ref³   (linearizes radiative loss near T_ref)
```

`h` = convective heat transfer coefficient [W/(m²K)], `A` = surface area. Combined:

```
C_th · dT/dt = Q_in(t) − G_th·(T − T_ambient)          τ = C_th/G_th
```

This is exactly the bolometer thermal equation from the other document — the same physics, just at a very different scale (a whole object instead of a detector pixel).

**Discretized form for coding (explicit Euler):**

```
T[n+1] = T[n] + (dt/C_th)·(Q_in(t_n) − G_th·(T[n] − T_ambient))
```

**Closed form for a step input** (e.g. `Q_in` switches from 0 to a constant `Q_run` at t=0, starting from `T_ambient`) — this is the warm-up curve:

```
T(t) = T_ambient + (Q_run/G_th)·(1 − exp(−t/τ))
T_ss = T_ambient + Q_run/G_th          (steady-state "fully warmed" temperature)
```

And cool-down, once `Q_in → 0` starting from `T_ss`:

```
T(t) = T_ambient + (T_ss − T_ambient)·exp(−t/τ_off)
```

**Worked example — a car engine:** while running, `Q_run` ≈ combustion heat not converted to work or exhausted; `G_th` while driving is boosted by forced airflow over the block/radiator. Once stopped, `G_th` drops (no more forced convection), so `τ_off` is typically *longer* than the warm-up `τ` — same equation, state-dependent conductance. `τ` itself is very build-specific (mass, material, airflow), so calibrate it to your object rather than assuming a universal number — trust the equation, not any particular time constant.

**If one node isn't enough:** an object with distinguishable parts at different temperatures (hot block, cooler chassis it's bolted to) can be modeled as multiple lumped nodes linked by thermal resistances — the same equation, chained, the way an RC ladder network extends a single RC circuit.

### 1.1 Terrain and Subsurface Heat Transfer

Lumped capacitance assumes uniform temperature, which breaks down for large environments like terrain or thick concrete walls. The sun heats the top millimeter of soil, but deep soil stays cool. At night, the heat from below diffuses back up. A single `C_th` cannot model this phase lag.

**The Solution:** For terrain, use a 1D heat equation (layered nodes: Surface, Shallow, Deep, Constant Core). The surface interacts with the sun/air and conducts to the shallow layer. This produces highly realistic **"thermal inertia"** where roads stay hot long after sunset.

**Thermal Shadows:** By implementing Subsurface Heat Transfer and tracking solar irradiance history per-texel on the ground, you naturally get "thermal shadows". If a car is parked on a road all day, the road underneath is shaded. When the car drives away, a striking cold thermal shadow remains.

## 2. Driving Inputs

`Q_in(t)` and `T_ambient(t)` are what you supply; a few practical starting forms:

```
Combustion/mechanical: Q_in(t) = Q_run · [engine on]              (step, or ramp with RPM/load for more realism)
Solar:                 Q_solar(t) ≈ S₀ · max(0, sin(elevation(t))) · α_solar · A_facing
Wind Convection:       h = h_natural + k_wind · max(0, dot(v_wind, n_surface))    (Windward sides cool much faster than leeward sides)
Ambient (diurnal):     T_ambient(t) ≈ T_mean − (ΔT/2)·cos(2π(t − t_peak)/24h)
```

`S₀` ≈ 1000 W/m² clear-sky peak; `t_peak` around mid-to-late afternoon; `elevation(t)` can be a simple half-sine over daylight hours unless real solar-position accuracy matters to you.

## 3. Water / Sea — Same Model, Very Different Constants

No new equation — just very different numbers plugged into §1, plus one extra loss term.

- `c_p,water ≈ 4186 J/(kg·K)` vs. ≈800 J/(kg·K) for dry soil/rock, ≈450–900 J/(kg·K) for common metals — for equal mass, water needs 5–9× more energy per degree of change
- Wave and current mixing continuously folds heat into a much deeper layer than conduction reaches in solid ground over a day, so the *effective* `C_th` for a patch of open water is larger still than the raw `c_p` difference alone suggests
- Net effect: give water points a much larger `C_th` (equivalently, much longer `τ`) than adjacent dry ground. This alone reproduces the classic thermal-imaging signature — water reads cool against sun-baked ground by day and warm against it at night, with a damped, phase-lagged daily swing
- Extra loss term specific to a liquid surface — evaporative cooling:

```
Q_evap ≈ L_v · A · k_evap · u_wind · (e_sat(T_water) − e_air)      L_v ≈ 2.26×10⁶ J/kg
```

  a bulk aerodynamic formula (evaporation scales with wind speed and vapor-pressure deficit). If you don't want to model humidity/wind explicitly, folding extra fixed conductance into `G_th` approximates the same effect.
- Reminder: water's directional emissivity (high near-normal, more mirror-like at grazing angles) is already covered in the other document's §14.

## 4. Ice — Phase Change and Latent Heat

Melting doesn't fit a single `C_th` — while ice and liquid water coexist, incoming energy goes into breaking bonds, not raising `T`, so temperature pins at the melting point until the phase change finishes. Two ways to handle it:

**(a) Quick version:** make heat capacity temperature-dependent, with a large spike right at `T_melt` sized to represent the latent heat. The same ODE from §1 then "just works," no extra bookkeeping needed.

**(b) Recommended version — track enthalpy, not temperature:**

```
dH/dt = Q_in(t) − Q_out(t)        (same energy balance, on enthalpy H instead of T — Q_out still needs T, computed from H below, each step)

H < 0:               T = T_melt + H/(m·c_ice)                        (solid)
0 ≤ H ≤ m·L_f:        T = T_melt,   melt_fraction = H/(m·L_f)         (plateau — the "different manner" part)
H > m·L_f:            T = T_melt + (H − m·L_f)/(m·c_water)            (liquid)
```

`L_f` (latent heat of fusion, water/ice) ≈ 334,000 J/kg — large relative to `c·ΔT` for any modest ΔT, which is exactly why slush sits at ~0°C for a long time even in direct sun: almost all the incoming energy is going into melting, not warming. `melt_fraction` is a useful free byproduct if you ever want to blend the object's appearance from ice to water as it melts.

This generalizes to any phase transition (boiling, condensation, freezing) — same plateau trick, different `T` and `L`.

Small aside, same flavor as the glass fact in the other document: despite looking bright/reflective in visible light, ice and snow are good LWIR emitters (ε ≈ 0.96–0.98, similar to water) — not the "cold and shiny = low emissivity" intuition one might reach for.

## 5. Fire — A Volumetric Emitter, Not a Surface at a Temperature

Every case above still assumes an opaque point with one `T` and one `ε`. Fire breaks that assumption entirely: it's a turbulent, semi-transparent 3D volume of hot gas and soot, not a solid surface — this is the "different manner" you flagged.

**Two emission mechanisms, not one:**
- **Soot (carbon particulate)** — near-graybody continuum emission, roughly following Planck's law at the local gas temperature. Soot is the dominant visible *and* infrared continuum source in most luminous flames.
- **Combustion gas molecular bands** — CO2 and H2O, the main combustion products, emit in narrow, strong bands rather than a smooth continuum: CO2 around 4.3–4.4 μm, H2O around 2.7–2.9 μm (both shifted slightly longer than their room-temperature atmospheric *absorption* bands, because the emitting gas is hot). This is genuinely why fire looks spectrally different from everything else in a scene — a smooth soot continuum with two sharp spikes on top — and it's the literal physical basis real flame detectors use to pick fire out from sunlight or other hot objects: narrow-band sensors tuned right onto the CO2 line. It matters more if you're simulating an MWIR sensor (where that CO2 band sits) than LWIR.

**Temperature** is set by combustion chemistry rather than being a free per-object parameter like a wall's: measured hydrocarbon diffusion-flame temperatures commonly run in the ~1400–2050 K range in controlled studies; open/real-world fires (campfires, structure fires) are often somewhat lower and less uniform, from incomplete combustion, air dilution, and radiative losses. Treat it as a fairly narrow, physically-motivated band (roughly 1000–2000 K) rather than a random per-object draw.

**Volumetric emission — reuse a formula you already have.** Because flame is a gas volume and not a surface, its effective emissivity along any line of sight depends on how much hot, sooty gas that ray passes through — structurally the same as the atmospheric path-radiance equation from the other document's §3:

```
L_flame(λ) = L_bb(λ, T_flame)·(1 − exp(−κ_soot(λ)·ℓ))
```

`ℓ` is the path length *through the flame volume* along that specific ray (not camera range — distance travelled inside the fire); `κ_soot` is a soot extinction coefficient (higher → thicker/sootier flame, closer to full blackbody; lower → thin flame edge, mostly see-through). Same math as `τ(λ,R) = exp(−γ·R)` from before, just relabeled: a flame is a small, very hot, local "atmosphere" that both emits and attenuates over a short path.

**Flicker — the part a static field can't give you.** Buoyant flames aren't steady: they show a well-documented "puffing" oscillation from periodic vortex shedding, at a frequency on the order of 10–20 Hz that's largely independent of fuel type and scales with source size roughly as `f ∝ D^(−1/2)` (`D` = burner/pool diameter) — bigger fires flicker slower, small ones (a candle) flicker faster. Drive `T_flame(x,y,z,t)` and `κ_soot(x,y,z,t)` with noise (e.g. Perlin/simplex advected upward, mimicking the buoyant plume) modulated by that ~10–20 Hz oscillation, rather than holding them fixed — this one addition is most of what reads as "fire" instead of "a static hot blob."

**Bonus realism:** the hot plume rising above the visible flame is usually invisible to the eye but still clearly hot in thermal. Worth a second, larger, cooler, more diffuse volumetric emitter above the flame base if the full thermal footprint matters, not just the visible flame shape.

## 6. Advanced Environmental Effects

To push the simulation to a truly physically-based "Full Version" state, consider these additional phenomena:

- **Directional Emissivity (Fresnel Effect for IR)**: By Fresnel's equations, emissivity drops dramatically at grazing angles, meaning the surface becomes highly reflective. A road viewed straight down might have `ε = 0.95`. But viewing the road near the horizon, `ε` might drop to `0.5`, meaning it reflects the cold sky (appearing cold). Ensure `ε` is evaluated as a function of the viewing angle: `ε(θ) = 1 - R_f(θ)`.
- **Radiosity (Thermal Bleeding)**: A hot engine doesn't just lose heat to the air; it radiates it to nearby surfaces. In an urban canyon, walls radiate heat to each other. A fast approximation is to update the `T_env` for a specific object based on the temperatures of nearby large bodies, simulating thermal inter-reflection.
- **Surface Moisture (Wetness)**: Rain makes surfaces wet, which increases their thermal inertia, reduces temperature via evaporative cooling, and changes their emissivity. Porous materials (wood, concrete) absorb moisture and show pronounced thermal variances, while non-porous materials shed it.

## References

**General thermal dynamics (§1–§2)**
- T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, Wiley — the standard heat-transfer textbook; Ch. 5 ("Transient Conduction / The Lumped Capacitance Method") is this section's source material. https://books.google.com/books/about/Fundamentals_of_Heat_and_Mass_Transfer.html?id=6piuzQEACAAJ

**Fire (§5)**
- B. M. Cetegen & T. A. Ahmed, "Experiments on the periodic instability of buoyant plumes and pool fires," *Combustion and Flame*, 93:157–184, 1993 — origin of the ~10–20 Hz, size-scaling "puffing" result used above. https://www.sciencedirect.com/science/article/abs/pii/001021809390090P
- "Dual channel multi-spectrum infrared optical fire and explosion detection system," US Patent 5,612,676 — a concrete, practical illustration of the CO2/H2O flame-band physics: real dual-band fire detectors are built directly on it. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5612676
- "Effects of gas and soot radiation on soot formation in a coflow laminar ethylene diffusion flame," *Combustion and Flame*, 2001 — good detail on why the optically-thin soot+gas radiation model is a reasonable approximation, and where it starts to break down. https://www.sciencedirect.com/science/article/abs/pii/S0022407301002059
