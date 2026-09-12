# Spec issues

Contradictions and gaps in `docs/physics-model.md`, the skills and the scaffold, raised for the spec
owner (roadmap open question 10) rather than silently resolved in code. Each row carries the
resolution the roadmap and the code assume; the **status** column below records what has been applied.
Line numbers refer to `docs/physics-model.md` at commit `3b653b3` unless stated. Opened by roadmap
step M0.10 on 2026-09-10 and maintained thereafter: when a step applies or overturns a resolution,
update the status here in the same commit.

## Status of the coding-blocking fixes (applied in M0.10, awaiting spec-owner review)

| # | change applied to `docs/physics-model.md` | why it blocked coding |
|---|---|---|
| S1 | §5.3(a): `T_sky(θ_el) … cos^q(θ_el)` → `T_sky(θ_zen) … cos^q(θ_zen)` | the sky is coldest at zenith (line 314, R13); the old form put the maximum depression at the horizon, and M6.5 / MS.2 tests encode coldest-at-zenith |
| S2 | §6.1: absorbed solar `(1−α_sol)^c Q_sol` → `α_sol Q_sol` in the displayed balance | the prose two lines below already said so; M6.7's monotone-α test encodes it |
| T4 | §12.2: `spectral_response: "responses/boson_vox.csv"` → `"spectra/responses/boson_vox.csv"` (relative to `data/`, µm, peak-normalised) | the path was stated four ways; M0.8's loader and ADR 0008 fix the layout as `data/spectra/responses/` |

Resolutions applied by code so far: T1 (M0.1), T2 (M0.2), T6 (M0.4, ADR 0005), T3 encode/decode half
(M0.5, ADR 0006), T7 fixtures (M0.6), T8 (M0.3), T11 `types-PyYAML` (M0.7), T12 HFOV test (M0.7),
S17 runtime regime switch (M0.9, ADR 0007), S27 both palette sets (M0.7). Everything else is open.

**Physics (fix the spec):**

| # | issue | proposed resolution |
|---|---|---|
| S1 | §5.3(a) line 311: `T_sky = T_air − ΔT_clear(1−cloud)cos^q(θ_el)` puts zero depression at zenith and maximum at the horizon — inverted vs line 314 ("at zenith"), line 305 (`L_sky(θ_zen)`) and R13 | Change to `cos^q(θ_zen)`; calibrate (ΔT_clear, q) per band to the digitised R13 profiles (the quoted 55–70 K, q 0.5–1 do not reproduce −40 °C at 15° elevation) — M6.5, MS.1 |
| S2 | §6.1 line 355 writes absorbed solar as `(1−α_sol)^c Q_sol` (stray exponent) while line 363, §6.4 and the skill use `α_sol Q_sol` | Use `α_sol Q_sol`; fix the displayed equation — M6.4 |
| S3 | §6.4 line 412: RK2 stability "60–200 s for thin painted metal" ignores the 1/R12 term; with §16.2 car paint and 1.2 mm the bound is 0.24 s | ADR 0036: thin panels are single nodes with a resistive back boundary; keep 1–60 s ticks — M6.8 |
| S4 | §6.4 uses R2d, T_deep, δ1, δ2 but defines none; §12.3 has one `thickness_m` | Define in ADR 0036 (R2d = δ2/(2k) + δ_deep/k; T_deep = spin-up-window mean of T_air; δ2 per material default) |
| S5 | §7.3 line 495 calls `w` "precipitable water (g m⁻³)"; the formula gives absolute humidity; RH unit (fraction vs percent) unstated | Rename to absolute humidity; RH is a fraction asserted at the loader — M8.2, M6.1 |
| S6 | §7.2 haze row: visible τ 0.45–0.65 at 200 m contradicts Koschmieder at 5 km visibility (0.855); implies V ≈ 1–1.8 km | Keep Koschmieder and the visibility label; re-derive the row — M8.4 (ADR 0049) |
| S7 | §9.4 line 661 quotes the paraxial 4F² datasheet NETD two paragraphs before the +1 form; §8.1 line 525 says "NETD scales as F²" | Implement only the +1 form; keep 4F² as a labelled datasheet conversion (ratio 1.25 at F/1) — M4.5 (ADR 0024) |
| S8 | §9.2 line 634 "smears over roughly 0.6 frames" is τ_th/Δt, not the smear extent (the IIR reaches 81 % in one frame); same phrase in the skill | Reword; tests use the closed form only — M9.1 |
| S9 | §9.2 line 629 uses R₀ for both responsivity (V/W) and bolometer resistance (Ω) | Rename one symbol |
| S10 | §13.4 SPG#4 emits `DN_ideal` and SPG#5 adds noise after quantisation, contradicting §9.1 (n_e before floor) and CLAUDE.md #3 for shot noise | Noise before quantisation; quantise last — ADR 0026 |
| S11 | §4.2 line 236 water n/k gives ε(0) = 0.990, ε_hemi = 0.951; §16.2 says 0.96. Bulk Al n/k gives ε ≈ 0.012 vs table 0.09. The table never says what it tabulates | ADR 0043 defines the convention and whether an effective k is fitted — M7.8/M7.9 |
| S12 | §4.1 defines ρ as the hemispherical DHR; §4.2 uses the specular Fresnel R as 1−ε — exact only for smooth surfaces, while §4.3 requires roughness for the same materials | ADR 0042: specular Fresnel for Level A (smooth) only; Level B/C for rough; error band recorded |
| S13 | §12.1 says glass is "opaque-ish" in MWIR; §4.4 gives only LWIR/SWIR; τ_mwir undefined for the headline material | Compute from Beer–Lambert over 5 mm of the checked-in k(λ) — M7.9 |
| S14 | §3.2(b) LUT 200–1000 K vs §5.3 cold sky below 200 K (T_air < 270 K) and §5.4 glint above 1000 K; the §13.5 kernel clamps silently | L_sky by direct quadrature (MS.1); inverse flags out-of-range; range widening evaluated in ADR 0011 |
| S15 | §1 line 55 says L3 reflection is "path-traced in T-domain"; §5.3(c) and §13.7 option 3 say radiance domain — temperatures do not add | Fix §1 to "radiance domain" |
| S16 | §2 line 86 reflected term `(1−ε) L↓` is the opaque-diffuse special case; §4.4 needs `ρ = 1−ε−τ`, §5.4 a specular lobe | Note in §2 that it is the L2 opaque-diffuse form |
| S17 | §5.2 "compile-time flag" for illumination culling vs §12.2 line 900 "no core code changes" | Runtime switch on `regime` — ADR 0063 |
| S18 | §13.5 hard-codes p = 4 with a float2 (ε0, a) table; §4.2 fits p ∈ 4–6 per class; §12.3 has no (a, p) fields | Table carries (ε0, a, p) — ADR 0042 |
| S19 | §13.3/§13.7 equate the AO AOV with V_s; §5.3(a) defines V_s as cosine-weighted sky visibility (wall: AO ≈ 1, V_s = 0.5) | V_s = occlusion·(1+n·up)/2 — ADR 0045 |
| S20 | §13.3 lists MotionVectors for "bolometer smear"; §9.2 defines smear as an inter-frame IIR | IIR from state; motion vectors feed MTF_motion for photon detectors only — ADR 0059 |
| S21 | §8.3 worked MTF_det(ξ_N) ≈ 0.64 assumes w = pitch; config fill factor 0.90 gives 0.669; the spec never says which | Box width = pitch, fill factor in A_d — ADR 0020 |
| S22 | §8.3 lists MTF_aberr, MTF_defocus, MTF_elec separately, defines one Gaussian and no MTF_elec | One Gaussian, MTF_elec = 1 — ADR 0059 |
| S23 | §11.1 pipeline omits DDE; the skill puts DDE after gamma; §13.4 before gamma; §11.1 names an undefined "temporal filter" | Order AGC → gamma → DDE → polarity → palette; no ISP temporal filter — ADRs 0031, 0058 |
| S24 | Four descriptions of post-NUC drift (§8.2, §9.2, §10.3, §11.2) with no budget; §8.2 both mandates and disparages the random walk | ADR 0053/0054: one budget; pixel-level OU drift only; global drift from self-emission |
| S25 | §10.2 "ratios in NETD units" (σ_TVH = NETD) vs §9.4 NETD from σ_total — 6 % apart for the Boson ratios; 5 % recovery for σ_T = 0.02 is statistically unattainable at 200 frames | ADR 0025 and 0023 |
| S26 | §11.3 plateau P is a per-bin count; §12.2 gives 0.012 dimensionless; "plateau = 0 behaves like linear" is degenerate | P = plateau·N_pixels; P→0 limit defined — ADR 0028 |
| S27 | §11.4 palettes include Arctic and omit gray; §12.2 the reverse | Support both sets — M5.4 |
| S28 | §7.1 says path radiance varies with elevation; the model has none; sky pixels have no defined owner | Uniform T_air on the path; sky pixels handed to the sky model — ADR 0050 |
| S29 | §5.4 `τ_atm^sun` (slant path) has no provider; §7 is horizontal only | ADR 0051/0064 — M8.8, M11.3 |
| S30 | §5.5 airglow in nW/cm² without saying in-band or total; no shape file | In-band definition, OH-Meinel shape with source — ADR 0065 |
| S31 | §9.1 cold-shield efficiency has no formula and lives under `optics:` while described as detector physics | ADR 0066 |
| S32 | §16.4 calls steps 1–5 "a defensible LWIR camera" although §5.3 says reflections are not to be skipped | Roadmap keeps the §16.4 order but M7 is part of the first *deployable* release |
| S33 | §5.3 models the sky by a cloud *fraction* only; for aerial targets clouds are the dominant clutter and no spatial cloud model exists anywhere in the spec | Author one and record it as an approximation (MS.3, ADR 0070); bound it by display-domain clutter statistics (ME.4) |
| S34 | §7.4 and Appendix A #2 bound the band-averaged atmosphere to ~500 m; ground-to-air sky targets sit at 0.5–5 km on slant paths, where a single grey γ_B is wrong by an order of magnitude for structured bands and the §5.3 sky formula disagrees with §7's own column emission | Exponential-sum band model with the sky derived from the same layered column (MS.1, ADR 0071); the cos^q sky kept only as a fitted fast path |
| S35 | The spec's radiometry is extended-source throughout (§2, §8.1); unresolved targets — the sky-target case — have no point-source treatment beyond §8.3's aliasing remark | Analytic injection below one native pixel in the fill-fraction form that reuses the §2 aperture factor (MS.6) |
| S36 | §15 Tier 4/5 assume owned hardware and ground-scene datasets; the data actually available for sky targets is 8-bit post-AGC anti-UAV video | Compare on the display output with matched AGC; Tier 2 benches remain self-consistency (ADR 0003) |
| S37 | CLAUDE.md #4 "author one of the three and derive the others" is unsatisfiable for τ > 0 materials (one closure equation, three unknowns) | Read as "author exactly one of ε, ρ, plus τ where it transmits; derive the rest" (ADR 0040, open question 12) |


**Repository / tooling (fix in M0):**

| # | issue | resolution |
|---|---|---|
| T1 | `np.trapezoid` needs NumPy ≥ 2.0; `pyproject` allows ≥ 1.24; suite red on 1.x; CHANGELOG claims green | M0.1 |
| T2 | `make golden-update` passes an unregistered option; `tests/golden` never runs in `make check` | M0.2 |
| T3 | CLAUDE.md #2 and #4 claim tests that do not exist | M0.5 (encode/decode), M7.2 (`test_committed_library_closes_in_every_band`, the library walk) |
| T4 | Spectral-response path stated four ways (`responses/`, `spectra/`, `data/spectra/`, skill) and the file is missing | M0.10 fixes the spec to `data/spectra/responses/`; M0.8 resolves it; M1.3 authors the file |
| T5 | `.gitignore` ignores `data/lut/*` while the ir-radiometry skill says LUTs are committed | ADR 0012 |
| T6 | Stefan–Boltzmann tolerance: spec 1e-6, skill 1e-12, test 2e-5 with no ADR | M0.4 (ADR 0005) |
| T7 | `tests/conftest.py` fixtures are two of the five the testing skill requires; `material_id = 0` will collide with the UNMAPPED sentinel | M0.6, M7.10 |
| T8 | `test_layering.py` forbids `carb` (CLAUDE.md does not) and neither forbids `irsim → irsim_isaac` | M0.3 |
| T9 | CLAUDE.md calls `irsim_isaac/pipeline` the "Warp-based reference pipeline" while both skills make the NumPy path the oracle; no `irsim/pipeline`, `irsim/validation`, `irsim/io` or `irsim_eval` in the layout; scopes `pipeline`/`validation`/`eval`/`io` not in the commit-scope list | ADR 0018; the CLAUDE.md changes are bundled into open question 12 (the user asked that CLAUDE.md not be edited by this plan) |
| T10 | `generate_luts.py` plans three tables; §9.4 needs the photon-form derivative (`dLb_q_dT`) too; `planck.py` lacks it | M1.1, M1.6 |
| T11 | Makefile `luts` passes no data root; `mypy --strict` will fail on `import yaml` (no stubs) | M1.9, M0.7 |
| T12 | Boson YAML: 14 mm/32° HFOV inconsistent (30.7°); `spectra/boson_vox.csv` missing; photon `null` keys omitted | M0.7 test documents it; M1.3; schema accepts both forms |
| T13 | ir-radiometry skill 0.1 % top-hat check ≠ the 5 mK LUT check (0.1 % ≈ 63 mK at 300 K); the ship-step example claims "< 2 mK" from a 0.1 % comparison | Both tests exist separately and print mK — M1.4, M1.6 |
| T14 | No config field declares a camera non-radiometric; `outputs.apparent_temperature` is the proxy | ADR 0021 |
| T15 | `conftest.py` says every random test must take `rng`; the noise skill seeds from (frame, sensor) | Physics tests seed via the stage API; `rng` for ad-hoc sampling — M4.2 |
| T16 | CLAUDE.md, README, §13 and the `isaac-sim-spg` skill say "Isaac Sim 6.0"; the build on the development machine is **6.1.0-rc.26** (Kit 110.3.0, `omni.rtx.spg` 0.4.0, `isaacsim.sensors.experimental.rtx` 1.9.0, Warp 1.16.0 bundled as a Kit extension) and every engine fact in ADR 0014 was measured there | ADR 0014 pins the measured build; README says 6.x and points at the ADR; the CLAUDE.md wording joins open question 12 (CLAUDE.md is not edited by this plan) |
| T17 | §13.1/§13.3 and the skill assume a float32 `PtSelfIllumination` (or `HdrColor` at 0 bounces) carrying the encoded temperature. Measured in 6.1: every colour AOV — `HdrColor`, `LdrColor`, all `Pt*` — is **float16** and exposure-scaled (readback ≈ 2.8e-4 × emissive colour × intensity), the float32 `GroundTruthEmission*` AOVs return no data, and path tracing at 0 bounces returns a constant. fp16 caps the round trip near 100 mK at 300 K, ten times the §13.3 bound | ADR 0014: the renderer transports **instance ids + float32 geometry** (`instance_segmentation` exact per pixel, `Camera3dPositionSD`/`DistanceToCameraSD` float32 at full resolution) and the per-facet temperature is looked up in a float32 Warp table — no quantisation at all; the §13.3 table and the skill need rewording by the spec owner; M10.1/M10.2/M10.18 carry the change |
| T18 | §7.2 table, haze row: the "5 km vis" label is inconsistent with its own τ columns — Koschmieder at 5 km gives τ_vis(200 m) = 0.855, not 0.45–0.65, and the SWIR column would need an aerosol extinction ratio above 1; the four columns are reproduced at V ≈ 1.5 km with wavelength-ordered ratios | ADR 0049 takes the τ values as authoritative; the `haze` preset is checked at V = 1.5 km |
| T19 | §7.3 calls w "precipitable water (g·m⁻³)"; precipitable water is a column amount (mm or kg m⁻²), the quantity in the Magnus expression is the absolute humidity (water-vapour density) | `irsim.atmosphere.humidity` names it `absolute_humidity_g_m3`; the 216.7 factor is derived from R_v (M8.2) |
| T20 | §5.3(a) says "under overcast, `T_sky → T_air`"; a cloud radiates at its **base** temperature, which is Γ_env z_LCL colder (6.5 K for a 1 km base, > 100× a Boson NETD). The spec's limit is the saturated-air special case (z_LCL = 0, i.e. fog) | MS.3 / ADR 0070: the overcast limit is `L_B(T_base)`; tests needing a sky at `T_air` use cloud = 1 with RH = 1 |

---

