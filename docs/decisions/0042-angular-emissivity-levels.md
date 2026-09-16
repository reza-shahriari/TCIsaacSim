# 0042 — Three angular-emissivity levels, declared per material

Date: 2026-09-16
Status: accepted
Roadmap: M7.6, RP.4 (§4.2, §13.3, §13.5)

## Context

`docs/physics-model.md` §4.2 offers three fidelity levels for the angular dependence of emissivity
and does not say who chooses between them, nor how the chosen one reaches a kernel. Shipped code
answered both questions, and four modules and five test files have cited this ADR for the answer
since M7.6 without it existing. This record is written after the fact, from the code, and is
therefore descriptive of what runs rather than a proposal.

Two contradictions in the spec forced part of the answer and are listed in `docs/spec-issues.md`
with this ADR named as their resolution:

* **S12.** §4.1 defines ρ as the hemispherical directional reflectance, while §4.2 uses the
  *specular* Fresnel R as 1 − ε. Those are the same number only for an optically smooth interface,
  and §4.3 asks for roughness on some of the same materials.
* **S18.** §13.5 hard-codes p = 4 behind a `float2` (ε₀, a) table; §4.2 says p is fitted per class
  over 4–6; §12.3's material schema has no (a, p) fields at all.

The physical constraint underneath is that **ε(θ) does not have one shape.** For a dielectric it
falls with angle; for a metal it *rises* — bare aluminium goes from 0.019 at normal to 0.030 at 70°.
No single two-parameter falloff can cover both.

## Options considered

1. **Fresnel everywhere (Level A only).** Exact, and the only form that represents a metal. Costs
   a checked-in n/k table per material; the library has proxies for two substances and nothing at
   all for asphalt, soil, vegetation, skin or cotton, so this would mean fabricating optical
   constants for most of the library — precisely what ADR 0041's provenance rule exists to stop.
2. **Empirical everywhere (Level B only).** Two instructions in a shader. Cannot represent a rising
   ε(θ) at any a ≥ 0, so every metal silently becomes a flat dielectric.
3. **Constant ε (Level C only).** §4.2 permits it "only for rough, high-emissivity surfaces
   (ε > 0.93) within ±50° of normal" and names vehicle bodies, glass and water as violating it.
4. **A level declared by the material, dispatched at one entry point.** More machinery, and the
   level becomes part of a material's authored data rather than a global fidelity knob.

## Decision

**Option 4.** The level is a property of the *material*, declared in its YAML as `angular_model`,
and `irsim.materials.directional.directional_emissivity` is the single entry point that dispatches
on it. A caller asks for ε_B(θ) and does not choose the fidelity; the material already did.

* **Level A — Fresnel** from a checked-in n/k table (`{type: fresnel, n_k_file: ...}`). The only
  level that can represent a metal. **This is the answer to S12:** the specular Fresnel R is used
  as 1 − ε *only* at Level A, i.e. only where the surface is optically smooth in the band in
  §4.3's sense (σ ≪ λ/8). Rough surfaces take Level B or C, and their reflection lobe is ADR 0067's
  business, not this one's.
* **Level B — empirical** ε(θ) = ε₀[1 − a(1 − cos θ)^p], fitted once per class against Level A and
  baked (`irsim.materials.angular`).
* **Level C — constant**, and **refused below ε_B = 0.93 rather than warned about**. The error it
  hides is large and one-sided: glass at 60° sits about 0.12 below its normal ε, so a constant ε
  renders the limb of every windscreen and every water surface too warm, everywhere, in a way that
  looks like a plausible scene rather than like a fault.

Four further choices inside Level B, each of which had an appealing wrong alternative:

* **The table carries (ε₀, a, p) as three columns, not (ε₀, a) with p pinned at 4.** This is the
  answer to S18. The authored library already uses p = 4 and p = 5, so pinning p would silently
  re-fit six materials at render time.
* **ε₀ is not fitted.** It is ε(0), which Level A gives exactly and which every other consumer of
  the material already uses as its band value. Letting the fit move it would make the
  normal-incidence emissivity depend on how well the *angular* model happened to fit.
* **The fit refuses on shape before it refuses on error.** A rising ε(θ) is a metal and Level B
  cannot represent it at any a ≥ 0, so fitting aluminium returns a = 0 and a flat curve with the
  wrong sign of slope. A residual bar does not catch this: aluminium's ε is so small that a
  completely wrong shape has an RMS residual of **0.0034**, comfortably inside the 0.02 bar. The
  monotonic direction is therefore checked first and on a *relative* scale (`RISE_FRACTION` = 0.02
  of ε₀, with a 1e-4 floor), and a rising curve is rejected whatever its residual.
* **p is searched exhaustively over (4, 5, 6) rather than optimised.** The model is linear in `a`
  given p, so each candidate is one least-squares solve and the best over the set is the exact
  global optimum. A gradient search on two parameters would be slower and could return a local one.

The fit range stops at **70°**, §4.2's own bound. Beyond it Fresnel turns over towards ε → 0 at
grazing, the two forms diverge quickly, and including that region would drag the fit away from the
angles a camera spends its time looking at to buy accuracy where a pixel is a sliver.

## Consequences

**What the library actually declares.** Of 19 materials: **3 at Level A** (bare aluminium, water,
glass windshield), **16 at Level B**, and **none at Level C**. The level the spec explicitly permits
is used by nothing, because the ε > 0.93 bound excludes every material in the library that would
otherwise want it. That is worth knowing before anyone spends effort optimising the Level C path.

**Glass is at Level A for a reason that will recur.** `angular_model` is one setting for a whole
material, and glass cannot be served by one empirical fit across four bands: the Si–O reststrahlen
band drives n to 0.35 at 8.8 µm inside the LWIR window, so a per-band fit wants a = 1.43 there
against 0.66–0.73 elsewhere, and an a > 1 clips ε to zero at 85.1° — calling a windscreen edge a
perfect mirror where Fresnel still has 0.37 of normal. Fresnel from a table is per-band by
construction, so Level A sidesteps the whole problem. **Any material whose ε(θ) shape differs
materially between bands must go to Level A**, not to a compromise fit.

**The fit contradicts the spec's own quoted range, and the fit wins.** §4.2 quotes a ≈ 0.15–0.35 for
painted metals and plastics. Fitting the four painted materials against Level A on the M7.5 paint
proxy gives a = 0.732–0.770 per band and 0.748 jointly, shipped as 0.75 — two to three times the
quoted range. The fit is preferred because §4.2 also *instructs* the fit, and because §4.3 puts a
clearcoat in the optically smooth regime in LWIR. The estimate it replaced (a = 0.25, p = 5) held ε
at 0.97 of normal at 70° where Fresnel on a clearcoat gives 0.86. Raised as spec issue **S40**.

**The error this introduces.** Level B is accepted only when its RMS residual against Level A over
0–70° is ≤ 0.02, and nothing is claimed for a Level B material beyond 70°. A scene whose radiometry
depends on grazing-incidence emissivity — a sea surface, a long ground plane running to the horizon
— must use Level A, which is why water is Level A and why `irsim.atmosphere.sea` goes through the
n/k path rather than the baked one (ADR 0079).

**What it makes easy.** Adding a material is authoring one `angular_model` line; no kernel branches
on the level, because the baked table carries the same three columns whatever the material chose.
`directional_emissivity` returns **float32**, matching the G-buffer's `normal_dot_view`, so the
float32 discipline of non-negotiable #2 is not quietly widened in the middle of the stage-1 chain.

**What it makes hard.** A material cannot declare different levels per band — see glass above. And
a caller who authors `empirical` for a metal gets a refusal rather than a plausible flat curve,
which is correct but is a failure at *bake* time for something that looks like a data entry choice.

## Revisit when

- A metal needs Level B for shader cost. This requires either signed `a` (changing the model's
  meaning) or a fourth level, and the shape guard is what would have to be relaxed.
- A material needs a different level in different bands, i.e. the glass problem recurs for a
  substance with no usable n/k table.
- Tier 4 comparison against public imagery shows a systematic bias at the limb of Level B surfaces,
  which would indicate the 70° bound or the 0.02 residual bar is too loose.
