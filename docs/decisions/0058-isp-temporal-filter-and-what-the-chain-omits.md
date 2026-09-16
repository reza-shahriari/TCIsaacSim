# 0058 — §11.1's temporal filter, and what the wired chain deliberately omits

- **Status:** Accepted
- Date: 2026-09-12
- Step: M9.8 (`feat(pipeline): wire IIR, FPA node, housing, drift, defects, NUC residual, FFC into run_frame`)
- Spec: docs/physics-model.md §11.1, §11.2, §9.2

## Context

M9.1–M9.7 each built one mechanism and tested it alone. Assembling them raises two questions that
no individual step could answer, because both are about what the *combination* should contain.

**First, §11.1 lists a stage it never defines.** The ISP order reads

    raw DN → bad-pixel replace → NUC (2-pt gain/offset) → temporal filter
           → [radiometric linearisation → T_app] → AGC / DRC → …

and nothing anywhere in the spec says what "temporal filter" does, what its coefficient is, or
whether real cores have one at all.

**Second, two mechanisms in the chain describe overlapping physics.** M9.2 gives the detector raw
`gain_of_t`/`offset_of_t` polynomials — the *uncorrected* FPA response, parts in 10³ per kelvin.
M9.6 gives the NUC residual — what survives correction, parts in 10⁴. A chain that applied both,
plus a two-point NUC between them, would be forming a large number and subtracting almost all of it
back to arrive at something the residual already gives directly.

## Decision

### The temporal filter is the identity, until ME.5 says otherwise

`SensorChain._temporal_filter` returns its input. It is a named stage with a docstring rather than
an absent one, so the gap is visible.

Inventing a coefficient here would be worse than leaving it out. The stage sits directly on
σ_TVH — the quantity NETD is *defined* from — so any low-pass would reduce measured NETD by
√(α/(2−α)) while leaving σ_VH alone. That is a large, plausible-looking change, and nobody looking
at the resulting imagery later could distinguish it from a detector-model error. The right evidence
already has a home in the plan: **ME.5's temporal PSD on flat sky**, which shows directly whether a
real core low-passes its output. If it does, this becomes an optional first-order stage with its
coefficient **ESTIMATED from the measured roll-off** — and the test asserting the √(α/(2−α))
reduction, already written into M9.8's exit criteria, is what would gate it.

### The chain applies the NUC residual, not the raw polynomials

Only `NucResidual` runs. `gain_of_t` and `offset_of_t` stay available for an uncorrected-FPA bench
and do not run in the pipeline. This is ADR 0053's split carried into the assembly: the residual is
*defined* as the part that survives, so applying the raw response and then a correction for it
would model one effect twice and lose precision doing so.

### The budget the assembly is tested against

On a uniform scene, the post-correction spatial noise must be

    σ_spatial = √(σ_V² + σ_H² + σ_VH²  +  σ_residual²)

within 10 % — **two mechanisms in quadrature, not four added up**. The residual contributes two
*terms* (§2's multiplicative `g_ij` and additive `o_ij`, comparable in size at a realistic ΔT_FPA)
but is one mechanism with one parameter set, reset by one event.

Three things had to be got right for that test to mean anything, and each was a real trap:

- **Subtract the flat-field reference.** `vignetting_cos4` alone puts a 0.9 % cos⁴ falloff across
  the array — a larger spatial standard deviation than the entire noise budget. Measuring the raw
  std of a frame measures the lens. A real NUC measurement subtracts a flat field, and so does this.
- **Average frames.** A single frame's spatial std also contains the temporal per-pixel noise,
  which in one frame is indistinguishable from fixed pattern.
- **Test at two drift rates.** At the roadmap's 0.05 K/s the residual is nearly thirty times the
  3-D spatial term, so the budget is dominated by one mechanism and a spurious third could hide
  inside it. A second rate is tuned to make the two equal, which is where the quadrature is
  actually under test: a linear sum would be 41 % high against a 10 % tolerance.

### `chain=None` stays the default

Attaching is explicit (`attach_sensor_chain`). `chain=None` is not "unwired", it is a meaningful
configuration — the ideal camera every M9 mechanism is measured against, and the one the
radiometric goldens describe. Those goldens exist to pin millikelvin-level physics, and a defect
map or a drifting pattern would make their claims untestable. The assembled chain gets its **own**
golden (`boson_sensor_chain_*`) instead, which regression-locks the wiring without diluting the
radiometry.

## Consequences

- The radiometric goldens are unchanged by this step. The roadmap anticipated updating them; that
  turned out to be the wrong trade, because a golden that contains both the radiometry and the
  defect map can no longer fail *informatively* for either.
- `T_FPA` falls back to `T_housing` when no `fpa_temp_mode` is configured — which the committed
  Boson does not configure. That is right for a camera (the focal plane sits in the housing) but it
  means the residual inherits the housing's 900 s lag, which damps a 179 s drive to a tenth of the
  air's excursion. Tests that need the residual to have grown give the FPA its own, much shorter
  time constant, as a real focal plane has relative to its body.
- The chain measures its frame interval from successive `t_s` values rather than from the frame
  rate, so a bench stepping at an irregular cadence still gets the right drift.
- Defects are applied on the quantised plane and carried back onto the float signal only where they
  actually bit, so the un-quantised signal keeps full precision everywhere else and the radiometric
  branch retains its sub-LSB meaning.

## Revisit when

ME.5's temporal PSD on flat sky is measured — the trigger for the temporal filter, in either
direction. Or when a camera is modelled with no two-point NUC at all, in which case the raw M9.2
polynomials become the right thing to apply and this ADR's second decision is reversed for that
configuration rather than in general.
