# SPG assets — and the question that has to be answered before there are any

`.cu` kernels, `.cu.lua` launch scripts, `.usda` shader definitions. **This directory is empty on
purpose.** M10.12 and M10.13a/b/e are the four steps that would fill it, and they are blocked on one
unresolved capability rather than on effort. Writing the kernels first and discovering the blocker
afterwards would produce shaders that cannot carry the data they need, so the question is written
down here first.

## What M2.3 already established (ADR 0014, ADR 0061)

* **(a) float32 passes through bit-identically.** `DistanceToCameraSD`, `DistanceToImagePlaneSD`,
  `Camera3dPositionSD` and `LdrColor`/`HdrColor` all survive a pass-through kernel unchanged, so the
  precision non-negotiable is safe on this path.
* **(b) There is no cross-frame device state.** A `cuda.static` buffer is re-initialised every
  frame; `cuda.empty` is an output descriptor, not a kernel argument; and an output AOV wired back
  into its own node's input produces nothing at all. Lua *file-scope* variables do persist, one call
  per rendered frame. This is why the stateful stages (4 and 5 — the membrane, the drifting fixed
  pattern, the FFC) stay in Warp.
* **(c) A baked table works and a loaded one does not.** `io.` is a forbidden token — the whole file
  is rejected by a textual validator — and a 16 001-entry Lua literal **crashes the Kit process**
  under the default 1 000-instruction sandbox limit. A `__device__ const float[16001]` baked into
  the `.cu` compiles under NVRTC in about a second and matches NumPy exactly.
* **Hazard.** A kernel that fails to load yields a **zero-filled output with status ok**. Any
  in-sim equivalence test must therefore assert against a *known non-zero* reference, never merely
  that a buffer came back.

## The open question: how does a per-frame float32 table reach the kernel?

Stage 1 is `ε₀ L_B(T) + (1 − ε₀) L_env`. Two of its inputs are per-frame and neither is an AOV:

| input | why it is not an AOV |
|---|---|
| the **facet temperature table**, keyed by instance id | ADR 0014: the renderer transports ids and geometry only; emissive colour is fp16 and exposure-scaled, so it cannot carry kelvin |
| the **environment radiance** `L_env`, per frame | it comes from the sky model on the host, not from the scene |

The Planck LUT is static and (c) says bake it. The facet table is **not** static — it changes every
frame as the thermal solver advances — and (b) says there is no device state to keep it in. So it
must arrive as a kernel argument each frame, and the unanswered question is:

> **Can the `.cu.lua` launch script upload a host-supplied float32 array as a kernel argument on
> every frame, and if so through which call?** The probe exercised `cuda.int`, `cuda.float`,
> `cuda.TextureObject`, `cuda.SurfaceObject`, `cuda.array`, `cuda.zeros` and `cuda.image`. None of
> those was shown to carry *host* data in, and `io.` is forbidden, so reading the table from disk is
> not an option either.

### Candidate answers, cheapest first

1. **A scalar sweep.** If the count of facets is small (the aerial scenes have under twenty nodes),
   pass the table as N `cuda.float()` scalars, rebuilt per frame by the authoring side. Bounded, but
   only if the argument list can be rebuilt per frame rather than fixed at authoring time.
2. **A one-row input image.** Write the table into a 1 × N float32 render var the graph already
   owns, and read it in the kernel with `tex2D<float>`. This reuses the (a) path that is known to be
   bit-exact. The question becomes whether a *host-written* image can be bound as a graph input.
3. **Re-bake per frame.** Regenerate and recompile the `.cu` with the table baked in. NVRTC took
   ≈ 1 s in (c), which is unusable at 60 Hz and possibly fine for a time-lapse; it would also make
   the shader path slower than the Warp path it is meant to accelerate.

### How to answer it

Extend `irsim_isaac/spg_probe.py` with a fourth experiment in the same shape as the others — author
a throwaway graph, try each candidate, read the output back and compare against NumPy. It costs one
Isaac session and it decides the design of all four SPG steps. Until it is run, the kernels here
would be written against a guess, and the project's rule is to flag that rather than code it
(CLAUDE.md, "Flag uncertainty rather than guessing": Isaac Sim 6.0's SPG API is new and the public
documentation is incomplete).

## Meanwhile

The Warp path (ADR 0018's CPU oracle and the `irsim_isaac.pipeline.warp_stages` equivalence
harness) renders every band today, including the four-band demo matrix. The SPG lane is an
optimisation of stages 1–3 and 6, not a prerequisite for anything that has shipped.
