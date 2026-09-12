# ADR 0061 — Warp first: GPU stages are twins of the CPU oracle; SPG only for proven stages

**Status:** Accepted
**Date:** 2026-09-12

## Context

ADR 0018 made the NumPy pipeline (`irsim.pipeline.run_frame`) the oracle every accelerated path is
compared against. ADR 0014 measured the two GPU routes on the pinned build (Isaac Sim 6.1.0-rc.26,
Kit 110.3.0): `omni.rtx.spg` 0.4.0 passes float32 through bit-exactly but holds **no cross-frame
state**, rejects a whole Lua file on one forbidden token, crashes Kit on a large Lua literal, and
takes its LUT only baked into the `.cu`; NVIDIA Warp 1.16.0 is the `omni.warp.core` Kit extension
and imports **only inside a running Kit application**. `pyproject.toml` still carried a
provisional `isaac = ["warp-lang>=1.5"]` extra from before those measurements, contradicting ADR
0014's "no pip Warp". Roadmap M10.4 is the first GPU stage; the pattern it sets is the pattern
for all six.

## Options considered

1. **SPG first**, as §13.4/§13.5 sketch it — rejected for now: stages 4–5 need state SPG cannot
   hold, the Lua sandbox failures are silent or fatal, and a kernel that fails to load still
   emits a zero-filled buffer with status ok. Nothing in it is testable without a render product.
2. **Warp inside Kit, each stage an op-for-op twin of the CPU oracle, one equivalence harness** —
   chosen.
3. **A standalone pip `warp-lang` so GPU tests could run without Kit** — rejected: a second Warp
   on `sys.path` shadows the Kit one, so the harness would pass against a runtime the renderer
   never uses. Runnable-without-Kit is not on offer on this build.
4. CuPy / Torch kernels — rejected: a third runtime, and the ML stack is barred from the core.

## Decision

- **One Warp twin per stage**, in `irsim_isaac.pipeline.warp_stages`, registered in
  `EQUIVALENCE_STAGES` as `(CPU stage, GPU stage)`. The twin repeats the oracle's float32
  arithmetic in the same order — for stage 1 that is `BandLUT.lookup` (float32 index, clamp to
  `n − KERNEL_CLAMP_MARGIN`, linear interpolation) and `ε·L_B + (1 − ε)·L_env` with ε = 1 under the
  sky mask. Anything the CPU raises on (UNMAPPED id 0, out-of-table ids, float16 temperature) is
  refused on the host before upload, because a kernel cannot raise.
- **One harness for every stage**: `tests/integration/test_kernels_vs_reference.py` runs the pair
  on the synthetic G-buffers (ramp, sphere, two-material) on `cuda:0` **and** on Warp's `cpu`
  device (same source through the C++ backend: a compiler-independent second check) and demands
  ≤ 1e-4 relative and ≤ 5 mK through dL_B/dT (ir-sim-testing budget). Every later stage adds a
  registry row, not a new test file.
- **Measured on stage 1** (`outputs/isaac_probe/warp_stage1_measure.json`): CUDA vs CPU oracle
  max 1.25 × 10⁻⁷ relative, 0.015 mK, 83 % of ramp pixels bit-identical, worst 2 ulp (fused
  multiply-add contraction); the Warp CPU device is bit-identical on every fixture; uniform
  fixtures are bit-identical on both. The 1e-4 budget therefore has three orders of magnitude of
  headroom, and a disagreement above ~1e-6 is a kernel bug, not tolerance.
- **Device tables are uploaded once** (`DeviceTables`, keyed by the SHA-1 of the LUT column and
  the ε column plus device), so the LUT pointer is unchanged across frames (tested over 10
  frames). The reflected-environment plane `l_env` stays a host computation
  (`environment_radiance`: a tilt LUT over V_s plus one ground radiance) uploaded per frame; it
  moves on-device when M10.9a's profile says so.
- **Dependencies and markers**: the `isaac` extra is **empty** and documented — no `warp-lang` —
  resolving the contradiction with ADR 0014. Only the existing `gpu` marker is used; a separate
  `warp` marker was considered and dropped because every Warp path needs Kit anyway. `gpu` tests
  take the `simulation_app` fixture even when they do not render.
- **SPG stays optional**: a stage gets an SPG triple only once its Warp twin passes the harness
  and it needs no cross-frame state (M10.12 stage 1, M10.13a/b); stages 4–5 stay in Warp
  (M10.13c/d deferred) and stage 6 keeps its temporal ISP state in Warp too.

## Consequences

- Two implementations per stage to keep in step; the harness is the guard, and a pull request
  that adds a kernel without a registry row is incomplete by definition.
- Kernel modules cannot use `from __future__ import annotations` (Warp evaluates real
  annotations) and disable three mypy codes (`valid-type`, `no-untyped-def`, `untyped-decorator`)
  for the kernel signatures only.
- A GPU test session costs one Kit boot (14–35 s); the harness itself runs in well under a
  second. NumPy-in/NumPy-out stage calls cost 0.3–2.3 ms per frame at 64²–256², transfer-
  dominated; the on-device entry point (`launch_band_radiance`) is what the M10.9a chain uses.
- The error introduced by the GPU path is bounded by the measured 2 ulp (≈ 1 × 10⁻⁷ relative);
  it cannot be seen by any downstream stage.

## Revisit when

- `omni.rtx.spg` documents persistent buffers or feedback edges (SPG stages 4–5 come back).
- A CI GPU job without Kit is wanted: pin `warp-lang` **exactly** to the Kit version and add a
  test that, inside Kit, `warp.__file__` resolves under the Kit `extscache`.
- The per-frame `l_env` upload or the host validation shows in the M10.9a profile.
