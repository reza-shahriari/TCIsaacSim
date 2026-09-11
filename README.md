# irsim — physically-based multi-band IR camera simulator

Simulates what a real LWIR / MWIR / SWIR / NIR camera would see, with radiometry that closes in physical
units. Targets NVIDIA Isaac Sim 6.x (every engine fact is measured on the 6.1.0-rc.26 source build,
ADR 0014); the physics core is engine-free so an Unreal Engine port is a rewrite of the glue only.

**Physics specification:** [`docs/physics-model.md`](docs/physics-model.md) — the source of truth for
every equation here. Code cites it by section.

**Spec issues:** [`docs/spec-issues.md`](docs/spec-issues.md) — contradictions found in the spec, with the
resolution the code assumes and what has been applied; raise new ones there, never diverge silently.

**Plan:** [`docs/roadmap.md`](docs/roadmap.md) — milestones, one-commit steps, risks, ADR backlog. Phase 1
targets objects in the sky against a sky background, validated against public anti-UAV thermal video;
phase 2 is ground scenes and automotive (ADR 0003).

---

## Status

Validation tiers: **T1** unit/analytic · **T2** radiometric bench · **T3** phenomenology ·
**T4** vs. real data · **T5** task-level. See `docs/physics-model.md` §15.

| Component | State | Tier | Notes |
|---|---|---|---|
| `radiometry` | 🟢 done | T1–T2 | Planck (both forms, derivatives, exitances; σ, σ_q to 1e-6), encoding (0.05 mK), R(λ) contract, Simpson oracle, band averaging, float32 LUT (0.03 mK) + inverse (< 1 mK), bundles + `make luts`; golden LUT slice at 1 mK; ADRs 0005–0013 |
| `materials` | 🟡 partial | T1 | Minimal per-band `MaterialTable` (id → ε₀, id 0 = UNMAPPED refused); spectral model pending M7 |
| `thermal` | ⬜ not started | — | |
| `atmosphere` | ⬜ not started | — | |
| `optics` | 🟡 partial | T1–T2 | Aperture factor π/(4F²+1) defined once (AST guard), FPA irradiance, pixel power; pinhole field angles and cos⁴ vignetting (ADR 0015); self-emission single-lens form + Kirchhoff-closed element stack, 87 mK/K shutterless drift (ADR 0016); box downsample + composed optics stage with inverse, blackbody round trip < 1 mK (ADR 0020) |
| `detector` | 🟡 partial | T1–T2 | `FpaParams`; ideal bolometer/photon transfers (ADR 0019), shared quantiser; NETD predictor (ADR 0024), anchoring (ADR 0025); `PhotonDetector` / `MicrobolometerDetector` responses with seeded per-pixel noise in physical units, two-blackbody NETD benches reproduce the anchor and the 0.576 derivative ratio (ADR 0026) |
| `noise` | 🟢 done | T2 | Counter-based per-pixel hash RNG (ADR 0022); NVESD synthesiser; `NoiseStage` (correlated 3-D terms on the detector's σ_TVH) wired into `run_frame`; DN-domain NETD within 10 % of the anchor, ratios recovered within floors; drift/NUC/bad pixels are M9 |
| `isp` | 🟡 partial | T1 | Radiometric branch: calibrated DN ↔ radiance (housing at T_cal = NUC level), T_app via LUT inverse, float32 and ½-LSB DN16 routes (ADR 0021); NUC/AGC/palette pending |
| `config` | 🟡 partial | T1 | `GBuffer` contract; pydantic `SensorConfig` (ADR 0007); YAML loader with data-root resolution, `config_hash`/`band_hash` (ADR 0008); band registry with derived/checked `band.id` and regime-driven illumination terms; optics/detector/noise extensions (schema v4, ADR 0017); 3-D ratio vector and variance closure `total_over_tvh` |
| `pipeline` | 🟡 partial | T1–T2 | Engine-free NumPy oracle (ADR 0018): `run_frame` composes band radiance → optics → detector noise → correlated noise → ADC → radiometric ISP (stage 2 identity; `noise_enabled=False` for the ideal chain); display8 pending M5 |
| `validation` | 🟡 partial | T2 | Tier 2 SITF and two-blackbody NETD benches; leakage-corrected NVESD 3-D decomposition, spatial/temporal PSD, `compare_psd` (ADR 0023) |
| `irsim_isaac` | 🟡 partial | T1 | `env.py` probes; **M2 gate spike done (ADR 0014):** no float32 colour AOV carries temperature (all fp16, exposure-scaled) → the renderer transports **instance ids + float32 geometry** and temperature comes from a Warp table; `omni.rtx.spg` 0.4.0 present, float32 pass-through bit-exact, **no cross-frame state** (stateful stages stay in Warp), LUT baked into the `.cu`. `probe.py`/`spg_probe.py` + `scripts/probe_isaac_*.py` reproduce it; `tests/integration` (10 tests, one Kit per session) pin it. Normals/AO/motion semantics still open (M10.1) |

Tier 2 today is self-consistency (no camera, ADR 0003): SITF strictly increasing, linear in L_B(T) to 0.29 LSB rms, blackbody T_app < 10 mK.

Bands configured: **LWIR** (`flir_boson_640_lwir`, estimated VOx response — ADR 0013; `make luts` builds the table) · Cameras modelled: _none yet_ (radiometry only; the pipeline starts at M3)

---

## Quick start

```bash
# The project interpreter is Isaac Sim's bundled Python (docs/decisions/0002); any CPython >= 3.10
# works for the engine-free core. Every make target honours PYTHON=.
export PYTHON=/home/hunter/IsaacSim/_build/linux-x86_64/release/python.sh
make install
make check        # lint + typecheck + unit tests — must be green before any commit
```

Isaac Sim is **not** required for anything in `src/irsim/` or `tests/unit/`. Requires NumPy ≥ 2.0.

Interpreter matrix (ADR 0002): the Isaac Sim interpreter (CPython 3.12) locally for everything
including integration tests; plain CPython 3.10 and 3.12 in CI (`.github/workflows/check.yml`, no GPU,
no Isaac). `make ci` reproduces the CI job locally in a `.venv-ci` built from `python3.10`.

## Commands

| Command | Does |
|---|---|
| `make install` | Editable install + dev dependencies |
| `make test` | Unit + golden tests (fast, no GPU) |
| `make test-all` | Adds integration tests (requires Isaac Sim) |
| `make lint` / `make fmt` | ruff check / ruff format |
| `make typecheck` | mypy on `src/irsim` and `src/irsim_isaac` |
| `make check` | lint + typecheck + test |
| `make ci` | The CI job locally: plain CPython 3.10 venv + `make check` (no GPU, no Isaac) |
| `make luts` | Regenerate band LUTs from configs and spectral data into `data/lut/` (gitignored; loader detects stale bundles, ADR 0012) |
| `make golden-update` | Regenerate golden reference arrays deliberately (ADR 0004) |

## Layout

```
src/irsim/          engine-free physics core (pure Python + NumPy); irsim.pipeline is the reference oracle (ADR 0018)
src/irsim_isaac/    Isaac Sim glue — the only place engine imports are allowed
tests/unit/         fast, no GPU, no Isaac Sim (default gate, with tests/golden)
tests/conftest.py   synthetic G-buffer fixtures: ramp, uniform, two-material, grazing sphere,
                    4x-supersampled 5.5° step edge, moving edge — the engine-free kernel test bed
tests/integration/  requires Isaac Sim; auto-marked isaac, skipped cleanly when absent
tests/golden/       regression fixtures: .npy + JSON sidecar with config hash (ADR 0004)
configs/            sensor / material / atmosphere YAML
data/               data root (`$IRSIM_DATA_DIR` overrides): spectra/responses/, n/k tables, LUTs, weather
docs/               physics-model.md and ADRs
```

## Current limitations

Stated deliberately — see `docs/physics-model.md` Appendix A for the full list and reasoning.

- No 3-D conduction. Engine bay and exhaust are prescribed, not solved.
- Band-averaged atmosphere (Beer-Lambert). Valid under ~500 m; not for airborne work.
- Emissivity is grey within a band.
- Reflections are approximate — sky-view-factor blending, not full path tracing.
- No polarisation, no atmospheric turbulence.
- NETD is anchored to datasheet values, not predicted from first principles.
- Weather is prescribed; there is no coupling back from the scene to the atmosphere.
- No IR camera is available: validation uses public datasets (8-bit, lossy-coded, some through an
  unknown ISP), so absolute radiometry (SITF, NETD) is checked for self-consistency only, and the
  target-vs-range behaviour is validated by data only to ~200 m — 0.5–5 km is modelled, not measured
  (ADR 0003).
- Clouds, slant-path atmosphere beyond 500 m and point-target radiometry are additions the physics
  specification does not cover; each carries its own ADR and error statement.

## Contributing

Read `CLAUDE.md` first — it defines the non-negotiables (engine-free core, checked in both directions by
`tests/unit/test_layering.py`, float32 everywhere
temperature flows, noise in radiance space, Kirchhoff closure) and the per-step workflow.

Every step: `make check` green → README status updated → CHANGELOG entry → ADR if a decision was made →
one commit.
