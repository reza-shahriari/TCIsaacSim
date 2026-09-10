# Subsystem maps

Frozen snapshots of the twelve subsystem analyses that produced `docs/roadmap.md` (2026-09-10). Each
file is the structured output of one reader that went through `docs/physics-model.md`, the skills and
the repository for a single area and recorded:

- `existing_state` — what existed in the repo at the time, file by file;
- `key_equations_and_data` — every equation, constant, data file and config field the area must contain,
  with spec sections;
- `work_items` — a finer decomposition than the roadmap's steps, each with the tests that would fail if
  the physics were wrong (the roadmap's **what** column cites these ids: `R-06`, `THM-08`, `IU-11`, …);
- `decisions_needing_adr`, `ambiguities_and_risks`, `non_negotiables_touched`,
  `spec_internal_inconsistencies`.

They are **not maintained**: the roadmap and the ADRs are the living documents. Use them when a roadmap
step needs splitting (risk R15), when a step's test list looks thin, or to see why a spec issue was
raised. Where a map and the roadmap disagree, the roadmap wins.

| file | area |
|---|---|
| `radiometry.json` | §3 Planck, band integration, LUT, inverse |
| `materials-surface.json` | §4, §12.3 materials, Kirchhoff, Fresnel, USD mapping |
| `scene-radiance.json` | §5 sky, reflection, solar, night illumination |
| `thermal.json` | §6 energy balance, weather, vehicle regimes |
| `atmosphere.json` | §7 Beer–Lambert, humidity, presets |
| `optics.json` | §8 aperture, vignetting, self-emission, MTF |
| `detector.json` | §9 photon and bolometer models, NETD |
| `noise.json` | §10 3-D noise, drift, bad pixels, NUC residual |
| `isp.json` | §11 NUC, AGC, DDE, palette |
| `band-config.json` | §12 configuration schemas, band registry, LUT files |
| `isaac-and-unreal.json` | §13, §14 AOVs, Warp, SPG, Unreal mapping |
| `validation.json` | §15 tiers, fixtures, benches, golden data |
