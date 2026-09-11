# ADR 0032 — Weather: one `WeatherSeries` object, injected; project CSV format; RH as a fraction

**Status:** Accepted
**Date:** 2026-09-11

## Context

CLAUDE.md #6: the temperature model and the atmosphere must run on the same weather. §6.4 and §7.3
list the fields each needs (air temperature, humidity, wind, cloud, direct/diffuse solar irradiance;
the atmosphere adds visibility). Appendix A #8 says weather is prescribed from a data file, never
simulated. The failure this ADR prevents is silent: a scene built with summer weather in the solver
and winter weather in the atmosphere renders a plausible image.

## Options considered

1. A global weather singleton — hidden state, untestable, and two scenes in one process collide.
2. Each consumer loads its own file from a shared path — the same file *today*; nothing stops a
   different path tomorrow, and the identity cannot be tested.
3. **One immutable `WeatherSeries` instance, built once by a loader, injected into every consumer**
   (chosen). Consumers take the object, never a path; the `Scene` builder (M6.17) constructs it once
   and asserts `x.weather is y.weather` across consumers.

## Decision

`irsim.thermal.weather.WeatherSeries`: frozen dataclass of read-only float64 arrays on a common
`time_s` axis (seconds since an aware UTC `epoch_utc`), validated at construction, linearly
interpolated by `at(t)`, extrapolation refused, `content_hash` over the bytes. It has no file-reading
code at all (tested by introspection). Fields and units:

| column | unit | guard |
|---|---|---|
| `t_air_k` | K | 150–350 (a Celsius value raises) |
| `rh_fraction` | 0–1 | a percentage raises |
| `wind_speed_m_s` | m s⁻¹ | ≥ 0 |
| `cloud_fraction` | 0–1 | |
| `dni_w_m2`, `dhi_w_m2` | W m⁻² | 0 … `SOLAR_CONSTANT_W_M2` (1361, Kopp & Lean 2011) |
| `visibility_m` | m | > 0, ∞ allowed |
| `precip_mm_h` | mm h⁻¹ | ≥ 0 |

**RH is a fraction** everywhere in the code (the Magnus routine of M8.2 raises on 80), and the
file carries the unit in the column name so a percent column cannot be mistaken for a fraction.

**Project CSV format** (`irsim.thermal.weather_io`, M6.2): UTF-8, `#` comment lines, one header
row of unit-suffixed column names, ISO-8601 UTC timestamps (`2024-06-21T05:00:00Z`), hourly or finer,
strictly increasing. The header may use `t_air_c` and `rh_percent` instead of `t_air_k` and
`rh_fraction`; the conversion happens once, at load. Floats are written with the shortest
round-trip representation so write → read is bit-exact. The sample
`data/weather/clear_midlat_summer_48h.csv` is synthetic (`synthetic_clear_day`) and a test regenerates
it, so file and generator cannot drift.

## Consequences

- Any consumer signature with a `weather_file`/path parameter is a defect (tests grep for it).
- Sub-hourly detail (gusts, passing clouds) is interpolated away; for that, supply finer samples.
- Irradiance in the file is what a pyranometer would give at the surface: DNI/DHI already include
  the solar-path atmosphere. The solar-path transmittance of M11 is for the *reflected* term's
  spectral shape, not for scaling the file's irradiance twice.
- Extrapolation refusal means spin-up (24–48 h before t = 0) needs the file to start early.

## Revisit when

A weather source with sub-hourly cadence or a gridded/forecast product is adopted, or when the
thermal solver needs soil moisture / precipitation history as state.
