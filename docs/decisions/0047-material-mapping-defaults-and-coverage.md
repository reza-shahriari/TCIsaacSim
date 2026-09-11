# ADR 0047 — Material mapping: precedence, the UNMAPPED default (magenta, never a silent ε), 95 % coverage gate

**Status:** Accepted
**Date:** 2026-09-11

## Context

No imported asset carries thermal properties; every prim must be assigned a library material.
Doing that by hand per object is the real per-object cost of the project and it is where silent
errors enter: a default emissivity applied to a forgotten prim produces a plausible image that is
wrong. The mapping must be automatic, auditable, and loud when it fails.

## Options considered

1. A default material for anything unmatched — silent, and the failure mode the project exists
   to avoid.
2. Fail the render on the first unmapped prim — correct but unusable while a scene is being built.
3. **Resolve by precedence, mark misses with the sentinel, render them magenta, and gate CI on
   coverage** (chosen).

## Decision

- **Precedence:** explicit override (`thermal:material` attribute; must name a library material or
  the import fails) → semantic class → ordered case-insensitive glob on the material name → miss.
  Rules live in `configs/materials/mapping.yaml`; targets are validated against the library.
- **The miss:** material id 0 (`UNMAPPED`), recorded with the prim path and material name. The
  NumPy oracle refuses to render id 0 (`MaterialTable.emissivity_for` raises); the Isaac debug
  render (M10.2) paints those pixels magenta so anyone can spot them. Where a debug image must
  still be produced with misses present, the kernel uses ε = 1 (the prim's own temperature as a
  blackbody) *under* the magenta overlay — never a "typical" emissivity that could pass as real.
- **Ids** are the packed table's sorted-name order (M7.18), so the resolver, the table and the
  G-buffer agree and ids are stable across loads.
- **Coverage gate:** `scripts/audit_materials.py` reports mapped/total, misses grouped and counted
  by material name, and exits non-zero below the threshold — **95 %** by default (in the rules
  file). New assets run it before they enter a scene.
- The prim dump is engine-free JSON (`path`, `material_name`, `semantic_class`, `override`);
  the Isaac adapter writes it and nothing in `irsim` imports USD to read it.

## Consequences

- Adding an object is a 30-second check: run the audit, add a pattern or override, re-run.
- A 95 % threshold still lets a few percent of prims render magenta in a scene; the image is
  honest about it, and the threshold can be raised per project.
- Pattern rules on material names are heuristic (`*metal*` → bare aluminium); the semantic
  route is preferred where a semantics schema exists.

## Revisit when

The material library grows past the point where a flat pattern list is maintainable (then a
per-asset mapping file), or when the debug AOV needs more than one sentinel colour.
