# ADR 0001 — Record architecture decisions

**Status:** Accepted
**Date:** (fill in)

## Context

This project makes many deliberate approximations: cheaper solvers, band-averaged atmosphere,
sky-view-factor reflections instead of path tracing, NETD anchored rather than predicted. Six months
later nobody remembers which of these were considered and chosen versus never thought about. That
distinction matters — a reviewer needs to tell a deliberate approximation from an oversight, and a
future contributor needs to know what would have to change to raise fidelity.

## Decision

Record every significant decision as a short ADR in `docs/decisions/NNNN-title.md`, using this template.

## Template

```markdown
# ADR NNNN — Title

**Status:** Proposed | Accepted | Superseded by ADR NNNN
**Date:** YYYY-MM-DD

## Context
What forced a choice. Include the physical or engineering constraint, not just the code situation.

## Options considered
1. Option A — cost, fidelity, risk
2. Option B — ...

## Decision
What we chose.

## Consequences
What this makes easy, what it makes hard, and **what error it introduces**. If the error cannot be
bounded, say so explicitly — that is itself important information.

## Revisit when
The condition that should make someone reopen this. E.g. "when we need ranges beyond 500 m" or
"when Tier 4 shows a per-class bias above 2 K".
```

## Consequences

Small ongoing cost per decision. Makes the fidelity ladder in `docs/physics-model.md` §1 auditable
rather than aspirational.
