"""The roadmap's "do next" queue resolves to exactly one step, deterministically.

The roadmap's stated picking rule -- "the first step in phase order whose deps are all ticked" --
does not pick a step. On revision 4 the day it landed, **51 of 109** open steps satisfied it at
once, 13 of them in phase 0. What was actually choosing was the row's position in the table, which
is not a rule: it changes when anyone reorders a table, and it cannot be argued with.

`scripts/next_step.py` replaces that with a topological sort over a documented total order. These
tests are what make it safe to type "do next" and act on the answer without reading the document:

* the order is **total** -- one head, no ties, and two runs agree;
* it is **sound** -- no step is ever emitted before something it depends on;
* it is **complete** -- every open step appears exactly once, so nothing is silently dropped;
* the **promotions** are real ids with real reasons, and there are few of them.

docs/roadmap.md ("Do next"); scripts/next_step.py
"""

from __future__ import annotations

import collections
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "next_step.py"
ROADMAP = REPO / "docs" / "roadmap.md"


def _module():
    spec = importlib.util.spec_from_file_location("next_step", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ns():
    return _module()


@pytest.fixture(scope="module")
def steps(ns):
    return ns.load(ROADMAP)


@pytest.fixture(scope="module")
def queue(ns, steps):
    return ns.order(steps)


def test_the_roadmap_still_parses(steps: dict) -> None:
    """Guards the guard: a table-format change would make every assertion below vacuous."""
    assert len(steps) > 80, f"only {len(steps)} step rows parsed; the table format changed"
    lanes = {s.lane for s in steps.values()}
    assert len(lanes) >= 8, f"only found lanes {sorted(lanes)}"


def test_every_named_dependency_exists(steps: dict) -> None:
    """Revision 3 had eleven rows whose deps named ids that existed only as letter-suffixed rows,
    which broke its picking rule for those eleven. A dangling dep is a step nobody can start."""
    dangling = {(sid, dep) for sid, s in steps.items() for dep in s.deps if dep not in steps}
    assert not dangling, "\n".join(
        f"  {sid} depends on {dep}, which has no row" for sid, dep in sorted(dangling)
    )


def test_the_queue_is_sound(queue: list, steps: dict) -> None:
    """No step before something it depends on -- what lets you act on the head without reading."""
    position = {sid: i for i, sid in enumerate(queue)}
    violations = [
        (sid, dep)
        for sid in queue
        for dep in steps[sid].deps
        if dep in position and position[dep] > position[sid]
    ]
    assert not violations, "\n".join(
        f"  {sid} is queued before its dependency {dep}" for sid, dep in violations
    )


def test_the_queue_is_complete(queue: list, steps: dict) -> None:
    """Every open step appears exactly once. A queue that drops steps still answers, which is worse
    than one that fails: the dropped work is invisible rather than late."""
    open_ids = {sid for sid, s in steps.items() if not s.done}
    assert collections.Counter(queue) == collections.Counter(open_ids), (
        f"queued {len(queue)}, open {len(open_ids)}; "
        f"missing {sorted(open_ids - set(queue))[:5]}, extra {sorted(set(queue) - open_ids)[:5]}"
    )


def test_the_queue_is_deterministic(ns, steps: dict, queue: list) -> None:
    """Two runs over the same document agree, so "do next" means the same thing to two sessions."""
    assert ns.order(ns.load(ROADMAP)) == queue
    assert ns.order(steps) == queue


def test_there_is_exactly_one_next_step(queue: list, steps: dict, ns) -> None:
    """The point of the whole exercise: a head, not a set of equally-first candidates."""
    assert queue, "nothing open -- the roadmap is finished, which would be a surprise"
    score = ns.leverage(steps)
    head = queue[0]

    def key(sid: str) -> tuple:
        return (
            0 if sid in ns.PROMOTED else 1,
            ns.PHASE_RANK.get(steps[sid].phase, 9),
            -score[sid],
            ns.SIZE_RANK[steps[sid].size],
            sid,
        )

    pending = set(queue)
    startable = [s for s in queue if all(d not in pending for d in steps[s].deps)]
    assert head == min(startable, key=key), "the head is not the best startable step by the rule"


def test_every_pick_is_the_best_available_one(queue: list, steps: dict, ns) -> None:
    """The invariant, checked at every position and not just the head.

    Replay the queue: at each point, the step emitted must be the minimum by the stated key over
    everything startable at that moment. This is the property that makes the order *arguable* --
    if you disagree with a pick, you disagree with the key, not with a table's row order.

    It subsumes phase ordering. A later-phase step precedes an earlier-phase one only when the
    earlier one was not yet startable, which is the honest reason and not the one a naive phase
    check would report: `PT.7` (phase C) legitimately precedes `XD.3` (phase B) because `XD.3` is
    still waiting on its own dependencies at that point.
    """
    score = ns.leverage(steps)

    def key(sid: str) -> tuple:
        return (
            0 if sid in ns.PROMOTED else 1,
            ns.PHASE_RANK.get(steps[sid].phase, 9),
            -score[sid],
            ns.SIZE_RANK[steps[sid].size],
            sid,
        )

    pending = {sid for sid, s in steps.items() if not s.done}
    emitted: set[str] = set()
    for position, sid in enumerate(queue):
        startable = [
            other
            for other in pending
            if other not in emitted
            and all(dep not in pending or dep in emitted for dep in steps[other].deps)
        ]
        best = min(startable, key=key)
        assert sid == best, (
            f"position {position}: queued {sid} (key {key(sid)}) but {best} was available "
            f"with a better key {key(best)}"
        )
        emitted.add(sid)


def test_promotions_are_few_real_and_reasoned(ns, steps: dict) -> None:
    """Promotion is the only place judgement overrides the rule, so it is bounded and auditable.
    A long list means the rule is wrong and should be changed instead."""
    for sid, reason in ns.PROMOTED.items():
        assert sid in steps, f"{sid} is promoted but has no row"
        assert len(reason) > 80, f"{sid}'s promotion needs a reason a reader can disagree with"
    assert len(ns.PROMOTED) <= 3, (
        f"{len(ns.PROMOTED)} promotions: the mechanical order is being overridden too often, "
        "which means the order itself should change"
    )


def test_the_roadmap_points_at_the_script(steps: dict) -> None:
    """A rule nobody can find is not a rule."""
    text = ROADMAP.read_text(encoding="utf-8")
    assert "next_step.py" in text, "the roadmap must tell a reader how to get the next step"
