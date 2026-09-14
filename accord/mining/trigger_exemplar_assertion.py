"""The one-line trigger/exemplar assertion, and the two selectors it is meant
to catch a divergence between.

Context (see accord/phase3_mining.py, module docstring, lines 1-60): the
original mining code decides a rule fires over the SET of actions seen
across every mention of a drug in a note, but then shows the annotator,
labeller, and model the FIRST mention of that drug by row position
(`.iloc[0]`) -- which need not be a mention whose own action is the one that
made the set-level rule true. `positional_selector` below reproduces that
bug in the abstract; `trigger_selector` reproduces the rebuild's fix, which
restricts the exemplar to mentions whose own field actually satisfies the
rule predicate.

`assert_exemplar_in_trigger_set` is the one-line assertion the thesis says
"would have caught" the bug had the mining code's schema recorded, alongside
each candidate, which mention indices actually satisfied the rule (the
"trigger set") rather than only the winning exemplar index.

IMPORTANT SCOPE NOTE: everything here operates on abstract synthetic
fixtures only -- dicts like {"drug": "D1", "action": "held"}, never real
mention text. This module and accord/mining/run_assertion_fixture.py
demonstrate that the assertion runs, and show it separating a buggy
selector from a fixed one on synthetic data. They do NOT measure how often
the trigger-exemplar dissociation defect actually occurred in the thesis's
real mining output -- that prevalence question is out of scope here and is
not answered by this module.
"""
from __future__ import annotations

from typing import Callable, Sequence


def assert_exemplar_in_trigger_set(exemplar_index, trigger_indices) -> None:
    """The one-line check: the chosen exemplar must be one of the mentions
    that actually satisfied the rule predicate (the trigger set), not merely
    some mention of the drug.
    """
    assert exemplar_index in trigger_indices, (
        f"exemplar index {exemplar_index!r} not in trigger set {list(trigger_indices)!r} "
        "-- exemplar/trigger dissociation"
    )


def trigger_set(mentions: Sequence[dict], predicate: Callable[[dict], bool]) -> list[int]:
    """Indices of every mention whose own fields satisfy the rule predicate."""
    return [i for i, m in enumerate(mentions) if predicate(m)]


def positional_selector(mentions: Sequence[dict], predicate: Callable[[dict], bool]) -> int:
    """THE ORIGINAL BUG: always the first mention of the drug, by row
    position, regardless of whether that mention is the one whose action
    made the rule fire. `predicate` is accepted only to keep the same
    signature as `trigger_selector`; it is not consulted.
    """
    del predicate  # unused: this is exactly the bug -- position, not trigger
    return 0


def trigger_selector(mentions: Sequence[dict], predicate: Callable[[dict], bool]) -> int | None:
    """THE FIX: the first mention (by position, for determinism) whose own
    fields satisfy the rule predicate. Returns None if no mention qualifies.
    """
    for i, m in enumerate(mentions):
        if predicate(m):
            return i
    return None
