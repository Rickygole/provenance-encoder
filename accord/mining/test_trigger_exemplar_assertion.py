"""Hand-made cases for the trigger/exemplar assertion. Abstract synthetic
mentions only (see trigger_exemplar_assertion.py's module docstring for
scope). Run with:

    python3 -m pytest accord/mining/test_trigger_exemplar_assertion.py -v

or, if pytest is unavailable, run this file directly with python3 -- the
functions below are plain `assert`-based and every `test_*` function is
called from `__main__`.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.mining.trigger_exemplar_assertion import (  # noqa: E402
    assert_exemplar_in_trigger_set,
    positional_selector,
    trigger_selector,
    trigger_set,
)

PREDICATE = lambda m: m["action"] == "held"  # noqa: E731


def test_trigger_at_position_0_passes():
    mentions = [
        {"drug": "D1", "action": "held"},
        {"drug": "D1", "action": "continue"},
    ]
    triggers = trigger_set(mentions, PREDICATE)
    assert triggers == [0]

    pos_exemplar = positional_selector(mentions, PREDICATE)
    fix_exemplar = trigger_selector(mentions, PREDICATE)
    assert pos_exemplar == 0
    assert fix_exemplar == 0

    # neither selector should trip the assertion when the trigger is first
    assert_exemplar_in_trigger_set(pos_exemplar, triggers)
    assert_exemplar_in_trigger_set(fix_exemplar, triggers)


def test_trigger_at_position_2_positional_selector_fails_assertion():
    mentions = [
        {"drug": "D1", "action": "continue"},
        {"drug": "D1", "action": "continue"},
        {"drug": "D1", "action": "held"},
    ]
    triggers = trigger_set(mentions, PREDICATE)
    assert triggers == [2]

    pos_exemplar = positional_selector(mentions, PREDICATE)
    assert pos_exemplar == 0  # the bug: grabs the first mention, not the trigger

    raised = False
    try:
        assert_exemplar_in_trigger_set(pos_exemplar, triggers)
    except AssertionError:
        raised = True
    assert raised, "positional selector's exemplar should fail the assertion here"


def test_trigger_at_position_2_fixed_selector_passes():
    mentions = [
        {"drug": "D1", "action": "continue"},
        {"drug": "D1", "action": "continue"},
        {"drug": "D1", "action": "held"},
    ]
    triggers = trigger_set(mentions, PREDICATE)

    fix_exemplar = trigger_selector(mentions, PREDICATE)
    assert fix_exemplar == 2  # the fix: finds the mention that actually triggered

    # must not raise
    assert_exemplar_in_trigger_set(fix_exemplar, triggers)


def test_assertion_does_not_fire_on_correct_exemplars():
    # a batch of correct (exemplar, trigger_set) pairs -- the assertion
    # must be silent on every one of them (no false alarms)
    cases = [
        (0, [0]),
        (0, [0, 1]),
        (3, [1, 3, 5]),
        (5, [5]),
    ]
    for exemplar, triggers in cases:
        assert_exemplar_in_trigger_set(exemplar, triggers)  # must not raise


def test_assertion_fires_when_exemplar_outside_trigger_set():
    raised = False
    try:
        assert_exemplar_in_trigger_set(0, [1, 2])
    except AssertionError:
        raised = True
    assert raised


if __name__ == "__main__":
    tests = [
        test_trigger_at_position_0_passes,
        test_trigger_at_position_2_positional_selector_fails_assertion,
        test_trigger_at_position_2_fixed_selector_passes,
        test_assertion_does_not_fire_on_correct_exemplars,
        test_assertion_fires_when_exemplar_outside_trigger_set,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
