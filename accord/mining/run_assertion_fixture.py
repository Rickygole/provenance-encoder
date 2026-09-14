"""Run the trigger/exemplar assertion (accord/mining/trigger_exemplar_assertion.py)
against deterministic, seeded sets of ABSTRACT SYNTHETIC candidates -- not
real mention text, not credentialed data -- and record whether it ever
actually runs, and what it catches.

This module is synthetic only; it is not a prevalence measurement. It
answers the committee's question ("has the assertion ever run on anything,
even a synthetic fixture?") with a yes and a receipt: it does NOT measure
the trigger-exemplar dissociation defect's prevalence in the thesis's real
mining output. See the module docstring in trigger_exemplar_assertion.py
for that scope note.

Two fixtures are built and both are reported:

MULTI-TRIGGER (the primary fixture, "multi_trigger" in the output). The
bug this assertion is meant to catch is a rule firing on a SET of mentions
-- any mention satisfying the predicate belongs to the trigger set, and a
candidate's trigger set can have more than one member. Each synthetic
candidate is a list of 1-6 abstract mentions of a single drug, e.g.
[{"drug": "D1", "action": "continue"}, {"drug": "D1", "action": "held"}].
Between 1 and 3 of a candidate's mentions are drawn, at random positions,
to be triggers (action == "held", satisfying the rule predicate); the rest
are non-triggering ("continue"). Because trigger positions are drawn
uniformly from all mention positions, some candidates end up with their
first mention (position 0) among the triggers, and some do not.

SINGLE-TRIGGER (the original fixture, kept for comparison as
"single_trigger" in the output, unchanged). Exactly one mention per
candidate is the trigger; this is the special case of a trigger set of
size 1, and is retained so the earlier receipt is not lost.

    python3 accord/mining/run_assertion_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.mining.trigger_exemplar_assertion import (  # noqa: E402
    assert_exemplar_in_trigger_set,
    positional_selector,
    trigger_selector,
    trigger_set,
)

SEED = 20260914
N_CANDIDATES = 1000
OUT_PATH = ROOT / "results" / "accord" / "mining" / "trigger_exemplar_assertion.json"

PREDICATE = lambda m: m["action"] == "held"  # noqa: E731 -- the rule predicate

MAX_TRIGGER_SET_SIZE = 3
MAX_MENTIONS = 6


# --------------------------------------------------------------------------
# MULTI-TRIGGER fixture: trigger sets of size 1 to MAX_TRIGGER_SET_SIZE,
# placed at random positions among 1 to MAX_MENTIONS mentions.
# --------------------------------------------------------------------------

def make_multi_candidate(rng: random.Random) -> list[dict]:
    """One synthetic candidate: 1-6 mentions of drug D1, of which 1-3 are
    triggers (action == "held"), placed at positions drawn uniformly at
    random (without replacement) from all mention positions -- so a
    candidate's first mention is a trigger whenever position 0 happens to
    be drawn, and is not otherwise. The rest of the mentions are
    non-triggering (action == "continue").
    """
    n_mentions = rng.randint(1, MAX_MENTIONS)
    k = rng.randint(1, min(MAX_TRIGGER_SET_SIZE, n_mentions))
    trigger_positions = set(rng.sample(range(n_mentions), k))
    mentions = []
    for i in range(n_mentions):
        action = "held" if i in trigger_positions else "continue"
        mentions.append({"drug": "D1", "action": action})
    return mentions


def build_multi_fixture(seed: int, n_candidates: int) -> list[list[dict]]:
    rng = random.Random(seed)
    return [make_multi_candidate(rng) for _ in range(n_candidates)]


def run_multi() -> dict:
    candidates = build_multi_fixture(SEED, N_CANDIDATES)

    size_counts = {1: 0, 2: 0, 3: 0}
    n_first_mention_in_trigger_set = 0
    positional_failures = 0
    trigger_selector_failures = 0
    construction_check_no_alarm_on_trigger_members = True

    for mentions in candidates:
        triggers = trigger_set(mentions, PREDICATE)
        k = len(triggers)
        assert 1 <= k <= MAX_TRIGGER_SET_SIZE, (
            "fixture invariant: trigger-set size in "
            f"[1, {MAX_TRIGGER_SET_SIZE}], got {k}"
        )
        size_counts[k] += 1
        if 0 in triggers:
            n_first_mention_in_trigger_set += 1

        pos_exemplar = positional_selector(mentions, PREDICATE)
        try:
            assert_exemplar_in_trigger_set(pos_exemplar, triggers)
        except AssertionError:
            positional_failures += 1

        fix_exemplar = trigger_selector(mentions, PREDICATE)
        try:
            assert_exemplar_in_trigger_set(fix_exemplar, triggers)
        except AssertionError:
            trigger_selector_failures += 1

        # Construction check, not a result: the assertion must never fire
        # when the exemplar IS a member of the trigger set. This is
        # verified over EVERY member of every trigger set (not only the
        # member a selector happened to pick), because it is true by
        # construction of assert_exemplar_in_trigger_set and trigger_set
        # -- a failure here would mean the assertion itself is broken, not
        # that a selector is buggy.
        for member in triggers:
            try:
                assert_exemplar_in_trigger_set(member, triggers)
            except AssertionError:
                construction_check_no_alarm_on_trigger_members = False

    return {
        "n_candidates": N_CANDIDATES,
        "trigger_set_size_1_count": size_counts[1],
        "trigger_set_size_2_count": size_counts[2],
        "trigger_set_size_3_count": size_counts[3],
        "n_first_mention_in_trigger_set": n_first_mention_in_trigger_set,
        "positional_failures": positional_failures,
        "trigger_selector_failures": trigger_selector_failures,
        "construction_check_no_alarm_on_trigger_members": (
            construction_check_no_alarm_on_trigger_members
        ),
        "seed": SEED,
    }


# --------------------------------------------------------------------------
# SINGLE-TRIGGER fixture (original, kept unchanged for comparison): exactly
# one triggering mention per candidate.
# --------------------------------------------------------------------------

def make_single_candidate(rng: random.Random, force_first: bool) -> list[dict]:
    """One synthetic candidate: 1-6 mentions of drug D1, exactly one of
    which is the trigger (action == "held"); the rest are non-triggering
    (action == "continue").
    """
    n_mentions = rng.randint(1, 6)
    trigger_pos = 0 if force_first else rng.randint(0, n_mentions - 1)
    mentions = []
    for i in range(n_mentions):
        action = "held" if i == trigger_pos else "continue"
        mentions.append({"drug": "D1", "action": action})
    return mentions


def build_single_fixture(seed: int, n_candidates: int) -> list[list[dict]]:
    rng = random.Random(seed)
    candidates = []
    for i in range(n_candidates):
        # ~20% of candidates deliberately force the trigger to position 0,
        # the case where positional_selector happens to be right anyway.
        force_first = rng.random() < 0.2
        candidates.append(make_single_candidate(rng, force_first))
    return candidates


def run_single() -> dict:
    candidates = build_single_fixture(SEED, N_CANDIDATES)

    n_first_mention_is_trigger = 0
    positional_failures = 0
    trigger_selector_failures = 0
    false_alarms = 0

    for mentions in candidates:
        triggers = trigger_set(mentions, PREDICATE)
        assert len(triggers) == 1, "fixture invariant: exactly one trigger per candidate"
        if triggers[0] == 0:
            n_first_mention_is_trigger += 1

        pos_exemplar = positional_selector(mentions, PREDICATE)
        try:
            assert_exemplar_in_trigger_set(pos_exemplar, triggers)
        except AssertionError:
            positional_failures += 1

        fix_exemplar = trigger_selector(mentions, PREDICATE)
        try:
            assert_exemplar_in_trigger_set(fix_exemplar, triggers)
        except AssertionError:
            trigger_selector_failures += 1

        # the assertion must never fire when the exemplar IS in the
        # trigger set -- no false alarms. The fixed selector is, by
        # construction, always in the trigger set, so this is the sanity
        # check that the assertion itself is not over-strict.
        if fix_exemplar in triggers:
            try:
                assert_exemplar_in_trigger_set(fix_exemplar, triggers)
            except AssertionError:
                false_alarms += 1

    return {
        "n_candidates": N_CANDIDATES,
        "n_first_mention_is_trigger": n_first_mention_is_trigger,
        "positional_failures": positional_failures,
        "trigger_selector_failures": trigger_selector_failures,
        "false_alarms": false_alarms,
        "seed": SEED,
    }


def run() -> dict:
    return {
        "multi_trigger": run_multi(),
        "single_trigger": run_single(),
    }


def main() -> None:
    results = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    for block, d in results.items():
        print(f"{block}:")
        for k, v in d.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
