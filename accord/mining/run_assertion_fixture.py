"""Run the trigger/exemplar assertion (accord/mining/trigger_exemplar_assertion.py)
against a deterministic, seeded set of ABSTRACT SYNTHETIC candidates -- not
real mention text, not credentialed data -- and record whether it ever
actually runs, and what it catches.

This answers the committee's question ("has the assertion ever run on
anything, even a synthetic fixture?") with a yes and a receipt: it does NOT
measure the trigger-exemplar dissociation defect's prevalence in the
thesis's real mining output. See the module docstring in
trigger_exemplar_assertion.py for that scope note.

Each synthetic candidate is a list of abstract mentions of a single drug,
e.g. [{"drug": "D1", "action": "continue"}, {"drug": "D1", "action": "held"}].
Exactly one mention per candidate is the "trigger" (action == "held",
satisfying the rule predicate); the rest are non-triggering ("continue").
The trigger's position is randomized 1-6 mentions deep, with a fraction of
candidates deliberately constructed so the trigger sits at position 0 (the
case where the original positional bug happens to be harmless).

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


def make_candidate(rng: random.Random, force_first: bool) -> list[dict]:
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


def build_fixture(seed: int, n_candidates: int) -> list[list[dict]]:
    rng = random.Random(seed)
    candidates = []
    for i in range(n_candidates):
        # ~20% of candidates deliberately force the trigger to position 0,
        # the case where positional_selector happens to be right anyway.
        force_first = rng.random() < 0.2
        candidates.append(make_candidate(rng, force_first))
    return candidates


def run() -> dict:
    candidates = build_fixture(SEED, N_CANDIDATES)

    n_first_mention_is_trigger = 0
    positional_failures = 0
    trigger_selector_failures = 0
    false_alarms = 0

    for mentions in candidates:
        triggers = trigger_set(mentions, PREDICATE)
        assert len(triggers) == 1, "fixture invariant: exactly one trigger per candidate"
        if triggers[0] == 0:
            n_first_mention_is_trigger += 1

        # (b) run both selectors, (c) count assertion failures for each
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

        # (d) the assertion must never fire when the exemplar IS in the
        # trigger set -- no false alarms. The fixed selector is, by
        # construction, always in the trigger set, so this is the sanity
        # check that the assertion itself is not over-strict.
        if fix_exemplar in triggers:
            try:
                assert_exemplar_in_trigger_set(fix_exemplar, triggers)
            except AssertionError:
                false_alarms += 1

    results = {
        "n_candidates": N_CANDIDATES,
        "n_first_mention_is_trigger": n_first_mention_is_trigger,
        "positional_failures": positional_failures,
        "trigger_selector_failures": trigger_selector_failures,
        "false_alarms": false_alarms,
        "seed": SEED,
    }
    return results


def main() -> None:
    results = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
