"""Register the synthetic trigger/exemplar assertion-fixture counts from
results/accord/mining/trigger_exemplar_assertion.json into
results/accord/VERIFIED_FACTS.json.

Mirrors the pattern in accord/exemplar_direct/register_exemplardirect.py:
delete the "assertfixture" prefix wholesale, re-add from the committed
artifact only.

Registers both fixtures the artifact reports:
  assertfixture.multi.*   the primary fixture -- trigger sets of size 1-3
  assertfixture.single.*  the original fixture -- trigger sets of size 1,
                           kept for comparison

Counts only -- see run_assertion_fixture.py's module docstring for the scope
note (synthetic fixture, not a prevalence measurement).

    python3 accord/mining/register_assertfixture.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/mining/trigger_exemplar_assertion.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"
PREFIX = "assertfixture"

MULTI_FIELDS = (
    "n_candidates",
    "trigger_set_size_1_count",
    "trigger_set_size_2_count",
    "trigger_set_size_3_count",
    "n_first_mention_in_trigger_set",
    "positional_failures",
    "trigger_selector_failures",
    "construction_check_no_alarm_on_trigger_members",
    "seed",
)

SINGLE_FIELDS = (
    "n_candidates",
    "n_first_mention_is_trigger",
    "positional_failures",
    "trigger_selector_failures",
    "false_alarms",
    "seed",
)


def main() -> None:
    d = json.loads((ROOT / SRC).read_text())
    facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
    for k in [k for k in facts if k.split(".")[0] == PREFIX]:
        del facts[k]

    multi = d["multi_trigger"]
    for field in MULTI_FIELDS:
        facts[f"{PREFIX}.multi.{field}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::multi_trigger.{field}"), ("value", multi[field])]
        )

    single = d["single_trigger"]
    for field in SINGLE_FIELDS:
        facts[f"{PREFIX}.single.{field}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::single_trigger.{field}"), ("value", single[field])]
        )

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}.multi: "
          + ", ".join(f"{f}={multi[f]}" for f in MULTI_FIELDS))
    print(f"registered {PREFIX}.single: "
          + ", ".join(f"{f}={single[f]}" for f in SINGLE_FIELDS))


if __name__ == "__main__":
    main()
