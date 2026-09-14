"""Register the synthetic trigger/exemplar assertion-fixture counts from
results/accord/mining/trigger_exemplar_assertion.json into
results/accord/VERIFIED_FACTS.json.

Mirrors the pattern in accord/exemplar_direct/register_exemplardirect.py:
delete the "assertfixture" prefix wholesale, re-add from the committed
artifact only.

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

FIELDS = (
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

    for field in FIELDS:
        facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{field}"), ("value", d[field])]
        )

    FACTS.write_text(json.dumps(facts, indent=1) + "\n")
    print(f"registered {PREFIX}: " + ", ".join(f"{f}={d[f]}" for f in FIELDS))


if __name__ == "__main__":
    main()
