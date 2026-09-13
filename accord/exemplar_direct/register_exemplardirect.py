"""Register the C106 direct-exemplar-check outcome from its artifact.
See C106_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task (out of edit scope for
results/accord/VERIFIED_FACTS.json). Mirrors the pattern in
accord/differential/register_graded_score.py: delete this prefix wholesale,
re-add from the committed artifact only.

    python3 accord/exemplar_direct/register_exemplardirect.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/exemplar_direct/exemplar_direct.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "exemplardirect"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

if d.get("outcome") == "C":
    facts[f"{PREFIX}.outcome"] = collections.OrderedDict(
        [("artifact", f"{SRC}::outcome"), ("value", "C")]
    )
    facts[f"{PREFIX}.reason"] = collections.OrderedDict(
        [("artifact", f"{SRC}::reason"), ("value", d["reason"])]
    )
else:
    for field in (
        "outcome", "n_total_rule1_rule2", "n_decidable", "n_undecidable",
        "n_wrong", "n_correct", "n_wrong_involves_unclassified",
        "n_wrong_no_unclassified_both_sides_classified", "defect_rate",
        "wilson_ci_95", "n_at_cap_200_discharge", "n_decidable_excl_capped",
        "n_wrong_excl_capped", "defect_rate_excl_capped",
        "wilson_ci_95_excl_capped",
    ):
        facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
            [("artifact", f"{SRC}::{field}"), ("value", d[field])]
        )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d.get('outcome')}")
