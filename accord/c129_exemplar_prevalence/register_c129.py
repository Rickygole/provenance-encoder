"""Register C129 (prefix c129; deletes and re-adds) from its artifact.
See C129_PREREGISTRATION.md.

    python3 accord/c129_exemplar_prevalence/register_c129.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c129_exemplar_prevalence/c129.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c129"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]


def add(key, artifact_path, value):
    facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{artifact_path}"), ("value", value)]
    )


add("outcome", "outcome", d["outcome"])
add("gate_counts_match", "gate_counts_match", d["gate_counts_match"])

if d["outcome"] == "A":
    for block_name in ("rule1", "rule2", "pooled"):
        b = d[block_name]
        add(f"{block_name}.n_candidates", f"{block_name}.n_candidates", b["n_candidates"])
        for side in ("nursing_side", "discharge_side", "either_side"):
            r = b["positional_outside_trigger_set"][side]
            for field in ("n", "count", "rate", "wilson_lo", "wilson_hi"):
                add(f"{block_name}.positional_outside_trigger_set.{side}.{field}",
                    f"{block_name}.positional_outside_trigger_set.{side}.{field}", r[field])
            rc = b["rebuild_selector_outside_trigger_set_construction_check"][side]
            add(f"{block_name}.rebuild_selector_outside_trigger_set.{side}.count",
                f"{block_name}.rebuild_selector_outside_trigger_set_construction_check.{side}.count",
                rc["count"])
        for side, v in b["single_mention_per_side_share"].items():
            add(f"{block_name}.single_mention_per_side_share.{side}",
                f"{block_name}.single_mention_per_side_share.{side}", v)

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d['outcome']}; "
      f"{sum(1 for k in facts if k.startswith(PREFIX + '.'))} keys under prefix {PREFIX}")
