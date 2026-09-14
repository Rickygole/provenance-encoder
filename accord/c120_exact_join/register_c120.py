"""Register C120 (prefix c120; deletes and re-adds). Moves the superseded C88 reconstructed-join keys
(weaklabeller.*) under withdrawn.superseded_c120 so the register keeps them on record but not citable.

    python3 accord/c120_exact_join/register_c120.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c120_exact_join/c120.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c120"
d = json.loads((ROOT / SRC).read_text())
assert d["outcome"] == "A", d
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]
facts[f"{PREFIX}.outcome"] = {"artifact": f"{SRC}::outcome", "value": d["outcome"]}
for s in ("g299", "g253"):
    for f, v in d[s].items():
        facts[f"{PREFIX}.{s}.{f}"] = {"artifact": f"{SRC}::{s}.{f}", "value": v}
SUP = "withdrawn.superseded_c120."
for k in [k for k in list(facts) if k.startswith("weaklabeller.")]:
    facts[SUP + k] = facts.pop(k)
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered c120; superseded weaklabeller keys: {sum(1 for k in facts if k.startswith(SUP))}; total {len(facts)}")
