"""Register deployed_vs_frontier_zeroshot.json under prefix deployedvsfzs (deletes prefix, re-adds).

    python3 accord/stats/register_deployed_vs_fzs.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/stats/deployed_vs_frontier_zeroshot.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "deployedvsfzs"
d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]
for k, v in d.items():
    facts[f"{PREFIX}.{k}"] = {"artifact": f"{SRC}::{k}", "value": v}
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {len(d)} {PREFIX} keys; total {len(facts)}")
