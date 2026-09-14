"""Register C116 from its artifact (prefix c116, deleted and re-added).

    python3 accord/c116_rank_paired/register_c116.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c116_rank_paired/c116.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c116"
d = json.loads((ROOT / SRC).read_text())
assert d["outcome"] == "valid", d
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]
paths = {"t1_keep_tfidf_claim": "t1_keep_tfidf_claim", "t2_nli_claim_allowed": "t2_nli_claim_allowed"}
for t in ("t1_tfidf_minus_detector", "t2_nli_minus_detector", "t3_tfidf_minus_nli"):
    for f in ("observed_ap_system1", "observed_ap_system2", "observed_diff", "p_permutation_two_sided", "B"):
        paths[f"{t}.{f}"] = f"{t}.{f}"
for s in ("tfidf", "nli", "det"):
    paths[f"raw_score_scale_g253.{s}.median"] = f"raw_score_scale_g253.{s}.median"


def get(o, dotted):
    for p in dotted.split("."):
        o = o[p]
    return o


for k, p in paths.items():
    facts[f"{PREFIX}.{k}"] = {"artifact": f"{SRC}::{p}", "value": get(d, p)}
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {len(paths)} {PREFIX} keys; total {len(facts)}")
