#!/usr/bin/env python3
"""Register the per-class confirmed rates and dictionary density."""
import collections, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/clinical_core/class_base_rates.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "classrate"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

n = 0
def put(key, value, path):
    global n
    facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
        [("value", value), ("artifact", f"{SRC}::{path}")])
    n += 1

put("n_annotated", d["n_annotated"], "n_annotated")
put("n_pairs_with_no_dictionary_class", d["n_pairs_with_no_dictionary_class"],
    "n_pairs_with_no_dictionary_class")
put("n_classes_in_dictionary", d["n_classes_in_dictionary"], "n_classes_in_dictionary")
for c, v in d["per_class"].items():
    for f in ("n_pairs", "n_confirmed", "confirmed_rate", "n_dictionary_agents"):
        put(f"{c}.{f}", v[f], f"per_class.{c}.{f}")
    put(f"{c}.confirmed_wilson_lo", v["confirmed_wilson95"][0],
        f"per_class.{c}.confirmed_wilson95[0]")
    put(f"{c}.confirmed_wilson_hi", v["confirmed_wilson95"][1],
        f"per_class.{c}.confirmed_wilson95[1]")
put("n_classes_with_a_confirmed_contradiction",
    sum(1 for v in d["per_class"].values() if v["n_confirmed"] > 0),
    "per_class, derived: count of classes with n_confirmed above 0")
put("max_dictionary_agents_in_any_class", max(d["dictionary_agents_per_class"].values()),
    "dictionary_agents_per_class, derived: max")
put("anticoagulants.is_largest_dictionary_class",
    d["anticoagulants"]["is_largest_dictionary_class"],
    "anticoagulants.is_largest_dictionary_class")

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys; register now holds {len(facts)}")
