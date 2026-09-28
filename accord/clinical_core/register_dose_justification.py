#!/usr/bin/env python3
"""Register the Rule 3 justification profile. Counts only, never the free text."""
import collections, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.cloud_paths import assert_local_only  # noqa: E402

SRC = "results/accord/clinical_core/dose_justification_profile.json"
FACTS = assert_local_only(ROOT / "results/accord/VERIFIED_FACTS.json")
PREFIX = "dosejust"

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

put("n_confirmed_rule3", d["n_confirmed_rule3"], "n_confirmed_rule3")
put("n_with_a_justification", d["n_with_a_justification"], "n_with_a_justification")
lens = [r["justification_word_count"] for r in d["rows"]]
put("justification_words_min", min(lens), "rows[*].justification_word_count, derived: min")
put("justification_words_max", max(lens), "rows[*].justification_word_count, derived: max")
for k, v in d["theme_counts"].items():
    put(f"theme.{k}", v, f"theme_counts.{k}")
put("n_themes_with_zero_matches",
    sum(1 for v in d["theme_counts"].values() if v == 0),
    "theme_counts, derived: count of themes with no match")

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {n} keys; register now holds {len(facts)}")
