"""Register the C98 graded-score outcome from its artifact. See C98_PREREGISTRATION.md."""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/differential/graded_score_granite33_g253.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "gradedscore"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]
for field in ("n", "n_positives", "auroc", "mannwhitney_p_one_sided",
              "n_degenerate", "n_distinct_scores"):
    facts[f"{PREFIX}.granite33_g253.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", d[field])])
FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: auroc={d['auroc']:.4f} p={d['mannwhitney_p_one_sided']:.4f} outcome={d['outcome']}")
