"""Register the C107 tfidf-permutation outcome from its artifact.
See C107_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task (out of edit scope for
results/accord/VERIFIED_FACTS.json). Mirrors the pattern in
accord/differential/register_graded_score.py: delete this prefix wholesale,
re-add from the committed artifact only.

    python3 accord/tfidf_permutation/register_tfidfperm.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/tfidf_permutation/tfidf_permutation.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "tfidfperm"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = {
    "task_a.g299.auroc": d["task_a"]["g299"]["observed_auroc"],
    "task_a.g299.ap": d["task_a"]["g299"]["observed_ap"],
    "task_a.g299.p_permutation_auroc": d["task_a"]["g299"]["p_permutation_auroc_one_sided"],
    "task_a.g299.p_permutation_ap": d["task_a"]["g299"]["p_permutation_ap_one_sided"],
    "task_a.g253.auroc": d["task_a"]["g253"]["observed_auroc"],
    "task_a.g253.ap": d["task_a"]["g253"]["observed_ap"],
    "task_a.g253.p_permutation_auroc": d["task_a"]["g253"]["p_permutation_auroc_one_sided"],
    "task_a.g253.p_permutation_ap": d["task_a"]["g253"]["p_permutation_ap_one_sided"],
    "task_a.decision": d["task_a"]["decision"],
    "task_b.pooled_out_of_fold_auroc": d["task_b"]["pooled_out_of_fold_auroc"],
    "task_b.pooled_out_of_fold_ap": d["task_b"]["pooled_out_of_fold_ap"],
    "task_b.mannwhitney_p_one_sided_greater": d["task_b"]["mannwhitney_p_one_sided_greater"],
    "task_b.mde_auroc_reference": d["task_b"]["mde_auroc_reference"],
    "task_b.decision": d["task_b"]["decision"],
}
for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: task_a={d['task_a']['decision']} task_b={d['task_b']['decision']}")
