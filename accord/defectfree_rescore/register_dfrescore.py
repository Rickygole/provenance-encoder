"""Register the C110 defect-free-rescore outcome from its artifact.
See C110_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task -- another agent owns
results/accord/VERIFIED_FACTS.json and is writing the register separately.
Mirrors the pattern in accord/tfidf_permutation/register_tfidfperm.py: delete
this prefix wholesale, re-add from the committed artifact only.

    python3 accord/defectfree_rescore/register_dfrescore.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/defectfree_rescore/dfrescore_strata.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "dfrescore"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = {}
for stratum_name, entry in d["strata"].items():
    entries[f"{stratum_name}.n"] = entry["n"]
    entries[f"{stratum_name}.n_pos"] = entry["n_pos"]
    for system in ("detector", "nli", "tfidf"):
        sys_entry = entry.get(system)
        if sys_entry is None:
            continue
        entries[f"{stratum_name}.{system}.auroc"] = sys_entry["auroc"]
        entries[f"{stratum_name}.{system}.ap"] = sys_entry["ap"]
        entries[f"{stratum_name}.{system}.p_permutation_auroc"] = sys_entry["p_permutation_auroc"]
        entries[f"{stratum_name}.{system}.p_permutation_ap"] = sys_entry["p_permutation_ap"]

entries["tfidf_reproduction_check.passed"] = d["tfidf_reproduction_check"]["passed"]
entries["decision.outcome"] = d["decision"]["outcome"]
entries["decision.decision_cell"] = d["decision"]["decision_cell"]
entries["decision.detector_g253_auroc_reference"] = d["decision"]["detector_g253_auroc_reference"]
entries["decision.detector_stratum_auroc"] = d["decision"]["detector_stratum_auroc"]
entries["decision.detector_stratum_p_permutation_auroc"] = d["decision"]["detector_stratum_p_permutation_auroc"]
for stratum_name, cmp_entry in d["ordering_comparison_vs_g253"].items():
    entries[f"ordering_comparison.{stratum_name}.matches_g253_order"] = cmp_entry["matches_g253_order"]

for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d['decision']['outcome']}")
