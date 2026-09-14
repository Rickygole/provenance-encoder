"""Register the C112 outcome from its artifact.
See C112_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task -- another agent owns
results/accord/VERIFIED_FACTS.json and is writing the register separately.
Mirrors the pattern in accord/defectfree_rescore/register_dfrescore.py: delete
this prefix wholesale, re-add from the committed artifact only.

    python3 accord/c112_decontam_defectfree/register_c112.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c112_decontam_defectfree/c112_defectfree.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c112"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = {}

if d.get("outcome") == "C":
    entries["outcome"] = "C"
    entries["abort_reason"] = d["abort_reason"]
else:
    entries["reproduction_check.passed"] = d["reproduction_check"]["passed"]
    entries["reproduction_check.recomputed_ensemble_auroc_g253"] = (
        d["reproduction_check"]["recomputed_ensemble_auroc_g253_full_precision"]
    )
    entries["label_agreement_check.pharm_label_vs_label_clean_mismatches_n299"] = (
        d["label_agreement_check"]["pharm_label_vs_label_clean_mismatches_n299"]
    )
    entries["label_agreement_check.y_true_vs_label_clean_mismatches_n253"] = (
        d["label_agreement_check"]["y_true_vs_label_clean_mismatches_n253"]
    )
    entries["g253_ensemble_auroc_reference"] = d["g253_ensemble_auroc_reference"]

    for stratum_name, entry in d["strata"].items():
        entries[f"{stratum_name}.n"] = entry["n"]
        entries[f"{stratum_name}.n_pos"] = entry["n_pos"]
        for system, sys_entry in entry["systems"].items():
            entries[f"{stratum_name}.{system}.auroc"] = sys_entry["auroc"]
            entries[f"{stratum_name}.{system}.ap"] = sys_entry["ap"]
            entries[f"{stratum_name}.{system}.p_permutation_auroc"] = sys_entry["p_permutation_auroc"]
            entries[f"{stratum_name}.{system}.p_permutation_ap"] = sys_entry["p_permutation_ap"]

    entries["decision.outcome"] = d["decision"]["outcome"]
    entries["decision.decision_cell"] = d["decision"]["decision_cell"]
    entries["decision.ensemble_stratum_auroc"] = d["decision"]["ensemble_stratum_auroc"]
    entries["decision.ensemble_stratum_p_permutation_auroc"] = d["decision"]["ensemble_stratum_p_permutation_auroc"]
    entries["decision.ensemble_g253_auroc_reference"] = d["decision"]["ensemble_g253_auroc_reference"]
    entries["decision.auroc_gap_vs_g253"] = d["decision"]["auroc_gap_vs_g253"]

for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d.get('outcome') or d.get('decision', {}).get('outcome')}")
