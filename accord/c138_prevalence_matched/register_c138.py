"""Register the C138 outcome from its artifact.
See accord/c138_prevalence_matched/C138_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task: another agent owns
results/accord/VERIFIED_FACTS.json and is writing the register concurrently.
Run this once that agent has finished. Mirrors
accord/c130_reproducible_student/register_c130.py exactly: delete the c138
prefix wholesale, then re-add every key from the committed artifact only.

    python3 accord/c138_prevalence_matched/register_c138.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c138_prevalence_matched/c138.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c138"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = collections.OrderedDict()
entries["mde_auroc_253"] = d["mde_auroc_253"]
entries["teacher.model"] = d["teacher"]["model"]
entries["teacher.prompt"] = d["teacher"]["prompt"]
entries["teacher.gold_g253_auroc_of_this_config"] = d["teacher"]["gold_g253_auroc_of_this_config"]
entries["teacher.pool_graded_scores_existed_before"] = d["teacher"]["pool_graded_scores_existed_before"]

for gname, gate in d["gates"].items():
    entries[f"gates.{gname}.passed"] = gate["passed"]

if d.get("outcome") == "C":
    entries["outcome"] = "C"
    entries["outcome_reason"] = d["outcome_reason"]
else:
    for arm, info in d["arms"].items():
        entries[f"arms.{arm}.n_pos_realised"] = info["n_pos_realised"]
        entries[f"arms.{arm}.positive_rate"] = info["positive_rate"]
        for stratum in ("g253", "g299"):
            for system, block in info[stratum].items():
                p = f"arms.{arm}.{stratum}.{system}"
                entries[f"{p}.n"] = block["n"]
                entries[f"{p}.n_pos"] = block["n_pos"]
                entries[f"{p}.auroc"] = block["auroc"]
                entries[f"{p}.ap"] = block["ap"]
                if block.get("bootstrap"):
                    entries[f"{p}.ci95_lo"] = block["bootstrap"]["ci95"][0]
                    entries[f"{p}.ci95_hi"] = block["bootstrap"]["ci95"][1]

    ref = d["reference_c130_reproducible_label_student"]
    entries["reference_c130.auroc"] = ref["recomputed_here"]["auroc"]
    entries["reference_c130.matches_registered_4dp"] = ref["matches_registered_4dp"]

    for cname, c in d["paired_comparisons_g253"].items():
        entries[f"paired.{cname}.diff"] = c["diff"]
        entries[f"paired.{cname}.ci95_lo"] = c["ci95_lo"]
        entries[f"paired.{cname}.ci95_hi"] = c["ci95_hi"]
        entries[f"paired.{cname}.excludes_zero"] = c["excludes_zero"]

    entries["decision.P1_satisfied"] = d["decision"]["P1_satisfied"]
    entries["decision.P2_satisfied"] = d["decision"]["P2_satisfied"]
    entries["outcome"] = d["outcome"]
    entries["outcome_reason"] = d["outcome_reason"]

for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: {len(entries)} keys, outcome={d.get('outcome')}")
