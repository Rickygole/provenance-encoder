"""Register the C130 outcome from its artifact.
See C130_PREREGISTRATION.md.

WRITTEN, NOT RUN as part of this task -- another agent owns
results/accord/VERIFIED_FACTS.json and is writing the register separately,
and this script has nothing to read yet: results/accord/c130_reproducible_student/c130.json
does not exist until c130_run.py's full (non-dry-run) mode has completed.
Mirrors accord/c112_decontam_defectfree/register_c112.py and
accord/c126_teacher_student/c126_run.py's register step: delete this prefix
wholesale, re-add from the committed artifact only.

    python3 accord/c130_reproducible_student/register_c130.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c130_reproducible_student/c130.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c130"

d = json.loads((ROOT / SRC).read_text())
facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]

entries = {}

if d.get("outcome") == "C":
    entries["outcome"] = "C"
    entries["abort_reason"] = d["abort_reason"]
else:
    entries["seeds"] = d["seeds"]
    for stratum in ("g299", "g253"):
        for system, block in d[stratum].items():
            prefix = f"{stratum}.{system}"
            entries[f"{prefix}.n"] = block["n"]
            entries[f"{prefix}.n_pos"] = block["n_pos"]
            entries[f"{prefix}.auroc"] = block["auroc"]
            entries[f"{prefix}.ap"] = block["ap"]
            if block.get("bootstrap"):
                entries[f"{prefix}.ci95_lo"] = block["bootstrap"]["ci95"][0]
                entries[f"{prefix}.ci95_hi"] = block["bootstrap"]["ci95"][1]

    entries["teacher_g299.auroc_binary_score"] = d["teacher_g299"]["auroc_binary_score"]
    entries["teacher_g299.recall"] = d["teacher_g299"]["recall"]
    entries["teacher_g299.specificity"] = d["teacher_g299"]["specificity"]

    if d["teacher_g253"].get("available", True) is not False:
        entries["teacher_g253.n"] = d["teacher_g253"]["n"]
        entries["teacher_g253.n_pos"] = d["teacher_g253"]["n_pos"]
        entries["teacher_g253.teacher_auroc"] = d["teacher_g253"]["teacher_auroc"]
        entries["teacher_g253.student_minus_teacher.diff"] = d["teacher_g253"]["student_minus_teacher"]["diff"]
        entries["teacher_g253.student_minus_teacher.ci95_lo"] = d["teacher_g253"]["student_minus_teacher"]["ci95_lo"]
        entries["teacher_g253.student_minus_teacher.ci95_hi"] = d["teacher_g253"]["student_minus_teacher"]["ci95_hi"]
    else:
        entries["teacher_g253.available"] = False
        entries["teacher_g253.reason"] = d["teacher_g253"]["reason"]

    entries["decision.stratum"] = d["decision"]["stratum"]
    entries["decision.power_floor"] = d["decision"]["power_floor"]
    entries["decision.outcome"] = d["decision"]["outcome"]
    entries["decision.reason"] = d["decision"]["reason"]

for field, value in entries.items():
    facts[f"{PREFIX}.{field}"] = collections.OrderedDict(
        [("artifact", f"{SRC}::{field}"), ("value", value)]
    )

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d.get('outcome') or d.get('decision', {}).get('outcome')}")
