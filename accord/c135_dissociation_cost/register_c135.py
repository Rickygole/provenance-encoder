"""Register C135 (prefix c135; deletes and re-adds) from its artifacts.
See C135_PREREGISTRATION.md.

    python3 accord/c135_dissociation_cost/register_c135.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c135_dissociation_cost/c135_bias_curve.json"
SRC_LAB = "results/accord/c135_dissociation_cost/c135_labeller.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c135"

d = json.loads((ROOT / SRC).read_text())
lab_path = ROOT / SRC_LAB
lab = json.loads(lab_path.read_text()) if lab_path.exists() else None

facts = json.loads(FACTS.read_text(), object_pairs_hook=collections.OrderedDict)
for k in [k for k in facts if k.split(".")[0] == PREFIX]:
    del facts[k]


def add(key, src, artifact_path, value):
    facts[f"{PREFIX}.{key}"] = collections.OrderedDict(
        [("artifact", f"{src}::{artifact_path}"), ("value", value)]
    )


add("outcome", SRC, "outcome", d["outcome"])
add("void", SRC, "kill_conditions.void", d["kill_conditions"]["void"])
add("seed", SRC, "seed", d["seed"])
add("n_usable_premise_groups", SRC, "data.n_usable_premise_groups",
    d["data"]["n_usable_premise_groups"])
add("pool_n", SRC, "pool_n", d["pool_n"])

if d["outcome"] != "VOID":
    for f in ("gap_measured_minus_true", "gap_boot_lo", "gap_boot_hi",
              "understatement", "understatement_lo", "understatement_hi", "relative_bias"):
        add(f"headline.{f}", SRC, f"headline.{f}", d["headline"][f])
    for name, block in d["pools"].items():
        add(f"{name}.realized_true_precision", SRC, f"pools.{name}.realized_true_precision",
            block["realized_true_precision"])
        add(f"{name}.n_true_positive", SRC, f"pools.{name}.n_true_positive",
            block["n_true_positive"])
        for i, b in enumerate(block["by_rate"]):
            r = f"{b['planted_rate']:.3f}".replace(".", "p")
            for f in ("n_planted", "measured_precision", "measured_precision_wilson_lo",
                      "measured_precision_wilson_hi", "true_precision",
                      "gap_measured_minus_true", "gap_boot_lo", "gap_boot_hi", "relative_bias",
                      "analytic_gap", "analytic_gap_residual",
                      "assertion_flagged_on_true_positives",
                      "assertion_flagged_equals_planted_set"):
                add(f"{name}.rate{r}.{f}", SRC, f"pools.{name}.by_rate.{i}.{f}", b[f])
        if "student" in block:
            for i, s in enumerate(block["student"]["by_rate"]):
                r = f"{s['planted_rate']:.3f}".replace(".", "p")
                for tag in ("A_displayed_evidence_labels", "B_candidate_fired_labels"):
                    for f in ("auroc", "train_positive_rate"):
                        add(f"{name}.student.{tag}.rate{r}.{f}", SRC,
                            f"pools.{name}.student.by_rate.{i}.{tag}.{f}", s[tag][f])

if lab is not None:
    for model, b in lab["models"].items():
        m = model.replace(":", "_").replace(".", "p")
        add(f"labeller.{m}.skipped", SRC_LAB, f"models.{model}.skipped", b["skipped"])
        for i, row in enumerate(b["by_rate"]):
            r = f"{row['planted_rate']:.3f}".replace(".", "p")
            for f in ("flag_rate", "accuracy_vs_true_label", "recall_on_true_positives",
                      "specificity_on_true_negatives",
                      "agreement_with_displayed_evidence_judge"):
                add(f"labeller.{m}.rate{r}.{f}", SRC_LAB, f"models.{model}.by_rate.{i}.{f}",
                    row[f])

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d['outcome']}; "
      f"{sum(1 for k in facts if k.startswith(PREFIX + '.'))} keys under prefix {PREFIX}")
