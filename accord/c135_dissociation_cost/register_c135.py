"""Register C135 (prefix c135; deletes and re-adds) from its three artifacts:
the bias curve, the labeller arm, and the labeller-versus-gold-judge arm.
See C135_PREREGISTRATION.md.

    python3 accord/c135_dissociation_cost/register_c135.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = "results/accord/c135_dissociation_cost/c135_bias_curve.json"
SRC_LAB = "results/accord/c135_dissociation_cost/c135_labeller.json"
SRC_VG = "results/accord/c135_dissociation_cost/c135_labeller_vs_gold.json"
FACTS = ROOT / "results/accord/VERIFIED_FACTS.json"
PREFIX = "c135"

d = json.loads((ROOT / SRC).read_text())
lab_path = ROOT / SRC_LAB
lab = json.loads(lab_path.read_text()) if lab_path.exists() else None
vg_path = ROOT / SRC_VG
vg = json.loads(vg_path.read_text()) if vg_path.exists() else None

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
        # A skipped model's by_rate rows are not measurements of anything: the run
        # did not happen, so whatever the artifact carries for it must not become a
        # registered fact that prose can then cite. Register the skip and stop. This
        # has bitten once already.
        if b["skipped"]:
            if b.get("skip_reason") is not None:
                add(f"labeller.{m}.skip_reason", SRC_LAB,
                    f"models.{model}.skip_reason", b["skip_reason"])
            continue
        for i, row in enumerate(b["by_rate"]):
            r = f"{row['planted_rate']:.3f}".replace(".", "p")
            for f in ("flag_rate", "accuracy_vs_true_label", "recall_on_true_positives",
                      "specificity_on_true_negatives",
                      "agreement_with_displayed_evidence_judge"):
                add(f"labeller.{m}.rate{r}.{f}", SRC_LAB, f"models.{model}.by_rate.{i}.{f}",
                    row[f])

if vg is not None:
    # The gold-judge arm: the same 400 candidate slots scored by the SNLI gold label
    # instead of by the labeller, so the labeller's measured-precision collapse can be
    # separated from the collapse an error-free judge sees. Same skip rule as above.
    for f in ("n_items_per_rate", "slot_true_precision", "headline_rate",
              "n_models_complete"):
        add(f"labeller_vs_gold.{f}", SRC_VG, f, vg[f])
    for model, b in vg["models"].items():
        m = model.replace(":", "_").replace(".", "p")
        add(f"labeller_vs_gold.{m}.skipped", SRC_VG, f"models.{model}.skipped", b["skipped"])
        if b["skipped"]:
            if b.get("skip_reason") is not None:
                add(f"labeller_vs_gold.{m}.skip_reason", SRC_VG,
                    f"models.{model}.skip_reason", b["skip_reason"])
            continue
        for i, row in enumerate(b["by_rate"]):
            r = f"{row['planted_rate']:.3f}".replace(".", "p")
            for f in ("llm_measured_precision", "llm_measured_precision_wilson_lo",
                      "llm_measured_precision_wilson_hi", "llm_gap_measured_minus_true",
                      "gold_measured_precision", "gold_measured_precision_wilson_lo",
                      "gold_measured_precision_wilson_hi", "gold_gap_measured_minus_true",
                      "llm_gap_minus_gold_gap", "accuracy_vs_true_label",
                      "recall_on_true_positives", "specificity_on_true_negatives",
                      "agreement_with_gold_judge"):
                add(f"labeller_vs_gold.{m}.rate{r}.{f}", SRC_VG,
                    f"models.{model}.by_rate.{i}.{f}", row[f])
        for f in ("planted_rate", "llm_drop_from_rate0", "llm_drop_newcombe_lo",
                  "llm_drop_newcombe_hi", "llm_drop_excludes_zero",
                  "gold_drop_from_rate0", "gold_drop_newcombe_lo",
                  "gold_drop_newcombe_hi", "llm_drop_minus_gold_drop",
                  "llm_drop_over_gold_drop"):
            add(f"labeller_vs_gold.{m}.headline.{f}", SRC_VG,
                f"models.{model}.headline.{f}", b["headline"][f])

FACTS.write_text(json.dumps(facts, indent=1) + "\n")
print(f"registered {PREFIX}: outcome={d['outcome']}; "
      f"{sum(1 for k in facts if k.startswith(PREFIX + '.'))} keys under prefix {PREFIX}")
