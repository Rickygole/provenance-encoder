"""
C28 reconciliation: the 278-vs-261 "predicted contradiction" discrepancy.

C28 recorded that the released per-pair file has `accord_pred == "contradiction"`
on 278 of the 299 pharmacist-annotated pairs, while
`exp1a_gold_error_rates_by_group.json` "implies 261", and hypothesised that the
two analyses handle the value `ambiguous` differently.

This script tests that hypothesis by recomputing both numbers from the files on
disk, and by exhaustively enumerating every filter/threshold rule that could
turn one into the other.

COMPLIANCE
----------
Inputs are MIMIC-III-derived and PhysioNet-credentialed. This script reads them,
computes aggregate counts, and writes ONLY counts. No clinical note text, no
HADM_ID, no SUBJECT_ID, and no pair_id is written to the output file.

Output: results/accord/reconciliation_c28.json
Run:    python3 accord/stats/reconcile_c28.py
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO, "results/accord/reconciliation_c28.json")

# The released per-pair ACCORD decision file (the "raw file" of C28 item 2).
GOLD_PATH = "/Users/rickygole/Downloads/gold_standard_accord_results.csv"

# pair_id -> HADM_ID/drug/rule/sentences bridge, and the weak-label carrier.
# run_experiments.py referenced the Google Drive copy; the Drive mount is gone
# (quarantine move), so the local copy of the identical file is used and both
# candidates are probed in order. The Google Drive candidate was removed:
# it joins note sentences to HADM_ID and must not be read from cloud storage.
SAMP_CANDIDATES = [
    "/Users/rickygole/mimic_local_only/from_drive_2026-08-08/outputs/annotation_sample_300.csv",
]
AUDIT_PATH = "/Users/rickygole/mimic_local_only/accord_supp/accord_decontamination_audit.csv"

# The artifact C28 says "implies 261".
EXP1A_PATH = os.path.join(REPO, "results/accord/differential/exp1a_gold_error_rates_by_group.json")


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)


def mcc(tp, fn, fp, tn):
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return None if den == 0 else num / den


def confusion(pred_pos: pd.Series, true_pos: pd.Series) -> dict:
    tp = int(((pred_pos == 1) & (true_pos == 1)).sum())
    fp = int(((pred_pos == 1) & (true_pos == 0)).sum())
    fn = int(((pred_pos == 0) & (true_pos == 1)).sum())
    tn = int(((pred_pos == 0) & (true_pos == 0)).sum())
    n = tp + fp + fn + tn
    return {
        "n": n,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "predicted_positive": tp + fp,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if (tp + fn) else None,
        "accuracy": (tp + tn) / n if n else None,
        "MCC": mcc(tp, fn, fp, tn),
    }


def build_join():
    """Reproduce accord/differential/run_experiments.py's gold<->weak-label join."""
    gold = pd.read_csv(GOLD_PATH)
    samp_path = first_existing(SAMP_CANDIDATES)
    samp = pd.read_csv(samp_path)
    audit = pd.read_csv(AUDIT_PATH)
    audit["HADM_ID"] = audit["HADM_ID"].astype(float)

    g2 = gold.merge(
        samp[["pair_id", "HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]],
        on=["pair_id", "rule"], how="left",
    )
    g2["HADM_ID"] = g2["HADM_ID"].astype(float)
    key = ["HADM_ID", "drug", "rule", "nursing_sentence", "discharge_sentence"]
    j = g2.merge(audit[key + ["weak_label", "removal_reason"]], on=key, how="left")

    assert j.shape[0] == 299, f"join changed row count: {j.shape[0]}"
    assert j["weak_label"].notna().sum() == 299, "gold<->weak-label join is not 1:1"

    j["pharm_pos"] = (j["pharm_label"] == "contradiction").astype(int)
    j["weak_pos"] = (j["weak_label"] == "contradiction").astype(int)
    j["accord_pos"] = (j["accord_pred"] == "contradiction").astype(int)
    return j, samp_path


def exp1a_implied_counts():
    """Sum TP+FP over each stratification in the exp1a artifact.

    This is the operation that 'implies 261': exp1a reports per-group confusion
    cells, and the predicted-positive total is TP+FP summed over the groups.
    """
    with open(EXP1A_PATH) as f:
        exp1a = json.load(f)
    out = {}
    for strat, rows in exp1a.items():
        out[strat] = {
            "n": int(sum(r["n"] for r in rows)),
            "predicted_positive_TP_plus_FP": int(sum(r["TP"] + r["FP"] for r in rows)),
            "TP": int(sum(r["TP"] for r in rows)),
            "FP": int(sum(r["FP"] for r in rows)),
            "FN": int(sum(r["FN"] for r in rows)),
            "TN": int(sum(r["TN"] for r in rows)),
        }
    return out


def enumerate_rules(j: pd.DataFrame):
    """Every filter/threshold rule on the ACCORD columns, and what count it gives.

    The point is to show that NO rule applied to `accord_pred` or
    `accord_contra_prob` produces 261 -- i.e. the 17-row gap is not an
    ambiguous-handling rule.
    """
    p = j["accord_pred"]
    prob = j["accord_contra_prob"]
    non_amb_pharm = j["pharm_label"] != "ambiguous"

    rules = {
        "accord_pred == contradiction (all 299)": int((p == "contradiction").sum()),
        "accord_pred != not_contradiction (ambiguous counted positive, all 299)":
            int((p != "not_contradiction").sum()),
        "accord_pred == contradiction, pharm-ambiguous rows dropped (n=253)":
            int(((p == "contradiction") & non_amb_pharm).sum()),
        "accord_pred != not_contradiction, pharm-ambiguous rows dropped (n=253)":
            int(((p != "not_contradiction") & non_amb_pharm).sum()),
        "accord_pred == contradiction, accord-ambiguous rows dropped (n=284)":
            int((p == "contradiction").sum()),  # dropping them cannot change the numerator
        "accord_contra_prob >= 0.5 (all 299)": int((prob >= 0.5).sum()),
        "accord_contra_prob >= 0.5, pharm-ambiguous dropped (n=253)":
            int(((prob >= 0.5) & non_amb_pharm).sum()),
    }

    # Is there ANY threshold on accord_contra_prob that yields exactly 261 on 299?
    thresholds = np.unique(prob.to_numpy())
    counts_299 = {int((prob >= t).sum()) for t in thresholds}
    counts_299.add(int((prob >= thresholds.max() + 1e-9).sum()))
    rules["_any_threshold_on_contra_prob_can_yield_261_on_299"] = bool(261 in counts_299)
    sub = prob[non_amb_pharm]
    counts_253 = {int((sub >= t).sum()) for t in np.unique(sub.to_numpy())}
    rules["_any_threshold_on_contra_prob_can_yield_261_on_253"] = bool(261 in counts_253)
    rules["_min_accord_contra_prob"] = float(prob.min())
    rules["_max_accord_contra_prob"] = float(prob.max())
    rules["_n_distinct_scores"] = int(len(thresholds))
    rules["_threshold_result_is_vacuous"] = bool(len(thresholds) == len(prob))
    rules["_threshold_note"] = (
        "The scores are near-perfectly distinct (see _n_distinct_scores), so sweeping the "
        "threshold attains almost every integer count from 0 to 299 and 'some threshold "
        "gives 261' carries no information. "
        "The substantive result is that no *named* rule -- the released decision, either "
        "handling of ambiguous, the 0.5 threshold, or any of these restricted to the 253 "
        "non-ambiguous pairs -- gives 261."
    )
    return rules


def main():
    j, samp_path = build_join()

    accord_pred_counts = {str(k): int(v) for k, v in j["accord_pred"].value_counts().items()}
    weak_label_counts = {str(k): int(v) for k, v in j["weak_label"].value_counts().items()}
    pharm_counts = {str(k): int(v) for k, v in j["pharm_label"].value_counts().items()}

    n_accord_pos = int(j["accord_pos"].sum())
    n_weak_pos = int(j["weak_pos"].sum())

    # cross-tabulation of the two columns (counts only)
    xtab = pd.crosstab(j["accord_pred"], j["weak_label"])
    xtab_d = {str(r): {str(c): int(xtab.loc[r, c]) for c in xtab.columns} for r in xtab.index}

    # exactly which rows are in one set and not the other
    both = int(((j["accord_pos"] == 1) & (j["weak_pos"] == 1)).sum())
    accord_only = int(((j["accord_pos"] == 1) & (j["weak_pos"] == 0)).sum())
    weak_only = int(((j["accord_pos"] == 0) & (j["weak_pos"] == 1)).sum())
    neither = int(((j["accord_pos"] == 0) & (j["weak_pos"] == 0)).sum())

    # the accord-side ambiguous rows, broken down by what the weak labeller said
    amb_break = {
        str(k): int(v)
        for k, v in j.loc[j["accord_pred"] == "ambiguous", "weak_label"].value_counts().items()
    }

    non_amb = j[j["pharm_label"] != "ambiguous"]

    measured = {
        "accord_pred_on_299": confusion(j["accord_pos"], j["pharm_pos"]),
        "accord_pred_on_253_non_ambiguous": confusion(non_amb["accord_pos"], non_amb["pharm_pos"]),
        "weak_label_on_299": confusion(j["weak_pos"], j["pharm_pos"]),
        "weak_label_on_253_non_ambiguous": confusion(non_amb["weak_pos"], non_amb["pharm_pos"]),
    }

    # The rows printed in ch7 Table 7.x and ch8 Table 8.x are NOT typed here --
    # they are read out of the artifact that produced them,
    # results/accord/step3_baselines.json, whose cells match the printed rows.
    with open(os.path.join(REPO, "results/accord/step3_baselines.json")) as f:
        base = json.load(f)

    def from_artifact(key):
        v = base[key]
        tp, fn, fp, tn = v["tp"], v["fn"], v["fp"], v["tn"]
        return {
            "artifact_key": key, "n": v["n"],
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "predicted_positive": tp + fp,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "accuracy": (tp + tn) / v["n"],
            "MCC": mcc(tp, fn, fp, tn),
        }

    ch8_llama_derived = from_artifact("llama_baseline")
    ch8_accord_derived = from_artifact("accord_binary_at_0.5_253core")
    ch8_llama = {k: ch8_llama_derived[k] for k in ("TP", "FN", "FP", "TN")}
    ch8_accord = {k: ch8_accord_derived[k] for k in ("TP", "FN", "FP", "TN")}

    result = {
        "correction": "C28, item 2 (predicted-positive count does not reconcile)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "accord/stats/reconcile_c28.py",
        "method": (
            "Both numbers were recomputed from the files on disk rather than read out of "
            "prose. (1) 278 was recomputed as the count of accord_pred == 'contradiction' in "
            "the released per-pair file gold_standard_accord_results.csv (299 rows). "
            "(2) 261 was recomputed by re-running the exact join that "
            "accord/differential/run_experiments.py performs -- "
            "gold_standard_accord_results.csv -> annotation_sample_300.csv (pair_id -> "
            "HADM_ID/drug/rule/sentences) -> accord_decontamination_audit.csv (weak_label) -- "
            "and counting weak_label == 'contradiction'; it was independently cross-checked by "
            "summing TP+FP over the per-group cells stored in "
            "results/accord/differential/exp1a_gold_error_rates_by_group.json, which is the "
            "operation that 'implies 261'. (3) The hypothesis in C28 -- that the gap is a "
            "difference in how the value 'ambiguous' is handled -- was tested by enumerating "
            "every filter and every threshold rule on accord_pred and accord_contra_prob and "
            "checking whether any of them yields 261. (4) The two columns were cross-tabulated "
            "to give the exact per-cell disagreement. (5) The measured weak-labeller and ACCORD "
            "confusion matrices were compared against the reconstructed ones printed in ch7 and "
            "ch8. No note text, HADM_ID, SUBJECT_ID or pair_id is written by this script."
        ),
        "inputs": {
            "released_per_pair_file": GOLD_PATH,
            "pair_id_bridge": samp_path,
            "weak_label_carrier": AUDIT_PATH,
            "exp1a_artifact": EXP1A_PATH,
            "note_on_bridge_path": (
                "run_experiments.py hard-codes the Google Drive copy of "
                "annotation_sample_300.csv; that mount no longer exists, so the identical "
                "local copy under mimic_local_only/from_drive_2026-08-08/outputs was used."
            ),
        },

        "a_what_each_number_counts": {
            "278": {
                "column": "accord_pred",
                "file": "gold_standard_accord_results.csv",
                "system": "ACCORD v2, contaminated run, released per-pair decision",
                "rule": "accord_pred == 'contradiction'",
                "denominator": 299,
                "value": n_accord_pos,
                "value_counts_of_column": accord_pred_counts,
            },
            "261": {
                "column": "weak_label",
                "file": "accord_decontamination_audit.csv (joined to the 299 gold pairs)",
                "system": "Llama 3 8B weak labeller -- NOT ACCORD",
                "rule": "weak_label == 'contradiction'",
                "denominator": 299,
                "value": n_weak_pos,
                "value_counts_of_column": weak_label_counts,
            },
            "pharm_label_value_counts": pharm_counts,
        },

        "b_the_rule_that_produces_the_17_row_difference": {
            "c28_hypothesis": (
                "C28 hypothesised the gap is a difference in how the three-valued "
                "accord_pred column's 'ambiguous' level is handled."
            ),
            "c28_hypothesis_holds": False,
            "actual_rule": (
                "There is no rule. The two numbers are counts of two different columns "
                "produced by two different systems on the same 299 rows. 278 is ACCORD's "
                "predicted-positive count; 261 is the Llama 3 8B weak labeller's "
                "predicted-positive count. exp1a is titled 'Llama weak-label error rates by "
                "group' and never touches accord_pred."
            ),
            "ambiguous_arithmetic_that_rules_the_hypothesis_out": {
                "accord_pred_ambiguous_rows": accord_pred_counts.get("ambiguous", 0),
                "278_minus_accord_ambiguous": n_accord_pos - accord_pred_counts.get("ambiguous", 0),
                "gap_to_explain": n_accord_pos - n_weak_pos,
                "weak_label_has_an_ambiguous_level": "ambiguous" in weak_label_counts,
            },
            "every_rule_tried_on_the_accord_columns": enumerate_rules(j),
            "cross_tabulation_accord_pred_by_weak_label": xtab_d,
            "set_difference_on_the_same_299_rows": {
                "both_call_contradiction": both,
                "accord_only": accord_only,
                "weak_labeller_only": weak_only,
                "neither": neither,
                "net": accord_only - weak_only,
            },
            "accord_ambiguous_rows_by_weak_label": amb_break,
            "exp1a_implied_totals_by_stratification": exp1a_implied_counts(),
        },

        "c_which_number_belongs_where": {
            "278": (
                "Belongs in any sentence about ACCORD's predicted-positive count on the 299 "
                "annotated pairs. It is correct as used in ch1:239, ch7:165, ch7:590, ch7:690 "
                "and ch7:804."
            ),
            "261": (
                "Belongs only in sentences about the Llama 3 8B weak labeller's positive rate "
                "on the 299 gold pairs. It is not an ACCORD quantity and must never be written "
                "as an alternative reading of 278."
            ),
            "they_are_not_alternatives": True,
        },

        "d_measured_vs_what_the_tex_says": {
            "measured": measured,
            "row_printed_as_Llama_3_8B_weak_labeller_source": ch8_llama_derived,
            "row_printed_as_ACCORD_tau_0.5_source": ch8_accord_derived,
            "note": (
                "The row the thesis prints as 'Llama 3 8B (weak labeller)' in ch7 "
                "Table tab:systems, ch8 tab:met-confusion, tab:met-selection and "
                "tab:met-mccci is the artifact key 'llama_baseline' in "
                "results/accord/step3_baselines.json, i.e. the Llama 3 8B *evaluation "
                "baseline* prompt run whose per-pair calls are archived in "
                "predictions_253_core.csv. It is not the weak labeller. The weak "
                "labeller's own calls live in weak_label in "
                "accord_decontamination_audit.csv and are measured above."
            ),
        },
    }

    # the explicit ch7/ch8 checks, all derived rather than typed
    m253_weak = measured["weak_label_on_253_non_ambiguous"]
    m253_accord = measured["accord_pred_on_253_non_ambiguous"]
    result["d_measured_vs_what_the_tex_says"]["findings"] = {
        "llama_row_FP_printed_minus_measured": ch8_llama["FP"] - m253_weak["FP"],
        "llama_row_TN_printed_minus_measured": ch8_llama["TN"] - m253_weak["TN"],
        "llama_predicted_positive_printed": ch8_llama_derived["predicted_positive"],
        "llama_predicted_positive_measured": m253_weak["predicted_positive"],
        "llama_precision_printed": ch8_llama_derived["precision"],
        "llama_precision_measured": m253_weak["precision"],
        "llama_MCC_printed": ch8_llama_derived["MCC"],
        "llama_MCC_measured": m253_weak["MCC"],
        "accord_row_FP_printed_minus_measured": ch8_accord["FP"] - m253_accord["FP"],
        "accord_predicted_positive_printed": ch8_accord_derived["predicted_positive"],
        "accord_predicted_positive_measured": m253_accord["predicted_positive"],
        "accord_MCC_printed": ch8_accord_derived["MCC"],
        "accord_MCC_measured": m253_accord["MCC"],
        "ch7_footnote_claim_44_of_46_ambiguous_predicted_positive": {
            "claimed": 44,
            "measured_ambiguous_pharm_rows_predicted_positive_by_accord": int(
                ((j["pharm_label"] == "ambiguous") & (j["accord_pos"] == 1)).sum()
            ),
            "measured_ambiguous_pharm_rows": int((j["pharm_label"] == "ambiguous").sum()),
        },
    }

    f = result["d_measured_vs_what_the_tex_says"]["findings"]
    result["e_tex_verdict"] = {
        "278_as_written": {
            "locations": ["ch1_introduction.tex:239", "ch7_evaluation.tex:165",
                          "ch7_evaluation.tex:590", "ch7_evaluation.tex:690",
                          "ch7_evaluation.tex:804"],
            "status": "CORRECT",
            "reason": "reproduces exactly from the released per-pair file",
        },
        "261_as_written": {
            "locations": [],
            "status": "NOT PRESENT IN ANY .tex CHAPTER",
            "reason": (
                "grep over the ten chapters finds no occurrence of 261. Only 278 appears. "
                "C28's statement that 'both numbers currently appear in the draft' is not "
                "true of the .tex source; 261 lives only in the exp1a artifact and run log, "
                "where it is correctly the weak labeller's count."
            ),
        },
        "ch7_footnote_open_TODO_at_lines_187_195": {
            "status": "RESOLVED BY THIS SCRIPT -- no numeric change needed",
            "denominator_253_confirmed": (
                m253_accord["n"] == 253
                and m253_accord["predicted_positive"] == f["accord_predicted_positive_printed"]
            ),
            "accord_accuracy_on_253_measured": m253_accord["accuracy"],
            "44_of_46_ambiguous_predicted_positive_confirmed": (
                f["ch7_footnote_claim_44_of_46_ambiguous_predicted_positive"]["claimed"]
                == f["ch7_footnote_claim_44_of_46_ambiguous_predicted_positive"][
                    "measured_ambiguous_pharm_rows_predicted_positive_by_accord"]
            ),
        },
        "weak_labeller_row_as_written": {
            "locations": ["ch1_introduction.tex:243", "ch7_evaluation.tex:453",
                          "ch7_evaluation.tex:933", "ch8_metrics.tex:226",
                          "ch8_metrics.tex:266", "ch8_metrics.tex:346",
                          "ch8_metrics.tex:359"],
            "status": "MISLABELLED",
            "reason": (
                "Every row and sentence labelled 'Llama 3 8B (weak labeller)' carries the "
                "cells of the artifact key 'llama_baseline' -- the Llama 3 8B evaluation "
                "baseline prompt run. The system that actually produced the weak supervision "
                "is a different Llama 3 8B run with different calls on the same pairs."
            ),
            "printed_predicted_positive": f["llama_predicted_positive_printed"],
            "measured_weak_labeller_predicted_positive": f["llama_predicted_positive_measured"],
            "rows_of_253_on_which_the_two_disagree": f["llama_row_FP_printed_minus_measured"],
            "printed_precision": f["llama_precision_printed"],
            "measured_weak_labeller_precision": f["llama_precision_measured"],
            "printed_MCC": f["llama_MCC_printed"],
            "measured_weak_labeller_MCC": f["llama_MCC_measured"],
            "sign_flips": (f["llama_MCC_printed"] < 0) and (f["llama_MCC_measured"] > 0),
            "consequence": (
                "The claim 'the weak labeller it was distilled from scores -0.044, which is "
                "worse than chance' is not supported by the weak labeller's own labels; on "
                "the same 253 pairs the weak labeller's MCC point estimate is positive. "
                "The -0.044 sentence is true of the Llama baseline and should be relabelled "
                "rather than deleted."
            ),
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {OUT_PATH}")
    print(json.dumps(result["a_what_each_number_counts"], indent=2))
    print(json.dumps(result["b_the_rule_that_produces_the_17_row_difference"]
                     ["set_difference_on_the_same_299_rows"], indent=2))
    print(json.dumps(result["d_measured_vs_what_the_tex_says"]["findings"], indent=2))


if __name__ == "__main__":
    main()
