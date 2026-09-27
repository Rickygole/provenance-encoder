"""ANALYSIS 2. Where the over-calling lands: false positives by generating rule
and by pharmacologic class.

Everything the project has said about over-calling so far is a single pooled
number (a calibration ratio of roughly 21x to 24x against the 4 percent gold
base rate). Pooled ratios say how much, never where. This script breaks the
over-call down by the rule that generated the pair and by the pharmacologic
class of the drug, for the deployed weak labeller and for each replication
labeller whose per-pair calls have been persisted.

DEFINITIONS, per cell (a rule, or a pharmacologic class):
  n_pairs            gold pairs in the cell
  n_confirmed        pairs the pharmacist called a contradiction
  n_flagged          pairs the labeller called a contradiction
  over_call_ratio    n_flagged / n_confirmed, null when n_confirmed is 0
  n_false_positive   flagged and not pharmacist-confirmed
  precision          n_true_positive / n_flagged, null when nothing is flagged
  fp_share_of_cell   n_false_positive / n_pairs
Ambiguous pairs are NOT counted as confirmed. They are reported separately as
n_ambiguous_flagged so a reader can see how much of the over-call sits on pairs
the pharmacist herself could not call either way, which is the most charitable
reading available to the labeller.

PUBLISHABILITY. Rule ids, dictionary class names and counts only.

OUT: results/accord/clinical_core/false_positives_by_category.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from accord.clinical_core.clinical_core_data import (  # noqa: E402
    REPLICATION_MODELS, RULE_SHORT, attach_replication, load_gold)

OUT = REPO / "results/accord/clinical_core/false_positives_by_category.json"
REGISTERED = REPO / "results/accord/multillm/per_model_results.json"
BANNED_RE = re.compile(r"HADM|SUBJECT_ID|nursing_sentence|discharge_sentence", re.IGNORECASE)

# Minimum gold pairs for a pharmacologic class to get its own reporting row.
MIN_CELL = 10


def cell(sub, flag_col: str) -> dict:
    n = int(len(sub))
    conf = (sub["pharm_label"] == "contradiction")
    amb = (sub["pharm_label"] == "ambiguous")
    flag = (sub[flag_col] == "contradiction")
    n_conf = int(conf.sum())
    n_flag = int(flag.sum())
    tp = int((flag & conf).sum())
    fp = int((flag & ~conf).sum())
    return {
        "n_pairs": n,
        "n_confirmed": n_conf,
        "n_ambiguous": int(amb.sum()),
        "n_flagged": n_flag,
        "over_call_ratio": (None if n_conf == 0 else round(n_flag / n_conf, 3)),
        "n_true_positive": tp,
        "n_false_positive": fp,
        "n_ambiguous_flagged": int((flag & amb).sum()),
        "n_false_positive_excluding_ambiguous": int((flag & ~conf & ~amb).sum()),
        "precision": (None if n_flag == 0 else round(tp / n_flag, 4)),
        "recall": (None if n_conf == 0 else round(tp / n_conf, 4)),
        "fp_share_of_cell": round(fp / n, 4),
        "flagged_rate": round(n_flag / n, 4),
        "n_unparseable": int((sub[flag_col] == "UNPARSEABLE").sum()),
    }


def main() -> None:
    g = load_gold()
    g, attached, missing = attach_replication(g)

    labellers = {"deployed_weak_labeller_llama3_8b_v1": "weak_label"}
    for m in attached:
        labellers[f"replication_{m}"] = f"pred_{m}"

    # Gate: each labeller's pooled confusion on the gold set must match what is
    # already registered, so a per-cell breakdown cannot be built on calls that
    # are not the calls the thesis reports.
    registered = json.loads(REGISTERED.read_text())["per_model"]
    gates = {}
    for m in attached:
        c = cell(g, f"pred_{m}")
        reg = registered[m]["gold"]["confusion"]
        got = {"TP": c["n_true_positive"], "FP": c["n_false_positive"],
               "FN": c["n_confirmed"] - c["n_true_positive"],
               "TN": c["n_pairs"] - c["n_confirmed"] - c["n_false_positive"]}
        gates[m] = {"registered": reg, "recomputed": got, "agrees": got == reg}
        assert got == reg, f"{m} pooled confusion does not match the registered aggregate: {got} vs {reg}"

    # Gate the deployed weak labeller too, against the two places the register
    # already pins it: the count of weak-label contradictions on the gold set,
    # and the per-rule positive rates from the calibration join.
    facts = json.loads((REPO / "results/accord/VERIFIED_FACTS.json").read_text())
    reg_total = facts["gold.weaklabel_contradiction"]["value"]
    got_total = int((g["weak_label"] == "contradiction").sum())
    weak_gate = {
        "registered_n_weak_label_contradictions_on_gold": reg_total,
        "recomputed": got_total,
        "agrees": got_total == reg_total,
        "per_rule_positive_rate": {},
    }
    assert got_total == reg_total, f"weak labeller total {got_total} != registered {reg_total}"
    for r in sorted(g["rule"].unique()):
        k = str(int(r))
        regr = facts[f"goldstrata.rule_stratified_calibration.{k}."
                     "llama_weak_label_pos_rate_on_these_299_pairs"]["value"]
        sub = g[g["rule"] == r]
        gotr = round(float((sub["weak_label"] == "contradiction").mean()), 4)
        weak_gate["per_rule_positive_rate"][k] = {
            "registered": regr, "recomputed": gotr, "agrees": abs(gotr - regr) < 1e-4}
        assert abs(gotr - regr) < 1e-4, f"rule {k} weak positive rate {gotr} != registered {regr}"
    gates["deployed_weak_labeller_llama3_8b_v1"] = weak_gate

    out_by_rule: dict[str, dict] = {}
    out_by_class: dict[str, dict] = {}
    out_by_class_grouped: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    for name, col in labellers.items():
        pooled[name] = cell(g, col)
        out_by_rule[name] = {
            str(int(r)): {"rule_short": RULE_SHORT[int(r)], **cell(g[g["rule"] == r], col)}
            for r in sorted(g["rule"].unique())}
        out_by_class[name] = {
            c: cell(g[g["pharm_class"] == c], col)
            for c in sorted(g["pharm_class"].unique())}
        # A reporting view: classes with at least MIN_CELL gold pairs kept
        # separate, everything thinner pooled, so a table can be read without
        # 18 rows of n=1 cells.
        big = [c for c in sorted(g["pharm_class"].unique())
               if int((g["pharm_class"] == c).sum()) >= MIN_CELL]
        grouped = {c: out_by_class[name][c] for c in big}
        rest = g[~g["pharm_class"].isin(big)]
        grouped[f"other_classes_under_{MIN_CELL}_pairs"] = {
            "n_classes_pooled": int(rest["pharm_class"].nunique()),
            **cell(rest, col)}
        out_by_class_grouped[name] = grouped

    # Which cells over-call worst, ranked. A cell with no confirmed positive at
    # all has an undefined ratio, so it is ranked by the count of false
    # positives instead and listed in its own block.
    def rank(d: dict, key: str) -> dict:
        defined = [(k, v) for k, v in d.items() if v["over_call_ratio"] is not None]
        undefined = [(k, v) for k, v in d.items() if v["over_call_ratio"] is None]
        return {
            "ranked_by_over_call_ratio": [
                {key: k, "n_pairs": v["n_pairs"], "n_confirmed": v["n_confirmed"],
                 "n_flagged": v["n_flagged"], "over_call_ratio": v["over_call_ratio"],
                 "n_false_positive": v["n_false_positive"]}
                for k, v in sorted(defined, key=lambda x: -x[1]["over_call_ratio"])],
            "no_confirmed_positive_ranked_by_false_positives": [
                {key: k, "n_pairs": v["n_pairs"], "n_flagged": v["n_flagged"],
                 "n_false_positive": v["n_false_positive"]}
                for k, v in sorted(undefined, key=lambda x: -x[1]["n_false_positive"])
                if v["n_false_positive"] > 0],
        }

    art = {
        "analysis": "Analysis 2: false positives by generating rule and pharmacologic class",
        "generated_by": "accord/clinical_core/false_positives_by_category.py",
        "inputs": {
            "canonical_gold_labels": "$MIMIC_LOCAL_ONLY_DIR/derived/gold_standard_accord_results.csv",
            "candidates_table": "$MIMIC_LOCAL_ONLY_DIR/from_drive_2026-08-08/outputs/annotation_sample_300.csv",
            "deployed_weak_labeller_calls": "$MIMIC_LOCAL_ONLY_DIR/accord_supp/accord_decontamination_audit.csv (weak_label)",
            "replication_labeller_calls": {m: str(p).replace(str(Path.home()), "$HOME")
                                           for m, p in REPLICATION_MODELS.items()},
            "registered_pooled_aggregates": "results/accord/multillm/per_model_results.json",
        },
        "definitions": {
            "over_call_ratio": "n_flagged divided by n_confirmed in the same cell; null when the cell has no pharmacist-confirmed positive",
            "n_false_positive": "flagged by the labeller and not pharmacist-confirmed; ambiguous pairs count as false positives here",
            "n_false_positive_excluding_ambiguous": "flagged, and the pharmacist called it not a contradiction",
        },
        "labellers": list(labellers),
        "replication_labellers_attached": attached,
        "replication_labellers_missing": missing,
        "pooled_confusion_gate": gates,
        "n_gold_pairs": int(len(g)),
        "gold_label_distribution": {k: int(v) for k, v in g["pharm_label"].value_counts().items()},
        "pooled": pooled,
        "by_rule": out_by_rule,
        "by_pharmacologic_class": out_by_class,
        "by_pharmacologic_class_grouped_for_reporting": out_by_class_grouped,
        "min_pairs_for_own_reporting_row": MIN_CELL,
        "worst_cells": {name: {"by_rule": rank(out_by_rule[name], "rule"),
                               "by_pharmacologic_class": rank(out_by_class[name], "pharmacologic_class")}
                        for name in labellers},
        "data_agreement": (
            "Rule ids, dictionary class names and integer counts only. No clinical note text and "
            "no patient or admission identifier appears in this artifact; asserted in code before "
            "writing."),
    }

    blob = json.dumps(art, indent=1, sort_keys=False)
    m = BANNED_RE.search(blob)
    assert m is None, f"unpublishable token {m.group(0)!r} reached the artifact"
    for ch in ("\u2013", "\u2014"):
        assert ch not in blob, "en or em dash in artifact"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(blob + "\n")
    print(f"wrote {OUT}")
    print("attached:", attached, "missing:", missing)
    for name in labellers:
        p = pooled[name]
        print(f"  {name}: flagged {p['n_flagged']}/{p['n_pairs']} ratio {p['over_call_ratio']} "
              f"precision {p['precision']}")


if __name__ == "__main__":
    main()
