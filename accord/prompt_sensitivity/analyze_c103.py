"""C103 analysis: McNemar (asserting vs neutral), AUROC/Mann-Whitney on
G_253, marginal feasibility floors, and the decision rule.

Opens the local numeric-only per-pair cache under
~/mimic_local_only/accord_prompt_sensitivity_cache/ (pair_id, flag label,
flag log-odds, degenerate, parse-failure -- never sentence text) and the
gold CSV (only to recover label_clean per pair_id, for the G_253 basis and
the McNemar/AUROC labels). Writes AGGREGATE ONLY to
results/accord/prompt_sensitivity/analysis.json.

Per the task's compliance rule 2, every process that opens
~/mimic_local_only must send its full stdout/stderr to a log file under
~/mimic_local_only/logs/; run this script with that redirection, e.g.:

    python3 accord/prompt_sensitivity/analyze_c103.py \\
        > ~/mimic_local_only/logs/c103_analyze_<ts>.log 2>&1

This script does not redirect itself, so it can be run interactively during
development; the invoking shell is responsible for the redirection at
run time, exactly as documented in C103_PREREGISTRATION.md's compliance
notes.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from accord.stats.marginal_feasibility_gate import (  # noqa: E402
    attainable_floor_p, best_table, mcc,
)

GOLD = pathlib.Path.home() / "mimic_local_only/from_drive_2026-08-08/outputs/pharmacist_labeled_clean.csv"
CACHE_DIR = pathlib.Path.home() / "mimic_local_only/accord_prompt_sensitivity_cache"
OVERCALL_PATH = ROOT / "results" / "accord" / "stats" / "overcall_decomposition.json"
MDE_PATH = ROOT / "results" / "accord" / "mde_recomputed.json"
OUT_PATH = ROOT / "results" / "accord" / "prompt_sensitivity" / "analysis.json"

MODELS = ["llama3:8b", "granite3.3:8b", "qwen3:8b"]
MODEL_TAGS = {"llama3:8b": "_llama3_8b_nothink",
              "granite3.3:8b": "_granite33_8b_nothink",
              "qwen3:8b": "_qwen3_8b_nothink"}
PROMPTS = ["P_asserting", "P_neutral", "P_reversed"]
GOLD_BASE_RATE_299 = 12 / 299


def load_cache(tag, prompt_name):
    path = CACHE_DIR / f"{tag}__{prompt_name}.jsonl"
    recs = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {r["pair_id"]: r for r in recs}


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    pair_ids_253 = [pid for pid in pair_ids_299 if label_by_pair[pid] != "ambiguous"]
    y_253 = {pid: int(label_by_pair[pid] == "contradiction") for pid in pair_ids_253}

    overcall = json.loads(OVERCALL_PATH.read_text())
    ceiling_ratio = overcall["max_over_call_ratio"]
    mde = json.loads(MDE_PATH.read_text())
    mde_auc = mde["strata"]["gold_ambiguous_excluded"]["minimum_detectable_auc"]

    cells = {}
    per_model = {}
    invalid_cells = []
    for model in MODELS:
        tag = MODEL_TAGS[model]
        by_prompt = {}
        for prompt_name in PROMPTS:
            recs = load_cache(tag, prompt_name)
            n = len(recs)
            n_parse_fail = sum(1 for r in recs.values() if r["parse_failure"])
            n_degenerate = sum(1 for r in recs.values() if r["degenerate"])
            n_thinking = sum(1 for r in recs.values()
                              if r.get("thinking_nonempty"))
            n_truncated = sum(1 for r in recs.values() if r.get("truncated"))
            n_flag = sum(1 for r in recs.values() if r.get("flag_label") == 1)
            n_scored = sum(1 for r in recs.values() if r.get("flag_label") is not None)
            parse_fail_rate = n_parse_fail / n if n else None
            valid = (n == 299 and parse_fail_rate is not None
                      and parse_fail_rate <= 0.05 and n_thinking == 0
                      and n_truncated == 0)
            if not valid:
                invalid_cells.append(f"{model}::{prompt_name}")

            # G_253 basis: AUROC + one-sided Mann-Whitney vs MDE.
            scores_253 = [recs[pid]["flag_logodds"] for pid in pair_ids_253
                          if recs.get(pid) is not None]
            labels_253 = [y_253[pid] for pid in pair_ids_253 if recs.get(pid) is not None]
            auc = mw_p = None
            if len(set(labels_253)) == 2:
                auc = float(roc_auc_score(labels_253, scores_253))
                pos = [s for s, y in zip(scores_253, labels_253) if y == 1]
                neg = [s for s, y in zip(scores_253, labels_253) if y == 0]
                _, mw_p = mannwhitneyu(pos, neg, alternative="greater")
                mw_p = float(mw_p)

            n_flag_299 = n_flag
            tp, tn, fp, fn = best_table(n, sum(1 for pid in pair_ids_299
                                                if label_by_pair[pid] == "contradiction"),
                                         n_flag_299)
            floor_p = attainable_floor_p(n, sum(1 for pid in pair_ids_299
                                                 if label_by_pair[pid] == "contradiction"),
                                          n_flag_299)

            by_prompt[prompt_name] = {
                "n": n, "n_parse_failures": n_parse_fail,
                "parse_failure_rate": round(parse_fail_rate, 4) if parse_fail_rate is not None else None,
                "n_degenerate": n_degenerate,
                "n_thinking_nonempty": n_thinking, "n_truncated": n_truncated,
                "n_flagged_299": n_flag_299, "n_scored_299": n_scored,
                "positive_rate_299": round(n_flag_299 / n, 4) if n else None,
                "auroc_253": auc, "mannwhitney_p_one_sided_253": mw_p,
                "marginal_feasibility_floor_p": floor_p,
                "valid": bool(valid),
            }
        per_model[model] = by_prompt

    # Primary decision rule: for each model, compare P_asserting vs P_neutral.
    model_verdicts = {}
    for model in MODELS:
        tag = MODEL_TAGS[model]
        rec_a = load_cache(tag, "P_asserting")
        rec_n = load_cache(tag, "P_neutral")
        common = [pid for pid in pair_ids_299
                  if rec_a.get(pid) and rec_n.get(pid)
                  and rec_a[pid]["flag_label"] is not None
                  and rec_n[pid]["flag_label"] is not None]
        n10 = sum(1 for pid in common
                  if rec_a[pid]["flag_label"] == 1 and rec_n[pid]["flag_label"] == 0)
        n01 = sum(1 for pid in common
                  if rec_a[pid]["flag_label"] == 0 and rec_n[pid]["flag_label"] == 1)
        n_disc = n10 + n01
        mcnemar_p = float(binomtest(min(n10, n01), n_disc, 0.5,
                                     alternative="two-sided").pvalue) if n_disc else 1.0
        rate_a = per_model[model]["P_asserting"]["positive_rate_299"]
        rate_n = per_model[model]["P_neutral"]["positive_rate_299"]
        margin_direction_ok = bool(rate_n is not None and rate_a is not None
                                     and rate_n < rate_a)
        neutral_over_call_ratio = (rate_n / GOLD_BASE_RATE_299) if rate_n is not None else None
        ratio_ok = bool(neutral_over_call_ratio is not None
                          and neutral_over_call_ratio <= ceiling_ratio)
        satisfies_a = bool(margin_direction_ok and mcnemar_p < 0.05 and ratio_ok)
        model_verdicts[model] = {
            "n_common_scored": len(common), "n10_assert_pos_neutral_neg": n10,
            "n01_assert_neg_neutral_pos": n01,
            "mcnemar_p_two_sided_exact": mcnemar_p,
            "positive_rate_asserting": rate_a, "positive_rate_neutral": rate_n,
            "margin_direction_ok": margin_direction_ok,
            "neutral_over_call_ratio": neutral_over_call_ratio,
            "ceiling_ratio": ceiling_ratio, "ratio_at_or_below_ceiling": ratio_ok,
            "satisfies_outcome_a_criteria": satisfies_a,
        }

    valid_models = [m for m in MODELS
                     if all(per_model[m][p]["valid"] for p in PROMPTS)]
    n_satisfying = sum(1 for m in valid_models if model_verdicts[m]["satisfies_outcome_a_criteria"])

    if len(valid_models) < 2:
        outcome = "C"
        outcome_reason = (f"only {len(valid_models)}/3 models valid across all "
                           f"three prompt cells; invalid cells: {invalid_cells}")
    elif n_satisfying >= 2:
        outcome = "A"
        outcome_reason = (f"{n_satisfying} of {len(valid_models)} valid models "
                           f"satisfy the margin + McNemar + ceiling-ratio criteria")
    else:
        outcome = "B"
        outcome_reason = (f"only {n_satisfying} of {len(valid_models)} valid models "
                           f"satisfy the criteria (need >= 2 of 3)")

    out = {
        "preregistration": "accord/prompt_sensitivity/C103_PREREGISTRATION.md",
        "gold_base_rate_299": GOLD_BASE_RATE_299,
        "ceiling_ratio_source": "results/accord/stats/overcall_decomposition.json::max_over_call_ratio",
        "ceiling_ratio": ceiling_ratio,
        "mde_auroc_253": mde_auc,
        "per_model_prompt": per_model,
        "model_verdicts": model_verdicts,
        "invalid_cells": invalid_cells,
        "valid_models": valid_models,
        "n_models_satisfying_outcome_a": n_satisfying,
        "outcome": outcome,
        "outcome_reason": outcome_reason,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}; outcome={outcome} ({outcome_reason})")


if __name__ == "__main__":
    main()
