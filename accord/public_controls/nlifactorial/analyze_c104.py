"""C104 analysis: TPR/FPR per hardness level, implied flag rate and
over-call at a set of prevalences, log-odds AUROC per level, ceiling MCC /
attainable floor p at n=253, and the three primary contrasts H, P, S.

Reads only results/accord/public_controls/nlifactorial/{cells,raw}; never
opens anything under mimic_local_only. Writes
results/accord/public_controls/nlifactorial/analysis.json.

OPERATIONALIZATION NOTE for "ceiling MCC and one-sided floor p at n=253 with
round(253p) positives (min 1)" (pre-registration, Metrics): for each
prevalence p in P_GRID, this script rescales the cell's own TPR/FPR to a
synthetic n=253 evaluation at prevalence p -- n_pos = max(1, round(253*p)),
implied flag count n_flag = round(253 * f(p)) where f(p) = TPR*p +
FPR*(1-p) -- and reads that (n, n_pos, n_flag) triple through
accord/stats/marginal_feasibility_gate.py's best_table/attainable_floor_p/mcc
(imported, not reimplemented), exactly as that module already does for the
clinical hard-label rows. This answers "if this model+prompt's TPR/FPR were
applied at prevalence p to an evaluation the size of G_253, what ceiling
could it reach and could it have rejected at 0.05" -- the same question
marginal_feasibility_gate.py asks of the clinical rows, asked here of every
public cell instead.
"""
import collections
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.stats.marginal_feasibility_gate import (  # noqa: E402
    attainable_floor_p, best_table, mcc,
)

HERE = pathlib.Path(__file__).resolve().parent
RAW_DIR = ROOT / "results" / "accord" / "public_controls" / "nlifactorial" / "raw"
CELLS_DIR = ROOT / "results" / "accord" / "public_controls" / "nlifactorial" / "cells"
OUT_PATH = ROOT / "results" / "accord" / "public_controls" / "nlifactorial" / "analysis.json"

MODELS = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3:14b",
          "llama3:8b", "granite3.3:8b"]
PROMPTS = ["P0", "P1", "P2"]
HARDNESS_LEVELS = ["easy", "medium", "hard"]
P_GRID = [0.005, 0.01, 0.02, 0.04, 0.10, 12 / 253]
B = 10_000
SEED = 42


def _tag(model):
    return model.replace(":", "_").replace(".", "")


def load_cell(model, prompt):
    tag = _tag(model)
    meta_path = CELLS_DIR / f"{tag}__{prompt}_meta.json"
    raw_path = RAW_DIR / f"{tag}__{prompt}.jsonl"
    if not meta_path.exists() or not raw_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    recs = [json.loads(l) for l in raw_path.read_text().splitlines() if l]
    return meta, recs


def split_by_hardness(recs):
    by_h = collections.defaultdict(list)
    for r in recs:
        by_h[r["hardness"]].append(r)
    return by_h


def flag_array(recs):
    return np.array([1 if r["hard_label"] == 1 else 0 for r in recs], dtype=float)


def logodds_array(recs):
    return np.array([r["logodds"] if r["logodds"] is not None else -20.0
                       for r in recs], dtype=float)


def bootstrap_rate_dist(flags, B, seed):
    rng = np.random.default_rng(seed)
    n = len(flags)
    if n == 0:
        return np.zeros(B)
    idx = rng.integers(0, n, size=(B, n))
    return flags[idx].mean(axis=1)


def percentile_ci(dist, lo=2.5, hi=97.5):
    return float(np.percentile(dist, lo)), float(np.percentile(dist, hi))


def cell_summary(model, prompt, meta, recs):
    by_h = split_by_hardness(recs)
    pos_recs = by_h.get("positives", [])
    tpr_flags = flag_array(pos_recs)
    tpr = float(tpr_flags.mean()) if len(tpr_flags) else None
    tpr_boot = bootstrap_rate_dist(tpr_flags, B, SEED)

    levels = {}
    for level in HARDNESS_LEVELS:
        level_recs = by_h.get(level, [])
        fpr_flags = flag_array(level_recs)
        fpr = float(fpr_flags.mean()) if len(fpr_flags) else None
        fpr_boot = bootstrap_rate_dist(fpr_flags, B, SEED + 1)

        auc = None
        all_recs = pos_recs + level_recs
        if pos_recs and level_recs:
            y = np.r_[np.ones(len(pos_recs)), np.zeros(len(level_recs))]
            s = logodds_array(all_recs)
            if len(set(y.tolist())) == 2:
                auc = float(roc_auc_score(y, s))

            f_p = {}
            ceiling = {}
            for p in P_GRID:
                fp_val = tpr * p + fpr * (1 - p)
                f_p[str(p)] = {"flag_rate": fp_val,
                               "over_call_ratio": (fp_val / p) if p else None}
                n253 = 253
                n_pos = max(1, round(n253 * p))
                n_flag = int(round(n253 * fp_val))
                n_flag = min(max(n_flag, 0), n253)
                tp, tn, fpn, fn = best_table(n253, n_pos, n_flag)
                ceiling[str(p)] = {
                    "n_pos": n_pos, "n_flag": n_flag,
                    "ceiling_mcc": mcc(tp, tn, fpn, fn),
                    "attainable_floor_p_one_sided": attainable_floor_p(n253, n_pos, n_flag),
                }
        else:
            f_p, ceiling = {}, {}

        levels[level] = {
            "n": len(level_recs), "fpr": fpr,
            "fpr_ci95": percentile_ci(fpr_boot) if len(fpr_flags) else None,
            "auroc_vs_positives": auc,
            "implied_flag_rate_by_p": f_p,
            "ceiling_by_p": ceiling,
        }

    return {
        "model": model, "prompt": prompt, "valid": meta.get("valid"),
        "n_positives": len(pos_recs), "tpr": tpr,
        "tpr_ci95": percentile_ci(tpr_boot) if len(tpr_flags) else None,
        "levels": levels,
        "_tpr_flags": tpr_flags, "_by_hardness_flags": {
            level: flag_array(by_h.get(level, [])) for level in HARDNESS_LEVELS},
    }


def contrast_h(cells):
    """FPR(hard) - FPR(easy) under P0, per model; interval via percentile
    bootstrap of the difference (independent resampling of the two disjoint
    item sets)."""
    out = {}
    for model in MODELS:
        c = cells.get((model, "P0"))
        if c is None or not c["valid"]:
            continue
        hard_flags = c["_by_hardness_flags"]["hard"]
        easy_flags = c["_by_hardness_flags"]["easy"]
        if len(hard_flags) == 0 or len(easy_flags) == 0:
            continue
        hard_boot = bootstrap_rate_dist(hard_flags, B, SEED + 10)
        easy_boot = bootstrap_rate_dist(easy_flags, B, SEED + 11)
        diff = hard_boot - easy_boot
        lo, hi = percentile_ci(diff)
        out[model] = {"point": float(hard_flags.mean() - easy_flags.mean()),
                       "ci95": [lo, hi], "interval_above_zero": bool(lo > 0)}
    n_above = sum(1 for v in out.values() if v["interval_above_zero"])
    return {"per_model": out, "n_models_valid": len(out),
            "n_models_interval_above_zero": n_above,
            "outcome": "A" if n_above >= 5 else "B"}


def contrast_p(cells):
    """FPR(P1) - FPR(P0) at medium level, per model."""
    out = {}
    for model in MODELS:
        c0 = cells.get((model, "P0"))
        c1 = cells.get((model, "P1"))
        if c0 is None or c1 is None or not c0["valid"] or not c1["valid"]:
            continue
        med0 = c0["_by_hardness_flags"]["medium"]
        med1 = c1["_by_hardness_flags"]["medium"]
        if len(med0) == 0 or len(med1) == 0:
            continue
        boot0 = bootstrap_rate_dist(med0, B, SEED + 20)
        boot1 = bootstrap_rate_dist(med1, B, SEED + 21)
        diff = boot1 - boot0
        lo, hi = percentile_ci(diff)
        out[model] = {"point": float(med1.mean() - med0.mean()),
                       "ci95": [lo, hi], "interval_above_zero": bool(lo > 0)}
    n_above = sum(1 for v in out.values() if v["interval_above_zero"])
    return {"per_model": out, "n_models_valid": len(out),
            "n_models_interval_above_zero": n_above,
            "outcome": "A" if n_above >= 5 else "B"}


def contrast_s(cells):
    """Implied over-call at 12/253 under P1, medium level, qwen3:14b - qwen3:8b."""
    p = 12 / 253
    c14 = cells.get(("qwen3:14b", "P1"))
    c8 = cells.get(("qwen3:8b", "P1"))
    if c14 is None or c8 is None or not c14["valid"] or not c8["valid"]:
        return {"available": False}

    def overcall_boot(c):
        tpr_boot = bootstrap_rate_dist(c["_tpr_flags"], B, SEED + 30)
        fpr_boot = bootstrap_rate_dist(c["_by_hardness_flags"]["medium"], B, SEED + 31)
        f_p = tpr_boot * p + fpr_boot * (1 - p)
        return f_p / p

    oc14 = overcall_boot(c14)
    oc8 = overcall_boot(c8)
    diff = oc14 - oc8
    lo, hi = percentile_ci(diff)
    interval_includes_or_above_zero = bool(hi >= 0)
    entirely_negative = bool(hi < 0)
    return {
        "available": True, "point": float(np.mean(oc14) - np.mean(oc8)),
        "ci95": [lo, hi],
        "interval_includes_or_above_zero": interval_includes_or_above_zero,
        "entirely_negative": entirely_negative,
        "outcome": "A" if interval_includes_or_above_zero else "B",
    }


def main():
    cells = {}
    cell_json = {}
    for model in MODELS:
        for prompt in PROMPTS:
            meta, recs = load_cell(model, prompt)
            if meta is None:
                continue
            summary = cell_summary(model, prompt, meta, recs)
            cells[(model, prompt)] = summary
            s2 = {k: v for k, v in summary.items() if not k.startswith("_")}
            cell_json[f"{model}::{prompt}"] = s2

    out = {
        "preregistration": "accord/public_controls/nlifactorial/C104_PREREGISTRATION.md",
        "n_cells_available": len(cells),
        "cells": cell_json,
        "contrast_H_fpr_hard_minus_easy_P0": contrast_h(cells),
        "contrast_P_fpr_p1_minus_p0_medium": contrast_p(cells),
        "contrast_S_overcall_14b_minus_8b_p1_medium": contrast_s(cells),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}; {len(cells)} cells available")
    print(f"H: {out['contrast_H_fpr_hard_minus_easy_P0']['outcome']} "
          f"({out['contrast_H_fpr_hard_minus_easy_P0']['n_models_interval_above_zero']}/"
          f"{out['contrast_H_fpr_hard_minus_easy_P0']['n_models_valid']} models above 0)")
    print(f"P: {out['contrast_P_fpr_p1_minus_p0_medium']['outcome']} "
          f"({out['contrast_P_fpr_p1_minus_p0_medium']['n_models_interval_above_zero']}/"
          f"{out['contrast_P_fpr_p1_minus_p0_medium']['n_models_valid']} models above 0)")
    print(f"S: {out['contrast_S_overcall_14b_minus_8b_p1_medium']}")


if __name__ == "__main__":
    main()
