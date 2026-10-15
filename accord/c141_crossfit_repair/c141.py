#!/usr/bin/env python3
"""C141. Repairs C140's Q3 (leave-one-out target leak) and Q4 (fixed variance).

Runs the check pre-registered in C141_PREREGISTRATION.md. No model inference.
Aggregates only.

    python3 accord/c141_crossfit_repair/c141.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from accord.labeller_interval_closure.intervals_17 import GOLD, DERIVED  # noqa: E402

SEED = 141
REPEATS = 200
PERM = 2000
ALPHA = 0.05
IN_SAMPLE = 0.6478215767634855
OUT = ROOT / "results/accord/c141_crossfit_repair/c141.json"


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    p, q = s[y == 1], s[y == 0]
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    return float(((p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum()) / (len(p) * len(q)))


def rates(y, grp, k, fallback):
    out = np.empty(k)
    for i in range(k):
        m = grp == i
        out[i] = y[m].mean() if m.any() else fallback
    return out


def stratified_folds(y, rng, K):
    idx_pos = rng.permutation(np.where(y == 1)[0])
    idx_neg = rng.permutation(np.where(y == 0)[0])
    folds = [[] for _ in range(K)]
    for j, i in enumerate(idx_pos):
        folds[j % K].append(i)
    for j, i in enumerate(idx_neg):
        folds[j % K].append(i)
    return [np.array(sorted(f)) for f in folds]


def hm_var(A, npos, nneg):
    """Hanley and McNeil variance of an AUC."""
    Q1 = A / (2 - A)
    Q2 = 2 * A * A / (1 + A)
    return (A * (1 - A) + (npos - 1) * (Q1 - A * A) + (nneg - 1) * (Q2 - A * A)) / (npos * nneg)


def main():
    g = pd.read_csv(GOLD)
    lab = dict(zip(g["pair_id"], g["label_clean"]))
    dg = pd.read_csv(DERIVED)
    rule = dict(zip(dg["pair_id"], dg["rule"]))
    ids = [p for p in g["pair_id"].tolist() if lab[p] != "ambiguous"]
    y = np.array([int(lab[p] == "contradiction") for p in ids])
    rs = sorted({rule[p] for p in ids})
    rmap = {r: i for i, r in enumerate(rs)}
    grp = np.array([rmap[rule[p]] for p in ids])
    k = len(rs)
    n_by = np.array([(grp == i).sum() for i in range(k)])
    k_by = np.array([y[grp == i].sum() for i in range(k)])

    kills = []
    if len(ids) != 253: kills.append(f"pairs {len(ids)} not 253")
    if y.sum() != 12: kills.append(f"positives {int(y.sum())} not 12")
    if sorted(n_by.tolist()) != sorted([117, 28, 63, 45]): kills.append(f"strata {n_by.tolist()}")
    a_in = auroc(y, rates(y, grp, k, y.mean())[grp])
    if not kills and abs(a_in - IN_SAMPLE) > 1e-12:
        kills.append(f"in-sample {a_in!r} does not reproduce")
    if kills:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"verdict": "KILLED", "kill_reasons": kills}, indent=1) + "\n")
        print("KILLED:", kills); return

    rng = np.random.default_rng(SEED)
    out = {
        "what": "C141 repair of C140 Q3 (LOO target leak) and Q4 (fixed variance).",
        "preregistration": "accord/c141_crossfit_repair/C141_PREREGISTRATION.md",
        "replaces": ["c140.json::q3_cross_fitted_rival", "c140.json::q4_detectability"],
        "no_new_inference": True, "seed": SEED, "in_sample_auroc": a_in,
        "strata": {"n": n_by.tolist(), "positives": k_by.tolist()},
    }

    # ---- Q3R genuine cross-fit ----
    def crossfit_auroc(yv, K, rr):
        folds = stratified_folds(yv, rr, K)
        s = np.empty(len(yv), float)
        for f in folds:
            m = np.ones(len(yv), bool); m[f] = False
            r = rates(yv[m], grp[m], k, yv[m].mean())
            s[f] = r[grp[f]]
        return auroc(yv, s)

    q3 = {}
    for K in (5, 3):
        vals, disc = [], 0
        for _ in range(REPEATS):
            v = crossfit_auroc(y, K, rng)
            if np.isnan(v): disc += 1
            else: vals.append(v)
        vals = np.array(vals)
        # permutation null by the C140 Q2 refit rule, applied to the cross-fitted statistic
        obs = float(vals.mean())
        cnt = 0
        for _ in range(PERM):
            yp = rng.permutation(y)
            if crossfit_auroc(yp, K, rng) >= obs - 1e-12: cnt += 1
        q3[f"K{K}"] = {
            "mean_auroc": obs, "sd_across_repeats": float(vals.std()),
            "pct2_5": float(np.quantile(vals, 0.025)), "pct97_5": float(np.quantile(vals, 0.975)),
            "min": float(vals.min()), "max": float(vals.max()),
            "repeats_used": int(len(vals)), "repeats_discarded": disc,
            "in_sample_minus_crossfit": a_in - obs,
            "mean_above_half": bool(obs > 0.5),
            "pct2_5_above_half": bool(np.quantile(vals, 0.025) > 0.5),
            "pct97_5_below_half": bool(np.quantile(vals, 0.975) < 0.5),
            "refit_permutation_p": float((1 + cnt) / (1 + PERM)), "permutation_draws": PERM,
        }
    out["q3r_cross_fitted_rival"] = q3
    out["q3r_withdraws"] = {
        "c140_loo_auroc": 0.4412171507607192,
        "c140_loo_optimism": 0.2066044260027663,
        "reason": "leave-one-out scored each pair as (k_r - y_i)/(n_r - 1), a function of "
                  "its own label; within every stratum every positive is scored strictly "
                  "below every negative for any label vector.",
    }

    # ---- Q4R detectability with variance recomputed at the alternative ----
    c140 = json.loads((ROOT / "results/accord/c140_rival_adequacy/c140.json").read_text())
    cells = json.loads((ROOT / "results/accord/c139_rule_prior/c139.json").read_text())["cells"]
    kfam = c140["q4_detectability"]["n_distinct_cells"]
    npos, nneg = 12, 241
    var_rival = hm_var(a_in, npos, nneg)
    per, unreach = {}, {"alpha_0_05": 0, "bonferroni": 0}
    for name, c in cells.items():
        rec = {"observed_diff": c["auc_diff_graded_minus_prior"], "se_diff_observed": c["se_diff"]}
        for tag, al in (("alpha_0_05", ALPHA), ("bonferroni", ALPHA / kfam)):
            z = float(stats.norm.ppf(1 - al / 2))
            f = lambda A: (A - a_in) - z * np.sqrt(hm_var(A, npos, nneg) + var_rival)
            try:
                root = brentq(f, a_in + 1e-9, 1 - 1e-9)
                rec[f"min_auroc_for_significance_{tag}"] = float(root)
                rec[f"unreachable_at_auroc_1_{tag}"] = False
            except ValueError:
                rec[f"min_auroc_for_significance_{tag}"] = None
                rec[f"unreachable_at_auroc_1_{tag}"] = True
                unreach[tag] += 1
        per[name] = rec
    out["q4r_detectability"] = {
        "method": "Hanley-McNeil variance recomputed at the alternative, zero covariance "
                  "(optimistic for the stage)",
        "rival_auroc": a_in, "n_distinct_cells": kfam,
        "bonferroni_alpha_distinct": ALPHA / kfam,
        "cells_unreachable_at_auroc_1_alpha_0_05": unreach["alpha_0_05"],
        "cells_unreachable_at_auroc_1_bonferroni": unreach["bonferroni"],
        "n_cells_evaluated": len(cells),
        "min_auroc_for_significance_bonferroni_range": [
            min(v["min_auroc_for_significance_bonferroni"] for v in per.values()
                if v["min_auroc_for_significance_bonferroni"] is not None),
            max(v["min_auroc_for_significance_bonferroni"] for v in per.values()
                if v["min_auroc_for_significance_bonferroni"] is not None)],
        "per_cell": per,
    }
    out["q4r_withdraws"] = {
        "c140_cells_unreachable_bonferroni": c140["q4_detectability"]["cells_unreachable_at_auroc_1_bonferroni"],
        "reason": "C140 held each cell's observed se_diff fixed while asking what would "
                  "happen at AUROC 1.0; the DeLong variance shrinks as the AUC rises, so "
                  "the observed standard error does not apply in that counterfactual.",
    }

    m5 = q3["K5"]
    out["outcome"] = ("A" if m5["pct2_5_above_half"] else
                      "C" if m5["pct97_5_below_half"] else "B")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT, "outcome", out["outcome"])


if __name__ == "__main__":
    main()
