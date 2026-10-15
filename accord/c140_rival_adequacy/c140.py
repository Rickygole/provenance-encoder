#!/usr/bin/env python3
"""C140. Rival adequacy: correct null, cross-fitting, and detectability.

Runs the check pre-registered in C140_PREREGISTRATION.md. No model inference:
every score is read from a committed per-pair cache or derived from the stored
rule column and the pharmacist labels. Aggregates only.

    python3 accord/c140_rival_adequacy/c140.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.labeller_interval_closure.intervals_17 import GOLD, DERIVED  # noqa: E402

SEED = 140
B_PERM = 20000
B_BOOT = 10000
ALPHA = 0.05
IN_SAMPLE_AUROC_EXPECTED = 0.6478215767634855
C139 = ROOT / "results/accord/c139_rival_adequacy_input.json"
C139_REAL = ROOT / "results/accord/c139_rule_prior/c139.json"
OUT = ROOT / "results/accord/c140_rival_adequacy/c140.json"


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def fit_rates(y, grp, k):
    return np.array([y[grp == i].mean() if (grp == i).any() else 0.0 for i in range(k)])


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    dg = pd.read_csv(DERIVED)
    rule_by_pair = dict(zip(dg["pair_id"], dg["rule"]))

    ids253 = [p for p in g["pair_id"].tolist() if label_by_pair[p] != "ambiguous"]
    y = np.array([int(label_by_pair[p] == "contradiction") for p in ids253])
    rules_sorted = sorted({rule_by_pair[p] for p in ids253})
    rmap = {r: i for i, r in enumerate(rules_sorted)}
    grp = np.array([rmap[rule_by_pair[p]] for p in ids253])
    k = len(rules_sorted)
    n_by = np.array([(grp == i).sum() for i in range(k)])
    k_by = np.array([y[grp == i].sum() for i in range(k)])

    kills = []
    if len(ids253) != 253:
        kills.append(f"pair count {len(ids253)} not 253")
    if y.sum() != 12:
        kills.append(f"positive count {int(y.sum())} not 12")
    if sorted(n_by.tolist()) != sorted([117, 28, 63, 45]):
        kills.append(f"stratum sizes {n_by.tolist()} not 117/28/63/45")

    rates = fit_rates(y, grp, k)
    s_in = rates[grp]
    a_in = auroc(y, s_in)
    if not kills and abs(a_in - IN_SAMPLE_AUROC_EXPECTED) > 1e-12:
        kills.append(f"in-sample rival AUROC {a_in!r} does not reproduce C139")

    if kills:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"verdict": "KILLED", "kill_reasons": kills}, indent=1) + "\n")
        print("KILLED:", kills)
        return

    rng = np.random.default_rng(SEED)
    out = {
        "what": "C140 rival adequacy: rule-label association, refit permutation null, "
                "leave-one-out cross-fitted rival, detectability, deduplicated counts.",
        "preregistration": "accord/c140_rival_adequacy/C140_PREREGISTRATION.md",
        "no_new_inference": True,
        "seed": SEED, "alpha": ALPHA,
        "strata": {"rules": rules_sorted, "n": n_by.tolist(), "positives": k_by.tolist(),
                   "rates": rates.tolist()},
        "n": int(len(ids253)), "n_pos": int(y.sum()),
    }

    # ---- Q1 rule-label association ----
    table = np.array([k_by, n_by - k_by])
    chi2, p_chi, dof, exp = stats.chi2_contingency(table)
    base = y.copy()
    cnt = 0
    for _ in range(B_PERM):
        perm = rng.permutation(base)
        t = np.array([[perm[grp == i].sum() for i in range(k)],
                      [(1 - perm[grp == i]).sum() for i in range(k)]])
        if stats.chi2_contingency(t)[0] >= chi2 - 1e-12:
            cnt += 1
    out["q1_rule_label_association"] = {
        "chi2": float(chi2), "dof": int(dof), "p_chi2_asymptotic": float(p_chi),
        "expected_positives": exp[0].tolist(),
        "permutation_p": float((1 + cnt) / (1 + B_PERM)), "permutation_draws": B_PERM,
        "min_expected_cell": float(exp.min()),
        "association_detected_at_0_05": bool(p_chi < ALPHA),
    }

    # ---- Q2 correct (refit) permutation null ----
    cnt2 = 0
    vals = np.empty(B_PERM)
    for b in range(B_PERM):
        perm = rng.permutation(base)
        vals[b] = auroc(perm, fit_rates(perm, grp, k)[grp])
        if vals[b] >= a_in - 1e-12:
            cnt2 += 1
    out["q2_refit_permutation_null"] = {
        "in_sample_auroc": a_in,
        "refit_permutation_p": float((1 + cnt2) / (1 + B_PERM)),
        "c139_fixed_score_permutation_p": 0.03674816259187041,
        "null_mean_auroc": float(vals.mean()), "null_median_auroc": float(np.median(vals)),
        "null_q95_auroc": float(np.quantile(vals, 0.95)),
        "fraction_of_null_draws_at_or_above_0_5": float((vals >= 0.5).mean()),
        "statistic_is_bounded_below_by_0_5": bool((vals >= 0.5 - 1e-12).all()),
        "permutation_draws": B_PERM,
    }

    # ---- Q3 leave-one-out cross-fitted rival ----
    s_loo = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        r = fit_rates(y[m], grp[m], k)
        s_loo[i] = r[grp[i]]
    a_loo = auroc(y, s_loo)
    boot = []
    idx = np.arange(len(y))
    for _ in range(B_BOOT):
        bi = rng.choice(idx, len(idx), replace=True)
        if y[bi].sum() in (0, len(bi)):
            continue
        boot.append(auroc(y[bi], s_loo[bi]))
    boot = np.array(boot)
    c3 = 0
    for _ in range(B_PERM):
        perm = rng.permutation(base)
        sl = np.empty(len(perm))
        for i in range(len(perm)):
            m = np.ones(len(perm), bool); m[i] = False
            sl[i] = fit_rates(perm[m], grp[m], k)[grp[i]]
        if auroc(perm, sl) >= a_loo - 1e-12:
            c3 += 1
    lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    out["q3_cross_fitted_rival"] = {
        "method": "leave-one-pair-out refit of the four stratum rates",
        "auroc": a_loo, "in_sample_auroc": a_in,
        "optimism_in_sample_minus_loo": a_in - a_loo,
        "bootstrap_ci95": [lo, hi], "bootstrap_draws_valid": int(len(boot)),
        "lower_bound_above_half": bool(lo > 0.5),
        "refit_permutation_p": float((1 + c3) / (1 + B_PERM)),
    }

    # ---- Q4 detectability, from committed C139 se_diff ----
    c139 = json.loads(C139_REAL.read_text())
    cells = c139["cells"]
    dup_groups = json.loads((ROOT / "results/accord/labeller_interval_closure/intervals_17.json").read_text())["summary"]["duplicate_auroc_groups"]
    dup_drop = {grp_[1] for grp_ in dup_groups}
    distinct = [c for c in cells if c not in dup_drop]
    kfam = len(distinct)
    z05 = float(stats.norm.ppf(1 - ALPHA / 2))
    zbf = float(stats.norm.ppf(1 - (ALPHA / kfam) / 2))
    zpow = float(stats.norm.ppf(0.80))
    maxdelta = 1.0 - a_in
    per = {}
    unreach_bf = unreach_05 = 0
    for name, c in cells.items():
        se = c["se_diff"]; d = c["auc_diff_graded_minus_prior"]
        mde05 = (z05 + zpow) * se; mdebf = (zbf + zpow) * se
        sig05 = z05 * se; sigbf = zbf * se
        rec = {
            "se_diff": se, "observed_diff": d,
            "mde80_alpha_0_05": mde05, "mde80_bonferroni": mdebf,
            "min_diff_for_significance_0_05": sig05,
            "min_diff_for_significance_bonferroni": sigbf,
            "implied_auroc_for_significance_0_05": a_in + sig05,
            "implied_auroc_for_significance_bonferroni": a_in + sigbf,
            "power_at_observed_0_05": float(1 - stats.norm.cdf(z05 - abs(d) / se) + stats.norm.cdf(-z05 - abs(d) / se)),
            "power_at_observed_bonferroni": float(1 - stats.norm.cdf(zbf - abs(d) / se) + stats.norm.cdf(-zbf - abs(d) / se)),
            "unreachable_at_auroc_1_bonferroni": bool(sigbf > maxdelta),
            "unreachable_at_auroc_1_alpha_0_05": bool(sig05 > maxdelta),
            "is_determinism_duplicate": name in dup_drop,
        }
        if rec["unreachable_at_auroc_1_bonferroni"]:
            unreach_bf += 1
        if rec["unreachable_at_auroc_1_alpha_0_05"]:
            unreach_05 += 1
        per[name] = rec
    ses = np.array([c["se_diff"] for c in cells.values()])
    out["q4_detectability"] = {
        "rival_auroc": a_in, "max_attainable_difference": maxdelta,
        "n_cells": len(cells), "n_distinct_cells": kfam,
        "bonferroni_divisor_distinct": kfam,
        "bonferroni_alpha_distinct": ALPHA / kfam,
        "se_diff_mean": float(ses.mean()), "se_diff_min": float(ses.min()), "se_diff_max": float(ses.max()),
        "cells_unreachable_at_auroc_1_bonferroni": unreach_bf,
        "cells_unreachable_at_auroc_1_alpha_0_05": unreach_05,
        "per_cell": per,
    }

    # ---- Q5 deduplicated counts ----
    i17 = json.loads((ROOT / "results/accord/labeller_interval_closure/intervals_17.json").read_text())
    s17 = i17["summary"]
    surv = s17["surviving_bonferroni_at_registered_B_20000"]
    out["q5_deduplicated_counts"] = {
        "duplicate_groups": dup_groups,
        "n_configurations_reported": i17["n_configurations"],
        "n_distinct_configurations": kfam,
        "surviving_bonferroni_reported": len(surv),
        "surviving_bonferroni_distinct": len([x for x in surv if x not in dup_drop]),
        "surviving_bonferroni_distinct_list": [x for x in surv if x not in dup_drop],
        "reject_chance_reported": s17["n_configurations_rejecting_chance_at_0_05"],
        "bootstrap_lb_above_half_reported": s17["n_cells_with_bootstrap_lower_bound_above_0_5_on_g253"],
        "note_sidedness": "against-chance permutation p is one-sided; DeLong p is two-sided",
    }

    # ---- outcomes ----
    q2p = out["q2_refit_permutation_null"]["refit_permutation_p"]
    o = []
    if q2p < ALPHA and out["q3_cross_fitted_rival"]["lower_bound_above_half"]:
        o.append("A")
    if q2p >= ALPHA or not out["q3_cross_fitted_rival"]["lower_bound_above_half"]:
        o.append("B")
    if unreach_bf > len(cells) / 2:
        o.append("C")
    out["outcome"] = o
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT, "outcome", o)


if __name__ == "__main__":
    main()
