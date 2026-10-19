#!/usr/bin/env python3
"""Delivers the two C140 commitments that C140 did not emit.

C140_PREREGISTRATION.md Q3 specifies "Paired DeLong of every one of the 17 cells
against this out-of-sample rival." C140 did not emit it. Q5 specifies "every
reported p is restated one-sided and two-sided"; C140 emitted a note instead.
Both are delivered here against the CORRECTED cross-fitted rival of C141, since
C140's own leave-one-out rival is withdrawn.

No new pre-registration: this completes analyses already registered at fbadcac.
No new model inference; aggregates only.

    python3 accord/c141_crossfit_repair/c142_delivery.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))
from accord.control.delong import delong_roc_test  # noqa: E402
from accord.labeller_interval_closure.intervals_17 import (  # noqa: E402
    CONFIGS, MODEL_TAGS, DERIVED, GOLD, load_cache,
)

SEED = 141
REPEATS = 200
ALPHA = 0.05
OUT = ROOT / "results/accord/c141_crossfit_repair/c142_delivery.json"


def rates(y, grp, k, fb):
    return np.array([y[grp == i].mean() if (grp == i).any() else fb for i in range(k)])


def folds(y, rng, K):
    pos = rng.permutation(np.where(y == 1)[0]); neg = rng.permutation(np.where(y == 0)[0])
    f = [[] for _ in range(K)]
    for j, i in enumerate(pos): f[j % K].append(i)
    for j, i in enumerate(neg): f[j % K].append(i)
    return [np.array(sorted(x)) for x in f]


def main():
    c141 = json.loads((ROOT / "results/accord/c141_crossfit_repair/c141.json").read_text())
    c139 = json.loads((ROOT / "results/accord/c139_rule_prior/c139.json").read_text())

    g = pd.read_csv(GOLD)
    lab = dict(zip(g["pair_id"], g["label_clean"]))
    dg = pd.read_csv(DERIVED)
    rule = dict(zip(dg["pair_id"], dg["rule"]))
    ids = [p for p in g["pair_id"].tolist() if lab[p] != "ambiguous"]
    y = np.array([int(lab[p] == "contradiction") for p in ids])
    rs = sorted({rule[p] for p in ids}); rmap = {r: i for i, r in enumerate(rs)}
    grp = np.array([rmap[rule[p]] for p in ids]); k = len(rs)

    assert len(ids) == 253 and y.sum() == 12, "G253 did not reproduce"

    # One fixed cross-fit assignment at K=5, seed 141, for the paired test.
    rng = np.random.default_rng(SEED)
    s_cf = np.empty(len(y))
    for f in folds(y, rng, 5):
        m = np.ones(len(y), bool); m[f] = False
        s_cf[f] = rates(y[m], grp[m], k, y[m].mean())[grp[f]]

    out = {
        "what": "Delivery of C140 Q3 (paired DeLong vs the out-of-sample rival) and "
                "Q5 (p restated one-sided and two-sided).",
        "registered_in": "accord/c140_rival_adequacy/C140_PREREGISTRATION.md",
        "rival_used": "C141 cross-fitted rival, stratified 5-fold at seed 141; "
                      "C140's leave-one-out rival is withdrawn",
        "no_new_inference": True, "seed": SEED,
        "crossfit_rival_auroc_this_assignment": None,
        "crossfit_rival_auroc_mean_over_repeats":
            c141["q3r_cross_fitted_rival"]["K5"]["mean_auroc"],
    }

    cells = {}
    n_sig_05 = n_sig_bonf = 0
    cfg_list = list(c139["cells"].keys())
    kfam = c141["q4r_detectability"]["n_distinct_cells"]
    a_bonf = ALPHA / kfam
    for name in cfg_list:
        fam, tag, prompt = name.split("::")
        cache = load_cache(MODEL_TAGS.get(tag, tag), prompt)
        sc = np.array([cache[q]["flag_logodds"] if q in cache and "flag_logodds" in cache[q]
                       else np.nan for q in ids], dtype=float)
        if np.isnan(sc).any():
            cells[name] = {"skipped": "per-pair cache incomplete for this configuration"}
            continue
        r = delong_roc_test(y, sc, s_cf)
        p2 = float(r["p"])
        p1 = p2 / 2 if r["auc_diff"] > 0 else 1 - p2 / 2
        cells[name] = {
            "auroc_graded": float(r["auc_a"]), "auroc_crossfit_rival": float(r["auc_b"]),
            "auc_diff": float(r["auc_diff"]),
            "p_two_sided": p2, "p_one_sided_graded_greater": float(p1),
            "clears_0_05_two_sided": bool(p2 < ALPHA),
            "clears_bonferroni_two_sided": bool(p2 < a_bonf),
        }
        out["crossfit_rival_auroc_this_assignment"] = float(r["auc_b"])
        if p2 < ALPHA: n_sig_05 += 1
        if p2 < a_bonf: n_sig_bonf += 1

    scored = [c for c in cells.values() if "skipped" not in c]
    out["q3_delivery"] = {
        "n_cells_scored": len(scored), "n_cells_skipped": len(cells) - len(scored),
        "bonferroni_divisor_distinct": kfam, "bonferroni_alpha": a_bonf,
        "n_clearing_0_05_two_sided": n_sig_05,
        "n_clearing_bonferroni_two_sided": n_sig_bonf,
        "n_above_rival_pointwise": sum(1 for c in scored if c["auc_diff"] > 0),
        "max_auc_diff": max((c["auc_diff"] for c in scored), default=None),
        "min_p_two_sided": min((c["p_two_sided"] for c in scored), default=None),
    }
    # --- stability of the delivery across fold assignments ---
    dup = {g[1] for g in json.loads(
        (ROOT / "results/accord/labeller_interval_closure/intervals_17.json").read_text()
    )["summary"]["duplicate_auroc_groups"]}
    distinct = [c for c in cfg_list if c not in dup]
    sc_by = {}
    for name in cfg_list:
        fam, tag, prompt = name.split("::")
        cache = load_cache(MODEL_TAGS.get(tag, tag), prompt)
        sc_by[name] = np.array([cache[q]["flag_logodds"] for q in ids], float)
    rng2 = np.random.default_rng(SEED)
    n05, nbf, rivals = [], [], []
    for _ in range(REPEATS):
        sv = np.empty(len(y))
        for f in folds(y, rng2, 5):
            m = np.ones(len(y), bool); m[f] = False
            sv[f] = rates(y[m], grp[m], k, y[m].mean())[grp[f]]
        a = b = 0
        for name in distinct:
            r2 = delong_roc_test(y, sc_by[name], sv)
            if r2["p"] < ALPHA: a += 1
            if r2["p"] < a_bonf: b += 1
            rv = float(r2["auc_b"])
        n05.append(a); nbf.append(b); rivals.append(rv)
    n05 = np.array(n05); nbf = np.array(nbf); rivals = np.array(rivals)
    out["q3_stability_over_fold_assignments"] = {
        "repeats": REPEATS, "n_distinct_cells": len(distinct),
        # The per-assignment counts themselves, so the distribution the paper
        # reports can be drawn from the artifact instead of from its summary.
        "clearing_0_05_per_assignment": [int(x) for x in n05],
        "clearing_bonferroni_per_assignment": [int(x) for x in nbf],
        "clearing_0_05_mean": float(n05.mean()), "clearing_0_05_median": float(np.median(n05)),
        "clearing_0_05_min": int(n05.min()), "clearing_0_05_max": int(n05.max()),
        "clearing_0_05_pct2_5": float(np.quantile(n05, 0.025)),
        "clearing_0_05_pct97_5": float(np.quantile(n05, 0.975)),
        "clearing_bonferroni_mean": float(nbf.mean()),
        "clearing_bonferroni_max": int(nbf.max()),
        "fraction_of_assignments_with_zero_clearing_0_05": float((n05 == 0).mean()),
        "fraction_of_assignments_with_any_clearing_bonferroni": float((nbf > 0).mean()),
        "rival_auroc_mean": float(rivals.mean()), "rival_auroc_sd": float(rivals.std()),
    }

    out["q5_sidedness"] = {
        "against_chance_permutation_test": "one-sided by construction",
        "paired_delong_against_rival": "two-sided; one-sided restated per cell above",
        "note": "Both footings are reported so the two families are comparable.",
    }
    out["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT)
    print(json.dumps(out["q3_delivery"], indent=1))


if __name__ == "__main__":
    main()
