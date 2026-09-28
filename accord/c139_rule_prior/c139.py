#!/usr/bin/env python3
"""C139. Does the graded labeller score beat a pure rule-identity prior?

Runs the check pre-registered in C139_PREREGISTRATION.md. No model inference:
every score is read from a committed per-pair cache or derived from the stored
rule column and the pharmacist labels.

    python3 accord/c139_rule_prior/c139.py
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.control.delong import delong_roc_test  # noqa: E402
from accord.labeller_interval_closure.intervals_17 import (  # noqa: E402
    CONFIGS, MODEL_TAGS, DERIVED, GOLD, load_cache, block,
)

ALPHA = 0.05
OUT = ROOT / "results/accord/c139_rule_prior/c139.json"
PREREG = pathlib.Path(__file__).resolve().parent / "C139_PREREGISTRATION.md"


def main():
    g = pd.read_csv(GOLD)
    label_by_pair = dict(zip(g["pair_id"], g["label_clean"]))
    pair_ids_299 = g["pair_id"].tolist()
    dg = pd.read_csv(DERIVED)
    rule_by_pair = dict(zip(dg["pair_id"], dg["rule"]))

    ids253 = [p for p in pair_ids_299 if label_by_pair[p] != "ambiguous"]
    y_by_pair = {p: int(label_by_pair[p] == "contradiction") for p in ids253}

    kills = []
    if len(ids253) != 253:
        kills.append(f"G253 pair count is {len(ids253)}, not 253")
    if sum(y_by_pair.values()) != 12:
        kills.append(f"G253 positive count is {sum(y_by_pair.values())}, not 12")
    missing_rule = [p for p in ids253 if p not in rule_by_pair]
    if missing_rule:
        kills.append(f"{len(missing_rule)} of 253 pairs have no rule label")

    # ---- the rule-prior score: in-sample pharmacist precision of each rule ----
    strata = collections.defaultdict(list)
    for p in ids253:
        strata[rule_by_pair[p]].append(p)
    rule_precision = {r: sum(y_by_pair[p] for p in ps) / len(ps)
                      for r, ps in sorted(strata.items())}
    s_rule_by_pair = {p: rule_precision[rule_by_pair[p]] for p in ids253}
    if len(set(s_rule_by_pair.values())) < 2:
        kills.append("rule-prior score takes fewer than 2 distinct values")

    if kills:
        res = {"verdict": "KILLED", "kill_reasons": kills}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1) + "\n")
        print("KILLED:", kills)
        return

    y253 = [y_by_pair[p] for p in ids253]
    s253 = [s_rule_by_pair[p] for p in ids253]
    rule_block = block(y253, s253)

    out = {
        "what": "Paired DeLong test of each graded labeller cell against an "
                "in-sample rule-identity prior on the same G253 pairs.",
        "preregistration": "accord/c139_rule_prior/C139_PREREGISTRATION.md",
        "no_new_inference": True,
        "alpha": ALPHA,
        "n_configurations": len(CONFIGS),
        "bonferroni_alpha_over_n": ALPHA / len(CONFIGS),
        "rule_strata": {str(r): {"n_pairs": len(ps),
                                 "n_positives": sum(y_by_pair[p] for p in ps),
                                 "pharmacist_precision": rule_precision[r]}
                        for r, ps in sorted(strata.items())},
        "rule_prior": rule_block,
        "cells": collections.OrderedDict(),
    }

    thresh = ALPHA / len(CONFIGS)
    for family, model, prompt in CONFIGS:
        recs = load_cache(MODEL_TAGS[model], prompt)
        key = f"{family}::{model}::{prompt}"
        if not recs:
            out["cells"][key] = {"cache_present": False}
            continue
        ids = [p for p in ids253
               if p in recs and recs[p].get("flag_logodds") is not None]
        if len(ids) != len(ids253):
            out["cells"][key] = {"cache_present": True,
                                 "note": f"joined {len(ids)} of {len(ids253)} pairs"}
        y = [y_by_pair[p] for p in ids]
        graded = [recs[p]["flag_logodds"] for p in ids]
        prior = [s_rule_by_pair[p] for p in ids]
        d = delong_roc_test(np.array(y), np.array(graded), np.array(prior))
        cell = out["cells"].setdefault(key, {})
        cell.update({
            "cache_present": True, "n": len(ids), "n_pos": int(sum(y)),
            "auroc_graded": d["auc_a"], "auroc_rule_prior": d["auc_b"],
            "auc_diff_graded_minus_prior": d["auc_diff"], "se_diff": d["se_diff"],
            "z": d["z"], "p_two_sided": d["p"], "ci95_diff": d["ci95_diff"],
            "clears_bonferroni": bool(d["p"] < thresh),
            "graded_above_prior": bool(d["auc_diff"] > 0),
        })

    live = [c for c in out["cells"].values() if c.get("p_two_sided") is not None]
    winners = [k for k, c in out["cells"].items()
               if c.get("clears_bonferroni") and c.get("graded_above_prior")]
    losers = [k for k, c in out["cells"].items()
              if c.get("clears_bonferroni") and c.get("graded_above_prior") is False]
    best = max(live, key=lambda c: c["auroc_graded"])
    best_key = [k for k, c in out["cells"].items() if c is best][0]

    if winners:
        verdict = "A_GRADED_BEATS_RULE_PRIOR"
    elif losers:
        verdict = "C_RULE_PRIOR_WINS"
    else:
        verdict = "B_NOT_SEPARABLE"

    out["summary"] = {
        "n_cells_clearing_bonferroni_graded_above_prior": len(winners),
        "cells_clearing_bonferroni_graded_above_prior": winners,
        "n_cells_clearing_bonferroni_prior_above_graded": len(losers),
        "n_cells_with_graded_above_prior_pointwise":
            sum(1 for c in live if c["graded_above_prior"]),
        "n_cells_uncorrected_p_below_0_05":
            sum(1 for c in live if c["p_two_sided"] < ALPHA),
        "strongest_cell": best_key,
        "strongest_cell_auroc": best["auroc_graded"],
        "rule_prior_auroc": rule_block["auroc"],
        "strongest_cell_diff": best["auc_diff_graded_minus_prior"],
        "strongest_cell_p_two_sided": best["p_two_sided"],
        "verdict": verdict,
    }

    # ---- secondary, descriptive: the strongest cell within each rule stratum ----
    recs = load_cache(MODEL_TAGS[best_key.split("::")[1]], best_key.split("::")[2])
    per_stratum = {}
    for r, ps in sorted(strata.items()):
        ids = [p for p in ps if p in recs and recs[p].get("flag_logodds") is not None]
        y = [y_by_pair[p] for p in ids]
        if sum(y) < 2 or (len(y) - sum(y)) < 2:
            per_stratum[str(r)] = {"n": len(y), "n_pos": int(sum(y)),
                                   "auroc": None,
                                   "note": "not estimable: fewer than 2 in a class"}
            continue
        per_stratum[str(r)] = block(y, [recs[p]["flag_logodds"] for p in ids],
                                    with_bootstrap=False)
    out["secondary_within_rule_strata_descriptive"] = {
        "cell": best_key,
        "why": "rule identity is constant inside a stratum, so signal surviving "
               "here is not rule identity",
        "strata": per_stratum,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")

    print(f"rule strata (G253): {out['rule_strata']}")
    print(f"rule-prior AUROC {rule_block['auroc']:.4f}  "
          f"CI {rule_block['bootstrap']['ci95'][0]:.4f} to "
          f"{rule_block['bootstrap']['ci95'][1]:.4f}  "
          f"perm p {rule_block['permutation_p']:.4f}")
    print(f"corrected threshold {thresh:.10f}")
    for k, c in out["cells"].items():
        if c.get("p_two_sided") is None:
            continue
        flag = "*" if c["clears_bonferroni"] else " "
        print(f" {flag} {k:46} graded {c['auroc_graded']:.4f}  prior "
              f"{c['auroc_rule_prior']:.4f}  diff {c['auc_diff_graded_minus_prior']:+.4f}  "
              f"p {c['p_two_sided']:.4f}")
    print(f"VERDICT: {verdict}")
    print(f"within-rule strata for {best_key}:")
    for r, b in per_stratum.items():
        a = b.get("auroc")
        print(f"   rule {r}: n={b['n']} pos={b['n_pos']} "
              f"auroc={'n/a' if a is None else f'{a:.4f}'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
