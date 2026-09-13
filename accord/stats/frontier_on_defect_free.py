"""The frontier discrimination tests on the defect-free stratum.

WHY THIS EXISTS
---------------
The thesis built G_134 to neutralise the exemplar selection defect: Rules 3 and 4
cannot exhibit it, because their evidence is the dose difference or the absence
itself rather than a selected mention. It then ran that control on the negative
result, where the detector's chance-level ranking reproduced, and did not run it
on the positive one.

That asymmetry is the defect this project exists to describe. The frontier
discrimination tests on G_253 carry the whole task-versus-pipeline separation, by
the thesis's own accounting, and they are computed on a set that includes the rows
where the displayed evidence was not the evidence the rule found. The pharmacist
and the labeller saw the same displayed pair, so their agreement on a corrupted row
is agreement about the stimulus, not about a clinical contradiction. Applying the
strongest available control to the null and omitting it from the rejection is a
procedure chosen where it happens to be favourable, which is C85's defect with a
different instrument.

Nothing is transmitted to compute this. Every input is a committed prediction
artifact plus the local gold labels; no pair is re-sent to any model.

Run:  python3 accord/stats/frontier_on_defect_free.py
Writes: results/accord/stats/frontier_on_defect_free.json
"""

import glob
import json
import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
GOLD = pathlib.Path.home() / "Downloads" / "gold_standard_accord_results.csv"
OUT = ROOT / "results" / "accord" / "stats" / "frontier_on_defect_free.json"
DEFECT_FREE_RULES = (3, 4)
B, SEED = 10000, 42


def mcc(tp, tn, fp, fn):
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def boot_ci(tp, fp, fn, tn, alpha=0.05, seed=SEED, B=B):
    n = tp + fp + fn + tn
    p = np.array([tp, fp, fn, tn], float) / n
    d = np.random.default_rng(seed).multinomial(n, p, size=B)
    a, b, c, e = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    den = np.sqrt((a + b) * (a + c) * (e + b) * (e + c), dtype=float)
    m = np.where(den == 0, 0.0, (a * e - b * c) / np.where(den == 0, 1.0, den))
    return (float(np.percentile(m, 100 * alpha / 2)),
            float(np.percentile(m, 100 * (1 - alpha / 2))))


def main():
    gold = pd.read_csv(GOLD)
    sub = gold[gold.rule.isin(DEFECT_FREE_RULES)]
    # Same treatment of the ambiguous class as G_253: dropped, not recoded.
    sub = sub[sub.pharm_label != "ambiguous"]
    truth = dict(zip(sub.pair_id, (sub.pharm_label == "contradiction").astype(int)))

    rows = {}
    for f in sorted(glob.glob(str(DIFF / "frontier_*_shot.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        preds = {p["pair_id"]: p["pred"] for p in d.get("predictions", [])}
        ids = [i for i in truth if i in preds and preds[i] is not None]
        if not ids:
            continue
        y = np.array([truth[i] for i in ids])
        yh = np.array([preds[i] for i in ids])
        tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
        fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
        n, n_pos, n_flag = len(ids), int(y.sum()), int(yh.sum())
        p_one = float(hypergeom.sf(tp - 1, n, n_pos, n_flag)) if n_flag else 1.0
        best = float(hypergeom.sf(min(n_pos, n_flag) - 1, n, n_pos, n_flag)) if n_flag else 1.0
        lo, hi = boot_ci(tp, fp, fn, tn)
        rows[pathlib.Path(f).stem.replace("frontier_", "")] = {
            "model": d.get("model"), "strategy": d.get("strategy"),
            "n": n, "n_positives": n_pos, "n_flagged": n_flag,
            "cells": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "mcc": round(mcc(tp, tn, fp, fn), 4),
            "mcc_ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "p_one_sided": p_one,
            "best_attainable_p": best,
            "can_reach_significance_at_0.05": bool(best < 0.05),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]["p_one_sided"]):
        star = "" if v["can_reach_significance_at_0.05"] else "  <-- zero power"
        print(f"{k[:44]:44s} n={v['n']:>3} pos={v['n_positives']} flag={v['n_flagged']:>3} "
              f"mcc={v['mcc']:+.4f} CI[{v['mcc_ci95'][0]:+.3f},{v['mcc_ci95'][1]:+.3f}] "
              f"p={v['p_one_sided']:.5g}{star}")


if __name__ == "__main__":
    main()
