"""Intervals around the DIFFERENCE between systems, not around each system.

WHY THIS EXISTS
---------------
Section m-stats commits this thesis to claiming an ordering among systems only on
an interval around the difference, and no such interval was ever computed. The
chapter and the conclusion instead argue from non-overlap of two separate
intervals, which is a weaker and different thing: two intervals can overlap while
the interval around their difference excludes zero, and they can fail to overlap
while it does not. The conclusion says so and calls the non-overlap "the weaker
evidence that is available". It is available, and this computes it.

Paired bootstrap over the same pairs, so the two systems are resampled together
and their correlation is preserved. That correlation is the whole point: both
systems score the same rows, so an unpaired comparison throws away the pairing the
design already has.

Nothing is transmitted. Inputs are the committed frontier predictions and the
locally held gold labels and detector scores.

Run:  python3 accord/stats/system_difference_intervals.py
Writes: results/accord/stats/system_difference_intervals.json
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIFF = ROOT / "results" / "accord" / "differential"
GOLD = pathlib.Path.home() / "Downloads" / "gold_standard_accord_results.csv"
OUT = ROOT / "results" / "accord" / "stats" / "system_difference_intervals.json"
B, SEED = 10000, 42


def mcc_vec(a, b, c, d):
    den = np.sqrt((a + b) * (a + c) * (d + b) * (d + c), dtype=float)
    return np.where(den == 0, 0.0, (a * d - b * c) / np.where(den == 0, 1.0, den))


def cells(y, p):
    return (((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 0)).sum(),
            ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum())


def main():
    gold = pd.read_csv(GOLD)
    g = gold[gold.pharm_label != "ambiguous"].copy()
    g["y"] = (g.pharm_label == "contradiction").astype(int)
    # The detector at its configured operating point, the comparison the thesis makes.
    g["det"] = (g.accord_contra_prob >= 0.5).astype(int)

    out = {"basis": "G_253", "n": int(len(g)), "n_positives": int(g.y.sum()),
           "detector_arm_caveat": (
               "The detector arm is the RELEASED run at its configured threshold of 0.5 on "
               "accord_contra_prob, because that is the only detector artifact carrying a "
               "per-pair score for all 253 rows. It is not the decontaminated three-seed run "
               "behind the operating-point table, for which no per-pair score file exists: "
               "predictions_253_core.csv holds the 12 positives only. The two runs are "
               "different models and they differ in the direction of their degeneracy at 0.5, "
               "so this comparison should be read as labeller against the released detector "
               "and not as labeller against the detector the operating-point table reports."),
           "method": ("paired percentile bootstrap over pairs, B=%d, seed=%d; both "
                      "systems resampled together so their correlation on shared rows "
                      "is preserved" % (B, SEED)),
           "comparisons": {}}

    for name, f in [("sonnet_zero_shot", "frontier_anthropic_zero_shot.json"),
                    ("haiku_zero_shot", "frontier_anthropic_claudehaiku4520251001_zero_shot.json"),
                    ("llama70b_few_shot", "frontier_bedrock_metallama370binstructv1_few_shot.json")]:
        path = DIFF / f
        if not path.exists():
            continue
        preds = {p["pair_id"]: p["pred"] for p in json.loads(path.read_text())["predictions"]}
        sub = g[g.pair_id.isin(preds)].copy()
        sub["lab"] = sub.pair_id.map(preds)
        sub = sub[sub.lab.notna()]
        y = sub.y.to_numpy(); a = sub.lab.to_numpy().astype(int); b = sub.det.to_numpy()
        n = len(y)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(B, n))
        ys, as_, bs = y[idx], a[idx], b[idx]
        m_a = mcc_vec(((ys == 1) & (as_ == 1)).sum(1), ((ys == 0) & (as_ == 1)).sum(1),
                      ((ys == 1) & (as_ == 0)).sum(1), ((ys == 0) & (as_ == 0)).sum(1))
        m_b = mcc_vec(((ys == 1) & (bs == 1)).sum(1), ((ys == 0) & (bs == 1)).sum(1),
                      ((ys == 1) & (bs == 0)).sum(1), ((ys == 0) & (bs == 0)).sum(1))
        d = m_a - m_b
        tp, tn, fp, fn = cells(y, a); m_obs_a = mcc_vec(np.array([tp]), np.array([fp]), np.array([fn]), np.array([tn]))[0]
        tp, tn, fp, fn = cells(y, b); m_obs_b = mcc_vec(np.array([tp]), np.array([fp]), np.array([fn]), np.array([tn]))[0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["comparisons"][f"{name}_minus_detector"] = {
            "n": int(n),
            "mcc_labeller": round(float(m_obs_a), 4),
            "mcc_detector_at_0.5": round(float(m_obs_b), 4),
            "observed_difference": round(float(m_obs_a - m_obs_b), 4),
            "difference_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "difference_excludes_zero": bool(lo > 0 or hi < 0),
            "p_bootstrap_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"basis {out['basis']}  n={out['n']}  positives={out['n_positives']}\n")
    for k, v in out["comparisons"].items():
        mark = "EXCLUDES 0" if v["difference_excludes_zero"] else "includes 0"
        print(f"{k:34s} diff={v['observed_difference']:+.4f} "
              f"CI[{v['difference_ci95'][0]:+.4f},{v['difference_ci95'][1]:+.4f}]  {mark}")


if __name__ == "__main__":
    main()
