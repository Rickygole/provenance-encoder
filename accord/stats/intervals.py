"""E2 + E3: intervals the thesis reports point estimates for.

E2 -- bootstrap CIs on the four system MCCs. ch8 orders four systems by MCC with
no interval on any of them; with 12 positives on 253 pairs that ordering is not
identified, and ch8:442 has an open TODO for exactly this.

E3 -- the defect-free-stratum AUROC interval, reported WITH the degeneracy count.
At 6 positives in 134, a fraction of resamples contain zero positives and are
undefined. The thesis reports [0.222, 0.762] without saying how many draws were
dropped, and without the stratified alternative.

Aggregates only. No note text. Fixed seed.
"""
import json, os, pathlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

SEED, B = 42, 10_000
MIMIC_LOCAL_ONLY_DIR = os.environ.get("MIMIC_LOCAL_ONLY_DIR", os.path.expanduser("~/mimic_local_only"))
ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root
OUT = ROOT / "results/accord/stats"; OUT.mkdir(parents=True, exist_ok=True)


def mcc(tp, fn, fp, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def mcc_ci(tp, fn, fp, tn, rng):
    """Resample the 2x2 by resampling the n labelled units it came from."""
    y = np.r_[np.ones(tp + fn, int), np.zeros(fp + tn, int)]
    p = np.r_[np.ones(tp, int), np.zeros(fn, int), np.ones(fp, int), np.zeros(tn, int)]
    n, out = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        a, b = y[i], p[i]
        out.append(mcc(((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum(),
                       ((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()))
    o = np.array(out)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


res = {"bootstrap_B": B, "seed": SEED}

# ---- E2 -------------------------------------------------------------------
base = json.loads((ROOT / "results/accord/step3_baselines.json").read_text())
rng = np.random.default_rng(SEED)
e2 = {}
for k, v in base.items():
    if not isinstance(v, dict) or "tp" not in v:
        continue
    tp, fn, fp, tn = v["tp"], v["fn"], v["fp"], v["tn"]
    lo, hi = mcc_ci(tp, fn, fp, tn, rng)
    e2[k] = {"n": v["n"], "mcc": round(mcc(tp, fn, fp, tn), 4),
             "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo > 0 or hi < 0)}
res["E2_mcc_intervals"] = e2

# ---- E3 -------------------------------------------------------------------
g = pd.read_csv(os.path.join(MIMIC_LOCAL_ONLY_DIR, "derived/gold_standard_accord_results.csv"))
sub = g[g.rule.isin([3, 4])]
y = (sub.pharm_label == 1).astype(int).values if sub.pharm_label.dtype != object else \
    sub.pharm_label.astype(str).str.lower().isin(["1", "true", "contradiction", "yes"]).astype(int).values
s = sub.accord_contra_prob.values
rng = np.random.default_rng(SEED)
boots, dropped = [], 0
for _ in range(B):
    i = rng.integers(0, len(y), len(y))
    if len(np.unique(y[i])) < 2:
        dropped += 1
        continue
    boots.append(roc_auc_score(y[i], s[i]))
boots = np.array(boots)

pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
rng = np.random.default_rng(SEED)
strat = []
for _ in range(B):
    i = np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)]
    strat.append(roc_auc_score(y[i], s[i]))
strat = np.array(strat)

res["E3_defect_free_stratum"] = {
    "n": int(len(y)), "n_pos": int(y.sum()),
    "auroc": round(float(roc_auc_score(y, s)), 4),
    "unstratified_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                          round(float(np.percentile(boots, 97.5)), 4)],
    "draws_dropped_zero_positives": dropped,
    "drop_rate": round(dropped / B, 5),
    "stratified_ci95": [round(float(np.percentile(strat, 2.5)), 4),
                        round(float(np.percentile(strat, 97.5)), 4)],
}
(OUT / "intervals.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
