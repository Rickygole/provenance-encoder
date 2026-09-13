"""C105 analysis: delta-AP / delta-AUROC (S1 minus H) with a two-level
percentile bootstrap (replicates, then items within a replicate's
evaluation-set predictions), plus the secondary flag-rate and resampled
c98-rule numbers. Public data / artifacts only.
"""
import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from accord.public_controls.evaluation import rule_c98  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPLICATES_DIR = HERE / "replicates"
OUT_PATH = HERE.parent.parent.parent.parent / "results" / "accord" / "public_controls" / "priorfix" / "analysis.json"
ARMS = ["H", "S1", "T", "S0"]
B = 10_000
SEED = 42


def load_replicates():
    reps = []
    for p in sorted(REPLICATES_DIR.glob("rep*.json")):
        reps.append(json.loads(p.read_text()))
    return reps


def per_replicate_metrics(rep):
    y = np.array(rep["eval_y"])
    out = {}
    for arm in ARMS:
        s = np.array(rep["arms"][arm]["scores"])
        ap = float(average_precision_score(y, s))
        auc = float(roc_auc_score(y, s))
        out[arm] = {"ap": ap, "auc": auc, "scores": s}
    return out, y


def two_level_bootstrap_delta(reps_metrics, arm_a, arm_b, metric, B, seed):
    """Two-level percentile bootstrap: resample replicates with replacement,
    then within each resampled replicate resample evaluation items with
    replacement, recompute metric(arm_a) - metric(arm_b), collect B draws."""
    rng = np.random.default_rng(seed)
    n_rep = len(reps_metrics)
    deltas = np.empty(B)
    for b in range(B):
        rep_idx = rng.integers(0, n_rep)
        metrics, y = reps_metrics[rep_idx]
        n_items = len(y)
        item_idx = rng.integers(0, n_items, size=n_items)
        y_b = y[item_idx]
        if len(set(y_b.tolist())) < 2:
            deltas[b] = np.nan
            continue
        s_a = metrics[arm_a]["scores"][item_idx]
        s_b = metrics[arm_b]["scores"][item_idx]
        if metric == "ap":
            m_a = average_precision_score(y_b, s_a)
            m_b = average_precision_score(y_b, s_b)
        else:
            m_a = roc_auc_score(y_b, s_a)
            m_b = roc_auc_score(y_b, s_b)
        deltas[b] = m_a - m_b
    deltas = deltas[~np.isnan(deltas)]
    return deltas


def main():
    reps = load_replicates()
    if not reps:
        raise SystemExit("no replicates found under replicates/; run run_priorfix.py first")

    reps_metrics = [per_replicate_metrics(r) for r in reps]
    per_rep_summary = []
    for r, (m, y) in zip(reps, reps_metrics):
        row = {"replicate": r["replicate"], "pi_hat": r["pi_hat"]}
        for arm in ARMS:
            row[f"{arm}_ap"] = m[arm]["ap"]
            row[f"{arm}_auc"] = m[arm]["auc"]
            row[f"{arm}_flag_rate_0_5"] = r["arms"][arm]["flag_rate_at_0_5"]
        row["teacher_marginal_on_pool"] = r["teacher_marginal_on_pool"]
        per_rep_summary.append(row)

    delta_ap = two_level_bootstrap_delta(reps_metrics, "S1", "H", "ap", B, SEED)
    delta_auc = two_level_bootstrap_delta(reps_metrics, "S1", "H", "auc", B, SEED + 1)

    mean_delta_ap = float(np.mean(delta_ap))
    ci_ap = [float(np.percentile(delta_ap, 2.5)), float(np.percentile(delta_ap, 97.5))]
    mean_delta_auc = float(np.mean(delta_auc))
    ci_auc = [float(np.percentile(delta_auc, 2.5)), float(np.percentile(delta_auc, 97.5))]

    outcome_a = bool(mean_delta_ap > 0 and ci_ap[0] > 0 and mean_delta_auc > -0.01)

    # Secondary: resampled 12/253 c98-rule rejection rate per arm, over the
    # B bootstrap resamples already drawn for delta_ap (reuse the same
    # resampling scheme by redrawing independently per arm here).
    rng = np.random.default_rng(SEED + 2)
    c98_rates = {}
    for arm in ARMS:
        fires = 0
        n_trials = 2000  # cheaper than B=10000 since c98_rule is O(n log n)
        for _ in range(n_trials):
            rep_idx = rng.integers(0, len(reps_metrics))
            m, y = reps_metrics[rep_idx]
            n_items = len(y)
            item_idx = rng.integers(0, n_items, size=n_items)
            y_b, s_b = y[item_idx], m[arm]["scores"][item_idx]
            if len(set(y_b.tolist())) < 2:
                continue
            if rule_c98(y_b, s_b):
                fires += 1
        c98_rates[arm] = fires / n_trials

    out = {
        "preregistration": "accord/public_controls/priorfix/C105_PREREGISTRATION.md",
        "n_replicates": len(reps),
        "per_replicate": per_rep_summary,
        "delta_ap_s1_minus_h": {"mean": mean_delta_ap, "ci95": ci_ap},
        "delta_auc_s1_minus_h": {"mean": mean_delta_auc, "ci95": ci_auc},
        "outcome": "A" if outcome_a else "B",
        "c98_rule_rejection_rate_bootstrap": c98_rates,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    print(f"delta_ap(S1-H): mean={mean_delta_ap:.4f} ci95={ci_ap} "
          f"delta_auc: mean={mean_delta_auc:.4f} ci95={ci_auc} outcome={out['outcome']}")


if __name__ == "__main__":
    main()
