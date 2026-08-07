"""
ACCORD operating-point sweep -- Analysis 1.

For each of three continuous scorers (ACCORD contra_prob, subject-disjoint
vanilla control ensemble, TF-IDF+LR), sweep the decision threshold so the
predicted-positive rate runs ~4% -> ~90% in ~2.5pp steps. At every operating
point, compute by language_group3 (ENGLISH / MISSING / RECORDED_NON_ENGLISH):
  - demographic-parity gap (max-min predicted-positive rate)
  - FPR gap, TPR gap, precision gap (max-min across groups where defined)
  - four-fifths ratio (min/max predicted-positive rate)
  - bootstrap 95% CIs (2000 resamples, fixed seed, threshold held fixed)
  - cell counts
  - minimum detectable gap (80% power, alpha=0.05, worst-case p=0.5 variance)
    per group pair and per metric family

Outputs (aggregates only, no note text):
  /Users/rickygole/mimic_local_only/accord_opsweep/sweep_<scorer>.csv
  /Users/rickygole/mimic_local_only/accord_opsweep/sweep_summary.json
"""
import json
import os

import numpy as np
import pandas as pd

IN_PATH = "/Users/rickygole/mimic_local_only/accord_opsweep/gold_scores_language.csv"
OUT_DIR = "/Users/rickygole/mimic_local_only/accord_opsweep"
os.makedirs(OUT_DIR, exist_ok=True)

GROUPS = ["ENGLISH", "MISSING", "RECORDED_NON_ENGLISH"]
GSHORT = {"ENGLISH": "EN", "MISSING": "MI", "RECORDED_NON_ENGLISH": "RN"}
PAIRS = [("ENGLISH", "MISSING"), ("ENGLISH", "RECORDED_NON_ENGLISH"), ("MISSING", "RECORDED_NON_ENGLISH")]

SCORERS = {
    "accord": "accord_score",
    "control": "control_score",
    "tfidf": "tfidf_score",
}

N_BOOT = 2000
BOOT_SEED = 42
Z_ALPHA2 = 1.959964
Z_BETA = 0.841621  # 80% power
TOLERANCE = 0.05  # audit "pass" tolerance on DP/FPR gap; four-fifths pass = ratio >= 0.8

RATE_GRID = np.round(np.arange(0.04, 0.905, 0.025), 4)


def group_masks(df):
    return {g: (df["language_group3"].to_numpy() == g) for g in GROUPS}


def cell_stats(pred, y, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, n_pos=0, n_neg=0, n_pred_pos=0, TP=0, FP=0, FN=0, TN=0,
                    pred_pos_rate=np.nan, FPR=np.nan, TPR=np.nan, precision=np.nan)
    p, yy = pred[mask], y[mask]
    TP = int(((p == 1) & (yy == 1)).sum())
    FP = int(((p == 1) & (yy == 0)).sum())
    FN = int(((p == 0) & (yy == 1)).sum())
    TN = int(((p == 0) & (yy == 0)).sum())
    n_pos, n_neg = TP + FN, FP + TN
    n_pred_pos = TP + FP
    return dict(
        n=n, n_pos=n_pos, n_neg=n_neg, n_pred_pos=n_pred_pos, TP=TP, FP=FP, FN=FN, TN=TN,
        pred_pos_rate=n_pred_pos / n,
        FPR=(FP / n_neg) if n_neg > 0 else np.nan,
        TPR=(TP / n_pos) if n_pos > 0 else np.nan,
        precision=(TP / n_pred_pos) if n_pred_pos > 0 else np.nan,
    )


def gap_metrics(pred, y, masks):
    cells = {g: cell_stats(pred, y, masks[g]) for g in GROUPS}
    dp = np.array([cells[g]["pred_pos_rate"] for g in GROUPS])
    fpr = np.array([cells[g]["FPR"] for g in GROUPS])
    tpr = np.array([cells[g]["TPR"] for g in GROUPS])
    prec = np.array([cells[g]["precision"] for g in GROUPS])

    def rng(a):
        a = a[~np.isnan(a)]
        return (np.nan, np.nan) if len(a) < 2 else (float(np.max(a) - np.min(a)), len(a))

    def ff(a):
        a = a[~np.isnan(a)]
        if len(a) < 2 or np.max(a) == 0:
            return np.nan
        return float(np.min(a) / np.max(a))

    dp_gap, dp_n = rng(dp)
    fpr_gap, fpr_n = rng(fpr)
    tpr_gap, tpr_n = rng(tpr)
    prec_gap, prec_n = rng(prec)
    four_fifths = ff(dp)

    return dict(cells=cells, dp_gap=dp_gap, dp_n_groups=dp_n, fpr_gap=fpr_gap, fpr_n_groups=fpr_n,
                tpr_gap=tpr_gap, tpr_n_groups=tpr_n, prec_gap=prec_gap, prec_n_groups=prec_n,
                four_fifths=four_fifths)


def bootstrap_ci(score, y, lang, threshold, rng, n_boot=N_BOOT):
    n = len(score)
    dp_s, fpr_s, tpr_s, prec_s, ff_s = [], [], [], [], []
    idx_pool = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        s_b, y_b, l_b = score[idx], y[idx], lang[idx]
        pred_b = (s_b >= threshold).astype(int)
        masks_b = {g: (l_b == g) for g in GROUPS}
        m = gap_metrics(pred_b, y_b, masks_b)
        dp_s.append(m["dp_gap"]); fpr_s.append(m["fpr_gap"]); tpr_s.append(m["tpr_gap"])
        prec_s.append(m["prec_gap"]); ff_s.append(m["four_fifths"])

    def ci(a):
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) < 50:
            return (np.nan, np.nan)
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    return dict(dp_gap_ci=ci(dp_s), fpr_gap_ci=ci(fpr_s), tpr_gap_ci=ci(tpr_s),
                prec_gap_ci=ci(prec_s), four_fifths_ci=ci(ff_s))


def mde_two_prop(n1, n2, p_bar=0.5):
    """Minimum detectable difference in two independent proportions,
    two-sided alpha=0.05, 80% power, worst-case variance at p_bar=0.5
    (the conservative / largest-MDE choice used throughout)."""
    if n1 == 0 or n2 == 0:
        return np.nan
    se = np.sqrt(p_bar * (1 - p_bar) * (1.0 / n1 + 1.0 / n2))
    return float((Z_ALPHA2 + Z_BETA) * se)


def run_scorer(df, score_col, rng):
    y = df["pharm_pos"].to_numpy()
    lang = df["language_group3"].to_numpy()
    score = df[score_col].to_numpy()
    n = len(df)

    order_desc = np.argsort(-score)  # highest score first
    sorted_scores = score[order_desc]

    rows = []
    for target_rate in RATE_GRID:
        k = max(1, min(n - 1, int(round(target_rate * n))))
        threshold = sorted_scores[k - 1]
        pred = (score >= threshold).astype(int)
        achieved_rate = pred.sum() / n

        m = gap_metrics(pred, y, group_masks(df))
        boot = bootstrap_ci(score, y, lang, threshold, rng)

        row = dict(target_rate=float(target_rate), achieved_rate=float(achieved_rate),
                   k_pred_pos=int(pred.sum()), threshold=float(threshold),
                   dp_gap=m["dp_gap"], dp_gap_ci_lo=boot["dp_gap_ci"][0], dp_gap_ci_hi=boot["dp_gap_ci"][1],
                   fpr_gap=m["fpr_gap"], fpr_gap_ci_lo=boot["fpr_gap_ci"][0], fpr_gap_ci_hi=boot["fpr_gap_ci"][1],
                   tpr_gap=m["tpr_gap"], tpr_gap_ci_lo=boot["tpr_gap_ci"][0], tpr_gap_ci_hi=boot["tpr_gap_ci"][1],
                   tpr_n_groups_defined=m["tpr_n_groups"],
                   prec_gap=m["prec_gap"], prec_gap_ci_lo=boot["prec_gap_ci"][0], prec_gap_ci_hi=boot["prec_gap_ci"][1],
                   prec_n_groups_defined=m["prec_n_groups"],
                   four_fifths=m["four_fifths"], four_fifths_ci_lo=boot["four_fifths_ci"][0],
                   four_fifths_ci_hi=boot["four_fifths_ci"][1])

        for g in GROUPS:
            c = m["cells"][g]
            pre = GSHORT[g]
            for k2, v2 in c.items():
                row[f"{pre}_{k2}"] = v2

        # minimum detectable gap per group pair per metric family
        for g1, g2 in PAIRS:
            tag = f"{GSHORT[g1]}_{GSHORT[g2]}"
            c1, c2 = m["cells"][g1], m["cells"][g2]
            row[f"mde_dp_{tag}"] = mde_two_prop(c1["n"], c2["n"])
            row[f"mde_fpr_{tag}"] = mde_two_prop(c1["n_neg"], c2["n_neg"])
            row[f"mde_tpr_{tag}"] = mde_two_prop(c1["n_pos"], c2["n_pos"])
            row[f"mde_prec_{tag}"] = mde_two_prop(c1["n_pred_pos"], c2["n_pred_pos"])

        row["audit_pass_dp"] = bool(m["dp_gap"] <= TOLERANCE) if not np.isnan(m["dp_gap"]) else False
        row["audit_pass_fpr"] = bool(m["fpr_gap"] <= TOLERANCE) if not np.isnan(m["fpr_gap"]) else False
        row["audit_pass_four_fifths"] = bool(m["four_fifths"] >= 0.8) if not np.isnan(m["four_fifths"]) else False
        row["audit_pass_all"] = row["audit_pass_dp"] and row["audit_pass_fpr"] and row["audit_pass_four_fifths"]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(IN_PATH)
    print("Loaded", df.shape, "language_group3 counts:")
    print(df["language_group3"].value_counts().to_string())
    print("\nGround truth positives (pharm_pos) by group:")
    print(df.groupby("language_group3")["pharm_pos"].sum().to_string())
    print("\nRECORDED_NON_ENGLISH has 0 gold positives -> TPR/recall is structurally undefined")
    print("for that group at EVERY operating point, for every scorer. Noting this up front.\n")

    summary = {"tolerance": TOLERANCE, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
               "rate_grid": RATE_GRID.tolist(), "scorers": {}}

    for name, col in SCORERS.items():
        rng = np.random.default_rng(BOOT_SEED)
        print(f"=== scorer: {name} ({col}) ===")
        tbl = run_scorer(df, col, rng)
        out_path = os.path.join(OUT_DIR, f"sweep_{name}.csv")
        tbl.to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({tbl.shape[0]} operating points)")

        pass_all = tbl["audit_pass_all"]
        first_pass_rate = tbl.loc[pass_all, "achieved_rate"].min() if pass_all.any() else np.nan
        last_fail_rate = tbl.loc[~pass_all, "achieved_rate"].max() if (~pass_all).any() else np.nan
        n_pass = int(pass_all.sum())
        n_total = len(tbl)

        # Degenerate-region check: does it pass ONLY above some high rate (e.g. >0.65)?
        low_region = tbl[tbl["achieved_rate"] <= 0.5]
        high_region = tbl[tbl["achieved_rate"] > 0.5]
        low_pass_frac = float(low_region["audit_pass_all"].mean()) if len(low_region) else np.nan
        high_pass_frac = float(high_region["audit_pass_all"].mean()) if len(high_region) else np.nan

        summary["scorers"][name] = dict(
            n_operating_points=n_total,
            n_pass_all=n_pass,
            frac_pass_all=n_pass / n_total,
            first_passing_rate=float(first_pass_rate) if not np.isnan(first_pass_rate) else None,
            last_failing_rate=float(last_fail_rate) if not np.isnan(last_fail_rate) else None,
            low_region_pass_frac_le_50pct=low_pass_frac,
            high_region_pass_frac_gt_50pct=high_pass_frac,
            dp_gap_range=[float(tbl["dp_gap"].min()), float(tbl["dp_gap"].max())],
            fpr_gap_range=[float(tbl["fpr_gap"].min()), float(tbl["fpr_gap"].max())],
            four_fifths_range=[float(tbl["four_fifths"].min()), float(tbl["four_fifths"].max())],
        )
        print(f"  audit passes at {n_pass}/{n_total} operating points ({n_pass/n_total:.1%})")
        print(f"  low-rate region (<=50% predicted positive) pass frac: {low_pass_frac:.2%}")
        print(f"  high-rate region (>50% predicted positive) pass frac: {high_pass_frac:.2%}")
        print()

    with open(os.path.join(OUT_DIR, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(OUT_DIR, "sweep_summary.json"))


if __name__ == "__main__":
    main()
