"""
Defect-free control: final aggregation.

Reads per-seed gold scores (scores_seed{N}.csv, written by train.py) plus
baselines.json, and produces:

  - a metrics table across seeds (mean/std), for gold_all and the
    defect-free rule-3/4 subset
  - a bootstrap 95% CI on AUROC (>=10,000 resamples, fixed seed) for each
    seed's score AND for the seed-ensemble (mean of the 3 sigmoid scores),
    on both strata
  - a DeLong test of the ensemble score's AUROC against ACCORD's own
    accord_contra_prob (0.460 on gold_all), on the SAME 299 pairs (paired,
    correlated-ROC test -- not an independent two-sample test)
  - the majority-class and TF-IDF+LR floor baselines
  - power-ceiling context (referencing
    results/accord/section6_auroc_power.json, plus the same Hanley-McNeil
    formula applied to the strata actually evaluated here)
  - an explicit verdict per the two interpretations laid out in the task

Writes /Users/rickygole/mimic_local_only/accord_control/final_report.json.
Aggregates only -- pair_id + numeric scores/labels, no note text.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from delong import delong_roc_test

OUT_DIR = "/Users/rickygole/mimic_local_only/accord_control"
GOLD_PATH = "/Users/rickygole/Downloads/gold_standard_accord_results.csv"
POWER_REF_PATH = "/Users/rickygole/provenance-encoder/results/accord/section6_auroc_power.json"

N_BOOT = 10_000
BOOT_SEED = 20260806  # fixed, documented


def top_k_tp(scores, y, k):
    order = np.argsort(-scores)
    return int(np.asarray(y)[order[:k]].sum())


def bootstrap_auroc_ci(y, scores, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap 95% CI on AUROC, resampling the 299 (or 134)
    gold rows with replacement. Resamples with only one class present are
    skipped (undefined AUROC) and do not count toward n_boot."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    aucs = []
    attempts = 0
    max_attempts = n_boot * 20
    while len(aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        aucs.append(roc_auc_score(yy, scores[idx]))
    aucs = np.array(aucs)
    return {
        "n_boot_valid": int(len(aucs)),
        "ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "boot_mean": float(aucs.mean()),
        "boot_std": float(aucs.std()),
    }


def block_metrics(y, scores, mask):
    yy, ss = np.asarray(y)[mask], np.asarray(scores)[mask]
    n, pos = int(mask.sum()), int(yy.sum())
    out = {"n": n, "pos": pos}
    if pos == 0 or pos == n:
        out["auroc"] = None
        out["ap"] = None
        out["bootstrap"] = None
    else:
        out["auroc"] = float(roc_auc_score(yy, ss))
        out["ap"] = float(average_precision_score(yy, ss))
        out["bootstrap"] = bootstrap_auroc_ci(yy, ss)
    out["top15_tp"] = top_k_tp(ss, yy, min(15, n))
    return out


def hanley_mcneil_min_detectable_auc(n_pos, n_neg, alpha=0.05, power=0.8):
    se0 = np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    gap = z * se0
    return {"n_pos": n_pos, "n_neg": n_neg, "se_auc_under_null": float(se0),
             "minimum_detectable_auc": float(0.5 + gap), "gap_above_chance": float(gap)}


def main():
    score_files = sorted(glob.glob(os.path.join(OUT_DIR, "scores_seed*.csv")))
    metric_files = sorted(glob.glob(os.path.join(OUT_DIR, "metrics_seed*.json")))
    assert score_files, f"no scores_seed*.csv found in {OUT_DIR} -- run train.py first"

    seed_scores = {}
    for path in score_files:
        seed = int(os.path.basename(path).replace("scores_seed", "").replace(".csv", ""))
        seed_scores[seed] = pd.read_csv(path)

    seed_metrics = {}
    for path in metric_files:
        with open(path) as f:
            m = json.load(f)
        seed_metrics[m["seed"]] = m

    seeds = sorted(seed_scores.keys())
    print(f"Found {len(seeds)} seeds: {seeds}")

    ref = seed_scores[seeds[0]][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
    for s in seeds:
        cur = seed_scores[s][["pair_id", "rule", "pharm_pos"]].reset_index(drop=True)
        assert (cur["pair_id"].to_numpy() == ref["pair_id"].to_numpy()).all(), "pair_id order mismatch across seeds"

    y = ref["pharm_pos"].to_numpy()
    rule = ref["rule"].to_numpy()
    defect_free_mask = np.isin(rule, [3, 4])
    all_mask = np.ones_like(defect_free_mask, dtype=bool)

    score_matrix = np.vstack([seed_scores[s]["score"].to_numpy() for s in seeds])
    ensemble_scores = score_matrix.mean(axis=0)

    # ---- per-seed table ----
    per_seed_table = []
    for s in seeds:
        ss = seed_scores[s]["score"].to_numpy()
        row = {
            "seed": s,
            "gold_all": block_metrics(y, ss, all_mask),
            "gold_defect_free_rule34": block_metrics(y, ss, defect_free_mask),
        }
        per_seed_table.append(row)

    def mean_std(key_path, block_name):
        vals = [r[block_name]["auroc"] for r in per_seed_table if r[block_name]["auroc"] is not None]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "values": vals}

    auroc_summary = {
        "gold_all": mean_std("auroc", "gold_all"),
        "gold_defect_free_rule34": mean_std("auroc", "gold_defect_free_rule34"),
    }

    ap_vals_all = [r["gold_all"]["ap"] for r in per_seed_table]
    ap_vals_df = [r["gold_defect_free_rule34"]["ap"] for r in per_seed_table]
    ap_summary = {
        "gold_all": {"mean": float(np.mean(ap_vals_all)), "std": float(np.std(ap_vals_all, ddof=1)) if len(ap_vals_all) > 1 else 0.0},
        "gold_defect_free_rule34": {"mean": float(np.mean(ap_vals_df)), "std": float(np.std(ap_vals_df, ddof=1)) if len(ap_vals_df) > 1 else 0.0},
    }

    top15_vals_all = [r["gold_all"]["top15_tp"] for r in per_seed_table]
    top15_vals_df = [r["gold_defect_free_rule34"]["top15_tp"] for r in per_seed_table]

    # ---- ensemble (mean of sigmoid scores across seeds) ----
    ensemble_block = {
        "gold_all": block_metrics(y, ensemble_scores, all_mask),
        "gold_defect_free_rule34": block_metrics(y, ensemble_scores, defect_free_mask),
    }

    # ---- ACCORD's own contra_prob on the same 299 pairs ----
    gold_raw = pd.read_csv(GOLD_PATH)
    accord_df = gold_raw.merge(ref[["pair_id"]], on="pair_id", how="inner")
    assert len(accord_df) == len(ref), "ACCORD gold csv did not match ref pair_id set 1:1"
    accord_df = accord_df.set_index("pair_id").loc[ref["pair_id"]].reset_index()
    accord_scores = accord_df["accord_contra_prob"].to_numpy()
    accord_pharm_pos = (accord_df["pharm_label"] == "contradiction").astype(int).to_numpy()
    assert (accord_pharm_pos == y).all(), "ACCORD gold csv pharm_label does not match ref pharm_pos"

    accord_block = {
        "gold_all": block_metrics(y, accord_scores, all_mask),
        "gold_defect_free_rule34": block_metrics(y, accord_scores, defect_free_mask),
    }

    # ---- DeLong: ensemble control vs ACCORD, on the same 299 pairs ----
    delong_all = delong_roc_test(y[all_mask], ensemble_scores[all_mask], accord_scores[all_mask])
    delong_df = delong_roc_test(y[defect_free_mask], ensemble_scores[defect_free_mask], accord_scores[defect_free_mask])
    delong_per_seed = {
        s: delong_roc_test(y, seed_scores[s]["score"].to_numpy(), accord_scores)
        for s in seeds
    }

    # ---- baselines ----
    baselines_path = os.path.join(OUT_DIR, "baselines.json")
    with open(baselines_path) as f:
        baselines = json.load(f)

    # ---- power ceiling ----
    with open(POWER_REF_PATH) as f:
        power_ref = json.load(f)
    # n_neg = n - n_pos over the mask actually evaluated (all 299, ambiguous included as negative)
    n_pos_all = int(y[all_mask].sum())
    n_neg_all = int(all_mask.sum() - n_pos_all)
    power_gold_all_full = hanley_mcneil_min_detectable_auc(n_pos_all, n_neg_all)
    n_pos_df = int(y[defect_free_mask].sum())
    n_neg_df = int(defect_free_mask.sum() - n_pos_df)
    power_defect_free = hanley_mcneil_min_detectable_auc(n_pos_df, n_neg_df)

    # ---- verdict ----
    ens_auc = ensemble_block["gold_all"]["auroc"]
    ens_ci = ensemble_block["gold_all"]["bootstrap"]["ci95"]
    ens_auc_df = ensemble_block["gold_defect_free_rule34"]["auroc"]
    ens_ci_df = ensemble_block["gold_defect_free_rule34"]["bootstrap"]["ci95"]

    clears_chance_all = ens_ci[0] > 0.5
    clears_chance_df = ens_ci_df[0] > 0.5
    ci_excludes_point6_all = ens_ci[1] < 0.6
    ci_excludes_point6_df = ens_ci_df[1] < 0.6

    if clears_chance_all or clears_chance_df:
        interpretation = "CLEARS_CHANCE"
        verdict_text = (
            "The vanilla BioMedBERT control's bootstrap 95% CI excludes 0.5 on at least one "
            "gold stratum, i.e. it detectably beats chance where ACCORD's own contra_prob did "
            "not. This points toward ACCORD's architecture/implementation as the source of its "
            "chance-level result, not an intrinsic ceiling on the task. Under this reading, the "
            "honest framing of the thesis narrows to a post-mortem of one implementation, and the "
            "title/claims should be adjusted accordingly."
        )
    elif ci_excludes_point6_all and ci_excludes_point6_df:
        interpretation = "STAYS_AT_CHANCE"
        verdict_text = (
            "The vanilla control's bootstrap 95% CI includes 0.5 and its upper bound excludes "
            "~0.6 on both gold_all and the defect-free rule-3/4 subset. A boring, defect-free "
            "cross-encoder trained the same way ACCORD was, on the same data and split, ALSO "
            "lands at chance. This is the first evidence in the project that the negative result "
            "is not an artifact of ACCORD's thirteen documented defects, and it gives the "
            "no-signal finding external validity beyond this one implementation."
        )
    else:
        interpretation = "INCONCLUSIVE_WIDE_CI"
        verdict_text = (
            "The vanilla control's point estimate is near chance and its CI does not clearly "
            "exclude 0.5, but the CI upper bound is not tight enough to rule out a real "
            "sub-power effect either. With only 12 (gold_all) / 6 (defect-free) positives, the "
            "minimum detectable AUROC at 80% power is well above what a near-chance point "
            "estimate could distinguish from a moderate real effect -- see power_ceiling below. "
            "This evaluation cannot adjudicate between 'the task is genuinely near-impossible at "
            "this AUROC' and 'there may be a real but sub-threshold effect'; only a much larger "
            "gold set could."
        )

    report = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "bootstrap_n_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "per_seed": per_seed_table,
        "auroc_mean_std_across_seeds": auroc_summary,
        "ap_mean_std_across_seeds": ap_summary,
        "top15_tp_per_seed": {"gold_all": top15_vals_all, "gold_defect_free_rule34": top15_vals_df},
        "ensemble_mean_of_seed_scores": ensemble_block,
        "accord_contra_prob_same_299_pairs": accord_block,
        "delong_ensemble_vs_accord": {"gold_all": delong_all, "gold_defect_free_rule34": delong_df},
        "delong_per_seed_vs_accord_gold_all": delong_per_seed,
        "baselines": baselines,
        "power_ceiling": {
            "referenced_file": POWER_REF_PATH,
            "referenced_gold_all_ambiguous_excluded": power_ref,
            "recomputed_same_formula_gold_all_ambiguous_included_n299": power_gold_all_full,
            "recomputed_same_formula_defect_free_rule34_n134": power_defect_free,
        },
        "interpretation": interpretation,
        "verdict": verdict_text,
    }

    out_path = os.path.join(OUT_DIR, "final_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({
        "n_seeds": len(seeds),
        "auroc_mean_std_across_seeds": auroc_summary,
        "ensemble_gold_all": ensemble_block["gold_all"],
        "ensemble_gold_defect_free": ensemble_block["gold_defect_free_rule34"],
        "accord_gold_all": accord_block["gold_all"],
        "delong_ensemble_vs_accord_gold_all": delong_all,
        "delong_ensemble_vs_accord_defect_free": delong_df,
        "interpretation": interpretation,
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
