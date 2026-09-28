"""C134f: how much of the pooled 0.3434 is the pooling and how much is the estimator.

Chapter 4 reports the gold-only nested cross-validation as an out-of-fold AUROC of
0.3434, "worse than a constant-score floor", with no caveat. Pooling out-of-fold
scores across leave-one-positive-out folds is known to bias AUROC, so an expert
reads 0.3434 as an artifact of the pooling and discounts it. C134b measured the
two pooling-free alternatives. This script decides, from C134b's stored fold
statistics alone, which of two caveats the thesis should carry:

  (A) "pooling made it look bad", or
  (B) "in half the folds the fitted model is near-constant and the ranking is its
      rounding error".

Both are testable here and both are false as stated, in opposite directions:

  - Pooling did not make it look bad. The mean per-fold AUROC and the
    rank-normalised pooled AUROC, the two estimates that no between-fold score
    shift can touch, are BOTH BELOW 0.3434. Removing the pooling lowers the
    number. So the caveat cannot be that pooling depressed it.
  - The near-constant folds are not where the below-chance result comes from.
    Stratifying the 12 folds by their own out-of-fold score span puts the folds
    with a degenerate span NEAREST chance and the folds with real score spread
    FURTHEST below it. The rank correlation between a fold's score span and its
    AUROC is reported so the direction is not asserted from six numbers.

Input is results/accord/c134_reanalysis/c134b_perfold_auroc.json, a committed
artifact of fold-level statistics. No MIMIC data is read here, no identifier is
emitted, and nothing is recomputed from note text.

    python3 accord/c134_reanalysis/c134f_pooling_vs_estimator.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "accord" / "c134_reanalysis" / "c134b_perfold_auroc.json"
REGISTERED = ROOT / "results" / "accord" / "tfidf_permutation" / "tfidf_permutation.json"
OUT = ROOT / "results" / "accord" / "c134_reanalysis" / "c134f_pooling_vs_estimator.json"

DEGENERATE_SPAN = 1e-3
WIDE_SPAN = 1e-2


def stratum(name, folds, aurocs):
    a = np.asarray([aurocs[f - 1] for f in folds], dtype=float)
    return {
        "stratum": name,
        "folds": [int(f) for f in folds],
        "n_folds": int(a.size),
        "auroc_values": [float(x) for x in a],
        "mean_auroc": float(a.mean()) if a.size else None,
        "median_auroc": float(np.median(a)) if a.size else None,
    }


def main():
    b = json.loads(SRC.read_text())
    reg = json.loads(REGISTERED.read_text())["task_b"]

    pooled = b["reproduction_of_the_published_pooled_figures"][
        "pooled_out_of_fold_auroc_recomputed"]
    assert abs(pooled - reg["pooled_out_of_fold_auroc"]) < 1e-12, (
        "C134b's reproduction no longer matches the registered pooled figure")
    per_fold = b["per_fold_summary"]
    aurocs = per_fold["auroc_values"]
    assert len(aurocs) == 12, len(aurocs)

    spans = {d["fold"]: d["span_over_the_20_or_21_negatives"]
             for d in b["near_constant_predictor_diagnostic"]["per_fold_score_span"]}
    gaps = {d["fold"]: d["distance_positive_to_negative_mean"]
            for d in b["near_constant_predictor_diagnostic"]["per_fold_score_span"]}
    rank_pooled = b["alternative_2_within_fold_rank_normalised_pooling"]["pooled_auroc"]
    mean_per_fold = per_fold["mean"]

    degenerate = sorted(f for f, s in spans.items() if s < DEGENERATE_SPAN)
    informative = sorted(f for f, s in spans.items() if s >= DEGENERATE_SPAN)
    wide = sorted(f for f, s in spans.items() if s >= WIDE_SPAN)

    s_vals = [spans[f] for f in range(1, 13)]
    rho, p_rho = spearmanr(s_vals, aurocs)

    # if the degenerate folds are read as carrying no ranking information, their
    # AUROC has expectation 0.5; this is what the per-fold mean becomes under
    # that reading, and it is the strongest version of caveat (B)
    as_if_noise = float(np.mean([0.5 if f in degenerate else aurocs[f - 1]
                                 for f in range(1, 13)]))

    n_pos_below_neg_mean = int(sum(1 for f in range(1, 13) if gaps[f] < 0))

    strata = [
        stratum(f"degenerate out-of-fold score span, below {DEGENERATE_SPAN:g}",
                degenerate, aurocs),
        stratum(f"non-degenerate out-of-fold score span, at or above {DEGENERATE_SPAN:g}",
                informative, aurocs),
        stratum(f"wide out-of-fold score span, at or above {WIDE_SPAN:g}", wide, aurocs),
    ]

    caveat_a_holds = (mean_per_fold > pooled) and (rank_pooled > pooled)
    degen_mean = strata[0]["mean_auroc"]
    info_mean = strata[1]["mean_auroc"]
    caveat_b_holds = degen_mean is not None and info_mean is not None and \
        abs(degen_mean - 0.5) > abs(info_mean - 0.5)

    out = {
        "what": "decomposition of the pooled leave-one-positive-out AUROC of 0.3434 into "
                "the part attributable to pooling and the part attributable to the "
                "estimator, from C134b's stored fold statistics",
        "source_artifact": str(SRC.relative_to(ROOT)),
        "targets_claim": "Chapter 4 reports the gold-only nested cross-validation as "
                         "'0.3434, AP 0.0362, one-sided Mann-Whitney p = 0.9666, worse "
                         "than a constant-score floor' with no caveat",
        "three_estimates_of_the_same_quantity": {
            "pooled_out_of_fold_auroc": pooled,
            "mean_per_fold_auroc": mean_per_fold,
            "rank_normalised_pooled_auroc": rank_pooled,
            "per_fold_sd_ddof1": per_fold["sd_ddof1"],
            "per_fold_sem": per_fold["sem"],
            "per_fold_ci95_normal": per_fold["ci95_normal_on_the_mean"],
            "per_fold_min": per_fold["min"],
            "per_fold_max": per_fold["max"],
            "n_folds_above_0_5": per_fold["n_folds_above_0_5"],
            "n_folds_below_0_5": per_fold["n_folds_below_0_5"],
        },
        "how_much_is_the_pooling": {
            "pooled_minus_mean_per_fold": pooled - mean_per_fold,
            "pooled_minus_rank_normalised_pooled": pooled - rank_pooled,
            "direction": "pooling RAISES the figure",
            "reading": "both pooling-free estimates are below 0.3434, so the pooling "
                       "flatters the estimator rather than depressing it; a reader who "
                       "discounts 0.3434 as a pooling artifact is discounting it in the "
                       "wrong direction",
        },
        "how_much_is_a_degenerate_estimator": {
            "degenerate_span_threshold": DEGENERATE_SPAN,
            "n_folds_degenerate": len(degenerate),
            "n_folds_non_degenerate": len(informative),
            "per_stratum": strata,
            "spearman_fold_score_span_vs_fold_auroc": {
                "rho": float(rho), "p_two_sided": float(p_rho), "n": 12,
                "caveat": "12 folds, each AUROC on a 1/20 or 1/21 grid, folds not "
                          "independent; read as descriptive",
            },
            "per_fold_mean_if_degenerate_folds_are_read_as_chance": as_if_noise,
            "n_folds_positive_scored_below_the_negative_mean": n_pos_below_neg_mean,
            "reading": "the degenerate folds sit NEARER chance than the folds with real "
                       "score spread, so the below-chance result is not the degenerate "
                       "folds' rounding error; it is concentrated in the folds where the "
                       "fitted model does separate rows",
        },
        "which_caveat_is_true": {
            "A_pooling_made_it_look_bad": caveat_a_holds,
            "B_half_the_folds_are_constant_predictors_and_the_ranking_is_rounding_error":
                caveat_b_holds,
            "verdict": "neither as stated. Pooling flatters the number, and the "
                       "near-constant folds are the ones nearest chance. What the "
                       "estimator shows is a consistent below-chance ranking that "
                       "survives both pooling-free estimates and is strongest where the "
                       "fitted model is least degenerate.",
        },
        "sentence_the_thesis_should_carry":
            "That pooled figure is not the only estimate available, and it is the "
            "most favourable one: the mean of the 12 per-fold AUROCs is "
            f"{mean_per_fold:.4f} and pooling within-fold mid-ranks instead of raw "
            f"probabilities gives {rank_pooled:.4f}, so the pooling raises the figure "
            "rather than depressing it. Half the folds fit a near-constant model, with "
            "an out-of-fold score span below 0.001 over their 20 or 21 negatives, but "
            f"those folds average {degen_mean:.4f} and the folds with real score spread "
            f"average {info_mean:.4f}, so the below-chance ranking is not those folds' "
            "rounding error. At 12 positives and one positive per fold the estimate is "
            "too coarse to support a claim of inverse discrimination; what it does not "
            "support is any claim of discrimination.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"pooled {pooled:.4f}  mean per-fold {mean_per_fold:.4f}  "
          f"rank-normalised pooled {rank_pooled:.4f}")
    for s in strata:
        print(f"{s['stratum']:62s} n={s['n_folds']:2d} mean AUROC {s['mean_auroc']:.4f}")
    print(f"spearman(span, auroc) rho={rho:.4f} p={p_rho:.4f}")
    print(f"per-fold mean if degenerate folds read as chance: {as_if_noise:.4f}")
    print(f"caveat A holds: {caveat_a_holds}   caveat B holds: {caveat_b_holds}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
