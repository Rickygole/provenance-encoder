"""C134b: per-fold AUROC for the gold-only nested cross-validation whose pooled
out-of-fold AUROC the thesis reports as 0.3434.

Chapter 4: "$0.3434$, AP $0.0362$, one-sided Mann-Whitney $p = 0.9666$, worse
than a [coin flip]". That 0.3434 is
results/accord/tfidf_permutation/tfidf_permutation.json::task_b.pooled_out_of_fold_auroc,
produced by accord/tfidf_permutation/tfidfperm_run.py::task_b_nested_lopo_cv:
12 folds, each holding out exactly 1 of the 12 positives together with a
disjoint chunk of 20 or 21 negatives, each fold refitting its own
TfidfVectorizer and its own inner-CV-selected C, and then pooling the 253
out-of-fold probabilities into ONE AUROC. Probabilities from 12 separately
fitted models are not on one scale, so pooling them mixes between-fold score
shift into the ranking. That is the artifact; this script measures the quantity
without it.

A fold with 1 positive and 20 or 21 negatives DOES admit an AUROC: it is
P(positive scored above a random negative from that fold), ties at 0.5. So the
per-fold AUROC exists for all 12 folds, and their mean is an estimate of the
same quantity the pooled figure claims to estimate, computed within-fold and
therefore invariant to any monotone per-fold rescaling.

The pooled figure is reproduced first and must match the registered
0.3433609958506224 to 10 decimal places; if it does not, the fold partition is
not the one behind the published number and nothing is written. Fold assignment,
grid, inner CV and RNG tree are taken from tfidfperm_run.py unchanged
(SeedSequence(42).spawn(4)[2] is the fold child).

Local only. Reads credentialed gold text through accord/control/data.py
build_splits, exactly as tfidfperm_run.py does; all stdout/stderr while that
data is in scope is redirected to a log under ~/mimic_local_only/logs. The JSON
written under results/ holds fold-level statistics only: no text, no pair
identifiers, no HADM_ID, no SUBJECT_ID.

    python3 accord/c134_reanalysis/c134b_perfold_auroc.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import time

import numpy as np
from scipy.stats import rankdata, ttest_1samp, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "accord" / "control")):
    if p not in sys.path:
        sys.path.insert(0, p)

from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get(
    "MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = assert_local_only(LOG_DIR / f"c134b_{time.strftime('%Y%m%dT%H%M%S')}.log")

REGISTERED = ROOT / "results" / "accord" / "tfidf_permutation" / "tfidf_permutation.json"
OUT = ROOT / "results" / "accord" / "c134_reanalysis" / "c134b_perfold_auroc.json"

SEED = 42
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def fold_auroc(pos_score: float, neg_scores: np.ndarray) -> float:
    return float((neg_scores < pos_score).mean() + 0.5 * (neg_scores == pos_score).mean())


def rerun_folds(y253, text253, seed_seq_child):
    """tfidfperm_run.task_b_nested_lopo_cv, unchanged, with per-fold detail kept."""
    n = len(y253)
    pos_local = np.where(y253 == 1)[0]
    neg_local = np.where(y253 == 0)[0]
    n_pos, n_neg = len(pos_local), len(neg_local)
    assert n_pos == 12 and n_neg == 241, f"unexpected class counts: {n_pos} pos, {n_neg} neg"

    rng = np.random.default_rng(seed_seq_child)
    pos_order = pos_local[rng.permutation(n_pos)]
    neg_order = neg_local[rng.permutation(n_neg)]

    base_chunk = n_neg // n_pos
    remainder = n_neg % n_pos
    chunk_sizes = [base_chunk + 1 if i < remainder else base_chunk for i in range(n_pos)]
    assert sum(chunk_sizes) == n_neg

    text_arr = np.array(text253, dtype=object)
    oof = np.full(n, np.nan)
    fold_rank_scores = np.full(n, np.nan)
    folds = []
    cursor = 0
    for i in range(n_pos):
        size = chunk_sizes[i]
        test_neg = neg_order[cursor:cursor + size]
        cursor += size
        test_idx = np.concatenate([[pos_order[i]], test_neg])
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        grid = GridSearchCV(pipe, param_grid={"clf__C": C_GRID},
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            scoring="average_precision", n_jobs=1, refit=True)
        grid.fit(text_arr[train_idx].tolist(), y253[train_idx])
        s = grid.predict_proba(text_arr[test_idx].tolist())[:, 1]
        oof[test_idx] = s
        # within-fold rank normalisation, ties averaged, mapped to (0, 1)
        fold_rank_scores[test_idx] = (rankdata(s) - 0.5) / len(s)

        pos_s, neg_s = float(s[0]), s[1:]
        folds.append({
            "fold": i + 1,
            "n_test": int(len(test_idx)),
            "n_test_positives": 1,
            "n_test_negatives": int(len(neg_s)),
            "best_C": float(grid.best_params_["clf__C"]),
            "auroc": fold_auroc(pos_s, neg_s),
            "n_negatives_scored_below_the_positive": int((neg_s < pos_s).sum()),
            "n_negatives_tied_with_the_positive": int((neg_s == pos_s).sum()),
            "rank_of_positive_from_top": int(1 + (neg_s > pos_s).sum()),
            "positive_score": pos_s,
            "negative_score_mean": float(neg_s.mean()),
            "negative_score_min": float(neg_s.min()),
            "negative_score_max": float(neg_s.max()),
        })
    assert not np.isnan(oof).any()
    return oof, fold_rank_scores, folds


def main():
    t0 = time.time()
    reg = json.loads(REGISTERED.read_text())["task_b"]
    with LOG.open("w") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        from data import build_splits  # noqa: E402

        splits = build_splits()
        gold = splits.gold
        y_gold = gold["pharm_pos"].to_numpy()
        g253 = (gold["pharm_label"] != "ambiguous").to_numpy()
        gold_text = (gold["nursing_sentence"].astype(str) + " [SEP] "
                     + gold["discharge_sentence"].astype(str)).tolist()
        y253 = y_gold[g253]
        text253 = [gold_text[i] for i in range(len(gold_text)) if g253[i]]

        ss = np.random.SeedSequence(SEED)
        _, _, child_taskb_fold, _ = ss.spawn(4)
        oof, fold_rank_scores, folds = rerun_folds(y253, text253, child_taskb_fold)

        pooled_auroc = float(roc_auc_score(y253, oof))
        pooled_ap = float(average_precision_score(y253, oof))
        rank_pooled_auroc = float(roc_auc_score(y253, fold_rank_scores))
        rank_pooled_ap = float(average_precision_score(y253, fold_rank_scores))
        fold_sizes_match = [f["n_test"] for f in folds] == reg["fold_test_sizes"]
        best_C_match = [f["best_C"] for f in folds] == reg["fold_best_C"]

    ok = (round(pooled_auroc, 10) == round(reg["pooled_out_of_fold_auroc"], 10)
          and round(pooled_ap, 10) == round(reg["pooled_out_of_fold_ap"], 10))
    if not ok:
        raise SystemExit(
            "ABORT: the rerun does not reproduce the registered pooled figures to 10 dp "
            f"(got auroc={pooled_auroc!r} ap={pooled_ap!r}); the fold partition is not the "
            "one behind the published 0.3434. No datum written.")

    a = np.array([f["auroc"] for f in folds])
    n = len(a)
    mean, sd1 = float(a.mean()), float(a.std(ddof=1))
    se = sd1 / np.sqrt(n)
    t_res = ttest_1samp(a, 0.5)
    try:
        w_stat, w_p = wilcoxon(a - 0.5, alternative="two-sided", zero_method="wilcox")
        w = {"statistic": float(w_stat), "p_two_sided": float(w_p)}
    except ValueError as e:
        w = {"error": str(e)}
    n_above, n_below = int((a > 0.5).sum()), int((a < 0.5).sum())

    out = {
        "what": "per-fold AUROC for the gold-only nested CV whose pooled out-of-fold AUROC "
                "the thesis reports as 0.3434",
        "targets_claim": "accord/thesis/msu_ch4_results.tex line 1096: '$0.3434$, AP $0.0362$, "
                         "one-sided Mann-Whitney $p = 0.9666$, worse than a [coin flip]'",
        "does_a_single_positive_fold_admit_an_auroc": {
            "answer": "yes",
            "why": "AUROC with one positive and k negatives is P(positive above a random "
                   "negative of that fold), ties at 0.5; it is defined for every k >= 1. "
                   "Each of the 12 folds here holds 1 positive and 20 or 21 negatives.",
            "granularity_per_fold": "1/20 or 1/21, so a single fold's AUROC is a coarse "
                                    "statistic; the mean over 12 folds is not.",
        },
        "reproduction_of_the_published_pooled_figures": {
            "pooled_out_of_fold_auroc_recomputed": pooled_auroc,
            "pooled_out_of_fold_auroc_registered": reg["pooled_out_of_fold_auroc"],
            "pooled_out_of_fold_ap_recomputed": pooled_ap,
            "pooled_out_of_fold_ap_registered": reg["pooled_out_of_fold_ap"],
            "matched_to_10_decimals": ok,
            "fold_test_sizes_match_registered": fold_sizes_match,
            "fold_best_C_match_registered": best_C_match,
        },
        "per_fold": folds,
        "per_fold_summary": {
            "n_folds": n,
            "auroc_values": [float(x) for x in a],
            "mean": mean,
            "sd_ddof1": sd1,
            "sem": float(se),
            "ci95_normal_on_the_mean": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
            "median": float(np.median(a)),
            "min": float(a.min()),
            "max": float(a.max()),
            "n_folds_above_0_5": n_above,
            "n_folds_below_0_5": n_below,
            "n_folds_at_0_5": int(n - n_above - n_below),
            "t_test_vs_0_5": {"t": float(t_res.statistic), "p_two_sided": float(t_res.pvalue),
                              "df": n - 1,
                              "caveat": "12 folds, bounded statistic, folds share 241 negatives "
                                        "pairwise-disjointly but share training rows, so they are "
                                        "not independent; read as descriptive"},
            "wilcoxon_signed_rank_vs_0_5": w,
        },
        "near_constant_predictor_diagnostic": {
            "why": "if a fold's fitted model returns nearly the same probability for every "
                   "held-out row, that fold's AUROC ranks floating-point noise, not signal, "
                   "and neither the pooled figure nor the per-fold mean is interpretable as "
                   "discrimination",
            "per_fold_score_span": [
                {"fold": f["fold"], "best_C": f["best_C"],
                 "span_over_the_20_or_21_negatives":
                     f["negative_score_max"] - f["negative_score_min"],
                 "distance_positive_to_negative_mean":
                     f["positive_score"] - f["negative_score_mean"]}
                for f in folds],
            "n_folds_with_negative_score_span_below_1e_3": int(sum(
                1 for f in folds if f["negative_score_max"] - f["negative_score_min"] < 1e-3)),
            "n_folds_with_negative_score_span_below_1e_4": int(sum(
                1 for f in folds if f["negative_score_max"] - f["negative_score_min"] < 1e-4)),
            "n_folds_selecting_the_smallest_C_0_001": int(sum(
                1 for f in folds if f["best_C"] == 0.001)),
            "gold_base_rate_in_a_training_fold": float(11.0 / 232.0),
        },
        "alternative_2_within_fold_rank_normalised_pooling": {
            "description": "each fold's out-of-fold probabilities replaced by their within-fold "
                           "mid-ranks mapped to (0,1) before pooling, which removes between-fold "
                           "score shift while keeping a single pooled AUROC",
            "pooled_auroc": rank_pooled_auroc,
            "pooled_ap": rank_pooled_ap,
        },
        "verdict": {
            "pooled_figure": reg["pooled_out_of_fold_auroc"],
            "mean_per_fold_auroc": mean,
            "rank_normalised_pooled_auroc": rank_pooled_auroc,
            "statement": "the pooled 0.3434 is not a like-for-like estimate of the within-fold "
                         "ranking; the defensible alternatives are the mean per-fold AUROC and "
                         "the rank-normalised pooled AUROC, both reported here",
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"pooled reproduced: {pooled_auroc:.16f} vs registered "
          f"{reg['pooled_out_of_fold_auroc']:.16f}  match={ok}")
    print("per-fold AUROC:", ", ".join(f"{x:.4f}" for x in a))
    print(f"mean {mean:.4f}  sd(ddof=1) {sd1:.4f}  sem {se:.4f}  "
          f"95% CI [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}]  "
          f"t p={t_res.pvalue:.4f}  above/below 0.5: {n_above}/{n_below}")
    print(f"rank-normalised pooled AUROC {rank_pooled_auroc:.4f}  AP {rank_pooled_ap:.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
