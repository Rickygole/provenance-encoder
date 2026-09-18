"""C107: TF-IDF permutation test (task a) and a gold-only nested-CV baseline on
G_253 (task b). See C106... no -- see C107_PREREGISTRATION.md.

Local only. Reads credentialed gold/weak-label data through
accord.control.data.build_splits (imported, not modified). All stdout/stderr
produced while that data is in scope is redirected to a file under
~/mimic_local_only/logs; only the aggregate JSON this script writes under
results/ and a short numeric summary ever reach real stdout.

    python3 accord/tfidf_permutation/tfidfperm_run.py
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys
import time

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "accord" / "control"))

from accord.cloud_paths import assert_local_only  # noqa: E402

MIMIC_LOCAL_ONLY_DIR = pathlib.Path(os.environ.get("MIMIC_LOCAL_ONLY_DIR", pathlib.Path.home() / "mimic_local_only"))
LOG_DIR = MIMIC_LOCAL_ONLY_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = assert_local_only(LOG_DIR / f"tfidfperm_{time.strftime('%Y%m%dT%H%M%S')}.log")

OUT = ROOT / "results" / "accord" / "tfidf_permutation" / "tfidf_permutation.json"
REGISTERED_BASELINES = ROOT / "results" / "accord" / "control" / "baselines.json"
FACTS = ROOT / "results" / "accord" / "VERIFIED_FACTS.json"

B_PERM = 10000
SEED = 42
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def perm_null(y: np.ndarray, s: np.ndarray, B: int, seed) -> dict:
    """One-sided label-permutation null holding scores s fixed."""
    rng = np.random.default_rng(seed)
    n = len(y)
    obs_auc = float(roc_auc_score(y, s))
    obs_ap = float(average_precision_score(y, s))
    null_auc = np.empty(B)
    null_ap = np.empty(B)
    idx = np.arange(n)
    for b in range(B):
        yp = y[rng.permutation(idx)]
        null_auc[b] = roc_auc_score(yp, s)
        null_ap[b] = average_precision_score(yp, s)
    return {
        "n": n,
        "n_pos": int(y.sum()),
        "observed_auroc": obs_auc,
        "observed_ap": obs_ap,
        "null_auroc_mean": float(null_auc.mean()),
        "null_ap_mean": float(null_ap.mean()),
        "p_permutation_auroc_one_sided": float((null_auc >= obs_auc).mean()),
        "p_permutation_ap_one_sided": float((null_ap >= obs_ap).mean()),
        "B": B,
    }


def task_b_nested_lopo_cv(y253: np.ndarray, text253: list, seed_seq_child) -> dict:
    n = len(y253)
    pos_local = np.where(y253 == 1)[0]
    neg_local = np.where(y253 == 0)[0]
    n_pos, n_neg = len(pos_local), len(neg_local)
    assert n_pos == 12 and n_neg == 241, f"unexpected class counts: {n_pos} pos, {n_neg} neg"

    rng = np.random.default_rng(seed_seq_child)
    pos_order = pos_local[rng.permutation(n_pos)]
    neg_order = neg_local[rng.permutation(n_neg)]

    base_chunk = n_neg // n_pos          # 20
    remainder = n_neg % n_pos            # 1
    chunk_sizes = [base_chunk + 1 if i < remainder else base_chunk for i in range(n_pos)]
    assert sum(chunk_sizes) == n_neg

    text_arr = np.array(text253, dtype=object)
    oof_scores = np.full(n, np.nan)
    fold_best_C = []
    fold_test_sizes = []

    cursor = 0
    for i in range(n_pos):
        size = chunk_sizes[i]
        test_neg = neg_order[cursor:cursor + size]
        cursor += size
        test_idx = np.concatenate([[pos_order[i]], test_neg])
        train_idx = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)

        train_text = text_arr[train_idx].tolist()
        train_y = y253[train_idx]
        test_text = text_arr[test_idx].tolist()

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipe, param_grid={"clf__C": C_GRID}, cv=inner_cv,
            scoring="average_precision", n_jobs=1, refit=True,
        )
        grid.fit(train_text, train_y)
        scores = grid.predict_proba(test_text)[:, 1]
        oof_scores[test_idx] = scores
        fold_best_C.append(float(grid.best_params_["clf__C"]))
        fold_test_sizes.append(int(len(test_idx)))

    assert not np.isnan(oof_scores).any(), "some row never scored out-of-fold"

    auroc = float(roc_auc_score(y253, oof_scores))
    ap = float(average_precision_score(y253, oof_scores))
    mw_stat, mw_p = mannwhitneyu(
        oof_scores[y253 == 1], oof_scores[y253 == 0], alternative="greater"
    )

    return {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "grid": C_GRID,
        "fold_best_C": fold_best_C,
        "fold_test_sizes": fold_test_sizes,
        "pooled_out_of_fold_auroc": auroc,
        "pooled_out_of_fold_ap": ap,
        "mannwhitney_u": float(mw_stat),
        "mannwhitney_p_one_sided_greater": float(mw_p),
        "_oof_scores": oof_scores,  # consumed by caller for the permutation null, stripped before write
        "_y253": y253,
    }


def run() -> dict:
    from data import build_splits  # noqa: E402 (imported inside redirected scope)

    splits = build_splits()
    y_train = splits.train["y"].to_numpy()
    gold = splits.gold
    y_gold = gold["pharm_pos"].to_numpy()
    g299_mask = np.ones_like(y_gold, dtype=bool)
    g253_mask = (gold["pharm_label"] != "ambiguous").to_numpy()

    train_text = (
        splits.train["nursing_sentence"].astype(str) + " [SEP] " + splits.train["discharge_sentence"].astype(str)
    ).tolist()
    gold_text = (
        gold["nursing_sentence"].astype(str) + " [SEP] " + gold["discharge_sentence"].astype(str)
    ).tolist()

    # ---- task (a): reproduce the registered baseline exactly ----
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_text)
    X_gold = vec.transform(gold_text)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    scores_gold = clf.predict_proba(X_gold)[:, 1]

    registered = json.loads(REGISTERED_BASELINES.read_text())["tfidf_logreg"]["gold_all"]
    auroc_299 = float(roc_auc_score(y_gold[g299_mask], scores_gold[g299_mask]))
    ap_299 = float(average_precision_score(y_gold[g299_mask], scores_gold[g299_mask]))
    if round(auroc_299, 4) != round(registered["auroc"], 4) or round(ap_299, 4) != round(registered["ap"], 4):
        raise SystemExit(
            "ABORT: reproduction does not match results/accord/control/baselines.json to 4 decimals "
            f"(got auroc={auroc_299:.6f} ap={ap_299:.6f}, registered auroc={registered['auroc']:.6f} "
            f"ap={registered['ap']:.6f}). No datum written."
        )

    ss = np.random.SeedSequence(SEED)
    child_299, child_253, child_taskb_fold, child_taskb_perm = ss.spawn(4)

    perm_299 = perm_null(y_gold[g299_mask], scores_gold[g299_mask], B_PERM, child_299)
    perm_253 = perm_null(y_gold[g253_mask], scores_gold[g253_mask], B_PERM, child_253)

    # ---- task (b): gold-only nested leave-one-positive-out CV on G_253 ----
    y253 = y_gold[g253_mask]
    text253 = [gold_text[i] for i in range(len(gold_text)) if g253_mask[i]]
    task_b = task_b_nested_lopo_cv(y253, text253, child_taskb_fold)
    oof_scores = task_b.pop("_oof_scores")
    y253_check = task_b.pop("_y253")
    task_b["permutation_null"] = perm_null(y253_check, oof_scores, B_PERM, child_taskb_perm)

    mde = json.loads(FACTS.read_text())["mde_recomputed.gold_ambiguous_excluded.minimum_detectable_auc"]["value"]

    decision_a = "A" if perm_299["p_permutation_ap_one_sided"] < 0.05 else "B"
    decision_b = "A" if (task_b["pooled_out_of_fold_auroc"] >= mde and task_b["mannwhitney_p_one_sided_greater"] < 0.05) else "B"

    result = {
        "seed": SEED,
        "reproduction_check": {
            "auroc_g299_recomputed": auroc_299,
            "ap_g299_recomputed": ap_299,
            "auroc_g299_registered": registered["auroc"],
            "ap_g299_registered": registered["ap"],
            "matched_to_4_decimals": True,
        },
        "task_a": {
            "g299": perm_299,
            "g253": perm_253,
            "auroc_g253": perm_253["observed_auroc"],
            "ap_g253": perm_253["observed_ap"],
            "note": "g253 scored by the SAME fitted classifier as g299/g134, no refit -- the run the thesis says never happened.",
            "decision": decision_a,
            "decision_rule": "A iff p_permutation_ap_one_sided on G_299 < 0.05",
        },
        "task_b": {**task_b, "mde_auroc_reference": mde, "decision": decision_b,
                    "decision_rule": "A iff pooled_out_of_fold_auroc >= mde_auroc_reference AND mannwhitney_p_one_sided_greater < 0.05"},
        "n_train": int(len(splits.train)),
        "n_gold": int(len(gold)),
        "n_g253": int(g253_mask.sum()),
    }
    return result


def main():
    with LOG_PATH.open("w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        print(f"tfidfperm_run starting {time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = run()
        print("tfidfperm_run finished")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    print(f"log: {LOG_PATH}")
    print(f"task (a) decision: {result['task_a']['decision']} "
          f"(p_ap_g299={result['task_a']['g299']['p_permutation_ap_one_sided']:.4f}, "
          f"p_ap_g253={result['task_a']['g253']['p_permutation_ap_one_sided']:.4f})")
    print(f"task (b) decision: {result['task_b']['decision']} "
          f"(pooled_oof_auroc={result['task_b']['pooled_out_of_fold_auroc']:.4f}, "
          f"mde={result['task_b']['mde_auroc_reference']:.4f}, "
          f"mw_p={result['task_b']['mannwhitney_p_one_sided_greater']:.4f})")


if __name__ == "__main__":
    main()
